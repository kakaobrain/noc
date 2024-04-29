# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import logging
import os
from collections import defaultdict
from itertools import chain

import numpy as np
import pickle5 as pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from torch.distributions.categorical import Categorical

from evaluation.pycocoevalcap.eval import COCOEvalCap
from noc.models.bucketing.bucketing import AlignmentLevelBucket
from noc.models.clip import clip as clip
from noc.models.clip.simple_tokenizer import get_clip_tokenizer
from noc.models.decoder.crossDecoder import CrossDecoder
from noc.models.decoder.tokenizer import load_gpt_tokenizer
from noc.utils import distribute as custom_dist
from nocap_evaluation.evalai import NocapsEvaluator


@torch.no_grad()
def generate_greedy(
    model,
    tokenizer,
    prompt,
    max_new_token=67,  # maximum number of words
    temperature=1.0,
    stochastic=False,
    nucleus_sampling=False,
    top_p=0.8,
    stop_token: str = "<|endoftext|>",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]  # 50256
    filter_value = 0.0
    device = next(model.parameters()).device
    bsz = len(prompt)
    tokens = None
    dones = torch.tensor([False for _ in range(bsz)], dtype=torch.bool, device=device)

    for _i in range(max_new_token):  # max sequence length
        # outputs = model(inputs_embeds=generated)
        # logits = outputs.logits  # [B, 1, vocab]
        if tokens is None:
            tokens = torch.tensor(stop_token_index, device=device).repeat(bsz).unsqueeze(-1)

        caption_lengths = torch.ones_like(tokens).sum(1)
        outputs = model(
            visual_feature=prompt,
            caption_tokens=tokens,
            caption_lengths=caption_lengths,
            isInfer=True,
        )
        logits = outputs
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # [B, vocab]

        if stochastic:
            prob = F.softmax(logits, dim=-1)  # [B, vocab]
            if nucleus_sampling:
                sorted_prob, sorted_indices = torch.sort(prob, descending=True)
                cumulative_probs = torch.cumsum(sorted_prob, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                prob[:, indices_to_remove] = filter_value

            sampler = Categorical(prob)
            next_token = sampler.sample().unsqueeze(1)
        else:
            next_token = torch.argmax(logits, -1).unsqueeze(1)  # [B, 1]

        if stop_token_index is not None:
            is_done = next_token.squeeze(-1) == stop_token_index
            dones = torch.logical_or(dones, is_done)
        if dones.all():
            break

        if tokens is None:
            tokens = next_token
        else:
            tokens = torch.cat((tokens, next_token), dim=1)

    tokens = tokens[:, 1:]  # remove bos token
    if tokens is not None:
        output_list = list(tokens.squeeze(0).cpu().numpy())
        if tokens.shape[0] > 1:
            output_text = tokenizer.batch_decode(output_list)
        else:
            output_text = [tokenizer.decode(output_list)]
    else:
        output_text = [stop_token] * bsz
        return output_text

    # Remove eos token and dummies
    for idx in range(len(output_text)):
        caption = output_text[idx]
        stop_token_idx = caption.find(stop_token)
        output_text[idx] = caption[:stop_token_idx] if stop_token_idx > 0 else caption
    return output_text


class Captioner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_type = cfg.model.type

        # params for prefix embedding
        self.prefix_length = cfg.model.prefix_length
        # params for visual encoder
        self.dense_feat = self.cfg.model.encoder.dense_feat
        self.freeze_clip_param = cfg.model.encoder.freeze_clip_param
        self.avg_pool_scale = self.cfg.model.encoder.avg_pool_scale
        # params for inference
        self.stop_token = self.cfg.model.transformer.stop_token
        # params for misc
        self.ds_name = cfg.dataset.name_val  # validation dataset name

        # build cc decoder using VirTex-like cross attention-based decoder
        self.decoder = CrossDecoder(
            max_caption_length=cfg.dataset.cc_text_max_len,
            **cfg.model.transformer,
        )
        dec_emb_size = cfg.model.transformer.hidden_size
        self.cc_tokenizer = load_gpt_tokenizer("gpt2")

        # build encoder using CLIP model
        self.encoder = self.load_clip_model(**cfg.model.encoder)
        enc_emb_size = self.encoder.text_projection.shape[1]
        self.v_proj = (
            nn.Linear(enc_emb_size, dec_emb_size)
            if (enc_emb_size != dec_emb_size)
            else nn.Identity()
        )  # if embedding sizes are different, we use a linear projection layer.

        # if controllable cpationer
        # build bucketing for alignment level assignment
        self.use_control = self.model_type == "controllable"
        if self.use_control:
            self.clip_tokenizer = get_clip_tokenizer()
            self.clip_text_max_len = cfg.dataset.clip_text_max_len
            with open(cfg.model.bucket_path, "rb") as f:
                bucket_data = pickle.load(f)
            self.bucket = AlignmentLevelBucket(bucket_data)
            self.num_bins = bucket_data.n_bins_[0]
            self.control_embedding = nn.Embedding(self.num_bins, dec_emb_size)
            self.register_buffer(
                "control_signal_at_inference",
                torch.tensor([cfg.model.control_signal_at_inference], dtype=torch.long),
            )

            # for on-the-fly of computing CLIP similarity
            self.clip_sim_onthefly = cfg.model.clip_sim_onthefly
            self.is_shared_encoder = (
                cfg.model.encoder.model_file == cfg.model.clip.model_file
            ) and self.freeze_clip_param
            if self.clip_sim_onthefly:
                if self.is_shared_encoder:
                    # in this case, we use encoder to compute CLIP similarity
                    self.clip = None
                else:
                    self.clip = self.load_clip_model(cfg.model.clip.model_file)

        # for result items
        self.output = []
        self.prediction_filename = None

    def on_train_start(self) -> None:
        if self.global_rank == 0:
            try:
                from noc.utils import main_utils
                main_utils.save_config_to_disk(self.cfg)
            except FileNotFoundError as err:
                logging.info(err)

    def load_clip_model(
        self, model_file, dense_feat=False, avg_pool_scale=1, freeze_clip_param=True
    ):
        clip_model = clip.load(
            model_file, device="cpu", dense_feat=dense_feat, avg_pool_scale=avg_pool_scale
        )[0]

        if freeze_clip_param:
            for param in clip_model.parameters():
                param.requires_grad = False
        else:
            # Freeze only the text modality network
            for param in clip_model.transformer.parameters():
                param.requires_grad = False
            for param in clip_model.token_embedding.parameters():
                param.requires_grad = False
            for param in clip_model.ln_final.parameters():
                param.requires_grad = False
            clip_model.positional_embedding.requires_grad = False
            clip_model.text_projection.requires_grad = False

        return clip_model

    @torch.no_grad()
    def compute_CLIP_similarity(self, img, clip_token, img_emb=None):
        if self.is_shared_encoder:
            # default: we reuse image embedding that is computed for prefix embedding
            if img_emb is None:
                img_emb = self.encoder.encode_image(img)
            if self.dense_feat:
                img_emb = img_emb[:, 0, :]  # Only use cls token for getting alignment scores
            txt_emb = self.encoder.encode_text(clip_token)
        else:
            img_emb = self.clip.encode_image(img)
            txt_emb = self.clip.encode_text(clip_token)

        with torch.cuda.amp.autocast(enabled=False):
            clip_sim = F.cosine_similarity(img_emb.float(), txt_emb.float())  # [B, ]
        return clip_sim

    def get_prefix_embedding(self, img, clip_token, clip_sim=None, inference=False):
        """
        if controllable captioner, prefix embedding is concatenation of img + control embeddings.
        if vanilla captioner, prefix embedding is img embedding.
        """
        with torch.set_grad_enabled((not self.freeze_clip_param) and (self.training)):
            img_emb = self.encoder.encode_image(img)

        if self.use_control and not inference:
            if self.clip_sim_onthefly:
                clip_sim = self.compute_CLIP_similarity(
                    img, clip_token, img_emb=img_emb if self.is_shared_encoder else None
                )
            else:
                assert clip_sim is not None

        img_emb = self.v_proj(img_emb)
        if len(img_emb.shape) < 3:  # for unimodal prefix case.
            img_emb = img_emb[:, None, :]

        if self.use_control:
            if inference:
                # we use given control signal value
                bsz = img_emb.shape[0]
                control_signal = self.control_signal_at_inference.repeat(bsz)
            else:
                # at training time, we compute alignment level with bucketing on CLIP similarity
                alignment_level = self.bucket(clip_sim)
                # then, we use alignment level as control signal
                control_signal = alignment_level

            control_emb = self.control_embedding(control_signal)[:, None, :]
            prefix_emb = torch.cat((img_emb, control_emb), dim=1)
        else:
            prefix_emb = img_emb

        return prefix_emb

    def forward_backward(self, img, cc_token, cc_mask, clip_token, clip_sim=None):
        """we perform forward and then compute loss (i.e., backward)"""
        prefix_emb = self.get_prefix_embedding(img, clip_token, clip_sim, inference=False)

        # decoding captions from the prefix embedding
        with torch.set_grad_enabled(self.training):
            logits = self.decoder(prefix_emb, cc_token, cc_mask)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                cc_token[:, 1:].reshape(-1),
                ignore_index=0,  # 0 means padding
            )

        return loss

    def training_step(self, batch: any, batch_idx: int):
        """infer training batch
        Args (batch of dictionary includes):
            img (Tensor; [B 3 224 224]): raw input image
            clip_token (Tensor; [B num_cc L]): tokens for CLIP text encoder
            clip_mask (Tensor; [B num_cc L]): masks of tokens for CLIP text encoder
            cc_token (Tensor; [B num_cc L]): tokens for captioner
            cc_mask (Tensor; [B num_cc P+L]): masks of prefix+tokens for captioner
            clip_sim (Tensor; [B]): CLIP similarity (optional)
        Returns:
            loss: (Tensor; []): caption generation loss (summation of negative likelihood over batch)
        """
        img = batch["img"]
        cc_token = batch["cc_token"][:, 0]
        cc_mask = batch["cc_mask"][:, 0]
        clip_token = batch["clip_token"][:, 0] if self.use_control else None
        clip_sim = batch["clip_sim"] if self.use_control else None

        # compute loss
        loss = self.forward_backward(img, cc_token, cc_mask, clip_token, clip_sim)

        # Logging
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch: torch.tensor, batch_idx: int):
        """infer validation batch
        Args (batch of dictionary includes):
            img (Tensor; [B 3 224 224]): raw input image
            clip_token (Tensor; [B num_cc L]): tokens for CLIP text encoder
            clip_mask (Tensor; [B num_cc L]): masks of tokens for CLIP text encoder
            cc_token (Tensor; [B num_cc L]): tokens for captioner
            cc_mask (Tensor; [B num_cc P+L]): masks of prefix+tokens for captioner
            clip_sim (Tensor; [B]): CLIP similarity (optional)
        Returns:
            loss: (Tensor; []): caption generation loss (summation of negative likelihood over batch)
        """
        # model forward (feature extraction, loss head) -> loss computation
        img = batch["img"]
        cc_token = batch["cc_token"][:, 0]
        cc_mask = batch["cc_mask"][:, 0]
        clip_token = batch["clip_token"][:, 0] if self.use_control else None
        clip_sim = batch["clip_sim"] if self.use_control else None

        loss = self.forward_backward(img, cc_token, cc_mask, clip_token, clip_sim)

        # Logging
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=img.shape[0],
        )

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def generate_captions(self, prefix):
        return generate_greedy(
            model=self.decoder,
            tokenizer=self.cc_tokenizer,
            prompt=prefix,
            stop_token=self.stop_token,
            max_new_token=self.cfg.dataset.cc_text_max_len,
        )  # [B, string]

    def get_prediction_dir(self):
        ckpt_name = self.cfg.experiment.load_from
        pred_dir = os.path.join(
            self.cfg.experiment.log_path,
            ckpt_name.split("/")[-1][:-5] if len(ckpt_name) > 0 else "",
        )
        os.makedirs(pred_dir, exist_ok=True)
        return pred_dir

    def get_prediction_filename(self):
        postfix = f"greedy_{self.ds_name}"
        if self.model_type == "controllable":
            postfix += f"_bin{self.control_signal_at_inference.item()}"
        filename = f"prediction_{postfix}.json"
        return filename

    def test_step(self, batch: torch.tensor, batch_idx: int):
        # data for inference
        img = batch["img"]  # TorchTensor [B, 3, 224, 224]
        # data for misc such as metric calculation
        gt_caps = batch["gt_caps"]  # nested list of caption num_cc * [B]
        imgpath = batch["imgpath"]  # list of image path [B]
        img_id = batch["img_id"]  # list of img id [B] where string

        # obtain prefix, i.e., image (+ control signal embedding)
        prefix_emb = self.get_prefix_embedding(img, clip_token=None, inference=True)

        # generate captions from the prefix embedding
        caption = self.generate_captions(prefix_emb)

        # save output
        for bidx in range(img.shape[0]):
            # List of list of string [num_caps_per_img, B] -> list of string
            if isinstance(gt_caps[0], list):  # if multiple captions per given image
                gt = []
                for gt_cap in gt_caps:
                    gt.append(gt_cap[bidx])
            else:
                gt = [gt_caps[bidx]]

            # prepare instance-wise result
            res_dict = {
                "gt": gt,
                "caption": [caption[bidx]],
                "image_id": img_id[bidx],
                "imgpath": imgpath[bidx],
            }
            if self.ds_name == "nocap":
                res_dict["domain"] = batch["domain"][bidx]
            self.output.append(res_dict)

        # for debugging
        if batch_idx % 10 == 0:
            logging.info(f"GT       : {gt}")
            logging.info(f"generated: {caption[-1]}")

        return res_dict

    def test_epoch_end(self, test_step_outputs):
        pred_dir = self.get_prediction_dir()
        self.prediction_filename = self.get_prediction_filename()

        custom_dist.synchronize()
        with torch.inference_mode(False):
            # Merging prediction results
            merged_output = custom_dist.gather(data=self.output, dst=0)
            if custom_dist.is_primary():
                for idx in range(1, len(merged_output)):
                    merged_output[0].extend(merged_output[idx])
                # save merged prediction (caption) results
                with open(os.path.join(pred_dir, self.prediction_filename), "w") as f:
                    json.dump(merged_output[0], f, indent=4, ensure_ascii=False)

        custom_dist.synchronize()

        # reset output buffer
        self.output = []

    def inference(self, img: torch.tensor):
        # obtain prefix, i.e., image (+ control signal embedding)
        prefix_emb = self.get_prefix_embedding(img, clip_token=None, inference=True)

        # generate captions from the prefix embedding
        caption = self.generate_captions(prefix_emb)

        return caption

    @torch.no_grad()
    def prediction_for_CLIP_similarity(self, batch):
        """predict CLIP (cosine) similarity for given image and text to generate annotation"""
        # data for inference
        img = batch["img"]  # [B, 3, 224, 224]
        clip_token = batch["clip_token"]  # [B, num_cc, 77]
        # data for misc such as annotation creation
        img_id = batch["img_id"]
        gt_caps = batch["gt_caps"]  # [1, [num_cc, B]]
        imgpath = batch["imgpath"]  # [B]

        bsz, num_cc, token_len = clip_token.shape
        CLIP = self.encoder if self.is_shared_encoder else self.clip
        img_emb = CLIP.encode_image(img)
        img_emb = img_emb[:, None, :].repeat(1, num_cc, 1).view(bsz * num_cc, -1)
        txt_emb = CLIP.encode_text(clip_token.view(bsz * num_cc, -1))  # [B D]

        with torch.cuda.amp.autocast(enabled=False):
            clip_sim = F.cosine_similarity(img_emb.float(), txt_emb.float())
            # for multiple captions, we use averaged CLIP similarity
            clip_sim = clip_sim.view(bsz, num_cc).mean(dim=1)

        anno_dict = []
        for bidx in range(len(img)):
            if isinstance(gt_caps[0], list):  # if multiple captions per given image
                gt = []
                for gt_cap in gt_caps:
                    gt.append(gt_cap[bidx])
            else:
                gt = [gt_caps[bidx]]

            res_temp = {
                "imgpath": imgpath[bidx],
                "img_id": img_id[bidx],
                "gt": gt,
                "clip_sim": clip_sim[bidx].item(),
            }
            anno_dict.append(res_temp)
        return anno_dict

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        anno_dict = self.prediction_for_CLIP_similarity(batch)
        return anno_dict

    def calc_metric(self, pred_dir=None, pred_filename=None):
        # metric calculation for results from all gpus
        metric_scores = defaultdict(lambda: dict())  # noqa: C408
        if self.global_rank == 0:
            if pred_dir is None:
                pred_dir = self.get_prediction_dir()
            if pred_filename is None:
                pred_filename = self.get_prediction_filename()

            annotation_key = "img_id" if self.ds_name == "coco" else "enum"

            # compute scores of captioning metrics using COCO evaluator
            cocoEval = COCOEvalCap(
                os.path.join(pred_dir, pred_filename),
                annotation_key=annotation_key,
                metric=["CLIP"] if self.ds_name == "nocap" else ["B", "M", "S", "C", "CLIP"],
            )
            cocoEval.evaluate()
            print("Total predictions: %d" % (cocoEval.num_total_preds))

            # compute scores for language metrics
            if self.ds_name == "nocap":
                # for CLIPScore, we need to manually compute domain-wise scores
                tmp_score = {
                    "in-domain": [],
                    "near-domain": [],
                    "out-domain": [],
                    "entire": [],
                }
                with open(os.path.join(pred_dir, pred_filename), "r") as f:
                    predictions = json.load(f)
                for imgid, score in cocoEval.imgToEval.items():
                    domain = predictions[imgid]["domain"]
                    tmp_score[domain].append(score["CLIPScore"])
                    tmp_score["entire"].append(score["CLIPScore"])
                for domain in ["in-domain", "near-domain", "out-domain", "entire"]:
                    metric_scores[domain]["CLIPScore"] = float(np.mean(tmp_score[domain]))

                # compute other metrics using nocap official evaluator
                formatted_predictions = [
                    {"image_id": int(it["image_id"]), "caption": it["caption"][0]}
                    for it in predictions
                ]
                nocapEval = NocapsEvaluator("val")
                logging.info(f"   >>>> Evaluation for {os.path.join(pred_dir, pred_filename)}")
                eval_metrics = nocapEval.evaluate(formatted_predictions)

                for metric in eval_metrics:
                    for domain in eval_metrics[metric]:
                        metric_scores[domain][metric] = float(eval_metrics[metric][domain])

                logging.info(f"   >>>  {os.path.join(pred_dir, pred_filename)}")
                logging.info(metric_scores)

            else:
                for metric, score in cocoEval.eval.items():
                    print("%s: %.3f" % (metric, score))
                    metric_scores[metric] = float(score)

        return metric_scores

    def set_optim_hyperparam(self, named_params, weight_decay, lr):
        params = []
        excluded_params = []
        skip_list = []

        if not self.cfg.optimizer.regularize_bn:
            skip_list.append("bn")
        if not self.cfg.optimizer.regularize_bias:
            skip_list.append("bias")

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay, "lr": lr},
            {"params": excluded_params, "weight_decay": 0.0, "lr": lr},
        ]

    def configure_optimizers(self):
        # exclude weight decay and set lr for visual encoder
        weight_decay_enc = self.cfg.optimizer.optim_cfg_enc.weight_decay
        lr_enc = self.cfg.optimizer.optim_cfg_enc.base_lr
        params_enc = self.set_optim_hyperparam(
            chain(self.encoder.named_parameters(), self.v_proj.named_parameters()),
            weight_decay=weight_decay_enc,
            lr=lr_enc,
        )

        # exclude weight decay and set lr for caption decoder
        weight_decay_dec = self.cfg.optimizer.optim_cfg_dec.weight_decay
        lr_dec = self.cfg.optimizer.optim_cfg_dec.base_lr
        params_dec = self.set_optim_hyperparam(
            self.decoder.named_parameters(), weight_decay=weight_decay_dec, lr=lr_dec
        )

        # optimizer
        model_params = params_enc + params_dec
        if self.cfg.optimizer.name == "sgd":
            optimizer = torch.optim.SGD(
                model_params, lr=self.cfg.optimizer.optim_cfg_enc.base_lr, momentum=0.9
            )
        elif self.cfg.optimizer.name == "adam":
            optimizer = torch.optim.Adam(model_params, lr=self.cfg.optimizer.optim_cfg_enc.base_lr)
        elif self.cfg.optimizer.name == "adamW":
            optimizer = torch.optim.AdamW(model_params, lr=self.cfg.optimizer.optim_cfg_enc.base_lr)
        else:
            raise ValueError()

        if self.cfg.experiment.training_mode == "epoch":
            total_steps = int(
                self.cfg.experiment.train_iters_per_epoch * self.cfg.experiment.max_epochs
            )
        else:  # for iter-based training
            total_steps = self.cfg.trainer.max_steps

        # scheduler
        warmup_steps = int(total_steps * self.cfg.optimizer.scheduler.warmup)
        if self.cfg.optimizer.scheduler.name == "cosine_with_linear_warmup":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    linear_warmup_decay(warmup_steps, total_steps, cosine=True),
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise NotImplementedError

        return [optimizer], [scheduler]
