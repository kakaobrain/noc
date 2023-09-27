# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import logging

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from noc.models.clip import clip as clip
from noc.models.clip.simple_tokenizer import get_clip_tokenizer


class RetrievalModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # build CLIP
        self.clip_conf = cfg.model.clip
        self.clip_text_max_len = cfg.dataset.clip_text_max_len
        self.clip = self.load_clip_model()
        self.clip_tokenizer = get_clip_tokenizer()

    def load_clip_model(self):
        clip_model = clip.load(
            self.clip_conf.model_file, device="cpu", dense_feat=False, avg_pool_scale=1
        )[0]
        return clip_model

    def test_step(self, batch: torch.tensor, batch_idx: int):
        # model forward (feature extraction) -> metric computation
        img = batch["img"]  # [B, 3, 224, 224]
        clip_token = batch["clip_token"]  # [B, 1, clip_text_max_len(=77)]
        imgpath = batch["imgpath"]

        img_emb = self.clip.encode_image(img)  # [B, dim]
        txt_emb = self.clip.encode_text(clip_token[:, 0])  # [B, dim]

        # following result will be accumulated in test_step_outputs as list
        res_dict = {
            "imgs": img_emb.cpu(),
            "txts": txt_emb.cpu(),
            "img_ids": [idx.item() for idx in batch["imgid"]],
            "img_paths": imgpath,
        }
        return res_dict

    def test_epoch_end(self, test_step_outputs):
        # Note that we assume a single gpu is used for evaluation
        imgs = torch.vstack([output["imgs"] for output in test_step_outputs])
        txts = torch.vstack([output["txts"] for output in test_step_outputs])
        img_ids = sum([output["img_ids"] for output in test_step_outputs], [])
        img_paths = sum([output["img_paths"] for output in test_step_outputs], [])

        # compute similarity matrix
        imgs = F.normalize(imgs, dim=1)  # [N, D]
        txts = F.normalize(txts, dim=1)  # [N, D]
        t2i_sim = txts @ imgs.t()  # [N, N]
        logging.info(f"sim matrix: {t2i_sim.shape}")

        # compute recall scores
        gt = torch.arange(0, t2i_sim.shape[0])[:, None]
        top_idx = torch.argsort(-t2i_sim, dim=1)
        correct = top_idx.cpu() == gt
        metrics = {}
        metrics["R1"] = torch.mean((torch.sum(correct[:, :1], dim=1) >= 1).float()).item() * 100
        metrics["R5"] = torch.mean((torch.sum(correct[:, :5], dim=1) >= 1).float()).item() * 100
        metrics["R10"] = torch.mean((torch.sum(correct[:, :10], dim=1) >= 1).float()).item() * 100

        ind = (correct == 1).nonzero()  # (row, col)
        metrics["MR"] = (torch.median(ind[:, 1]) + 1).item()

        # save scores
        modelname = self.clip_conf.model_file.replace("/", "-")
        save_to = self.cfg.dataset.pred_file_path[:-5] + f"_retrieval_score_{modelname}.json"
        logging.info(json.dumps(metrics, indent=4))
        with open(save_to, "w") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        logging.info(f"saved to {save_to}")

        # save retrieval results
        retrieval_result = []
        for i in range(len(top_idx)):
            cur_res = {
                "img_id": img_ids[i],
                "img_path": img_paths[i],
                "retrieval_idx": top_idx[i][:100].tolist(),
            }
            retrieval_result.append(cur_res)
        save_to = self.cfg.dataset.pred_file_path[:-5] + f"_retrieval_result_{modelname}.json"
        with open(save_to, "w") as f:
            json.dump(retrieval_result, f, indent=4, ensure_ascii=False)
        logging.info(f"saved to {save_to}")
