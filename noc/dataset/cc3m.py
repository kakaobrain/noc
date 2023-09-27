# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from noc.dataset.transforms import build_transform
from noc.models.clip.simple_tokenizer import get_clip_tokenizer
from noc.models.decoder.tokenizer import GPTTokenizer


class CC3MDataset(Dataset):
    def __init__(
        self,
        cfg: dict,
        split: str,
        **ignore_kwargs,
    ):
        self.split = split
        self.transform = build_transform(split, cfg.dataset.transform_hparams)

        self.filtering_th = cfg.dataset.filtering_th
        self.clip_sim_onthefly = cfg.model.clip_sim_onthefly
        self.img_dir = cfg.dataset.image_dir

        # for clip text encoder
        self.clip_text_max_len = cfg.dataset.clip_text_max_len
        self.clip_tokenizer = get_clip_tokenizer()

        # for controllable captioner
        self.prefix_length = cfg.model.prefix_length
        self.cc_text_max_len = cfg.dataset.cc_text_max_len
        self.cc_tokenizer = GPTTokenizer(self.prefix_length, self.cc_text_max_len)

        self.items = []
        if split == "train":
            if self.filtering_th > 0:
                # currently support only 0.3
                ann_path = f"data/cc3m/train_list_filtered_by_{self.filtering_th}.txt"
            else:
                ann_path = "data/cc3m/train_list.txt"
        else:
            ann_path = "data/cc3m/val_list.txt"

        with open(ann_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                toks = line.strip().split("\t")
                assert len(toks) in [2, 3]

                imgpath, text = toks[0], toks[1]
                clip_sim = toks[2] if len(toks) == 3 else -1

                self.items.append((os.path.join(self.img_dir, imgpath), text, clip_sim))

        logging.info(f"total items (cc3m) / ({split}): {len(self.items)}")

    def __getitem__(self, item: int):
        imgpath, txt, clip_sim = self.items[item]

        # load images
        img = Image.open(imgpath).convert("RGB")
        img = self.transform(img)

        # tokenize for CLIP text encoder
        clip_token, clip_mask = self.clip_tokenizer.padded_tokens_and_mask(
            [txt], self.clip_text_max_len
        )

        # tokenize for captioner
        cc_token, cc_mask = self.cc_tokenizer(txt)
        cc_token, cc_mask = cc_token.unsqueeze(0), cc_mask.unsqueeze(0)  # [1, L]

        img_id = imgpath.split("/")[-1].split(".")[0]
        txt = [txt]

        return {
            "img": img,  # input image
            "imgpath": imgpath,  # path to input image for debugging
            "img_id": str(img_id),
            "clip_token": clip_token,  # token for CLIP text encoder
            "clip_mask": clip_mask,  # token mask for CLIP text encoder
            "cc_token": cc_token,  # token for captioner
            "cc_mask": cc_mask,  # token mask for captioner
            "gt_caps": txt,
            "clip_sim": torch.tensor(float(clip_sim)),
            "ds_name": "cc3m",
        }

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)
