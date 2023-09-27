# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging
import os
import random
from typing import Any, List, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO  # this is pip-installed version of official COCO
from torchvision.datasets import VisionDataset

from noc.dataset.transforms import build_transform
from noc.models.clip.simple_tokenizer import get_clip_tokenizer
from noc.models.decoder.tokenizer import GPTTokenizer


class COCOCaptionDataset(VisionDataset):
    def __init__(
        self,
        cfg: dict,
        split: str,
        **ignore_kwargs,
    ) -> None:
        # basically, split == "test", i.e., used for evaluation on COCO
        if split == "train":
            self.img_dir = os.path.join(cfg.dataset.image_dir, "train2017")
        elif split == "val":
            self.img_dir = os.path.join(cfg.dataset.image_dir, "val2017")
        elif split == "test":  # karpathy test split
            self.img_dir = os.path.join(cfg.dataset.image_dir, "val2014")
        else:
            raise NotImplementedError
        super().__init__(self.img_dir)

        self.transform = build_transform(split, cfg.dataset.transform_hparams)
        self.coco = COCO(f"data/coco/captions_{split}2017.json")
        self.ids = list(sorted(self.coco.imgs.keys()))  # noqa: C413
        self.split = split

        # for clip text encoder
        self.clip_text_max_len = cfg.dataset.clip_text_max_len
        self.clip_tokenizer = get_clip_tokenizer()

        # for controllable captioner
        self.prefix_length = cfg.model.prefix_length
        self.cc_text_max_len = cfg.dataset.cc_text_max_len
        self.cc_tokenizer = GPTTokenizer(self.prefix_length, self.cc_text_max_len)

        logging.info(f"total items (coco) / ({split}): {len(self.ids)}")

    def _load_image(self, id: int) -> Image.Image:
        img_filename = self.coco.loadImgs(id)[0]["file_name"]
        imgpath = os.path.join(self.img_dir, img_filename)
        return Image.open(imgpath).convert("RGB"), imgpath, img_filename

    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in self.coco.loadAnns(self.coco.getAnnIds(id))]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img_id = self.ids[index]

        # load image and apply transform
        img, imgpath, post_path = self._load_image(img_id)
        img = self.transform(img)

        # prepare GT captions (=txt)
        txt = self._load_target(img_id)
        if len(txt) > 5:
            txt = txt[:5]
        elif len(txt) < 5:
            dummy_txt = txt[-1]
            for _idx in range(5 - len(txt)):
                txt.append(dummy_txt)

        # when training, we randomly select one caption to train model
        if self.split == "train":
            rnd_idx = random.randint(0, len(txt) - 1)
            txt = [txt[rnd_idx]]

        instance = {
            "img_id": str(img_id),
            "img": img,  # input image
            "imgpath": imgpath,  # path to input image for debugging
            "gt_caps": txt,
            "ds_name": "coco",
        }

        if self.split == "train":
            # tokenize for captioner
            cc_token, cc_mask = [], []
            for cc in txt:
                temp_token, temp_mask = self.cc_tokenizer(cc)
                cc_token.append(temp_token)
                cc_mask.append(temp_mask)
            cc_token, cc_mask = torch.stack(cc_token), torch.stack(cc_mask)
            instance["cc_token"] = cc_token  # token for captioner
            instance["cc_mask"] = cc_mask  # token mask for captioner

            # tokenize for CLIP text encoder
            clip_token, clip_mask = self.clip_tokenizer.padded_tokens_and_mask(
                txt, self.clip_text_max_len
            )
            instance["clip_token"] = clip_token  # token for CLIP text encoder
            instance["clip_mask"] = clip_mask  # token mask for CLIP text encoder

        return instance

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.ids)
