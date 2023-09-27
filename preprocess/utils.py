# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging
import os

from PIL import Image
from torch.utils.data import Dataset

from noc.dataset.transforms import CLIPTransform
from noc.models.clip.simple_tokenizer import get_clip_tokenizer


class BasicDataset(Dataset):
    """
    Dataset for getting cosine similarities for all image-text pairs, which is used for dataset filtering and constructing the buecket file.
    """

    def __init__(self, cfg):
        self.transform = CLIPTransform(
            "val", resolution=cfg.resolution, clip_resolution=cfg.clip_resolution
        )
        self.img_dir = cfg.image_dir

        # for clip text encoder
        self.clip_text_max_len = cfg.clip_text_max_len
        self.clip_tokenizer = get_clip_tokenizer()

        ann_path = cfg.annotation_path
        self.items = []
        for line in open(ann_path, "r").readlines():  # noqa: SIM115
            toks = line.strip().split("\t")
            assert len(toks) == 2

            imgpath, text = toks[0], toks[1]
            self.items.append((imgpath, text))

        logging.info(f"total items (cc3m): {len(self.items)}")

    def __getitem__(self, item: int):
        img_path, txt = self.items[item]

        # load images
        img = Image.open(os.path.join(self.img_dir, img_path)).convert("RGB")
        img = self.transform(img)

        # tokenize for CLIP text encoder
        clip_token, clip_mask = self.clip_tokenizer.padded_tokens_and_mask(
            [txt], self.clip_text_max_len
        )

        return {
            "img": img,  # input image
            "imgpath": img_path,  # path to input image for debugging
            "clip_token": clip_token,  # token for CLIP text encoder
            "gt_caps": txt,
        }

    def __len__(self):
        return len(self.items)
