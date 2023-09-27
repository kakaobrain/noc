# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import logging

from PIL import Image
from torch.utils.data import Dataset

from noc.dataset.transforms import build_transform
from noc.models.clip.simple_tokenizer import get_clip_tokenizer


class SelfRetrievalDataset(Dataset):
    """
    Dataset for self-retrieval, which works based on output file of caption generation
    """

    def __init__(
        self,
        cfg: dict,
        **ignore_kwargs,
    ):
        self.transform = build_transform("val", cfg.dataset.transform_hparams)

        # for clip text encoder
        self.clip_text_max_len = cfg.dataset.clip_text_max_len
        self.clip_tokenizer = get_clip_tokenizer()

        # load prediction file that is generated one after caption generation.
        assert len(cfg.dataset.pred_file_path) > 0
        with open(cfg.dataset.pred_file_path, "r") as f:
            self.items = json.load(f)

        logging.info(f"total items (retrieval): {len(self.items)}")

    def __getitem__(self, idx: int):
        instance = self.items[idx]

        # load image
        imgpath = instance["imgpath"]
        img = Image.open(imgpath).convert("RGB")
        img = self.transform(img)

        # prepare text token
        if "generated" in instance:
            txt = instance["generated"]  # list that contains single sentence
        elif "caption" in instance:
            txt = instance["caption"]  # list that contains single sentence
        else:
            raise NotImplementedError("Generated captions should be given.")
        clip_token, clip_mask = self.clip_tokenizer.padded_tokens_and_mask(
            txt, self.clip_text_max_len
        )

        return {
            "imgid": idx,
            "img": img,
            "clip_token": clip_token,
            "clip_mask": clip_mask,
            "imgpath": imgpath,
        }

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)
