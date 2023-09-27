# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import logging
import os

from PIL import Image
from torch.utils.data import Dataset

from noc.dataset.transforms import build_transform


class NocapDataset(Dataset):
    """
    Nocap dataset is used for only evaluation purpose (i.e., no training).
    Because test set has no caption annotation for leaderboard competition,
    we use validataion set for evaluation.
    """

    def __init__(
        self,
        cfg: dict,
        split: str,
        **ignore_kwargs,
    ):
        assert split in ["val", "test"], "only support validation split for evaluation"
        if split == "test":
            split = "val"
        self.split = split
        self.transform = build_transform("val", cfg.dataset.transform_hparams)

        # load annotation file
        assert len(cfg.dataset.ann_file_path) > 0
        with open(cfg.dataset.ann_file_path, "r") as f:
            anns = json.load(f)
        img_anns = anns["images"]
        img_dir = os.path.join(cfg.dataset.image_dir, "validation")

        # build instance items
        self.items = []
        for ann in img_anns:
            filename = ann["file_name"]
            imgpath = os.path.join(img_dir, filename)
            item = {
                "id": ann["id"],
                "domain": ann["domain"],
                "imgpath": imgpath,
                "captions": [],
                "coco_url": ann["coco_url"],
            }
            self.items.append(item)

        # if validation, we include GT captions for qualitative check
        if split == "val":
            cap_anns = anns["annotations"]
            for ann in cap_anns:
                self.items[ann["image_id"]]["captions"].append(ann["caption"])

        logging.info(f"total items (nocap): {len(self.items)}")

    def __getitem__(self, idx: int):
        instance = self.items[idx]

        # load image
        imgpath = instance["imgpath"]
        img = Image.open(imgpath).convert("RGB")
        img = self.transform(img)

        # return should include img, imgpath, gt_caps, ds_name
        return {
            "img_id": str(instance["id"]),
            "img": img,
            "imgpath": imgpath,
            "gt_caps": instance["captions"],
            "ds_name": "nocap",
            "domain": instance["domain"],
        }

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)
