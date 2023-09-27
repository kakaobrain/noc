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


class Flickr30KDataset(Dataset):
    """
    Flickr30K dataset is used for only evaluation purpose (i.e., no training).
    """

    def __init__(
        self,
        cfg: dict,
        **ignore_kwargs,
    ):
        self.transform = build_transform("val", cfg.dataset.transform_hparams)

        # load annotation file
        """ example of each item: {
            "imgid": 67,
            "imgpath": "1018148011.jpg",
            "captions": [
                "A group of people stand in the back of a truck filled with cotton.",
                "Men are standing on and about a truck carrying a white substance.",
                "A group of people are standing on a pile of wool in a truck.",
                "A group of men are loading cotton onto a truck",
                "Workers load sheared wool onto a truck."
            ]
        }
        """
        assert len(cfg.dataset.ann_file_path) > 0
        with open(cfg.dataset.ann_file_path, "r") as f:
            self.items = json.load(f)
        self.img_dir = cfg.dataset.image_dir  # <path_to_flickr30k_images>

        logging.info(f"total items (flickr): {len(self.items)}")

    def __getitem__(self, idx: int):
        instance = self.items[idx]

        # load image
        imgpath = os.path.join(self.img_dir, instance["imgpath"])
        img = Image.open(imgpath).convert("RGB")
        img = self.transform(img)

        # return should include # img, imgpath, gt_caps, ds_name
        return {
            "img_id": str(instance["imgid"]),
            "img": img,
            "imgpath": imgpath,
            "gt_caps": instance["captions"],
            "ds_name": "flickr30k",
        }

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)
