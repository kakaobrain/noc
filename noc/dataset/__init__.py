# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import torch

from noc.dataset.cc3m import CC3MDataset
from noc.dataset.coco import COCOCaptionDataset
from noc.dataset.flickr30k import Flickr30KDataset
from noc.dataset.nocap import NocapDataset
from noc.dataset.retrieval import SelfRetrievalDataset


def get_dataset(cfg, split):
    # there are two splits for training and evaluation
    # e.g., training on CC3M and evaluation on MSCOCO

    # training dataset: cc3m
    # evaluation dataset: coco, nocap, flickr30k (capation gen.), retrieval
    ds_name = cfg.dataset.name_train if split == "train" else cfg.dataset.name_val
    if ds_name == "cc3m":
        dataset = CC3MDataset(cfg=cfg, split=split)

    elif ds_name == "coco":
        dataset = COCOCaptionDataset(cfg=cfg, split=split)

    elif ds_name == "nocap":
        dataset = NocapDataset(cfg=cfg, split=split)

    elif ds_name == "flickr30k":
        dataset = Flickr30KDataset(cfg=cfg)

    elif ds_name == "retrieval":
        dataset = SelfRetrievalDataset(cfg=cfg)

    else:
        raise NotImplementedError()

    assert dataset is not None
    return dataset


def get_collate_fn(cfg):
    return torch.utils.data._utils.collate.default_collate


__all__ = ["get_dataset", "get_collate_fn"]
