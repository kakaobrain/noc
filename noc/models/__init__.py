# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging

from .lightning.captioner import Captioner
from .lightning.retrieval import RetrievalModel


def build_model(cfg):
    name = cfg.model.name
    if "load_from" in cfg.experiment and len(cfg.experiment.load_from) > 0:
        ckpt_path = cfg.experiment.load_from
        logging.info(f"Checkpoint load from: {ckpt_path}")
        if name == "captioner":
            model = Captioner.load_from_checkpoint(cfg=cfg, checkpoint_path=ckpt_path, strict=False)
        else:
            raise NotImplementedError
    else:
        if name == "captioner":
            model = Captioner(cfg)
        elif name == "retrieval":
            model = RetrievalModel(cfg)
        else:
            raise NotImplementedError

    return model


__all__ = ["build_model"]
