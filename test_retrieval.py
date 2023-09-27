#!/usr/bin/env python3
# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
import logging
import pprint

import pytorch_lightning as pl
from omegaconf import DictConfig

from noc.utils.main_utils import init_data_loader, init_hydra_config, init_model, init_trainer

logging.info("Inference with config:")
logging.getLogger().setLevel(logging.DEBUG)


def main():
    # init cfg
    cfg = init_hydra_config(mode="test")

    # set random seed
    if "seed" in cfg.experiment and cfg.experiment.seed >= 0:
        pl.seed_everything(cfg.experiment.seed, workers=True)

    # init dataloader
    _, test_loader = init_data_loader(cfg, split="val")

    # init model
    cfg, model = init_model(cfg)

    # init trainer
    cfg, trainer = init_trainer(cfg)

    if trainer.global_rank == 0:
        # avoiding repeated print when using multiple gpus
        logging.info(f"MODEL: {model}")
        logging.info(cfg.pretty() if isinstance(cfg, DictConfig) else pprint.pformat(cfg))

    # test
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    main()
