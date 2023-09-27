#!/usr/bin/env python3
# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import logging
import pprint

import pytorch_lightning as pl
from omegaconf import DictConfig

from noc.utils.callbacks import CallbackTrainer
from noc.utils.main_utils import init_data_loader, init_hydra_config, init_model, init_trainer

logging.info("Training with config:")
logging.getLogger().setLevel(logging.DEBUG)


def main():
    # init cfg
    cfg = init_hydra_config(mode="train")

    # set random seed
    if "seed" in cfg.experiment and cfg.experiment.seed >= 0:
        pl.seed_everything(cfg.experiment.seed, workers=True)

    # init dataloader
    loaders = []
    custom_callbacks = []
    cfg, train_loader = init_data_loader(cfg, split="train")
    loaders.append(train_loader)
    if cfg.dataset.ds_type == "webdataset":
        custom_callbacks.append(CallbackTrainer(train_loader))  # set epoch for shuffling webdataset
    else:
        custom_callbacks = None

    # validation loader
    _, val_loader = init_data_loader(cfg, split="val")
    loaders.append(val_loader)

    # init model
    cfg, model = init_model(cfg)

    # init trainer
    cfg, trainer = init_trainer(cfg, custom_callbacks=custom_callbacks)

    if trainer.global_rank == 0:
        # avoiding repeated print when using multiple gpus
        logging.info(f"MODEL: {model}")
        logging.info(cfg.pretty() if isinstance(cfg, DictConfig) else pprint.pformat(cfg))

        # train
        logging.info(
            f"Start Training: Total Epoch - {cfg.trainer.max_epochs}, Precision: {cfg.trainer.precision}"
        )

    trainer.fit(
        model,
        *loaders,
        ckpt_path=cfg.experiment.resume_from if cfg.experiment.resume else None,
    )


if __name__ == "__main__":
    main()
