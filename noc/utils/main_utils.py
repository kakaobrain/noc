# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import logging
import os
import sys
from datetime import timedelta

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from noc.dataset import get_collate_fn, get_dataset
from noc.models import build_model
from noc.utils.attr_dict import AttrDict


def save_config_to_disk(cfg):
    filename = os.path.join(cfg.experiment.log_path, "config.json")
    with open(filename, "w") as fopen:
        json.dump(cfg, fopen, indent=4, ensure_ascii=False)
        fopen.flush()
    logging.info(f"Saved Config Data to File: {filename}")


def init_hydra_config(mode: str, from_file: str = ""):
    # set logging level
    logging.getLogger().setLevel(logging.DEBUG)

    overrides = sys.argv[1:]
    logging.info(f"####### overrides: {overrides}")
    with hydra.initialize_config_module(version_base=None, config_module="config"):
        cfg = hydra.compose("default", overrides=overrides)

    # convert the config to AttrDict
    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict(cfg)

    # assert the config and infer
    cfg.mode = mode
    cfg = infer_and_assert_hydra_config(cfg)

    return cfg


def infer_and_assert_hydra_config(cfg):
    # batch size for each proc (if distributed)
    cfg.distributed.world_size = int(cfg.distributed.num_nodes * cfg.distributed.num_proc_per_node)

    assert cfg.experiment.effective_batch_size_train % cfg.distributed.world_size == 0
    assert cfg.experiment.effective_batch_size_test % cfg.distributed.world_size == 0

    cfg.experiment.batch_size_per_proc_train = int(
        cfg.experiment.effective_batch_size_train / cfg.distributed.world_size
    )
    cfg.experiment.batch_size_per_proc_test = int(
        cfg.experiment.effective_batch_size_test / cfg.distributed.world_size
    )

    # overwrite options for pl trainer
    cfg.trainer.gpus = cfg.distributed.num_proc_per_node
    cfg.trainer.num_nodes = cfg.distributed.num_nodes
    cfg.trainer.sync_batchnorm = cfg.model.use_sync_bn
    if cfg.experiment.max_steps > 0:
        assert cfg.experiment.max_epochs is None
        assert cfg.experiment.val_check_interval is not None
        cfg.trainer.max_epochs = cfg.experiment.max_epochs
        cfg.trainer.max_steps = cfg.experiment.max_steps
        cfg.trainer.val_check_interval = cfg.experiment.val_check_interval
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.experiment.training_mode = "iter"
    else:
        cfg.trainer.max_epochs = cfg.experiment.max_epochs
        cfg.trainer.max_steps = -1
        cfg.experiment.training_mode = "epoch"

    # log path
    cfg.experiment.proj_root = os.environ["PYTHONPATH"]
    assert len(cfg.experiment.proj_root) > 0
    cfg.experiment.log_path = os.path.join(
        cfg.experiment.proj_root, "results", cfg.experiment.expr_name
    )

    # when beginning from checkpoint
    if "load_from" in cfg.experiment and len(cfg.experiment.load_from) > 0:  # noqa: SIM102
        if "results" not in cfg.experiment.load_from:  # noqa: SIM102
            # if only ckpt filename is given, we expand it with working experiment directory
            cfg.experiment.load_from = os.path.join(
                cfg.experiment.proj_root,
                "results",
                cfg.experiment.expr_name,
                cfg.experiment.load_from,
            )

    # resume from the last ckpt file
    if "resume" in cfg.experiment and cfg.experiment.resume:
        cfg.experiment.resume_from = os.path.join(
            cfg.experiment.proj_root, "results", cfg.experiment.expr_name, "last.ckpt"
        )

    # Adjust prefix length based on zizt condition
    if cfg.model.type == "controllable":
        # we add 1 to prefix length for control embedding
        cfg.model.prefix_length += 1

    # dry-run mode
    if cfg.experiment.dry_run:
        cfg.trainer.limit_train_batches = 20
        cfg.trainer.limit_val_batches = 10
        cfg.trainer.limit_test_batches = 10
        cfg.trainer.limit_predict_batches = 10
        cfg.trainer.num_sanity_val_steps = 0

    return cfg


def init_data_loader(cfg, split, drop_last=None):
    is_train = split == "train"
    ds_name = cfg.dataset.name_train if is_train else cfg.dataset.name_val
    drop_last = is_train if drop_last is None else drop_last

    dataset = get_dataset(cfg, split)
    if ds_name in ["cc3m"]:
        shuffle = (
            is_train and cfg.dataset.ds_type == "mapstyle"
        )  # No shuffle for webdatset. the webdataset has its own shuffling strategy.
    elif ds_name in ["coco", "nocap", "retrieval", "flickr30k"]:
        # we do not shuffle for evaluation datasets
        shuffle = False
    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=(
            cfg.experiment.batch_size_per_proc_train
            if split == "train"
            else cfg.experiment.batch_size_per_proc_test
        ),
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=drop_last,
        shuffle=shuffle,
        collate_fn=get_collate_fn(cfg),
    )

    if is_train:
        # need for warmup
        if cfg.dataset.ds_type == "mapstyle":
            iter_divider = cfg.experiment.effective_batch_size_train
        else:
            iter_divider = cfg.experiment.batch_size_per_proc_train
        cfg.experiment.train_iters_per_epoch = len(dataset) // iter_divider
    return cfg, data_loader


def init_model(cfg):
    model = build_model(cfg)
    return cfg, model


def setup_strategy(config):
    logging.info("   >>> strategy: ", config.experiment.get("strategy"))
    if config.experiment.get("strategy") is None or config.experiment.strategy.type == "ddp":
        return DDPStrategy(
            timeout=timedelta(hours=72),
            find_unused_parameters=True,  # TODO: switch to False for code release?
            ddp_comm_hook=default_hooks.fp16_compress_hook
            if config.experiment.fp16_grad_comp
            else None,
        )
    elif config.experiment.strategy.type == "ddp_sharded":
        return "ddp_sharded"
    elif config.experiment.strategy.type.startswith("zero"):
        # use deepspeed
        if config.experiment.strategy.type == "zero1":
            assert not config.experiment.strategy.offload_optimizer
            assert not config.experiment.strategy.offload_parameters
            strategy = DeepSpeedStrategy(
                stage=1,
                offload_optimizer=False,
                offload_parameters=False,
            )
        elif config.experiment.strategy.type == "zero2":
            strategy = DeepSpeedStrategy(
                stage=2,
                offload_optimizer=config.experiment.strategy.offload_optimizer,
                offload_parameters=config.experiment.strategy.offload_parameters,
            )
        elif config.experiment.strategy.type == "zero3":
            strategy = DeepSpeedStrategy(
                stage=3,
                offload_optimizer=config.experiment.strategy.offload_optimizer,
                offload_parameters=config.experiment.strategy.offload_parameters,
            )
        else:
            raise ValueError(config.experiment.strategy.type)

        logging.info(f"   >>> Deepspeed gradient clipping is configured: {strategy.config}")
        return strategy
    else:
        raise ValueError(config.experiment.strategy.type)


def init_trainer(cfg, custom_callbacks=None):
    # logger
    logger = None
    callbacks = []
    if cfg.mode == "train":
        os.makedirs(cfg.experiment.log_path, exist_ok=True)
        logger = pl.loggers.TensorBoardLogger(cfg.experiment.log_path, version=0)

        # checkpoint callback
        if cfg.experiment.training_mode == "epoch":
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=cfg.experiment.log_path,
                    monitor="val/loss",
                    filename="model-best-epoch{epoch:03d}-val_loss{val/loss:.3f}",
                    auto_insert_metric_name=False,
                    save_last=True,
                    save_top_k=cfg.experiment.save_top_k,
                    every_n_epochs=None
                    if cfg.experiment.ckpt_freq < 1
                    else cfg.experiment.ckpt_freq,
                )
            )
        else:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=cfg.experiment.log_path,
                    monitor="val/loss",
                    filename="model-best-step{step:06d}-val_loss{val/loss:.3f}",
                    auto_insert_metric_name=False,
                    save_last=True,
                    save_top_k=cfg.experiment.save_top_k,
                    every_n_epochs=None
                    if cfg.experiment.ckpt_freq < 1
                    else cfg.experiment.ckpt_freq,
                )
            )

        # learning rate callback
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

    # model size summary
    callbacks.append(ModelSummary(max_depth=2))

    if custom_callbacks is not None:
        callbacks.extend(custom_callbacks)
    if cfg.optimizer.gradient_clip_val is None:
        cfg.optimizer.gradient_clip_val = None

    # Trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        strategy=setup_strategy(cfg),
        logger=logger,
        log_every_n_steps=cfg.logging.log_freq,
        gradient_clip_val=cfg.optimizer.gradient_clip_val,
        gradient_clip_algorithm=cfg.optimizer.gradient_clip_algorithm,
    )
    return cfg, trainer
