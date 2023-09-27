#!/usr/bin/env python3
# ------------------------------------------------------------------------------------
# Copyright (c) 2023 KakaoBrain. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
import json
import logging
import os
import pprint

import easydict
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

    # overwrite model config from one of checkpoint.
    expr_dir = "/".join(cfg.experiment.load_from.split("/")[:-1])
    with open(os.path.join(expr_dir, "config.json"), "r") as f:
        ckpt_cfg = json.load(f)
        ckpt_cfg = easydict.EasyDict(ckpt_cfg)
    cfg.model.update(ckpt_cfg.model)

    # init dataloader
    _, test_loader = init_data_loader(cfg, split="test")

    # init model
    cfg, model = init_model(cfg)

    # init trainer
    cfg, trainer = init_trainer(cfg)

    if trainer.global_rank == 0:
        # avoiding repeated print when using multiple gpus
        logging.info(f"MODEL: {model}")
        logging.info(cfg.pretty() if isinstance(cfg, DictConfig) else pprint.pformat(cfg))

    # test
    pred_dir = model.get_prediction_dir()
    if not cfg.test.only_calc_score_from_files:
        if cfg.model.type == "controllable":
            scores = {}
            total_bins = model.num_bins
            for bin_idx in range(total_bins - 1, -1, -1):  # reverse order
                logging.info(f"   >>> Evaluation on the bin index {bin_idx+1} / {total_bins}")
                model.control_signal_at_inference[:] = bin_idx
                pred_filename = model.get_prediction_filename()
                trainer.test(model, dataloaders=test_loader)

                # calc captioning metric
                logging.info(f"Computing captioning metrics bin index {bin_idx+1} / {total_bins}")
                cur_score = model.calc_metric(pred_dir, pred_filename)
                logging.info(f"   >>> prediction saved to {os.path.join(pred_dir, pred_filename)}")
                scores[f"control_signal_{bin_idx+1}"] = cur_score

        else:  # cover specified bin index and non-zizt model testing
            pred_filename = model.get_prediction_filename()
            logging.info("Start Inference: ")
            trainer.test(model, dataloaders=test_loader)
            # calc captioning metric
            logging.info("Start getting captioning scores: ")
            scores = model.calc_metric(pred_dir, pred_filename)
    else:
        logging.info("Start getting captioning scores: ")
        scores = model.calc_metric(cfg.experiment.log_path, cfg.test.pred_file_pattern)

    if trainer.global_rank == 0:
        logging.info(json.dumps(scores, indent=4))
        ds_name = cfg.dataset.name_val
        with open(os.path.join(pred_dir, f"{ds_name}_scores.json"), "w") as f:
            json.dump(scores, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
