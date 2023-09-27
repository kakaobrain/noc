#!/usr/bin/env bash
# script for test self-retrieval

PRED_PATH=$1  #e.g., "results/cc3m_fp16/model-best-epoch005-val_loss2.474/prediction_greedy_img_coco2017_7.json"
WORK_DIR=$(pwd)  # == <path_to_root_of_github>

PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/test_retrieval.py \
	experiment.expr_name="eval_retrieval" \
	dataset.pred_file_path=${PRED_PATH} \
	+task=eval_retrieval
