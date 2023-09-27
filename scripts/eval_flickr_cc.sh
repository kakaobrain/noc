#!/usr/bin/env bash
# script for test captioning on flickr30k

EXPR_NAME=$1
MODEL_NAME="model.ckpt"
WORK_DIR=$(pwd)  # == <path_to_root_of_github>

PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/test_captioning.py \
	experiment.expr_name=${EXPR_NAME} \
	experiment.load_from=${MODEL_NAME} \
	+task=eval_captioning_flickr
