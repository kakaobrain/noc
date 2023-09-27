#!/usr/bin/env bash
# script for training vanilla captioning model

WORK_DIR=$(pwd) # == <path_to_root_of_github>
echo ${WORK_DIR}

EXPR_NAME="cc3m_vanilla"
PYTHONPATH=${WORK_DIR} python3 ${WORK_DIR}/train.py \
	distributed.num_nodes=16 distributed.num_proc_per_node=4 \
	experiment.expr_name=${EXPR_NAME} \
	model.type=vanilla
