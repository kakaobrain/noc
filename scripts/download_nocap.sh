#!/usr/bin/env bash

DATA_DIR=./data/nocap
mkdir -p ${DATA_DIR}/images

# download open-images raw images of validation split
aws s3 --no-sign-request sync s3://open-images-dataset/validation ${DATA_DIR}/images/validation
# download annotations for data loader
wget -N -P ${DATA_DIR} https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json
