#!/usr/bin/env bash

DATA_DIR=./data/flickr30k
mkdir -p ${DATA_DIR}/images

# download annotations for data loader
wget -N -P ${DATA_DIR} https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

# decompress
unzip ${DATA_DIR}/caption_datasets.zip -d ${DATA_DIR}/temp
mv ${DATA_DIR}/temp/dataset_flickr30k.json ${DATA_DIR}/

# parse annotation
python3 preprocess/parse_flickr_annotations.py \
    --anno_src_path ${DATA_DIR}/dataset_flickr30k.json \
    --anno_trg_path ${DATA_DIR}/ann_test.json

# delete unnecessary files
rm ${DATA_DIR}/caption_datasets.zip
rm ${DATA_DIR}/dataset_flickr30k.json
rm -r ${DATA_DIR}/temp
