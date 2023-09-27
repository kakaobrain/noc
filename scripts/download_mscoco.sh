#!/usr/bin/env bash

DATA_DIR=./data/coco
mkdir -p ${DATA_DIR}/images

# download MS-COCO raw images of validation split
wget -N -P ${DATA_DIR} http://images.cocodataset.org/zips/val2017.zip
# download annotations for data loader
wget -N -P ${DATA_DIR} http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# you should download karpathy test split on your own.
# you may refer this website to download karpathy test split
#   https://www.kaggle.com/datasets/shtvkumar/karpathy-splits
#   or https://github.com/karpathy/neuraltalk2

# decompress
unzip ${DATA_DIR}/annotations_trainval2017.zip -d ${DATA_DIR}
mv ${DATA_DIR}/annotations/captions_*.json ${DATA_DIR}/
unzip ${DATA_DIR}/val2017.zip -d ${DATA_DIR}/images

# delete unnecessary files
rm -r ${DATA_DIR}/annotations
rm ${DATA_DIR}/annotations_trainval2017.zip
rm ${DATA_DIR}/val2017.zip
