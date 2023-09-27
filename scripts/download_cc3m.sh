#!/usr/bin/env bash

DATA_DIR=./data/cc3m
mkdir -p ${DATA_DIR}/images

##########################################################
# 1. Download the annotations by yourself
##########################################################
# Download annotations from https://ai.google.com/research/ConceptualCaptions/download
# Then, move the files to ${DATA_DIR} as follows:
# mv Train_GCC-training.tsv ${DATA_DIR}
# mv Validation_GCC-1.1.0-Validation.tsv ${DATA_DIR}

##########################################################
# 2. Download the images using img2dataset package
##########################################################
# add tags
sed -i '1s/^/caption\turl\n/' ${DATA_DIR}/Train_GCC-training.tsv
sed -i '1s/^/caption\turl\n/' ${DATA_DIR}/Validation_GCC-1.1.0-Validation.tsv

# rename tsv files
mv ${DATA_DIR}/Train_GCC-training.tsv ${DATA_DIR}/cc3m_train.tsv
mv ${DATA_DIR}/Validation_GCC-1.1.0-Validation.tsv ${DATA_DIR}/cc3m_val.tsv

# download images
# it will take sevaral hours for downloading
pip install img2dataset
img2dataset --url_list ${DATA_DIR}/cc3m_train.tsv --input_format "tsv" \
	--url_col "url" --caption_col "caption" --output_format files \
	--output_folder ${DATA_DIR}/images/train --processes_count 16 --thread_count 64 \
	--image_size 346 --resize_mode keep_ratio

img2dataset --url_list ${DATA_DIR}/cc3m_val.tsv --input_format "tsv" \
	--url_col "url" --caption_col "caption" --output_format files \
	--output_folder ${DATA_DIR}/images/val --processes_count 16 --thread_count 64 \
	--image_size 346 --resize_mode keep_ratio

##########################################################
# 3. Generate annotations
##########################################################
# it approximately takes ~3 hour for training split
python3 preprocess/generate_cc3m_annotations.py \
	--split train --anno_trg_path ${DATA_DIR}/train_list.txt

python3 preprocess/generate_cc3m_annotations.py \
	--split val --anno_trg_path ${DATA_DIR}/val_list.txt

# delete unnecessary files
rm ${DATA_DIR}/cc3m_train.tsv
rm ${DATA_DIR}/cc3m_val.tsv
