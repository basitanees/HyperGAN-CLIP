#!/bin/bash

PRETRAINED_STYLEGAN_PATH="pretrained_models/ffhq.pt"
IMG_SIZE=1024
TARGET_TRAIN_DATA_DIR_PATH="target_data/raw_data"
DEVICE_NUM=0
ITER=15000 # ITER=1800
OUTPUT_DIR="output_residual_close_discdiff"

CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train_clip_random_disc_no_res.py --frozen_gen_ckpt=${PRETRAINED_STYLEGAN_PATH} --size=${IMG_SIZE} --output_dir=${OUTPUT_DIR} --style_img_dir=${TARGET_TRAIN_DATA_DIR_PATH} --iter=${ITER}  --human_face
#"output_res_features_res_weight" is actually old version