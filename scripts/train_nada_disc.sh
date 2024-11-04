#!/bin/bash

PRETRAINED_STYLEGAN_PATH="pretrained_models/ffhq.pt"
IMG_SIZE=1024
TARGET_TRAIN_DATA_DIR_PATH="/kuacc/users/aanees20/hpc_run/DynaGAN/target_data/hdn_20/"
DEVICE_NUM=0
ITER=1200 # ITER=1800 *26 # 250
OUTPUT_DIR="output_clip_hdn_20_2" #

CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train_clip_nada_disc.py --frozen_gen_ckpt=${PRETRAINED_STYLEGAN_PATH} --size=${IMG_SIZE} --output_dir=${OUTPUT_DIR} --style_img_dir=${TARGET_TRAIN_DATA_DIR_PATH} --iter=${ITER}  --human_face
