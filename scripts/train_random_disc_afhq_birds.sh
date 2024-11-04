#!/bin/bash

PRETRAINED_STYLEGAN_PATH="/kuacc/users/aanees20/hpc_run/StyleGAN2_rosinality/stylegan2-pytorch/checkpoint/350000.pt"
ID_MODEL_PATH="pretrained_models/moco_v2_800ep_pretrain.pt"
IMG_SIZE=256
TARGET_TRAIN_DATA_DIR_PATH="target_data/raw_data"
DEVICE_NUM=0
ITER=15000 # ITER=1800
OUTPUT_DIR="output_birds_final_3"

CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train_clip_random_disc_res.py --frozen_gen_ckpt=${PRETRAINED_STYLEGAN_PATH} --id_model_path=${ID_MODEL_PATH} --size=${IMG_SIZE} --output_dir=${OUTPUT_DIR} --style_img_dir=${TARGET_TRAIN_DATA_DIR_PATH} --iter=${ITER}
#"output_res_features_res_weight" is actually old version