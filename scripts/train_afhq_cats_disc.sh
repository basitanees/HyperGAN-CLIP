#!/bin/bash

PRETRAINED_STYLEGAN_PATH="pretrained_models/afhqcats.pt"
ID_MODEL_PATH="pretrained_models/moco_v2_800ep_pretrain.pt"
IMG_SIZE=512
TARGET_TRAIN_DATA_DIR_PATH="/kuacc/users/aanees20/hpc_run/DynaGAN/target_data/wild_data"
DEVICE_NUM=0
ITER=10000 # ITER=1800
OUTPUT_DIR="output_cats_final_wild_dogs_dyna"

CUDA_VISIBLE_DEVICES=${DEVICE_NUM} 
python train_clip_cats_disc.py --frozen_gen_ckpt=${PRETRAINED_STYLEGAN_PATH} --id_model_path=${ID_MODEL_PATH} --size=${IMG_SIZE} --stylegan_size=${IMG_SIZE} --output_dir=${OUTPUT_DIR} --style_img_dir=${TARGET_TRAIN_DATA_DIR_PATH} --iter=${ITER}
#"output_res_features_res_weight" is actually old version