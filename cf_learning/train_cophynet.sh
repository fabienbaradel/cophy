#!/usr/bin/env bash

# Command line
# ./cf_learning/train_cophynet.sh $1 $2 $3 $4

# Envs
PYTHONPATH=`pwd`

LOG_DIR=$1
COPHY=$2
DERENDERING=$3
PREXTRACTED_OBJ=$4

## BlocktowerCF
for k in 3 4
do
for type in normal generalization
  do
    python cf_learning/main.py \
      --dataset_dir $COPHY/blocktowerCF/ \
      --derendering_ckpt $DERENDERING/blocktowerCF/model_state_dict.pt \
      --log_dir $LOG_DIR/blocktowerCF/$k/$type/ \
      --preextracted_obj_vis_prop_dir $PREXTRACTED_OBJ/blocktowerCF \
      --dataset_name blocktower \
      --model cophynet --num_objects $k --type $type \
      --batch_size 32 --workers 10 --epochs 50
  done
done

## Special evaluation ##
# 4->3
python cf_learning/main.py \
        --dataset_dir $COPHY/blocktowerCF/ \
        --pretrained_ckpt $LOG_DIR/blocktowerCF/3/normal/model_state_dict.pt \
        --log_dir $LOG_DIR/blocktowerCF/4/normal_3 \
        --dataset_name blocktower \
        --model cophynet --num_objects 4 --type normal --evaluate

# 3->4
python cf_learning/main.py \
        --dataset_dir $COPHY/blocktowerCF/ \
        --pretrained_ckpt $LOG_DIR/blocktowerCF/4/normal/model_state_dict.pt \
        --log_dir $LOG_DIR/blocktowerCF/3/normal_4 \
        --dataset_name blocktower \
        --model cophynet --num_objects 3 --type normal --evaluate


# BallsCF
k=4
python cf_learning/main.py \
      --dataset_dir $COPHY/ballsCF/ \
      --derendering_ckpt $DERENDERING/ballsCF/model_state_dict.pt \
      --log_dir $LOG_DIR/ballsCF/$k/ \
      --preextracted_obj_vis_prop_dir $PREXTRACTED_OBJ/ballsCF \
      --dataset_name balls \
      --model cophynet --num_objects $k \
      --batch_size 32 --workers 10 --epochs 50

## Special evaluation ##
for k in 2 3 5 6
do
    python cf_learning/main.py \
            --dataset_dir $COPHY/ballsCF/ \
            --pretrained_ckpt $LOG_DIR/ballsCF/4/model_state_dict.pt \
            --log_dir $LOG_DIR/ballsCF/4_$k \
            --dataset_name balls \
            --model cophynet --num_objects $k --type normal --evaluate
done

# CollisionCF
for type in normal moving_cylinder moving_sphere
do
python cf_learning/main.py \
      --dataset_dir $COPHY/collisionCF/ \
      --derendering_ckpt $DERENDERING/collisionCF/model_state_dict.pt \
      --log_dir $LOG_DIR/collisionCF/$k/$type/ \
      --preextracted_obj_vis_prop_dir $PREXTRACTED_OBJ/ballsCF \
      --dataset_name collision \
      --model cophynet --num_objects $k \
      --batch_size 32 --workers 10 --epochs 50
