#!/usr/bin/env bash

# Command line
# ./derendering/extract_3d_pose.sh <COPHY> <DERENDERING-DIR> <OUT-DIR>

# Envs
PYTHONPATH=`pwd`
COPHY=$1
DERENDERING_DIR=$2
OUT_DIR=$3

# BallsCF
for k in 2 3 4 5 6
do
  for s in 'val' 'train' 'test'
  do
    python derendering/extract_object_visual_properties.py \
    --dataset_dir $COPHY/ballsCF \
    --out_dir $OUT_DIR/ballCF \
    --dataset_name balls \
    --derendering_ckpt $DERENDERING_DIR/ballsCF/model_state_dict.pt \
    --workers 8 \
    --num_objects $k \
    --split $s
  done
done


# CollisionCF
for k in moving_cylinder moving_sphere normal
do
  for s in 'val' 'train' 'test'
  do
    python derendering/extract_object_visual_properties.py \
    --dataset_dir $COPHY/collisionCF \
    --out_dir $OUT_DIR/collisionCF \
    --dataset_name collision \
    --derendering_ckpt $DERENDERING_DIR/collisionCF/model_state_dict.pt \
    --workers 8 \
    --type $k \
    --split $s
  done
done

# BlocktowerCF
for k in normal generalization
do
  for s in 'val' 'train' 'test'
  do
    for m in 3 4
    do
      python derendering/extract_object_visual_properties.py \
      --dataset_dir $COPHY/blocktowerCF \
      --out_dir $OUT_DIR/blocktowerCF \
      --dataset_name blocktower \
      --derendering_ckpt $DERENDERING_DIR/blocktowerCF/model_state_dict.pt \
      --workers 8 \
      --type $k \
      --num_objects $m \
      --split $s
    done
  done
done