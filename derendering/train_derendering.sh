#!/usr/bin/env bash
# ./derendering/train_derendering.sh <LOC-COPHY> <LOC-LOG_DIR>

# Envs
PYTHONPATH=`pwd`
COPHY=$1
LOGDIR=$2

# BlocktowerCF
python derendering/main.py \
--dataset_dir $COPHY/blocktowerCF \
--log_dir $LOGDIR/blocktowerCF \
--dataset_name blocktower \
--batch_size 16 \
--workers 10 \
--epochs 10

# BallsCF
python derendering/main.py \
--dataset_dir $COPHY/ballsCF \
--log_dir $LOGDIR/ballsCF \
--dataset_name balls \
--batch_size 16 \
--workers 10 \
--epochs 20

# CollisionCF
python derendering/main.py \
--dataset_dir $COPHY/collisionCF \
--log_dir $LOGDIR/collisionCF \
--dataset_name collision \
--batch_size 16 \
--workers 10 \
--epochs 20