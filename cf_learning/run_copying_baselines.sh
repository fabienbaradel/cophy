#!/usr/bin/env bash

# Command line
# ./cf_learning/run_copying_baselines.sh $1 $2 $3

# Envs
PYTHONPATH=`pwd`

LOG_DIR=$1
COPHY=$2
DERENDERING=$3

# BlocktowerCF
for k in 4 3
do
    for type in normal generalization
    do
        python cf_learning/main.py \
        --dataset_dir $COPHY/blocktowerCF/ \
        --derendering_ckpt $DERENDERING/blocktowerCF/model_state_dict.pt \
        --log_dir $LOG_DIR/blocktowerCF \
        --dataset_name blocktower \
        --model copy_c --num_objects $k --type $type --evaluate
    done
done

# CollisionCF
for type in normal moving_cylinder moving_sphere
do
    python cf_learning/main.py \
    --dataset_dir $COPHY/collisionCF/ \
    --derendering_ckpt $DERENDERING/collisionCF/model_state_dict.pt \
    --log_dir $LOG_DIR/collisionCF \
    --dataset_name collision \
    --model copy_c --type $type --evaluate
done


# BallsCF
for k in 2 3 4 5 6
do
    python cf_learning/main.py \
    --dataset_dir $COPHY/ballsCF/ \
    --derendering_ckpt $DERENDERING/ballsCF/model_state_dict.pt \
    --log_dir $LOG_DIR/collisionCF \
    --dataset_name balls \
    --model copy_c --num_objects $k --evaluate
done

# show in log
printf "\n\n\n*************** Results ***************\n\n"
tail $LOG_DIR/blocktowerCF/*/*/*/test.txt
tail $LOG_DIR/collisionCF/*/*/test.txt
tail $LOG_DIR/ballsCF/*/*/test.txt