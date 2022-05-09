#!/bin/bash

echo Running shell script that trains tiny network.

# Define python version
EXE_PYTHON=python3

# Define environment variables
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map

BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/trainer

PY_SCRIPT=train.py

DATA_DIR=/project/learningphysics/tartandrive_trajs

RUN_NAME=tiny2

TRAIN_SPLIT=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/${RUN_NAME}_train.txt

VAL_SPLIT=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/${RUN_NAME}_val.txt


# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .

# Login to Weights and Biases
wandb login b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6
export WANDB_MODE=offline
wandb init -p SARA

# Run split script
${EXE_PYTHON} $BASE_DIR/$PY_SCRIPT \
    --data_dir $DATA_DIR \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --log_dir $RUN_NAME \
    --num_epochs 50 \
    --batch_size 16 \
    --eval_interval 1 \
    --save_interval 1

echo Training tiny network shell script ends.

