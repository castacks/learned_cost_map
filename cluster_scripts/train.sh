#!/bin/bash

echo Running shell script that trains network with a given split.

# Define python version
EXE_PYTHON=python3

# Define environment variables

# Varibales to find code
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map
BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/trainer

# Variables for generating data split:
PY_SPLIT=create_split.py
NUM_TRAIN=30
NUM_VAL=30
ALL_TRAIN_FP=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/tartandrive_train.txt
ALL_VAL_FP=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/tartandrive_val.txt
OUTPUT_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits

# Variables for trainer
PY_TRAIN=train.py
DATA_DIR=/project/learningphysics/tartandrive_trajs
TRAIN_SPLIT=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/train${NUM_TRAIN}.txt
VAL_SPLIT=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/val${NUM_TRAIN}.txt
RUN_NAME=train${NUM_TRAIN}
NUM_EPOCHS=50
BATCH_SIZE=16
EVAL_INTERVAL=1
SAVE_INTERVAL=1
NUM_WORKERS=10



# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .

# Login to Weights and Biases
wandb login b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6
export WANDB_MODE=offline
wandb init -p SARA

# Run split script
${EXE_PYTHON} $BASE_DIR/$PY_SPLIT \
    --num_train $NUM_TRAIN \
    --num_val $NUM_VAL \
    --all_train_fp $ALL_TRAIN_FP \
    --all_val_fp $ALL_VAL_FP \
    --output_dir $OUTPUT_DIR

echo Done creating split

# Run trainer
${EXE_PYTHON} $BASE_DIR/$PY_TRAIN \
    --data_dir $DATA_DIR \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --log_dir $RUN_NAME \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --num_workers $NUM_WORKERS
    # --shuffle_train

echo Training tiny network shell script ends.