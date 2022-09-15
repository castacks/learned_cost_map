#!/bin/bash

echo Training CostFourierVelModel network with balanced data with a given split.

# Define python version
EXE_PYTHON=python3

# Define environment variables

# Varibales to find code
BASE_DIR=/home/mateo/phoenix_ws/src/learned_cost_map/scripts/learned_cost_map/trainer

# Variables for trainer
PY_TRAIN=train.py
DATA_DIR=/home/mateo/Data/SARA/tartancost_data
TRAIN_LC_DIR=lowcost_5k
TRAIN_HC_DIR=highcost_10k
VAL_LC_DIR=lowcost_val_1k
VAL_HC_DIR=highcost_val_2k
# TRAIN_LC_DIR=lowcost_val_1k
# TRAIN_HC_DIR=highcost_val_2k
# VAL_LC_DIR=lowcost_val_1k
# VAL_HC_DIR=highcost_val_2k
MODEL=BaselineVisualGeometricModel
RUN_NAME=train_${MODEL}_lr_3e-3_g_99e-1_bal_aug_l2
NUM_EPOCHS=15
BATCH_SIZE=256
LEARNING_RATE=0.003
WEIGHT_DECAY=0.0000001
GAMMA=0.99
EVAL_INTERVAL=1
SAVE_INTERVAL=1
NUM_WORKERS=1
MODELS_DIR=/home/mateo/phoenix_ws/src/learned_cost_map/learned_cost_map/models
MAP_CONFIG=/home/mateo/phoenix_ws/src/learned_cost_map/configs/map_params.yaml


# Login to Weights and Biases
wandb login b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6
wandb init -p SARA

echo Running standard split

# Run trainer
${EXE_PYTHON} $BASE_DIR/$PY_TRAIN \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --models_dir $MODELS_DIR \
    --log_dir $RUN_NAME \
    --map_config $MAP_CONFIG \
    --balanced_loader \
    --train_lc_dir $TRAIN_LC_DIR \
    --train_hc_dir $TRAIN_HC_DIR \
    --val_lc_dir $VAL_LC_DIR \
    --val_hc_dir $VAL_HC_DIR \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    -lr $LEARNING_RATE \
    --gamma $GAMMA \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --num_workers $NUM_WORKERS\
    --augment_data \
    # --pretrained