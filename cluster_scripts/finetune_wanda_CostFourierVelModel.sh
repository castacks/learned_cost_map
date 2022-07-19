#!/bin/bash

echo Running shell script that fine-tunes CostFourierVelModel network with Wanda data with a given split.

# Define python version
EXE_PYTHON=python3

# Define environment variables

# Varibales to find code
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map
BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/trainer

# Variables for trainer
PY_TRAIN=train.py
DATA_DIR=/project/learningphysics/tartancost_wanda_traj
TRAIN_SPLIT=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/wanda_train.txt
VAL_SPLIT=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/wanda_val.txt
MODEL=CostFourierVelModel
FOURIER_SCALE=10.0
RUN_NAME=finetune_wanda_${MODEL}_lr_3e-5_g_99e-1_aug_l2_scale_${FOURIER_SCALE}
NUM_EPOCHS=50
BATCH_SIZE=128
SEQ_LENGTH=1
LEARNING_RATE=0.00003
WEIGHT_DECAY=0.0000001
GAMMA=0.99
EVAL_INTERVAL=1
SAVE_INTERVAL=1
NUM_WORKERS=10
SAVED_MODEL=/data/datasets/mguamanc/learned_cost_map/models/train_CostFourierVelModel_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0/epoch_50.pt
SAVED_FREQS=/data/datasets/mguamanc/learned_cost_map/models/train_CostFourierVelModel_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0/fourier_freqs.pt
MODELS_DIR=/data/datasets/mguamanc/learned_cost_map/models
MAP_CONFIG=/data/datasets/mguamanc/learned_cost_map/configs/wanda_map_params.yaml


# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .

# Login to Weights and Biases
wandb login b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6
export WANDB_MODE=offline
wandb init -p SARA

# # Run split script
# ${EXE_PYTHON} $BASE_DIR/$PY_SPLIT \
#     --num_train $NUM_TRAIN \
#     --num_val $NUM_VAL \
#     --all_train_fp $ALL_TRAIN_FP \
#     --all_val_fp $ALL_VAL_FP \
#     --output_dir $OUTPUT_DIR

echo Running standard split

# Run trainer
${EXE_PYTHON} $BASE_DIR/$PY_TRAIN \
    --model $MODEL \
    --data_dir $DATA_DIR \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --models_dir $MODELS_DIR \
    --log_dir $RUN_NAME \
    --map_config $MAP_CONFIG \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    -lr $LEARNING_RATE \
    --gamma $GAMMA \
    --weight_decay $WEIGHT_DECAY \
    --eval_interval $EVAL_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --num_workers $NUM_WORKERS\
    --multiple_gpus \
    --augment_data \
    --fourier_scale $FOURIER_SCALE \
    --fine_tune \
    --saved_model $SAVED_MODEL \
    --saved_freqs $SAVED_FREQS \
    --wanda
    # --pretrained

echo Finetune Wanda CostFourierVelModel network shell script ends.
