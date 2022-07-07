#!/bin/bash

echo Running shell script that runs cost labeling for TartanDrive trajectories.

# Define python version
EXE_PYTHON=python3

# Define environment variables
PACKAGE_DIR=/home/mateo/phoenix_ws/src/learned_cost_map

BASE_DIR=/home/mateo/phoenix_ws/src/learned_cost_map/scripts/learned_cost_map/viz

PY_SCRIPT=costmaps_from_dir.py

MODEL=CostFourierVelModel

SAVED_MODEL=/home/mateo/phoenix_ws/src/learned_cost_map/scripts/learned_cost_map/trainer/models/train_CostFourierVelModel_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0/epoch_50.pt

SAVED_FREQS=/home/mateo/phoenix_ws/src/learned_cost_map/scripts/learned_cost_map/trainer/models/train_CostFourierVelModel_lr_3e-4_g_99e-1_bal_aug_l2_scale_10.0/fourier_freqs.pt

DATA_DIR=/home/mateo/Data/SARA/corl_video/wanda

VEL=3.0


# Install learned_cost_map package
cd $PACKAGE_DIR
# sudo pip3 install -e .

# Run labeling script
${EXE_PYTHON} $BASE_DIR/$PY_SCRIPT \
    --model $MODEL \
    --saved_model $SAVED_MODEL \
    --saved_freqs $SAVED_FREQS \
    --data_dir $DATA_DIR \
    --vel $VEL


echo Script to create learned costmaps.

