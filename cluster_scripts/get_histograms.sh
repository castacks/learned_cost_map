#!/bin/bash

echo Running shell script that creates histograms for cost and speed.

# Define python version
EXE_PYTHON=python3

# Define environment variables

# Varibales to find code
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map
BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/utils


# Variables for trainer
PY_TRAIN=get_histograms.py
DATA_DIR=/project/learningphysics/tartandrive_trajs
TRAIN_SPLIT=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/train_uniform.txt
VAL_SPLIT=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits/val_uniform.txt
NUM_BINS=20
OUTPUT_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits



# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .


# Run split script
${EXE_PYTHON} $BASE_DIR/$PY_SPLIT \
    --data_dir $DATA_DIR \
    --train_split $TRAIN_SPLIT \
    --val_split $VAL_SPLIT \
    --num_bins $NUM_BINS \
    --output_dir $OUTPUT_DIR
