#!/bin/bash

echo Running shell script that generates cost statistics for Wanda data

# Define python version
EXE_PYTHON=python3

# Define environment variables
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map

BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/data_processing

PY_SCRIPT=wanda_cost_stats.py

DATA_DIR1=/project/learningphysics/tartancost_wanda_traj
DATA_DIR2=/project/learningphysics/tartancost_wanda_traj0609

OUTPUT_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/ros


# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .

# Login to Weights and Biases
wandb login b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6

# Run labeling script
${EXE_PYTHON} $BASE_DIR/$PY_SCRIPT \
    --data_dir1 $DATA_DIR1 \
    --data_dir2 $DATA_DIR2 \
    --output_dir $COSTSTATS_DIR


echo Cost statistics shell script ends.

