#!/bin/bash

echo Running shell script that creates training split.

# Define python version
EXE_PYTHON=python3

# Define environment variables
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map

BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/data_processing

PY_SCRIPT=create_splits.py

DATA_DIR=/project/learningphysics/tartandrive_trajs

OUTPUT_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits


# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .

# Login to Weights and Biases
wandb login b47938fa5bae1f5b435dfa32a2aa5552ceaad5c6

# Run split script
${EXE_PYTHON} $BASE_DIR/$PY_SCRIPT \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR


echo Creating splits shell script ends.

