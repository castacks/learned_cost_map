#!/bin/bash

echo Running shell script that creates training split.

# Define python version
EXE_PYTHON=python3

# Define environment variables
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map

BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/data_processing

PY_SCRIPT=create_splits.py

DATA_DIR=/project/learningphysics/tartancost_wanda_traj

OUTPUT_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/splits

SPLIT_NAME=wanda_train.txt

# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .

# Run split script
${EXE_PYTHON} $BASE_DIR/$PY_SCRIPT \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --split_name $SPLIT_NAME


echo Creating splits shell script ends.

