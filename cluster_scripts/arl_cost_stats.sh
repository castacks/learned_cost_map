#!/bin/bash

echo Running shell script that generates cost statistics for Wanda data

# Define python version
EXE_PYTHON=python3

# Define environment variables
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map

BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/data_processing

PY_SCRIPT=arl_cost_stats.py

DATA_DIR=/project/learningphysics/arl_20220922_traj
OUTPUT_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/ros


# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .


# Run labeling script
${EXE_PYTHON} $BASE_DIR/$PY_SCRIPT \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR


echo Cost statistics shell script ends.

