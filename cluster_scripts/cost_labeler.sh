#!/bin/bash

echo Running shell script that runs cost labeling for TartanDrive trajectories.

# Define python version
EXE_PYTHON=python3

# Define environment variables
PACKAGE_DIR=/data/datasets/mguamanc/learned_cost_map

BASE_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/data_processing

PY_SCRIPT=cost_labeler.py

DATA_DIR=/project/learningphysics/tartandrive_trajs_test

COSTSTATS_DIR=/data/datasets/mguamanc/learned_cost_map/scripts/learned_cost_map/ros/cost_statistics.yaml


# Install learned_cost_map package
cd $PACKAGE_DIR
sudo pip3 install -e .

# Run labeling script
${EXE_PYTHON} $BASE_DIR/$PY_SCRIPT \
    --data_dir $DATA_DIR \
    --coststats_dir $COSTSTATS_DIR


echo Labeling shell script ends.

