#!/bin/bash

# SLURM Resource Parameters

#SBATCH -N 1  # CPU Cores
#SBATCH -t 0-01:00 # D-HH:MM
#SBATCH -p gpu # cpu/gpu/dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=8192  # MB
#SBATCH --job-name=create_split
#SBATCH -o /home/mguamanc/job_%j.out
#SBATCH -e /home/mguamanc/job_%j.err
#SBATCH --mail-type=ALL # BEGIN, END, FAIL, ALL
#SBATCH --mail-user=mguamanc@andrew.cmu.edu

# Executable
EXE=/bin/bash
WORKING_DIR=/data/datasets/mguamanc/learned_cost_map/cluster_scripts
EXE_SCRIPT=$WORKING_DIR/create_split.sh

USER=mguamanc

nvidia-docker run --rm --ipc=host -e CUDA_VISIBLE_DEVICES='echo $CUDA_VISIBLE_DEVICES' -v /data/datasets:/data/datasets -v /home/$USER:/home/$USER -v /project:/project mguamanc/sara $EXE $EXE_SCRIPT