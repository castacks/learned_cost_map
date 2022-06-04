#!/bin/bash

# SLURM Resource Parameters

#SBATCH -N 1  # CPU Cores
#SBATCH -t 1-00:00 # D-HH:MM
#SBATCH -p dgx # cpu/gpu/dgx
#SBATCH -w calculon 
#SBATCH --gres=gpu:1
#SBATCH --mem=65536  # MB
#SBATCH --job-name=train_CostFourierVelModelEfficientNet
#SBATCH -o /home/mguamanc/job_%j.out
#SBATCH -e /home/mguamanc/job_%j.err
#SBATCH --mail-type=ALL # BEGIN, END, FAIL, ALL
#SBATCH --mail-user=mguamanc@andrew.cmu.edu

# Executable
EXE=/bin/bash
WORKING_DIR=/data/datasets/mguamanc/learned_cost_map/cluster_scripts
EXE_SCRIPT=$WORKING_DIR/train_CostFourierVelModelEfficientNet.sh

USER=mguamanc

nvidia-docker run --rm --ipc=host -e CUDA_VISIBLE_DEVICES=`echo $CUDA_VISIBLE_DEVICES` -v /data/datasets:/data/datasets -v /home/$USER:/home/$USER -v /project:/project mguamanc/sara $EXE $EXE_SCRIPT