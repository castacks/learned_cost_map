#!/bin/bash

# SLURM Resource Parameters

#SBATCH -N 10  # CPU Cores
#SBATCH -t 1-00:00 # D-HH:MM
#SBATCH -p a100 # cpu/gpu/dgx
#SBATCH -w a100-gpu-full 
#SBATCH --gres=gpu:1
#SBATCH --mem=80G  # MB
#SBATCH --job-name=train_CostFourierVelModel_balanced_singularity
#SBATCH -o /home/mguamanc/job_%j.out
#SBATCH -e /home/mguamanc/job_%j.err
#SBATCH --mail-type=ALL # BEGIN, END, FAIL, ALL
#SBATCH --mail-user=mguamanc@andrew.cmu.edu

# Executable
EXE=/bin/bash
SINGULARITY_DIR=/data1/datasets/mguamanc/singularity
WORKING_DIR=/data1/datasets/mguamanc/learned_cost_map/cluster_scripts
EXE_SCRIPT=$WORKING_DIR/train_CostFourierVelModelBalancedSingularity.sh

USER=mguamanc

cd $SINGULARITY_DIR
singularity instance start --nv sara.sif sara
singularity run --nv instance://sara
$EXE $EXE_SCRIPT