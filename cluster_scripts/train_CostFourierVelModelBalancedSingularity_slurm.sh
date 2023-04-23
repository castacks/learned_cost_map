#!/bin/bash

# SLURM Resource Parameters

#SBATCH -n 10  # CPU Cores
#SBATCH -t 1-00:00 # D-HH:MM
#SBATCH -p dgx1-gpu # cpu/gpu/dgx 
#SBATCH --gres=gpu:1
#SBATCH --mem=32G  # MB
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
source /etc/profile.d/modules.sh
SIF="${SINGULARITY_DIR}/sara.sif"
S_EXEC="singularity exec -B /data1:/data1 --nv ${SIF}"

$S_EXEC $EXE_SCRIPT
