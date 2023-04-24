#!/bin/bash

#SBATCH -p GPU-small
#SBATCH -t 05:00:00
#SBATCH -n 5
#SBATCH -J train_CostFourierVelModel_balanced_singularity
#SBATCH --gpus=v100-32:1
#SBATCH -o /ocean/projects/cis220039p/guamanca/sbatch/outputs/learned_cost_map/job_%j.out
#SBATCH -e /ocean/projects/cis220039p/guamanca/sbatch/outputs/learned_cost_map/job_%j.err

'bash' /ocean/projects/cis220039p/guamanca/projects/learned_cost_map/cluster_scripts/train_psc.job
