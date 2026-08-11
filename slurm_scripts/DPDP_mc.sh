#!/bin/bash
#SBATCH -J DPDP_MC
#SBATCH -t 5:00:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=.
#SBATCH --output=logs/prediction_%j.out
#SBATCH --error=logs/prediction_%j.err
 
mkdir -p logs

source ~/Torch/bin/activate
cd /gpfs/home/zli/workspace/CDE_LMM/
 
python DPDP_mc_ci.py --sim_idx 0 --ledoit_wolf --n_mc 500

