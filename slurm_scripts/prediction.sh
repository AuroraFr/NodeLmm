#!/bin/bash
#SBATCH -J prediction_real3C
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
 
python prediction_ODE_real.py --compute_ci --ci_method delta --fisher_cache fisher.npy

