#!/bin/bash
#SBATCH -J train_base
#SBATCH -t 20:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH -p routage
#SBATCH --nodelist=bora[001-044]
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=.
#SBATCH --output=logs/train_base/train_%j.out
#SBATCH --error=logs/train_base/train_%j.err
#SBATCH --exclusive

START_IDX=$1
echo "dataset=$START_IDX"

module purge
source ~/torch/bin/activate
cd /beegfs/zli/workspace/CDE_LMM/

OPENBLAS_NUM_THREADS=8 python train.py --sim_start $START_IDX --config configs/config_basic.json