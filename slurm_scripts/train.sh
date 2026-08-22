#!/bin/bash
#SBATCH -J train_cumulative
#SBATCH -t 20:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH -p routage
#SBATCH --nodelist=bora[001-044]
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=.
#SBATCH --output=train_cumulative/train_%j.out
#SBATCH --error=train_cumulative/train_%j.err
#SBATCH --exclusive

START_IDX=$1
echo "dataset=$START_IDX"

module purge
source ~/torch/bin/activate
cd /beegfs/zli/workspace/CDE_LMM/

python train.py --sim_start $START_IDX --config configs/config_cumulative.json