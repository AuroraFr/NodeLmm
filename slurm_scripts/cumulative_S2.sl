#!/usr/bin/env bash
#SBATCH -J CUMU_ODELMM
# Asking for one node
#SBATCH -w bora019 --time=2-10:00:00 --exclusive
# Standard output
#SBATCH -o slurm.sh%j.out
# Standard error
#SBATCH -e slurm.sh%j.err

echo "=====my job information ===="
echo "Node List: " $SLURM_NODELIST
echo "my jobID: " $SLURM_JOB_ID
echo "Partition: " $SLURM_JOB_PARTITION
echo "submit directory:" $SLURM_SUBMIT_DIR
echo "submit host:" $SLURM_SUBMIT_HOST
echo "In the directory:" $PWD
echo "As the user:" $USER

module purge
source ~/torch/bin/activate
cd /beegfs/zli/workspace/CDE_LMM/

python train_ODE_cumulative.py