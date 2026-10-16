#!/usr/bin/env bash
#SBATCH -J var_base
# Asking for one node
#SBATCH -w bora040 --time=2-10:00:00 --exclusive
# Standard output
#SBATCH -o logs/baseline_variance.%j.out
# Standard error
#SBATCH -e logs/baseline_variance.%j.err

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

python PDP_run.py --config configs/pdp_sim_baseline.json
