#!/usr/bin/env bash
#SBATCH -J S2_2_sel
# Asking for one node
#SBATCH -w bora044 --time=2-10:00:00 --exclusive
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

python cumulative_model_selection.py --n_sims 3