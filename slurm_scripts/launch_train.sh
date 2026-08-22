#!/bin/bash

for START_IDX in 0 10 20 30 40 50 60 70 80 90; do
    echo "Submitting init=$START_IDX"
    sbatch train.sh $START_IDX
done
