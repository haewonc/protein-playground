#!/bin/bash

#SBATCH -p jepyc
#SBATCH -n 1
#SBATCH -J "haewon"
#SBATCH -o ../../output/%j.out 

module load anaconda3/2020.11 
module load python/3.7.2
module load cudatoolkit/11.1
module load gcc/10.2.0
source activate dhh

ulimit -s unlimited

srun python exp_svm.py