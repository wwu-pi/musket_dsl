#!/bin/bash
#SBATCH --job-name matmult_float-nodes-1-cpu-4
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition normal
#SBATCH --output ~/musket-build/matmult_float/GPU/out/matmult_float-nodes-1-cpu-4.out
#SBATCH --cpus-per-task 64
#SBATCH --mail-type ALL
#SBATCH --mail-user my@e-mail.de
#SBATCH --time 01:00:00

export OMP_NUM_THREADS=4

~/musket-build/matmult_float/GPU/build/bin/matmult_float
