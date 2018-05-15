#!/bin/bash
#SBATCH --job-name matmult_float-nodes-4-cpu-24
#SBATCH --ntasks 4
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --partition normal
#SBATCH --output ~/musket-build/matmult_float/CPU-MPMD/out/matmult_float-nodes-4-cpu-24.out
#SBATCH --cpus-per-task 64
#SBATCH --mail-type ALL
#SBATCH --mail-user my@e-mail.de
#SBATCH --time 01:00:00

export OMP_NUM_THREADS=24

mpirun ~/musket-build/matmult_float/CPU-MPMD/build/bin/matmult_float
