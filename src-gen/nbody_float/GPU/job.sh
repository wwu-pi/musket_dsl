#!/bin/bash
#SBATCH --job-name nbody_float-nodes-1-cpu-4
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition normal
#SBATCH --output ~/musket-build/nbody_float/GPU/out/nbody_float-nodes-1-cpu-4.out
#SBATCH --cpus-per-task 64
#SBATCH --mail-type ALL
#SBATCH --mail-user my@e-mail.de
#SBATCH --time 01:00:00

export OMP_NUM_THREADS=4

~/musket-build/nbody_float/GPU/build/bin/nbody_float