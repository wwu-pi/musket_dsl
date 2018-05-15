#!/bin/bash
#SBATCH --job-name double-nodes-4-cpu-8
#SBATCH --ntasks 4
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --partition normal
#SBATCH --output ~/musket-build/double/CPU-MPMD/out/double-nodes-4-cpu-8.out
#SBATCH --cpus-per-task 64
#SBATCH --mail-type ALL
#SBATCH --mail-user my@e-mail.de
#SBATCH --time 01:00:00

export OMP_NUM_THREADS=8

mpirun ~/musket-build/double/CPU-MPMD/build/bin/double
