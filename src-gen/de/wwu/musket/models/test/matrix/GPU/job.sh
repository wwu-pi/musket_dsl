#!/bin/bash
#SBATCH --job-name matrix-nodes-1-cpu-4
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition normal
#SBATCH --output ~/musket-build/de/wwu/musket/models/test/matrix/GPU/out/matrix-nodes-1-cpu-4.out
#SBATCH --cpus-per-task 64
#SBATCH --mail-type ALL
#SBATCH --mail-user my@e-mail.de
#SBATCH --time 01:00:00

export OMP_NUM_THREADS=4

~/musket-build/de/wwu/musket/models/test/matrix/GPU/build/bin/matrix
