#!/bin/bash
#SBATCH --job-name map-nodes-0-cpu-0
#SBATCH --ntasks 0
#SBATCH --nodes 0
#SBATCH --ntasks-per-node 1
#SBATCH --partition normal
#SBATCH --output ~/musket-build/de/wwu/musket/models/test/map/CPU/out/map-nodes-0-cpu-0.out
#SBATCH --cpus-per-task 64
#SBATCH --mail-type ALL
#SBATCH --mail-user my@e-mail.de
#SBATCH --time 01:00:00

export OMP_NUM_THREADS=0

~/musket-build/de/wwu/musket/models/test/map/CPU/build/bin/map
