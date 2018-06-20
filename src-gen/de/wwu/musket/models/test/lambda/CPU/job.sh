#!/bin/bash
#SBATCH --job-name lambda-nodes-4-cpu-8
#SBATCH --ntasks 4
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/musket-build/de/wwu/musket/models/test/lambda/CPU/out/lambda-nodes-4-cpu-8.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 05:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=8

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/de/wwu/musket/models/test/lambda/CPU/build/benchmark/bin/lambda
done		