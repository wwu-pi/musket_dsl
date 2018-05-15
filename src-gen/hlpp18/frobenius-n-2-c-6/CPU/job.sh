#!/bin/bash
#SBATCH --job-name frobenius-n-2-c-6-nodes-2-cpu-6
#SBATCH --ntasks 2
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/musket-build/hlpp18/frobenius-n-2-c-6/CPU/out/frobenius-n-2-c-6-nodes-2-cpu-6.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 05:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=6

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/hlpp18/frobenius-n-2-c-6/CPU/build/benchmark/bin/frobenius-n-2-c-6
done		
