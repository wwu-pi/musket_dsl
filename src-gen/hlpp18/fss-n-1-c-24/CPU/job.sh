#!/bin/bash
#SBATCH --job-name fss-n-1-c-24-nodes-1-cpu-24
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/musket-build/hlpp18/fss-n-1-c-24/CPU/out/fss-n-1-c-24-nodes-1-cpu-24.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 05:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=24

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/hlpp18/fss-n-1-c-24/CPU/build/benchmark/bin/fss-n-1-c-24
done		
