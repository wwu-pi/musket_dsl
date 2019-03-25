#!/bin/bash
#SBATCH --job-name matmult-n-16-g-4-CPU-MPMD-nodes-16-cpu-24
#SBATCH --ntasks 16
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclusive
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/musket-build/hlpp19/matmult-n-16-g-4/CPU-MPMD/out/matmult-n-16-g-4-nodes-16-cpu-24.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 05:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=24

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun --multi-prog /home/fwrede/musket/src-gen/hlpp19/matmult-n-16-g-4/CPU-MPMD/job.conf
done	
