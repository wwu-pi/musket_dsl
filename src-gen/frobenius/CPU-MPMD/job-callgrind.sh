#!/bin/bash
#SBATCH --job-name frobenius-CPU-MPMD-callgrind-nodes-1-cpu-4
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclusive
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/musket-build/frobenius/CPU-MPMD/out/frobenius-nodes-1-cpu-4.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 05:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=4

srun --multi-prog /home/fwrede/musket/src-gen/frobenius/CPU-MPMD/job-callgrind.conf
