#!/bin/bash
#SBATCH --job-name fss-n-1-g-1-CPU-MPMD-callgrind-nodes-1-cpu-1
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclusive
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/musket-build/hlpp19/fss-n-1-g-1/CPU-MPMD/out/fss-n-1-g-1-nodes-1-cpu-1.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 05:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=1

srun --multi-prog /home/fwrede/musket/src-gen/hlpp19/fss-n-1-g-1/CPU-MPMD/job-callgrind.conf