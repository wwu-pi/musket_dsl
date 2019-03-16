#!/bin/bash
#SBATCH --job-name nbody-n-1-g-1-CPU-MPMD-socket-nodes-1-cpu-24
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mincpus 24
#SBATCH --cpus-per-task 12
#SBATCH --cores-per-socket 12
#SBATCH --ntasks-per-socket 1
#SBATCH --threads-per-core 1
#SBATCH --partition haswell
#SBATCH --exclusive
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/musket-build/hlpp19/nbody-n-1-g-1/CPU-MPMD/out/nbody-n-1-g-1-socket-nodes-1-cpu-24.out
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 05:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=12

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun --multi-prog /home/fwrede/musket/src-gen/hlpp19/nbody-n-1-g-1/CPU-MPMD/job.conf
done	
