#!/bin/bash
#SBATCH --job-name nbody-n-16-g-4-GPU-nodes-16-gpu-4
#SBATCH --ntasks 16
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --exclusive
#SBATCH --output /home/fwrede/musket-build/hlpp19/nbody-n-16-g-4/GPU/out/nbody-n-16-g-4-nodes-16-gpu-4.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 03:00:00
#SBATCH -A p_algcpugpu
#SBATCH --gres gpu:4

export OMP_NUM_THREADS=24

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun --multi-prog /home/fwrede/musket/src-gen/hlpp19/nbody-n-16-g-4/GPU/job.conf
done
