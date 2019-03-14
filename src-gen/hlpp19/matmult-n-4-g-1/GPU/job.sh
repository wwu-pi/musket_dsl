#!/bin/bash
#SBATCH --job-name matmult-n-4-g-1-GPU-nodes-4-gpu-1
#SBATCH --ntasks 4
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --exclusive
#SBATCH --output /home/fwrede/musket-build/hlpp19/matmult-n-4-g-1/GPU/out/matmult-n-4-g-1-nodes-4-gpu-1.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 00:05:00
#SBATCH -A p_algcpugpu
#SBATCH --gres gpu:4

RUNS=1
for ((i=1;i<=RUNS;i++)); do
    srun --multi-prog /home/fwrede/musket/src-gen/hlpp19/matmult-n-4-g-1/GPU/job.conf
done
