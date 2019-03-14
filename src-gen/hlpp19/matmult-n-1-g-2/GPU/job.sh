#!/bin/bash
#SBATCH --job-name matmult-n-1-g-2-GPU-nodes-1-gpu-2
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --exclusive
#SBATCH --output /home/fwrede/musket-build/hlpp19/matmult-n-1-g-2/GPU/out/matmult-n-1-g-2-nodes-1-gpu-2.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 00:05:00
#SBATCH -A p_algcpugpu
#SBATCH --gres gpu:4

RUNS=1
for ((i=1;i<=RUNS;i++)); do
    srun --multi-prog /home/fwrede/musket/src-gen/hlpp19/matmult-n-1-g-2/GPU/job.conf
done
