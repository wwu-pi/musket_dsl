#!/bin/bash
#SBATCH --job-name frobenius-n-4-g-4-GPU-nodes-4-gpu-4
#SBATCH --ntasks 4
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --exclusive
#SBATCH --output /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-4/GPU/out/frobenius-n-4-g-4-nodes-4-gpu-4.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 01:00:00
#SBATCH -A p_algcpugpu
#SBATCH --gres gpu:4

export OMP_NUM_THREADS=24

RUNS=1
for ((i=1;i<=RUNS;i++)); do
    srun --multi-prog /home/fwrede/musket/src-gen/hlpp19/frobenius-n-4-g-4/GPU/job.conf
done
