#!/bin/bash
#SBATCH --job-name frobenius-n-1-g-4-GPU-nvprof-nodes-1-gpu-4
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --exclusive
#SBATCH --output /home/fwrede/musket-build/hlpp19/frobenius-n-1-g-4/CUDA/out/frobenius-n-1-g-4-nvprof-nodes-1-gpu-4.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 01:00:00
#SBATCH -A p_algcpugpu
#SBATCH --gres gpu:4

export OMP_NUM_THREADS=24

RUNS=1
for ((i=1;i<=RUNS;i++)); do
	nvprof ~/out/mnp/frobenius-n-1-g-4-cuda-%p.out --annotate-mpi openmpi ~/build/mnp/frobenius-n-1-g-4/cuda/bin/frobenius-n-1-g-4_0
done
