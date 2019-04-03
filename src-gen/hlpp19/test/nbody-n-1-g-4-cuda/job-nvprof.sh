#!/bin/bash
#SBATCH --job-name nbody-test-n-1-g-4-cuda-nvprof
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --exclusive
#SBATCH --output /home/fwrede/out/hlpp19/musket-test/nbody-n-1-g-4-cuda-nvprof.out
#SBATCH --error /home/fwrede/out/hlpp19/musket-test/nbody-n-1-g-4-cuda-nvprof.err
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 00:15:00
#SBATCH -A p_algcpugpu
#SBATCH --gres gpu:4

export OMP_NUM_THREADS=24

RUNS=1
for ((i=1;i<=RUNS;i++)); do
    nvprof /home/fwrede/build/hlpp19/musket/nbody-n-1-g-4-cuda/bin/nbody-n-1-g-4_0
done
