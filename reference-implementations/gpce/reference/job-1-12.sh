#!/bin/bash
#SBATCH --job-name fss-reference-nodes-1-cpu-12
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclusive
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/out/gpce/fss-reference/fss-reference-nodes-1-cpu-12.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 23:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=12

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/gpce/fss-reference/benchmark/bin/value_1
done

echo "---------------------------------------------------------------------------"

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/gpce/fss-reference/benchmark/bin/reference_1
done

echo "---------------------------------------------------------------------------"

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/gpce/fss-reference/benchmark/bin/reference_1_fused
done