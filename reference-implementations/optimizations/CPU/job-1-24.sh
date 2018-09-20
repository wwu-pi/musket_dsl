#!/bin/bash
#SBATCH --job-name opt-nodes-1-cpu-24
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclusive
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/out/sac19/opt-nodes-1-cpu-24.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 23:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=24

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/data
done

echo "---------------------------------------------------------------------------"

for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/data_opt
done

echo "---------------------------------------------------------------------------"

for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/data_opt_move
done

echo "---------------------------------------------------------------------------"
echo "###########################################################################"
echo "---------------------------------------------------------------------------"

for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/map
done

echo "---------------------------------------------------------------------------"

for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/map_opt
done

echo "---------------------------------------------------------------------------"
echo "###########################################################################"
echo "---------------------------------------------------------------------------"

for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/map_fold
done

echo "---------------------------------------------------------------------------"

for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/map_fold_opt
done

echo "---------------------------------------------------------------------------"
echo "###########################################################################"
echo "---------------------------------------------------------------------------"

for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/ref
done

echo "---------------------------------------------------------------------------"

for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/sac19/benchmark/bin/ref_opt
done