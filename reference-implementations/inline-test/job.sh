#!/bin/bash
#SBATCH --job-name inline-tests
#SBATCH --ntasks 4
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --partition haswell
#SBATCH --exclusive
#SBATCH --exclude taurusi[1001-1270],taurusi[3001-3180],taurusi[2001-2108],taurussmp[1-7],taurusknl[1-32]
#SBATCH --output /home/fwrede/out/hlpp18/inline-tests-nodes-4-cpu-24.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user fabian.wrede@mailbox.tu-dresden.de
#SBATCH --time 5-00:00:00
#SBATCH -A p_algcpugpu

export OMP_NUM_THREADS=24

RUNS=5
TESTS="frobenius_inline frobenius_inline_w_gather frobenius_skeleton_functions frobenius_user_functions frobenius_skeleton_user_functions frobenius_mapfold_inline frobenius_mapfold_skeleton_functions frobenius_mapfold_user_functions frobenius_mapfold_skeleton_user_functions matmult_inline matmult_skeleton_functions matmult_user_functions matmult_skeleton_user_functions nbody_inline nbody_skeleton_functions nbody_user_functions nbody_skeleton_user_functions"

for test in $TESTS; do
	echo "\n\n######################################################### ${test} #########################################################\n\n" 
	for ((i=1;i<=RUNS;i++)); do
    srun /home/fwrede/musket-build/inline-test/build/bin/${test}
	done
done
