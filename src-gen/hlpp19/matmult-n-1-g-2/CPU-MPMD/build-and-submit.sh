#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-1-g-2/CPU-MPMD/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/matmult-n-1-g-2/CPU-MPMD/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-1-g-2/CPU-MPMD/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/matmult-n-1-g-2/CPU-MPMD/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make matmult-n-1-g-2_0 && \
cd ${source_folder} && \

sbatch job.sh
