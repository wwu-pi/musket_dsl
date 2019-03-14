#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-4-g-1/CPU-MPMD/out/cg && \
rm -rf -- /home/fwrede/musket-build/hlpp19/matmult-n-4-g-1/CPU-MPMD/build/cg && \
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-4-g-1/CPU-MPMD/build/cg && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/matmult-n-4-g-1/CPU-MPMD/build/cg && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make matmult-n-4-g-1_0 && \
make matmult-n-4-g-1_1 && \
make matmult-n-4-g-1_2 && \
make matmult-n-4-g-1_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
