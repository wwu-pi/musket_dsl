#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/matmult_float/CPU-MPMD/out/cg && \
rm -rf -- /home/fwrede/musket-build/matmult_float/CPU-MPMD/build/cg && \
mkdir -p /home/fwrede/musket-build/matmult_float/CPU-MPMD/build/cg && \

# run cmake
cd /home/fwrede/musket-build/matmult_float/CPU-MPMD/build/cg && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make matmult_float_0 && \
make matmult_float_1 && \
make matmult_float_2 && \
make matmult_float_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
