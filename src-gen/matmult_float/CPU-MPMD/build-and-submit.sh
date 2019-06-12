#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/matmult_float/CPU-MPMD/out/ && \
rm -rf -- /home/fwrede/musket-build/matmult_float/CPU-MPMD/build/benchmark && \
mkdir -p /home/fwrede/musket-build/matmult_float/CPU-MPMD/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/matmult_float/CPU-MPMD/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make matmult_float_0 && \
make matmult_float_1 && \
make matmult_float_2 && \
make matmult_float_3 && \
cd ${source_folder} && \

sbatch job.sh
