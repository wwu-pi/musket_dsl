#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-16-g-1/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/matmult-n-16-g-1/CUDA/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-16-g-1/CUDA/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/matmult-n-16-g-1/CUDA/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make matmult-n-16-g-1_0 && \
make matmult-n-16-g-1_1 && \
make matmult-n-16-g-1_2 && \
make matmult-n-16-g-1_3 && \
make matmult-n-16-g-1_4 && \
make matmult-n-16-g-1_5 && \
make matmult-n-16-g-1_6 && \
make matmult-n-16-g-1_7 && \
make matmult-n-16-g-1_8 && \
make matmult-n-16-g-1_9 && \
make matmult-n-16-g-1_10 && \
make matmult-n-16-g-1_11 && \
make matmult-n-16-g-1_12 && \
make matmult-n-16-g-1_13 && \
make matmult-n-16-g-1_14 && \
make matmult-n-16-g-1_15 && \
cd ${source_folder} && \

sbatch job.sh
