#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-16-g-2/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/matmult-n-16-g-2/CUDA/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-16-g-2/CUDA/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/matmult-n-16-g-2/CUDA/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make matmult-n-16-g-2_0 && \
make matmult-n-16-g-2_1 && \
make matmult-n-16-g-2_2 && \
make matmult-n-16-g-2_3 && \
make matmult-n-16-g-2_4 && \
make matmult-n-16-g-2_5 && \
make matmult-n-16-g-2_6 && \
make matmult-n-16-g-2_7 && \
make matmult-n-16-g-2_8 && \
make matmult-n-16-g-2_9 && \
make matmult-n-16-g-2_10 && \
make matmult-n-16-g-2_11 && \
make matmult-n-16-g-2_12 && \
make matmult-n-16-g-2_13 && \
make matmult-n-16-g-2_14 && \
make matmult-n-16-g-2_15 && \
cd ${source_folder} && \

sbatch job.sh
