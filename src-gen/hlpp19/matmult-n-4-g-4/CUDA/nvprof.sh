#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-4-g-4/CUDA/out/ && \
rm -rf -- ~/build/mnp/matmult-n-4-g-4/cuda && \
mkdir -p ~/build/mnp/matmult-n-4-g-4/cuda && \

# run cmake
cd ~/build/mnp/matmult-n-4-g-4/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make matmult-n-4-g-4_0 && \
make matmult-n-4-g-4_1 && \
make matmult-n-4-g-4_2 && \
make matmult-n-4-g-4_3 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
