#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/matmult_float/CUDA/out/ && \
rm -rf -- ~/build/mnp/matmult_float/cuda && \
mkdir -p ~/build/mnp/matmult_float/cuda && \

# run cmake
cd ~/build/mnp/matmult_float/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make matmult_float_0 && \
make matmult_float_1 && \
make matmult_float_2 && \
make matmult_float_3 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
