#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/matmult_float/GPU/out/ && \
rm -rf -- ~/build/mnp/matmult_float/gpu && \
mkdir -p ~/build/mnp/matmult_float/gpu && \

# run cmake
cd ~/build/mnp/matmult_float/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make matmult_float_0 && \
make matmult_float_1 && \
make matmult_float_2 && \
make matmult_float_3 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
