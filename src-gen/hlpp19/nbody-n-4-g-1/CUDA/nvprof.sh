#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-4-g-1/CUDA/out/ && \
rm -rf -- ~/build/mnp/nbody-n-4-g-1/cuda && \
mkdir -p ~/build/mnp/nbody-n-4-g-1/cuda && \

# run cmake
cd ~/build/mnp/nbody-n-4-g-1/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make nbody-n-4-g-1_0 && \
make nbody-n-4-g-1_1 && \
make nbody-n-4-g-1_2 && \
make nbody-n-4-g-1_3 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
