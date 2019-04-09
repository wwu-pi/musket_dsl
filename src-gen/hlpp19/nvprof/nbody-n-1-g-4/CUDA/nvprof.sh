#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-1-g-4/CUDA/out/ && \
rm -rf -- ~/build/mnp/nbody-n-1-g-4/cuda && \
mkdir -p ~/build/mnp/nbody-n-1-g-4/cuda && \

# run cmake
cd ~/build/mnp/nbody-n-1-g-4/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make nbody-n-1-g-4_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh