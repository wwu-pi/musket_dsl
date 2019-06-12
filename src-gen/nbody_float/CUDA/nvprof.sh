#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/nbody_float/CUDA/out/ && \
rm -rf -- ~/build/mnp/nbody_float/cuda && \
mkdir -p ~/build/mnp/nbody_float/cuda && \

# run cmake
cd ~/build/mnp/nbody_float/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make nbody_float_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
