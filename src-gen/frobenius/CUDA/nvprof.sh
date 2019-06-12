#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/frobenius/CUDA/out/ && \
rm -rf -- ~/build/mnp/frobenius/cuda && \
mkdir -p ~/build/mnp/frobenius/cuda && \

# run cmake
cd ~/build/mnp/frobenius/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make frobenius_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh