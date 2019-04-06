#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-1-g-1/CUDA/out/ && \
rm -rf -- ~/build/mnp/frobenius-n-1-g-1/cuda && \
mkdir -p ~/build/mnp/frobenius-n-1-g-1/cuda && \

# run cmake
cd ~/build/mnp/frobenius-n-1-g-1/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make frobenius-n-1-g-1_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
