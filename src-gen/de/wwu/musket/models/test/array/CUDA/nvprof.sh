#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/de/wwu/musket/models/test/array/CUDA/out/ && \
rm -rf -- ~/build/mnp/array/cuda && \
mkdir -p ~/build/mnp/array/cuda && \

# run cmake
cd ~/build/mnp/array/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make array_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
