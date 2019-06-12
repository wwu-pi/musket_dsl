#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/de/wwu/musket/models/test/matrix/CUDA/out/ && \
rm -rf -- ~/build/mnp/matrix/cuda && \
mkdir -p ~/build/mnp/matrix/cuda && \

# run cmake
cd ~/build/mnp/matrix/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make matrix_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
