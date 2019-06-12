#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/de/wwu/musket/models/test/matrix/GPU/out/ && \
rm -rf -- ~/build/mnp/matrix/gpu && \
mkdir -p ~/build/mnp/matrix/gpu && \

# run cmake
cd ~/build/mnp/matrix/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make matrix_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
