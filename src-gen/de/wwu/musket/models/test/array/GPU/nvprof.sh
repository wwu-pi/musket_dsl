#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/de/wwu/musket/models/test/array/GPU/out/ && \
rm -rf -- ~/build/mnp/array/gpu && \
mkdir -p ~/build/mnp/array/gpu && \

# run cmake
cd ~/build/mnp/array/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make array_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
