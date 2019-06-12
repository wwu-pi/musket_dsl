#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/frobenius/GPU/out/ && \
rm -rf -- ~/build/mnp/frobenius/gpu && \
mkdir -p ~/build/mnp/frobenius/gpu && \

# run cmake
cd ~/build/mnp/frobenius/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make frobenius_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
