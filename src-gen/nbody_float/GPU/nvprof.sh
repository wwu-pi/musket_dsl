#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/nbody_float/GPU/out/ && \
rm -rf -- ~/build/mnp/nbody_float/gpu && \
mkdir -p ~/build/mnp/nbody_float/gpu && \

# run cmake
cd ~/build/mnp/nbody_float/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make nbody_float_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
