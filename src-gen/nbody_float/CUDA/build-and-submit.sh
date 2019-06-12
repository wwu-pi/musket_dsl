#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/nbody_float/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/nbody_float/CUDA/build/benchmark && \
mkdir -p /home/fwrede/musket-build/nbody_float/CUDA/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/nbody_float/CUDA/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make nbody_float_0 && \
cd ${source_folder} && \

sbatch job.sh
