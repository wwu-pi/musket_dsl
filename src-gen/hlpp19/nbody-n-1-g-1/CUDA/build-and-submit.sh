#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-1-g-1/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/nbody-n-1-g-1/CUDA/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-1-g-1/CUDA/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/nbody-n-1-g-1/CUDA/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make nbody-n-1-g-1_0 && \
cd ${source_folder} && \

sbatch job.sh
