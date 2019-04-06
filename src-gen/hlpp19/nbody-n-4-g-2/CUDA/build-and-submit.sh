#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-4-g-2/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/nbody-n-4-g-2/CUDA/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-4-g-2/CUDA/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/nbody-n-4-g-2/CUDA/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make nbody-n-4-g-2_0 && \
make nbody-n-4-g-2_1 && \
make nbody-n-4-g-2_2 && \
make nbody-n-4-g-2_3 && \
cd ${source_folder} && \

sbatch job.sh
