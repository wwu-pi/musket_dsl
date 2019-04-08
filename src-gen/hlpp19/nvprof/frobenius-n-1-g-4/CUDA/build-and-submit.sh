#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-1-g-4/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/frobenius-n-1-g-4/CUDA/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-1-g-4/CUDA/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/frobenius-n-1-g-4/CUDA/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make frobenius-n-1-g-4_0 && \
cd ${source_folder} && \

sbatch job.sh
