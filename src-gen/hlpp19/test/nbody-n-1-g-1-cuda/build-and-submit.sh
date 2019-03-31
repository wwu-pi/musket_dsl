#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/out/hlpp19/musket-test && \
rm -rf -- /home/fwrede/build/hlpp19/musket/nbody-n-1-g-1-cuda && \
mkdir -p /home/fwrede/build/hlpp19/musket/nbody-n-1-g-1-cuda && \

# run cmake
cd /home/fwrede/build/hlpp19/musket/nbody-n-1-g-1-cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make nbody-n-1-g-1_0 && \
cd ${source_folder} && \

sbatch job.sh
