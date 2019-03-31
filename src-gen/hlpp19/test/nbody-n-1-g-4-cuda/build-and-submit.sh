#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/out/hlpp19/musket-test && \
rm -rf -- /home/fwrede/build/hlpp19/musket/nbody-n-1-g-4-cuda && \
mkdir -p /home/fwrede/build/hlpp19/musket/nbody-n-1-g-4-cuda && \

# run cmake
cd /home/fwrede/build/hlpp19/musket/nbody-n-1-g-4-cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ && \

make nbody-n-1-g-4_0 && \
cd ${source_folder} && \

sbatch job.sh
