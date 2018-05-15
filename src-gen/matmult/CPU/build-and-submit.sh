#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/matmult/CPU/out/ && \
rm -rf -- /home/fwrede/musket-build/matmult/CPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/matmult/CPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/matmult/CPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make matmult && \
cd ${source_folder} && \

sbatch job.sh
