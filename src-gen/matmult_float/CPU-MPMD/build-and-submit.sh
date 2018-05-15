#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/matmult_float/CPU-MPMD/build/ && \
mkdir ~/musket-build/matmult_float/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/matmult_float/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make matmult_float && \
cd .. && \
mkdir -p ~/musket-build/matmult_float/CPU-MPMD/out/ && \
sbatch job.sh
