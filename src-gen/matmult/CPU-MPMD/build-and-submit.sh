#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/matmult/CPU-MPMD/build/ && \
mkdir ~/musket-build/matmult/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/matmult/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make matmult && \
cd .. && \
mkdir -p ~/musket-build/matmult/CPU-MPMD/out/ && \
sbatch job.sh
