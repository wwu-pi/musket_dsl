#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/matmult_float/GPU/build/ && \
mkdir ~/musket-build/matmult_float/GPU/build/ && \

# run cmake
cd ~/musket-build/matmult_float/GPU/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark -D CMAKE_CXX_COMPILER=pgc++ ../ && \

make matmult_float && \
cd .. && \
mkdir -p ~/musket-build/matmult_float/GPU/out/ && \
sbatch job.sh
