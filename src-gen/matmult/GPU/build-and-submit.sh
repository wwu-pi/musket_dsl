#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/matmult/GPU/build/ && \
mkdir ~/musket-build/matmult/GPU/build/ && \

# run cmake
cd ~/musket-build/matmult/GPU/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark -D CMAKE_CXX_COMPILER=pgc++ ../ && \

make matmult && \
cd .. && \
mkdir -p ~/musket-build/matmult/GPU/out/ && \
sbatch job.sh
