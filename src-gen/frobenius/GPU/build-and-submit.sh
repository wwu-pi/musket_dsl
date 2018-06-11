#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/frobenius/GPU/build/ && \
mkdir ~/musket-build/frobenius/GPU/build/ && \

# run cmake
cd ~/musket-build/frobenius/GPU/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark -D CMAKE_CXX_COMPILER=pgc++ ../ && \

make frobenius && \
cd .. && \
mkdir -p ~/musket-build/frobenius/GPU/out/ && \
sbatch job.sh
