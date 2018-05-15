#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/frobenius/CPU-MPMD/build/ && \
mkdir ~/musket-build/frobenius/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/frobenius/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make frobenius && \
cd .. && \
mkdir -p ~/musket-build/frobenius/CPU-MPMD/out/ && \
sbatch job.sh
