#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/nbody/CPU-MPMD/build/ && \
mkdir ~/musket-build/nbody/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/nbody/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make nbody && \
cd .. && \
mkdir -p ~/musket-build/nbody/CPU-MPMD/out/ && \
sbatch job.sh
