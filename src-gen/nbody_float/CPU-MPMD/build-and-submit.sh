#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/nbody_float/CPU-MPMD/build/ && \
mkdir ~/musket-build/nbody_float/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/nbody_float/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make nbody_float && \
cd .. && \
mkdir -p ~/musket-build/nbody_float/CPU-MPMD/out/ && \
sbatch job.sh
