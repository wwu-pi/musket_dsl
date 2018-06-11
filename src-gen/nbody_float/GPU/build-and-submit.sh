#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/nbody_float/GPU/build/ && \
mkdir ~/musket-build/nbody_float/GPU/build/ && \

# run cmake
cd ~/musket-build/nbody_float/GPU/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark -D CMAKE_CXX_COMPILER=pgc++ ../ && \

make nbody_float && \
cd .. && \
mkdir -p ~/musket-build/nbody_float/GPU/out/ && \
sbatch job.sh
