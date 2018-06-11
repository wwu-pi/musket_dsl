#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/double/GPU/build/ && \
mkdir ~/musket-build/double/GPU/build/ && \

# run cmake
cd ~/musket-build/double/GPU/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark -D CMAKE_CXX_COMPILER=pgc++ ../ && \

make double && \
cd .. && \
mkdir -p ~/musket-build/double/GPU/out/ && \
sbatch job.sh
