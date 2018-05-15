#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/double/CPU-MPMD/build/ && \
mkdir ~/musket-build/double/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/double/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make double && \
cd .. && \
mkdir -p ~/musket-build/double/CPU-MPMD/out/ && \
sbatch job.sh
