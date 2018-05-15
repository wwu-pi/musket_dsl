#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/fss/CPU-MPMD/build/ && \
mkdir ~/musket-build/fss/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/fss/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make fss && \
cd .. && \
mkdir -p ~/musket-build/fss/CPU-MPMD/out/ && \
sbatch job.sh
