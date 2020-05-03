#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/out/sac19 && \
rm -rf -- /home/fwrede/musket-build/sac19/benchmark && \
mkdir -p /home/fwrede/musket-build/sac19/benchmark && \

# run cmake
cd /home/fwrede/musket-build/sac19/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ${source_folder} && \

make && \
cd ${source_folder} && \

sbatch job-1-1.sh && \
sbatch job-1-12.sh && \
sbatch job-1-24.sh