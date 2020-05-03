#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/out/gpce/fss-reference && \
rm -rf -- /home/fwrede/musket-build/gpce/fss-reference/benchmark && \
mkdir -p /home/fwrede/musket-build/gpce/fss-reference/benchmark && \

# run cmake
cd /home/fwrede/musket-build/gpce/fss-reference/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ${source_folder} && \

make && \
cd ${source_folder} && \

sbatch job-1-1.sh && \
sbatch job-1-12.sh && \
sbatch job-1-24.sh && \
sbatch job-4-1.sh && \
sbatch job-4-12.sh && \
sbatch job-4-24.sh