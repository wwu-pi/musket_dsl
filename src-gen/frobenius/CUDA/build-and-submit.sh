#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/frobenius/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/frobenius/CUDA/build/benchmark && \
mkdir -p /home/fwrede/musket-build/frobenius/CUDA/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/frobenius/CUDA/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make frobenius_0 && \
cd ${source_folder} && \

sbatch job.sh
