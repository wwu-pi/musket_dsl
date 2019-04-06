#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-16-g-4/GPU/out/ && \
rm -rf -- ~/build/mnp/matmult-n-16-g-4/gpu && \
mkdir -p ~/build/mnp/matmult-n-16-g-4/gpu && \

# run cmake
cd ~/build/mnp/matmult-n-16-g-4/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make matmult-n-16-g-4_0 && \
make matmult-n-16-g-4_1 && \
make matmult-n-16-g-4_2 && \
make matmult-n-16-g-4_3 && \
make matmult-n-16-g-4_4 && \
make matmult-n-16-g-4_5 && \
make matmult-n-16-g-4_6 && \
make matmult-n-16-g-4_7 && \
make matmult-n-16-g-4_8 && \
make matmult-n-16-g-4_9 && \
make matmult-n-16-g-4_10 && \
make matmult-n-16-g-4_11 && \
make matmult-n-16-g-4_12 && \
make matmult-n-16-g-4_13 && \
make matmult-n-16-g-4_14 && \
make matmult-n-16-g-4_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
