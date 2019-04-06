#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-16-g-2/CUDA/out/ && \
rm -rf -- ~/build/mnp/nbody-n-16-g-2/cuda && \
mkdir -p ~/build/mnp/nbody-n-16-g-2/cuda && \

# run cmake
cd ~/build/mnp/nbody-n-16-g-2/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make nbody-n-16-g-2_0 && \
make nbody-n-16-g-2_1 && \
make nbody-n-16-g-2_2 && \
make nbody-n-16-g-2_3 && \
make nbody-n-16-g-2_4 && \
make nbody-n-16-g-2_5 && \
make nbody-n-16-g-2_6 && \
make nbody-n-16-g-2_7 && \
make nbody-n-16-g-2_8 && \
make nbody-n-16-g-2_9 && \
make nbody-n-16-g-2_10 && \
make nbody-n-16-g-2_11 && \
make nbody-n-16-g-2_12 && \
make nbody-n-16-g-2_13 && \
make nbody-n-16-g-2_14 && \
make nbody-n-16-g-2_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
