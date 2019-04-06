#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/fss-n-1-g-4/GPU/out/ && \
rm -rf -- ~/build/mnp/fss-n-1-g-4/gpu && \
mkdir -p ~/build/mnp/fss-n-1-g-4/gpu && \

# run cmake
cd ~/build/mnp/fss-n-1-g-4/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make fss-n-1-g-4_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh