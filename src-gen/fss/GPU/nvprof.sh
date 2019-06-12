#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/fss/GPU/out/ && \
rm -rf -- ~/build/mnp/fss/gpu && \
mkdir -p ~/build/mnp/fss/gpu && \

# run cmake
cd ~/build/mnp/fss/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make fss_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
