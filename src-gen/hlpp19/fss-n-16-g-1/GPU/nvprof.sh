#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/fss-n-16-g-1/GPU/out/ && \
rm -rf -- ~/build/mnp/fss-n-16-g-1/gpu && \
mkdir -p ~/build/mnp/fss-n-16-g-1/gpu && \

# run cmake
cd ~/build/mnp/fss-n-16-g-1/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make fss-n-16-g-1_0 && \
make fss-n-16-g-1_1 && \
make fss-n-16-g-1_2 && \
make fss-n-16-g-1_3 && \
make fss-n-16-g-1_4 && \
make fss-n-16-g-1_5 && \
make fss-n-16-g-1_6 && \
make fss-n-16-g-1_7 && \
make fss-n-16-g-1_8 && \
make fss-n-16-g-1_9 && \
make fss-n-16-g-1_10 && \
make fss-n-16-g-1_11 && \
make fss-n-16-g-1_12 && \
make fss-n-16-g-1_13 && \
make fss-n-16-g-1_14 && \
make fss-n-16-g-1_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
