#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/fss-n-16-g-4/GPU/out/ && \
rm -rf -- ~/build/mnp/fss-n-16-g-4/gpu && \
mkdir -p ~/build/mnp/fss-n-16-g-4/gpu && \

# run cmake
cd ~/build/mnp/fss-n-16-g-4/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make fss-n-16-g-4_0 && \
make fss-n-16-g-4_1 && \
make fss-n-16-g-4_2 && \
make fss-n-16-g-4_3 && \
make fss-n-16-g-4_4 && \
make fss-n-16-g-4_5 && \
make fss-n-16-g-4_6 && \
make fss-n-16-g-4_7 && \
make fss-n-16-g-4_8 && \
make fss-n-16-g-4_9 && \
make fss-n-16-g-4_10 && \
make fss-n-16-g-4_11 && \
make fss-n-16-g-4_12 && \
make fss-n-16-g-4_13 && \
make fss-n-16-g-4_14 && \
make fss-n-16-g-4_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
