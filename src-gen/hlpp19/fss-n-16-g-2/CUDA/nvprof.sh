#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/fss-n-16-g-2/CUDA/out/ && \
rm -rf -- ~/build/mnp/fss-n-16-g-2/cuda && \
mkdir -p ~/build/mnp/fss-n-16-g-2/cuda && \

# run cmake
cd ~/build/mnp/fss-n-16-g-2/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make fss-n-16-g-2_0 && \
make fss-n-16-g-2_1 && \
make fss-n-16-g-2_2 && \
make fss-n-16-g-2_3 && \
make fss-n-16-g-2_4 && \
make fss-n-16-g-2_5 && \
make fss-n-16-g-2_6 && \
make fss-n-16-g-2_7 && \
make fss-n-16-g-2_8 && \
make fss-n-16-g-2_9 && \
make fss-n-16-g-2_10 && \
make fss-n-16-g-2_11 && \
make fss-n-16-g-2_12 && \
make fss-n-16-g-2_13 && \
make fss-n-16-g-2_14 && \
make fss-n-16-g-2_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
