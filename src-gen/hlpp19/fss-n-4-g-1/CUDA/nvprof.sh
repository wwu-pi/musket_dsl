#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/fss-n-4-g-1/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/fss-n-4-g-1/CUDA/build/nvprof && \
mkdir -p /home/fwrede/musket-build/hlpp19/fss-n-4-g-1/CUDA/build/nvprof && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/fss-n-4-g-1/CUDA/build/nvprof && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make fss-n-4-g-1_0 && \
make fss-n-4-g-1_1 && \
make fss-n-4-g-1_2 && \
make fss-n-4-g-1_3 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
