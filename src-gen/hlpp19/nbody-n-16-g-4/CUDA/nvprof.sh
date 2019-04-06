#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-16-g-4/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/nbody-n-16-g-4/CUDA/build/nvprof && \
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-16-g-4/CUDA/build/nvprof && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/nbody-n-16-g-4/CUDA/build/nvprof && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make nbody-n-16-g-4_0 && \
make nbody-n-16-g-4_1 && \
make nbody-n-16-g-4_2 && \
make nbody-n-16-g-4_3 && \
make nbody-n-16-g-4_4 && \
make nbody-n-16-g-4_5 && \
make nbody-n-16-g-4_6 && \
make nbody-n-16-g-4_7 && \
make nbody-n-16-g-4_8 && \
make nbody-n-16-g-4_9 && \
make nbody-n-16-g-4_10 && \
make nbody-n-16-g-4_11 && \
make nbody-n-16-g-4_12 && \
make nbody-n-16-g-4_13 && \
make nbody-n-16-g-4_14 && \
make nbody-n-16-g-4_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
