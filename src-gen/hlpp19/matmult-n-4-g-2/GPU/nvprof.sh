#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-4-g-2/GPU/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/matmult-n-4-g-2/GPU/build/nvprof && \
mkdir -p /home/fwrede/musket-build/hlpp19/matmult-n-4-g-2/GPU/build/nvprof && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/matmult-n-4-g-2/GPU/build/nvprof && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make matmult-n-4-g-2_0 && \
make matmult-n-4-g-2_1 && \
make matmult-n-4-g-2_2 && \
make matmult-n-4-g-2_3 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
