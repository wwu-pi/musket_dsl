#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-2/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-2/CUDA/build/nvprof && \
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-2/CUDA/build/nvprof && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-2/CUDA/build/nvprof && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make frobenius-n-4-g-2_0 && \
make frobenius-n-4-g-2_1 && \
make frobenius-n-4-g-2_2 && \
make frobenius-n-4-g-2_3 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
