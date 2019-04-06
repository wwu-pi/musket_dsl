#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-4/CUDA/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-4/CUDA/build/nvprof && \
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-4/CUDA/build/nvprof && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-4/CUDA/build/nvprof && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make frobenius-n-16-g-4_0 && \
make frobenius-n-16-g-4_1 && \
make frobenius-n-16-g-4_2 && \
make frobenius-n-16-g-4_3 && \
make frobenius-n-16-g-4_4 && \
make frobenius-n-16-g-4_5 && \
make frobenius-n-16-g-4_6 && \
make frobenius-n-16-g-4_7 && \
make frobenius-n-16-g-4_8 && \
make frobenius-n-16-g-4_9 && \
make frobenius-n-16-g-4_10 && \
make frobenius-n-16-g-4_11 && \
make frobenius-n-16-g-4_12 && \
make frobenius-n-16-g-4_13 && \
make frobenius-n-16-g-4_14 && \
make frobenius-n-16-g-4_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
