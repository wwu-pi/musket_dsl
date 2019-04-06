#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-2/GPU/out/ && \
rm -rf -- ~/build/mnp/frobenius-n-16-g-2/gpu && \
mkdir -p ~/build/mnp/frobenius-n-16-g-2/gpu && \

# run cmake
cd ~/build/mnp/frobenius-n-16-g-2/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make frobenius-n-16-g-2_0 && \
make frobenius-n-16-g-2_1 && \
make frobenius-n-16-g-2_2 && \
make frobenius-n-16-g-2_3 && \
make frobenius-n-16-g-2_4 && \
make frobenius-n-16-g-2_5 && \
make frobenius-n-16-g-2_6 && \
make frobenius-n-16-g-2_7 && \
make frobenius-n-16-g-2_8 && \
make frobenius-n-16-g-2_9 && \
make frobenius-n-16-g-2_10 && \
make frobenius-n-16-g-2_11 && \
make frobenius-n-16-g-2_12 && \
make frobenius-n-16-g-2_13 && \
make frobenius-n-16-g-2_14 && \
make frobenius-n-16-g-2_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
