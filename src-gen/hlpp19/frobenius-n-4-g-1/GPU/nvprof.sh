#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-1/GPU/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-1/GPU/build/nvprof && \
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-1/GPU/build/nvprof && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/frobenius-n-4-g-1/GPU/build/nvprof && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make frobenius-n-4-g-1_0 && \
make frobenius-n-4-g-1_1 && \
make frobenius-n-4-g-1_2 && \
make frobenius-n-4-g-1_3 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
