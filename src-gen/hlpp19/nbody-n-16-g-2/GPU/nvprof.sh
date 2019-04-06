#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-16-g-2/GPU/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/nbody-n-16-g-2/GPU/build/nvprof && \
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-16-g-2/GPU/build/nvprof && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/nbody-n-16-g-2/GPU/build/nvprof && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make nbody-n-16-g-2_0 && \
make nbody-n-16-g-2_1 && \
make nbody-n-16-g-2_2 && \
make nbody-n-16-g-2_3 && \
make nbody-n-16-g-2_4 && \
make nbody-n-16-g-2_5 && \
make nbody-n-16-g-2_6 && \
make nbody-n-16-g-2_7 && \
make nbody-n-16-g-2_8 && \
make nbody-n-16-g-2_9 && \
make nbody-n-16-g-2_10 && \
make nbody-n-16-g-2_11 && \
make nbody-n-16-g-2_12 && \
make nbody-n-16-g-2_13 && \
make nbody-n-16-g-2_14 && \
make nbody-n-16-g-2_15 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
