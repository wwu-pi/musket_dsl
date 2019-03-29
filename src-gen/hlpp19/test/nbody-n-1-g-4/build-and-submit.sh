#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/out/hlpp19/musket-test && \
rm -rf -- /home/fwrede/build/hlpp19/musket/nbody-n-1-g-4 && \
mkdir -p /home/fwrede/build/hlpp19/musket/nbody-n-1-g-4 && \

# run cmake
cd /home/fwrede/build/hlpp19/musket/nbody-n-1-g-4 && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make nbody-n-1-g-4_0 && \
cd ${source_folder} && \

sbatch job.sh
