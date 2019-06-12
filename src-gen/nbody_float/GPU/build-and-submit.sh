#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/nbody_float/GPU/out/ && \
rm -rf -- /home/fwrede/musket-build/nbody_float/GPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/nbody_float/GPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/nbody_float/GPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make nbody_float_0 && \
cd ${source_folder} && \

sbatch job.sh
