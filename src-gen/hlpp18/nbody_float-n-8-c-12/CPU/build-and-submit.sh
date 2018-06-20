#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/nbody_float-n-8-c-12/CPU/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/nbody_float-n-8-c-12/CPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/nbody_float-n-8-c-12/CPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/nbody_float-n-8-c-12/CPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make nbody_float-n-8-c-12 && \
cd ${source_folder} && \

sbatch job.sh