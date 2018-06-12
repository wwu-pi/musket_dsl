#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/nbody_float-n-4-c-24/CPU-MPMD/out/callgrind && \
rm -rf -- /home/fwrede/musket-build/hlpp18/nbody_float-n-4-c-24/CPU-MPMD/build/callgrind && \
mkdir -p /home/fwrede/musket-build/hlpp18/nbody_float-n-4-c-24/CPU-MPMD/build/callgrind && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/nbody_float-n-4-c-24/CPU-MPMD/build/callgrind && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make nbody_float-n-4-c-24_0 && \
make nbody_float-n-4-c-24_1 && \
make nbody_float-n-4-c-24_2 && \
make nbody_float-n-4-c-24_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
