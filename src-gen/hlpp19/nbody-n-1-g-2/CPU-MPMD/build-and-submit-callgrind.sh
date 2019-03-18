#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-1-g-2/CPU-MPMD/out/cg && \
rm -rf -- /home/fwrede/musket-build/hlpp19/nbody-n-1-g-2/CPU-MPMD/build/cg && \
mkdir -p /home/fwrede/musket-build/hlpp19/nbody-n-1-g-2/CPU-MPMD/build/cg && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/nbody-n-1-g-2/CPU-MPMD/build/cg && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make nbody-n-1-g-2_0 && \
cd ${source_folder} && \

sbatch job-callgrind.sh