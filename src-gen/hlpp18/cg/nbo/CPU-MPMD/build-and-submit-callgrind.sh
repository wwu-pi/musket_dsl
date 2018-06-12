#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/out/cg && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/cg && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/cg && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/cg && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make nbo_0 && \
make nbo_1 && \
make nbo_2 && \
make nbo_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
