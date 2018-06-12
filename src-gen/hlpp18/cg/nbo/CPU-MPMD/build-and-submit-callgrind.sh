#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/out/callgrind && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/callgrind && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/callgrind && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/callgrind && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make nbo_0 && \
make nbo_1 && \
make nbo_2 && \
make nbo_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
