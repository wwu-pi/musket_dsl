#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/mat/CPU-MPMD/out/callgrind && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/mat/CPU-MPMD/build/callgrind && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/mat/CPU-MPMD/build/callgrind && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/mat/CPU-MPMD/build/callgrind && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make mat_0 && \
make mat_1 && \
make mat_2 && \
make mat_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
