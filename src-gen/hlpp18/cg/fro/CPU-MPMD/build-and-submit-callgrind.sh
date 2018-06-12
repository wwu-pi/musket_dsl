#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/fro/CPU-MPMD/out/callgrind && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/fro/CPU-MPMD/build/callgrind && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/fro/CPU-MPMD/build/callgrind && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/fro/CPU-MPMD/build/callgrind && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make fro_0 && \
make fro_1 && \
make fro_2 && \
make fro_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
