#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/fro-fused/CPU-MPMD/out/cg && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/fro-fused/CPU-MPMD/build/cg && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/fro-fused/CPU-MPMD/build/cg && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/fro-fused/CPU-MPMD/build/cg && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make fro-fused_0 && \
make fro-fused_1 && \
make fro-fused_2 && \
make fro-fused_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
