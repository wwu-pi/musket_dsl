#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/fro-fused/CPU-MPMD/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/fro-fused/CPU-MPMD/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/fro-fused/CPU-MPMD/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/fro-fused/CPU-MPMD/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make fro-fused_0 && \
make fro-fused_1 && \
make fro-fused_2 && \
make fro-fused_3 && \
cd ${source_folder} && \

sbatch job.sh
