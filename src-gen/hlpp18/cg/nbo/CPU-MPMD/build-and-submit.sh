#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/nbo/CPU-MPMD/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make nbo_0 && \
make nbo_1 && \
make nbo_2 && \
make nbo_3 && \
cd ${source_folder} && \

sbatch job.sh
