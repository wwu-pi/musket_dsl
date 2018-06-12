#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/mat/CPU-MPMD/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/mat/CPU-MPMD/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/mat/CPU-MPMD/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/mat/CPU-MPMD/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make mat_0 && \
make mat_1 && \
make mat_2 && \
make mat_3 && \
cd ${source_folder} && \

sbatch job.sh
