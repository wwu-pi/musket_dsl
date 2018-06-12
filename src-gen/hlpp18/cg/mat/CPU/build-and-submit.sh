#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/cg/mat/CPU/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/cg/mat/CPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/cg/mat/CPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/cg/mat/CPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make mat && \
cd ${source_folder} && \

sbatch job.sh
