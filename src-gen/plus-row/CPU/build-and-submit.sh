#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/plus-row/CPU/out/ && \
rm -rf -- /home/fwrede/musket-build/plus-row/CPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/plus-row/CPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/plus-row/CPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make plus-row && \
cd ${source_folder} && \

sbatch job.sh
