#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/fss-n-1-c-24/CPU/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/fss-n-1-c-24/CPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/fss-n-1-c-24/CPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/fss-n-1-c-24/CPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make fss-n-1-c-24 && \
cd ${source_folder} && \

sbatch job.sh
