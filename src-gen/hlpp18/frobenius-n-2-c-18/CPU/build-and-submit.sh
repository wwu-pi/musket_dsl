#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/frobenius-n-2-c-18/CPU/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/frobenius-n-2-c-18/CPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/frobenius-n-2-c-18/CPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/frobenius-n-2-c-18/CPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make frobenius-n-2-c-18 && \
cd ${source_folder} && \

sbatch job.sh