#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/frobenius/CPU-MPMD/out/cg && \
rm -rf -- /home/fwrede/musket-build/frobenius/CPU-MPMD/build/cg && \
mkdir -p /home/fwrede/musket-build/frobenius/CPU-MPMD/build/cg && \

# run cmake
cd /home/fwrede/musket-build/frobenius/CPU-MPMD/build/cg && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make frobenius_0 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
