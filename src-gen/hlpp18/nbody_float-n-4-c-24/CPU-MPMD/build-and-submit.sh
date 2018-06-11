#!/bin/bash
		
source_folder=${PWD} && \

# remove files and create folder
mkdir -p ~/musket-build/hlpp18/nbody_float-n-4-c-24/CPU-MPMD/out/ && \
rm -rf -- ~/musket-build/hlpp18/nbody_float-n-4-c-24/CPU-MPMD/build/benchmark && \
mkdir -p ~/musket-build/hlpp18/nbody_float-n-4-c-24/CPU-MPMD/build/benchmark && \

# run cmake
cd ~/musket-build/hlpp18/nbody_float-n-4-c-24/CPU-MPMD/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make nbody_float-n-4-c-24 && \
cd ${source_folder} && \

sbatch job.sh
