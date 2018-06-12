#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/valgrind/nbody_float-valgrind-n-4-c-24/CPU-MPMD/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/valgrind/nbody_float-valgrind-n-4-c-24/CPU-MPMD/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/valgrind/nbody_float-valgrind-n-4-c-24/CPU-MPMD/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/valgrind/nbody_float-valgrind-n-4-c-24/CPU-MPMD/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make nbody_float-valgrind-n-4-c-24_0 && \
make nbody_float-valgrind-n-4-c-24_1 && \
make nbody_float-valgrind-n-4-c-24_2 && \
make nbody_float-valgrind-n-4-c-24_3 && \
cd ${source_folder} && \

sbatch job.sh
