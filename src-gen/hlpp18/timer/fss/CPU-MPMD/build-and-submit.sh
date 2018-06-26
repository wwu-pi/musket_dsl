#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/timer/fss/CPU-MPMD/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp18/timer/fss/CPU-MPMD/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp18/timer/fss/CPU-MPMD/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/timer/fss/CPU-MPMD/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make fss_0 && \
make fss_1 && \
make fss_2 && \
make fss_3 && \
cd ${source_folder} && \

sbatch job.sh
