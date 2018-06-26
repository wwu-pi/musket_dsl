#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp18/timer/fss/CPU-MPMD/out/cg && \
rm -rf -- /home/fwrede/musket-build/hlpp18/timer/fss/CPU-MPMD/build/cg && \
mkdir -p /home/fwrede/musket-build/hlpp18/timer/fss/CPU-MPMD/build/cg && \

# run cmake
cd /home/fwrede/musket-build/hlpp18/timer/fss/CPU-MPMD/build/cg && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make fss_0 && \
make fss_1 && \
make fss_2 && \
make fss_3 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
