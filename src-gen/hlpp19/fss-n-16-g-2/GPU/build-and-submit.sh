#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/fss-n-16-g-2/GPU/out/ && \
rm -rf -- /home/fwrede/musket-build/hlpp19/fss-n-16-g-2/GPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/hlpp19/fss-n-16-g-2/GPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/fss-n-16-g-2/GPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make fss-n-16-g-2_0 && \
make fss-n-16-g-2_1 && \
make fss-n-16-g-2_2 && \
make fss-n-16-g-2_3 && \
make fss-n-16-g-2_4 && \
make fss-n-16-g-2_5 && \
make fss-n-16-g-2_6 && \
make fss-n-16-g-2_7 && \
make fss-n-16-g-2_8 && \
make fss-n-16-g-2_9 && \
make fss-n-16-g-2_10 && \
make fss-n-16-g-2_11 && \
make fss-n-16-g-2_12 && \
make fss-n-16-g-2_13 && \
make fss-n-16-g-2_14 && \
make fss-n-16-g-2_15 && \
cd ${source_folder} && \

sbatch job.sh
