#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-1/CPU-MPMD/out/cg && \
rm -rf -- /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-1/CPU-MPMD/build/cg && \
mkdir -p /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-1/CPU-MPMD/build/cg && \

# run cmake
cd /home/fwrede/musket-build/hlpp19/frobenius-n-16-g-1/CPU-MPMD/build/cg && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Callgrind ${source_folder} && \

make frobenius-n-16-g-1_0 && \
make frobenius-n-16-g-1_1 && \
make frobenius-n-16-g-1_2 && \
make frobenius-n-16-g-1_3 && \
make frobenius-n-16-g-1_4 && \
make frobenius-n-16-g-1_5 && \
make frobenius-n-16-g-1_6 && \
make frobenius-n-16-g-1_7 && \
make frobenius-n-16-g-1_8 && \
make frobenius-n-16-g-1_9 && \
make frobenius-n-16-g-1_10 && \
make frobenius-n-16-g-1_11 && \
make frobenius-n-16-g-1_12 && \
make frobenius-n-16-g-1_13 && \
make frobenius-n-16-g-1_14 && \
make frobenius-n-16-g-1_15 && \
cd ${source_folder} && \

sbatch job-callgrind.sh
