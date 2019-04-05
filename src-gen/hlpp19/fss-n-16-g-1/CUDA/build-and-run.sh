#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=g++ ../ && \

make fss-n-16-g-1_0 && \
make fss-n-16-g-1_1 && \
make fss-n-16-g-1_2 && \
make fss-n-16-g-1_3 && \
make fss-n-16-g-1_4 && \
make fss-n-16-g-1_5 && \
make fss-n-16-g-1_6 && \
make fss-n-16-g-1_7 && \
make fss-n-16-g-1_8 && \
make fss-n-16-g-1_9 && \
make fss-n-16-g-1_10 && \
make fss-n-16-g-1_11 && \
make fss-n-16-g-1_12 && \
make fss-n-16-g-1_13 && \
make fss-n-16-g-1_14 && \
make fss-n-16-g-1_15 && \

mpirun -np 1 bin/fss-n-16-g-1_0 : -np 1 bin/fss-n-16-g-1_1 : -np 1 bin/fss-n-16-g-1_2 : -np 1 bin/fss-n-16-g-1_3 : -np 1 bin/fss-n-16-g-1_4 : -np 1 bin/fss-n-16-g-1_5 : -np 1 bin/fss-n-16-g-1_6 : -np 1 bin/fss-n-16-g-1_7 : -np 1 bin/fss-n-16-g-1_8 : -np 1 bin/fss-n-16-g-1_9 : -np 1 bin/fss-n-16-g-1_10 : -np 1 bin/fss-n-16-g-1_11 : -np 1 bin/fss-n-16-g-1_12 : -np 1 bin/fss-n-16-g-1_13 : -np 1 bin/fss-n-16-g-1_14 : -np 1 bin/fss-n-16-g-1_15
