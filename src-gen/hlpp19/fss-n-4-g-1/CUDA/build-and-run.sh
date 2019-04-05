#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=g++ ../ && \

make fss-n-4-g-1_0 && \
make fss-n-4-g-1_1 && \
make fss-n-4-g-1_2 && \
make fss-n-4-g-1_3 && \

mpirun -np 1 bin/fss-n-4-g-1_0 : -np 1 bin/fss-n-4-g-1_1 : -np 1 bin/fss-n-4-g-1_2 : -np 1 bin/fss-n-4-g-1_3
