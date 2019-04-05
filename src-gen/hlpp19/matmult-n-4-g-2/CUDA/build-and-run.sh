#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=g++ ../ && \

make matmult-n-4-g-2_0 && \
make matmult-n-4-g-2_1 && \
make matmult-n-4-g-2_2 && \
make matmult-n-4-g-2_3 && \

mpirun -np 1 bin/matmult-n-4-g-2_0 : -np 1 bin/matmult-n-4-g-2_1 : -np 1 bin/matmult-n-4-g-2_2 : -np 1 bin/matmult-n-4-g-2_3
