#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make matmult_float-n-4-c-1 && \
mpirun -np 4 bin/matmult_float-n-4-c-1
