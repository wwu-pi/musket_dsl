#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make frobenius-fused-n-4-c-24 && \
mpirun -np 4 bin/frobenius-fused-n-4-c-24
