#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make nbody_float-valgrind-n-4-c-24 && \
mpirun -np 4 bin/nbody_float-valgrind-n-4-c-24
