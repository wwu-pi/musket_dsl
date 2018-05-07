#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev ../ && \

make mpi && \
mpirun -np 4 bin/mpi 
