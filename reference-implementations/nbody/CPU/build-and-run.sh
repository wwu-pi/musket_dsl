#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev ../ && \

make nbody && \
mpirun -np 4 bin/nbody 
