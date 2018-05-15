#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev ../ && \

make scatter_0 && \
make scatter_1 && \
make scatter_2 && \
make scatter_3 && \

mpirun -np 1 bin/scatter_0 : -np 1 bin/scatter_1 : -np 1 bin/scatter_2 : -np 1 bin/scatter_3
