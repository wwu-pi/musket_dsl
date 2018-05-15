#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev ../ && \

make map_0 && \
make map_1 && \
make map_2 && \
make map_3 && \

mpirun -np 1 bin/map_0 : -np 1 bin/map_1 : -np 1 bin/map_2 : -np 1 bin/map_3
