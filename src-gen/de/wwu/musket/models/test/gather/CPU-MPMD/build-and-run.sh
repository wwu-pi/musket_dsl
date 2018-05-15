#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev ../ && \

make gather_0 && \
make gather_1 && \
make gather_2 && \
make gather_3 && \

mpirun -np 1 bin/gather_0 : -np 1 bin/gather_1 : -np 1 bin/gather_2 : -np 1 bin/gather_3
