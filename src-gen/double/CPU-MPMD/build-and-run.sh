#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev ../ && \

make double_0 && \
make double_1 && \
make double_2 && \
make double_3 && \

mpirun -np 1 bin/double_0 : -np 1 bin/double_1 : -np 1 bin/double_2 : -np 1 bin/double_3
