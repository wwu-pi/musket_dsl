#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev ../ && \

make matrix_0 && \
make matrix_1 && \
make matrix_2 && \
make matrix_3 && \

mpirun -np 1 bin/matrix_0 : -np 1 bin/matrix_1 : -np 1 bin/matrix_2 : -np 1 bin/matrix_3
