#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make mat_0 && \
make mat_1 && \
make mat_2 && \
make mat_3 && \

mpirun -np 1 bin/mat_0 : -np 1 bin/mat_1 : -np 1 bin/mat_2 : -np 1 bin/mat_3
