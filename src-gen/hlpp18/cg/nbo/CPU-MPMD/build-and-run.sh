#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make nbo_0 && \
make nbo_1 && \
make nbo_2 && \
make nbo_3 && \

mpirun -np 1 bin/nbo_0 : -np 1 bin/nbo_1 : -np 1 bin/nbo_2 : -np 1 bin/nbo_3
