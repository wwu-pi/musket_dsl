#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make fro_0 && \
make fro_1 && \
make fro_2 && \
make fro_3 && \

mpirun -np 1 bin/fro_0 : -np 1 bin/fro_1 : -np 1 bin/fro_2 : -np 1 bin/fro_3
