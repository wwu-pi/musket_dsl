#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make nbody_float_0 && \
make nbody_float_1 && \
make nbody_float_2 && \
make nbody_float_3 && \

mpirun -np 1 bin/nbody_float_0 : -np 1 bin/nbody_float_1 : -np 1 bin/nbody_float_2 : -np 1 bin/nbody_float_3
