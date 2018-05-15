#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make frobenius_0 && \
make frobenius_1 && \
make frobenius_2 && \
make frobenius_3 && \

mpirun -np 1 bin/frobenius_0 : -np 1 bin/frobenius_1 : -np 1 bin/frobenius_2 : -np 1 bin/frobenius_3
