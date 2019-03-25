#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make matmult-n-16-g-2_0 && \
make matmult-n-16-g-2_1 && \
make matmult-n-16-g-2_2 && \
make matmult-n-16-g-2_3 && \
make matmult-n-16-g-2_4 && \
make matmult-n-16-g-2_5 && \
make matmult-n-16-g-2_6 && \
make matmult-n-16-g-2_7 && \
make matmult-n-16-g-2_8 && \
make matmult-n-16-g-2_9 && \
make matmult-n-16-g-2_10 && \
make matmult-n-16-g-2_11 && \
make matmult-n-16-g-2_12 && \
make matmult-n-16-g-2_13 && \
make matmult-n-16-g-2_14 && \
make matmult-n-16-g-2_15 && \

mpirun -np 1 bin/matmult-n-16-g-2_0 : -np 1 bin/matmult-n-16-g-2_1 : -np 1 bin/matmult-n-16-g-2_2 : -np 1 bin/matmult-n-16-g-2_3 : -np 1 bin/matmult-n-16-g-2_4 : -np 1 bin/matmult-n-16-g-2_5 : -np 1 bin/matmult-n-16-g-2_6 : -np 1 bin/matmult-n-16-g-2_7 : -np 1 bin/matmult-n-16-g-2_8 : -np 1 bin/matmult-n-16-g-2_9 : -np 1 bin/matmult-n-16-g-2_10 : -np 1 bin/matmult-n-16-g-2_11 : -np 1 bin/matmult-n-16-g-2_12 : -np 1 bin/matmult-n-16-g-2_13 : -np 1 bin/matmult-n-16-g-2_14 : -np 1 bin/matmult-n-16-g-2_15
