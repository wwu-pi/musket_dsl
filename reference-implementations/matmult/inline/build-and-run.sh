#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make matmult_inline && \
make matmult_skeleton_functions && \
make matmult_user_functions && \
make matmult_skeleton_user_functions && \

mpirun -np 4 bin/matmult_inline && \
mpirun -np 4 bin/matmult_skeleton_functions && \
mpirun -np 4 bin/matmult_user_functions && \
mpirun -np 4 bin/matmult_skeleton_user_functions
