#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=pgc++ -D MPI_CXX_COMPILER=/opt/pgi/linux86-64/18.10/mpi/openmpi/bin/mpic++ ../ && \

make matmult_float_0 && \
make matmult_float_1 && \
make matmult_float_2 && \
make matmult_float_3 && \

mpirun -np 1 bin/matmult_float_0 : -np 1 bin/matmult_float_1 : -np 1 bin/matmult_float_2 : -np 1 bin/matmult_float_3
