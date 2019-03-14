#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=pgc++ -D MPI_CXX_COMPILER=/opt/pgi/linux86-64/18.10/mpi/openmpi/bin/mpic++ ../ && \

make frobenius-n-4-g-4_0 && \
make frobenius-n-4-g-4_1 && \
make frobenius-n-4-g-4_2 && \
make frobenius-n-4-g-4_3 && \

mpirun -np 1 bin/frobenius-n-4-g-4_0 : -np 1 bin/frobenius-n-4-g-4_1 : -np 1 bin/frobenius-n-4-g-4_2 : -np 1 bin/frobenius-n-4-g-4_3
