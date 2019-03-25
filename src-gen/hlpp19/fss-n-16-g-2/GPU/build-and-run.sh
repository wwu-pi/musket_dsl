#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=pgc++ -D MPI_CXX_COMPILER=/opt/pgi/linux86-64/18.10/mpi/openmpi/bin/mpic++ ../ && \

make fss-n-16-g-2_0 && \
make fss-n-16-g-2_1 && \
make fss-n-16-g-2_2 && \
make fss-n-16-g-2_3 && \
make fss-n-16-g-2_4 && \
make fss-n-16-g-2_5 && \
make fss-n-16-g-2_6 && \
make fss-n-16-g-2_7 && \
make fss-n-16-g-2_8 && \
make fss-n-16-g-2_9 && \
make fss-n-16-g-2_10 && \
make fss-n-16-g-2_11 && \
make fss-n-16-g-2_12 && \
make fss-n-16-g-2_13 && \
make fss-n-16-g-2_14 && \
make fss-n-16-g-2_15 && \

mpirun -np 1 bin/fss-n-16-g-2_0 : -np 1 bin/fss-n-16-g-2_1 : -np 1 bin/fss-n-16-g-2_2 : -np 1 bin/fss-n-16-g-2_3 : -np 1 bin/fss-n-16-g-2_4 : -np 1 bin/fss-n-16-g-2_5 : -np 1 bin/fss-n-16-g-2_6 : -np 1 bin/fss-n-16-g-2_7 : -np 1 bin/fss-n-16-g-2_8 : -np 1 bin/fss-n-16-g-2_9 : -np 1 bin/fss-n-16-g-2_10 : -np 1 bin/fss-n-16-g-2_11 : -np 1 bin/fss-n-16-g-2_12 : -np 1 bin/fss-n-16-g-2_13 : -np 1 bin/fss-n-16-g-2_14 : -np 1 bin/fss-n-16-g-2_15
