#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make fss_0 && \
make fss_1 && \
make fss_2 && \
make fss_3 && \

mpirun -np 1 bin/fss_0 : -np 1 bin/fss_1 : -np 1 bin/fss_2 : -np 1 bin/fss_3
