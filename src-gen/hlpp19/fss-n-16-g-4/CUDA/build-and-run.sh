#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=g++ ../ && \

make fss-n-16-g-4_0 && \
make fss-n-16-g-4_1 && \
make fss-n-16-g-4_2 && \
make fss-n-16-g-4_3 && \
make fss-n-16-g-4_4 && \
make fss-n-16-g-4_5 && \
make fss-n-16-g-4_6 && \
make fss-n-16-g-4_7 && \
make fss-n-16-g-4_8 && \
make fss-n-16-g-4_9 && \
make fss-n-16-g-4_10 && \
make fss-n-16-g-4_11 && \
make fss-n-16-g-4_12 && \
make fss-n-16-g-4_13 && \
make fss-n-16-g-4_14 && \
make fss-n-16-g-4_15 && \

mpirun -np 1 bin/fss-n-16-g-4_0 : -np 1 bin/fss-n-16-g-4_1 : -np 1 bin/fss-n-16-g-4_2 : -np 1 bin/fss-n-16-g-4_3 : -np 1 bin/fss-n-16-g-4_4 : -np 1 bin/fss-n-16-g-4_5 : -np 1 bin/fss-n-16-g-4_6 : -np 1 bin/fss-n-16-g-4_7 : -np 1 bin/fss-n-16-g-4_8 : -np 1 bin/fss-n-16-g-4_9 : -np 1 bin/fss-n-16-g-4_10 : -np 1 bin/fss-n-16-g-4_11 : -np 1 bin/fss-n-16-g-4_12 : -np 1 bin/fss-n-16-g-4_13 : -np 1 bin/fss-n-16-g-4_14 : -np 1 bin/fss-n-16-g-4_15
