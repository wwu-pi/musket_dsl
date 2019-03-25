#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make frobenius-n-16-g-4_0 && \
make frobenius-n-16-g-4_1 && \
make frobenius-n-16-g-4_2 && \
make frobenius-n-16-g-4_3 && \
make frobenius-n-16-g-4_4 && \
make frobenius-n-16-g-4_5 && \
make frobenius-n-16-g-4_6 && \
make frobenius-n-16-g-4_7 && \
make frobenius-n-16-g-4_8 && \
make frobenius-n-16-g-4_9 && \
make frobenius-n-16-g-4_10 && \
make frobenius-n-16-g-4_11 && \
make frobenius-n-16-g-4_12 && \
make frobenius-n-16-g-4_13 && \
make frobenius-n-16-g-4_14 && \
make frobenius-n-16-g-4_15 && \

mpirun -np 1 bin/frobenius-n-16-g-4_0 : -np 1 bin/frobenius-n-16-g-4_1 : -np 1 bin/frobenius-n-16-g-4_2 : -np 1 bin/frobenius-n-16-g-4_3 : -np 1 bin/frobenius-n-16-g-4_4 : -np 1 bin/frobenius-n-16-g-4_5 : -np 1 bin/frobenius-n-16-g-4_6 : -np 1 bin/frobenius-n-16-g-4_7 : -np 1 bin/frobenius-n-16-g-4_8 : -np 1 bin/frobenius-n-16-g-4_9 : -np 1 bin/frobenius-n-16-g-4_10 : -np 1 bin/frobenius-n-16-g-4_11 : -np 1 bin/frobenius-n-16-g-4_12 : -np 1 bin/frobenius-n-16-g-4_13 : -np 1 bin/frobenius-n-16-g-4_14 : -np 1 bin/frobenius-n-16-g-4_15
