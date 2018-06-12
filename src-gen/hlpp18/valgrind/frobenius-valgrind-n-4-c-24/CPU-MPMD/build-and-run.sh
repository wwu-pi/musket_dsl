#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make frobenius-valgrind-n-4-c-24_0 && \
make frobenius-valgrind-n-4-c-24_1 && \
make frobenius-valgrind-n-4-c-24_2 && \
make frobenius-valgrind-n-4-c-24_3 && \

mpirun -np 1 bin/frobenius-valgrind-n-4-c-24_0 : -np 1 bin/frobenius-valgrind-n-4-c-24_1 : -np 1 bin/frobenius-valgrind-n-4-c-24_2 : -np 1 bin/frobenius-valgrind-n-4-c-24_3
