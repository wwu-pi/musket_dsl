#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev ../ && \

make zip_0 && \
make zip_1 && \
make zip_2 && \
make zip_3 && \

mpirun -np 1 bin/zip_0 : -np 1 bin/zip_1 : -np 1 bin/zip_2 : -np 1 bin/zip_3
