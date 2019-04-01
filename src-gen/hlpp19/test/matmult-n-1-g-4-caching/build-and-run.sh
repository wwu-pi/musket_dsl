#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=pgc++ ../ && \

make matmult-n-1-g-4_0 && \

bin/matmult-n-1-g-4_0
