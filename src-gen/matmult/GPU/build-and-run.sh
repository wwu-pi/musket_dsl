#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev -D CMAKE_CXX_COMPILER=pgc++ ../ && \

make matmult_0 && \

bin/matmult_0
