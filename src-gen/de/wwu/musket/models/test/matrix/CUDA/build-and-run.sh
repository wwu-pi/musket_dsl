#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Dev -D CMAKE_CXX_COMPILER=g++ ../ && \

make matrix_0 && \

bin/matrix_0
