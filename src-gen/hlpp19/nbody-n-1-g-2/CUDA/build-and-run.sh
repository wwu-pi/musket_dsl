#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test -D CMAKE_CXX_COMPILER=g++ ../ && \

make nbody-n-1-g-2_0 && \

bin/nbody-n-1-g-2_0
