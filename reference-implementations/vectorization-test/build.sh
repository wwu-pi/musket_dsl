#!/bin/bash
rm -rf -- build && \
mkdir -p build && \
# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test .. && \
make
