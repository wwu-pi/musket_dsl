#!/bin/bash
rm -rf build && \
mkdir build && \
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test .. && \
make
