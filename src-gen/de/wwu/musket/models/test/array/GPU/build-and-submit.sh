#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/de/wwu/musket/models/test/array/GPU/build/ && \
mkdir ~/musket-build/de/wwu/musket/models/test/array/GPU/build/ && \

# run cmake
cd ~/musket-build/de/wwu/musket/models/test/array/GPU/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark -D CMAKE_CXX_COMPILER=pgc++ ../ && \

make array && \
cd .. && \
mkdir -p ~/musket-build/de/wwu/musket/models/test/array/GPU/out/ && \
sbatch job.sh
