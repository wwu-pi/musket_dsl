#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/de/wwu/musket/models/test/zip/CPU-MPMD/build/ && \
mkdir ~/musket-build/de/wwu/musket/models/test/zip/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/de/wwu/musket/models/test/zip/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make zip && \
cd .. && \
mkdir -p ~/musket-build/de/wwu/musket/models/test/zip/CPU-MPMD/out/ && \
sbatch job.sh
