#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/de/wwu/musket/models/test/scatter/CPU-MPMD/build/ && \
mkdir ~/musket-build/de/wwu/musket/models/test/scatter/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/de/wwu/musket/models/test/scatter/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make scatter && \
cd .. && \
mkdir -p ~/musket-build/de/wwu/musket/models/test/scatter/CPU-MPMD/out/ && \
sbatch job.sh
