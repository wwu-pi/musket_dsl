#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/de/wwu/musket/models/test/gather/CPU-MPMD/build/ && \
mkdir ~/musket-build/de/wwu/musket/models/test/gather/CPU-MPMD/build/ && \

# run cmake
cd ~/musket-build/de/wwu/musket/models/test/gather/CPU-MPMD/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make gather && \
cd .. && \
mkdir -p ~/musket-build/de/wwu/musket/models/test/gather/CPU-MPMD/out/ && \
sbatch job.sh
