#!/bin/bash

# remove files and create folder
rm -rf -- ~/musket-build/de/wwu/musket/models/test/map/CPU/build/ && \
mkdir ~/musket-build/de/wwu/musket/models/test/map/CPU/build/ && \

# run cmake
cd ~/musket-build/de/wwu/musket/models/test/map/CPU/build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmark ../ && \

make map && \
cd .. && \
mkdir -p ~/musket-build/de/wwu/musket/models/test/map/CPU/out/ && \
sbatch job.sh
