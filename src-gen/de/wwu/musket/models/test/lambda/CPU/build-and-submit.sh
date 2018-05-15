#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/de/wwu/musket/models/test/lambda/CPU/out/ && \
rm -rf -- /home/fwrede/musket-build/de/wwu/musket/models/test/lambda/CPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/de/wwu/musket/models/test/lambda/CPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/de/wwu/musket/models/test/lambda/CPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus ${source_folder} && \

make lambda && \
cd ${source_folder} && \

sbatch job.sh
