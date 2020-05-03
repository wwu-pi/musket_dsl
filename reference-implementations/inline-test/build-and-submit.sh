#!/bin/bash

# remove files and create folder
mkdir -p /home/fwrede/out/hlpp18 && \
mkdir -p /home/fwrede/musket-build/inline-test/build && \

rm -rf -- /home/fwrede/musket-build/inline-test/build && \
mkdir /home/fwrede/musket-build/inline-test/build && \

# run cmake
cd /home/fwrede/musket-build/inline-test/build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus /home/fwrede/musket/reference-implementations/inline-test/ && \

make all && \

cd /home/fwrede/musket/reference-implementations/inline-test && \
sbatch job.sh
