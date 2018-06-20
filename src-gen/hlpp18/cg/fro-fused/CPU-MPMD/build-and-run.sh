#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make fro-fused_0 && \
make fro-fused_1 && \
make fro-fused_2 && \
make fro-fused_3 && \

mpirun -np 1 bin/fro-fused_0 : -np 1 bin/fro-fused_1 : -np 1 bin/fro-fused_2 : -np 1 bin/fro-fused_3
