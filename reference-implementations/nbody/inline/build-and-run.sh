#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make nbody_inline && \
make nbody_skeleton_functions && \
make nbody_user_functions && \
make nbody_skeleton_user_functions && \

mpirun -np 4 bin/nbody_inline && \
mpirun -np 4 bin/nbody_skeleton_functions && \
mpirun -np 4 bin/nbody_user_functions && \
mpirun -np 4 bin/nbody_skeleton_user_functions
