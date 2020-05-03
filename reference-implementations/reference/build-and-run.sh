#!/bin/bash

# remove files and create folder
rm -rf -- build && \
mkdir build && \

# run cmake
cd build/ && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Test ../ && \

make frobenius_inline && \
make frobenius_inline_w_gather && \
make frobenius_skeleton_functions && \
make frobenius_user_functions && \
make frobenius_skeleton_user_functions && \
make frobenius_mapfold_inline && \
make frobenius_mapfold_skeleton_functions && \
make frobenius_mapfold_user_functions && \
make frobenius_mapfold_skeleton_user_functions && \

mpirun -np 4 bin/frobenius_inline && \
mpirun -np 4 bin/frobenius_inline_w_gather && \
mpirun -np 4 bin/frobenius_skeleton_functions && \
mpirun -np 4 bin/frobenius_user_functions && \
mpirun -np 4 bin/frobenius_skeleton_user_functions && \
mpirun -np 4 bin/frobenius_mapfold_inline && \
mpirun -np 4 bin/frobenius_mapfold_skeleton_functions && \
mpirun -np 4 bin/frobenius_mapfold_user_functions && \
mpirun -np 4 bin/frobenius_mapfold_skeleton_user_functions
