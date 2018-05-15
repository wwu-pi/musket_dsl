#!/bin/bash
cd build && \

make nbody_inline && \
make nbody_skeleton_functions && \
make nbody_user_functions && \
make nbody_skeleton_user_functions && \

mpirun -np 4 bin/nbody_inline && \
mpirun -np 4 bin/nbody_skeleton_functions && \
mpirun -np 4 bin/nbody_user_functions && \
mpirun -np 4 bin/nbody_skeleton_user_functions
