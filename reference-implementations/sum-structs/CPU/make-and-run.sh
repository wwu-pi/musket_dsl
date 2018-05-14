#!/bin/bash
cd build && \
make mpi && \
mpirun -np 4 bin/mpi 
