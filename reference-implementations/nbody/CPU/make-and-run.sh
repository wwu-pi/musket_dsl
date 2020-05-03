#!/bin/bash
cd build && \
make nbody && \
mpirun -np 4 bin/nbody 
