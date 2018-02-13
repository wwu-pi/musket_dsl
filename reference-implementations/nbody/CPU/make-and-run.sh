#!/bin/bash
cd build && \
make nbody && \
mpirun -np 2 bin/nbody 
