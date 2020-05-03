#!/bin/bash
cd build && \
make double && \
mpirun -np 4 bin/double 
