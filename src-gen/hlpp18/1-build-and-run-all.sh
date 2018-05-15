#!/bin/bash

for benchmark in frobenius fss matmult_float nbody_float; do
  for node in 1 2 4 8 16; do
    for core in 1 6 12 18 24; do
    	chmod +x ${benchmark}-n-${node}-c-${core}/CPU/build-and-submit.sh && \
      ${benchmark}-n-${node}-c-${core}/CPU/build-and-submit.sh
    done
  done
done 
