#!/bin/bash

for benchmark in frobenius fss; do
  for node in 1 4 16; do
    for core in 1 6 12 18 24; do
    	cd /home/fwrede/musket/src-gen/hlpp18/high/${benchmark}-n-${node}-c-${core}/CPU/ && \
    	chmod +x build-and-submit.sh && \
      ./build-and-submit.sh
    done
  done
done 
