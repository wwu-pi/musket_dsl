#!/bin/bash

for benchmark in frobenius fss matmult nbody; do
  for node in 1 4 16; do
    for gpu in 1 2 4; do
      DIR="${benchmark}-n-${node}-g-${gpu}/GPU"
      if [ -d "${DIR}" ]; then
    	  chmod +x ${DIR}/build-and-submit.sh && \
        ${DIR}/build-and-submit.sh
      fi
    done
  done
done 
