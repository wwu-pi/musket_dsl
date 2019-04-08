#!/bin/bash

current_folder=${PWD} && \

#build
for benchmark in frobenius matmult nbody; do
  for node in 1; do
    for gpu in 4; do
      DIR="nvprof/${benchmark}-n-${node}-g-${gpu}/CUDA"
      if [ -d "${DIR}" ]; then
        cd ${DIR} && \
    	  chmod +x nvprof.sh && \
        ./nvprof.sh && \
        cd ${current_folder}
      fi
    done
  done
done 
