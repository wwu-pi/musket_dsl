#!/bin/bash

current_folder=${PWD} && \

#build
for benchmark in frobenius; do
  for node in 1 4 16; do
    for gpu in 1 2 4; do
      DIR="${benchmark}-n-${node}-g-${gpu}/CUDA"
      if [ -d "${DIR}" ]; then
        cd ${DIR} && \
    	  chmod +x build-and-submit.sh && \
        ./build-and-submit.sh && \
        cd ${current_folder}
      fi
    done
  done
done 
