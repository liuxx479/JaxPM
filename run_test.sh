#!/bin/bash

if [ "$2" == "-log" ]; then
    log_file=$(basename $1 .py)
    python $1 -p 2x2 -g 256 -o results &> ${log_file}_${SLURM_PROCID}.log
else
    python $1 -p 2x2 -g 256 -o results
fi
