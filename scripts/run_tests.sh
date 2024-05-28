#!/bin/bash

log_file=$(basename $1 .py)
python $1 &> ${log_file}_${SLURM_PROCID}.log