#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 40
#SBATCH -p gpu

mpirun --bind-to none ../bin/cannonMM -n 8192 -gx 2 -gy 2 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0
mpirun --bind-to none ../bin/cannonMM -n 8192 -gx 2 -gy 2 -ll:ocpu 2 -ll:gpu 0 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0 -dslmapper -mapping mappings_cpu
