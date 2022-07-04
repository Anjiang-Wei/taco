#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 40
#SBATCH -p gpu

# mpirun ./circuit -mapping mappings_circuit -level nsmapper=debug -ll:gpu 4 -ll:cpu 2

# mpirun ./bin/cannonMM -n 8192 -gx 2 -gy 1 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0
mpirun ./bin/cannonMM -n 1024 -gx 2 -gy 1 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0
