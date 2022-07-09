#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 40
#SBATCH -p gpu

# mpirun ./circuit -mapping mappings_circuit -level nsmapper=debug -ll:gpu 4 -ll:cpu 2

# ./bin/cannonMM -n 8192 -gx 2 -gy 1 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0
# mpirun -H g0002.stanford.edu --bind-to none ./bin/cannonMM -n 8192 -gx 2 -gy 1 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 
# GASNET_BACKTRACE=1 mpirun -H g0002,g0003 --bind-to none ./bin/cannonMM -n 8192 -gx 2 -gy 1 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0 -mapping mappings -level nsmapper=debug
mpirun -H g0002,g0003 --bind-to none ./bin/cannonMM -n 8192 -gx 2 -gy 1 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0 -mapping mappings -level nsmapper=debug
