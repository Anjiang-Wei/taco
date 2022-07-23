#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 40
#SBATCH -p gpu

# mpirun --bind-to none ./bin/cannonMM-cuda -n 8192 -ll:gpu 4 -ll:fsize 15G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -gx 1 -gy 2
# -lg:partcheck -lg:safe_mapper
# -level mapper=debug

# mpirun -H g0002,g0003 --bind-to none ./bin/cannonMM -n 8192 -gx 2 -gy 2 -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0 -mapping mappings -level nsmapper=debug
# mpirun --bind-to none ./bin/cannonMM-cuda -n 8192 -lg:partcheck -lg:safe_mapper -ll:gpu 4 -ll:fsize 15G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -gx 1 -gy 2 -mapping mappings -level nsmapper=debug -level mapper=debug
mpirun --bind-to none ./bin/cannonMM-cuda -n 8192 -ll:gpu 4 -ll:fsize 15G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -gx 1 -gy 2 -mapping mappings -level nsmapper=debug
