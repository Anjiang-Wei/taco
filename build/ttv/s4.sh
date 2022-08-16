#!/bin/bash
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 40
#SBATCH -p gpu

mpirun -n 4 -npernode 1 --bind-to none ../bin/ttv -n 400 -gx 4 -gy 2 -tm:numa_aware_alloc -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0 -level mapper=debug -logfile mapper44_taco%.log

mpirun -n 4 -npernode 1 --bind-to none ../bin/ttv -n 400 -gx 4 -gy 2 -tm:numa_aware_alloc -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0 -level mapper=debug -level nsmapper=debug -dslmapper -mapping mappings -logfile mapper44_dsl%.log
