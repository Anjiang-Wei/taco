#!/bin/bash
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 40
#SBATCH -p gpu

mpirun -n 4 -npernode 1 --bind-to none ../bin/johnsonMM-cuda -n 10000 -gdim 4 -tm:untrack_valid_regions -ll:ocpu 1 -ll:othr 1 -ll:gpu 4 -ll:fsize 15G -ll:nsize 30G -ll:ncsize 0 -tm:multiple_shards_per_node -level mapper=debug -logfile mapper44_taco%.log

mpirun -n 4 -npernode 1 --bind-to none ../bin/johnsonMM-cuda -n 10000 -gdim 4 -tm:untrack_valid_regions -ll:ocpu 1 -ll:othr 1 -ll:gpu 4 -ll:fsize 15G -ll:nsize 30G -ll:ncsize 0 -tm:multiple_shards_per_node -level mapper=debug -level nsmapper=debug -dslmapper -mapping mappings -logfile mapper44_dsl%.log
