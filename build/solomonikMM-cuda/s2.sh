#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 40
#SBATCH -p gpu

mpirun -n 2 -npernode 1 --bind-to none ../bin/solomonikMM-cuda -n 1000 -rpoc 2 -c 2 -rpoc3 2 -tm:untrack_valid_regions -ll:ocpu 1 -ll:othr 1 -ll:gpu 4 -ll:fsize 15G -ll:nsize 30G -ll:ncsize 0 -level mapper=debug -logfile mapper24_taco%.log -wrapper

mpirun -n 2 -npernode 1 --bind-to none ../bin/solomonikMM-cuda -n 1000 -rpoc 2 -c 2 -rpoc3 2 -tm:untrack_valid_regions -ll:ocpu 1 -ll:othr 1 -ll:gpu 4 -ll:fsize 15G -ll:nsize 30G -ll:ncsize 0 -level mapper=debug -level nsmapper=debug -dslmapper -mapping mappings -logfile mapper24_dsl%.log -wrapper
