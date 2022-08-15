#!/bin/bash
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 40
#SBATCH -p gpu

mpirun -n 4 -npernode 1 --bind-to none ../bin/solomonikMM -n 8192 -rpoc 4 -c 1 -rpoc3 2 -tm:untrack_valid_regions -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0 -level mapper=debug -logfile mapper44_taco%.log

mpirun -n 4 -npernode 1 --bind-to none ../bin/solomonikMM -n 8192 -rpoc 4 -c 1 -rpoc3 2 -tm:untrack_valid_regions -ll:ocpu 2 -ll:othr 9 -ll:nsize 3G -ll:ncsize 0 -level mapper=debug -level nsmapper=debug -dslmapper -mapping mappings -logfile mapper44_dsl%.log
