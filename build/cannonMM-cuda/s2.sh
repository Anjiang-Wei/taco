#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 40
#SBATCH -p gpu

mpirun --bind-to none ../bin/cannonMM-cuda -n 20000 -ll:gpu 4 -ll:fsize 15G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -gx 1 -gy 2 -dm:exact_region -tm:untrack_valid_regions -lg:spy -logfile spy2_cannon_%.log -lg:prof 2 -lg:prof_logfile prof2_cannon_%.gz
# mpirun --bind-to none ../bin/cannonMM-cuda -n 20000 -ll:gpu 4 -ll:fsize 15G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -gx 1 -gy 2 -mapping mappings -dslmapper -level nsmapper=debug -level mapper=debug -logfile mapper24_dsl%.log -wrapper -dm:exact_region -tm:untrack_valid_regions
