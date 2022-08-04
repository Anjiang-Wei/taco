#!/bin/bash
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 40
#SBATCH -p gpu

# -lg:partcheck -lg:safe_mapper
# -level mapper=debug -logfile slurm_mytry_incorrect%.log
# -lg:spy -logfile spy_%.log
# -lg:safe_ctrlrepl -lg:warn
#  -dm:replicate 0
# -level mapper=debug -logfile mapper_taco%.log
# -lg:prof 2 -lg:prof_logfile prof_taco%.gz
# 
# -level mapper=debug -logfile mapper_dsl%.log
# -lg:prof 2 -lg:prof_logfile prof_dsl%.gz
# -level nsmapper=debug

# mpirun --bind-to none ./bin/pummaMM-cuda -n 20000 -ll:gpu 4 -ll:fsize 15G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -gx 1 -gy 2 -mapping mappings
mpirun --bind-to none ./bin/pummaMM-cuda -n 20000 -ll:gpu 4 -ll:fsize 15G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -gx 1 -gy 2 -level mapper=debug -logfile mapper_taco%.log -lg:prof 2 -lg:prof_logfile prof_taco%.gz
mpirun --bind-to none ./bin/pummaMM-cuda -n 20000 -ll:gpu 4 -ll:fsize 15G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -gx 1 -gy 2 -mapping mappings -dslmapper -level nsmapper=debug -level mapper=debug -logfile mapper_dsl%.log -lg:prof 2 -lg:prof_logfile prof_dsl%.gz
