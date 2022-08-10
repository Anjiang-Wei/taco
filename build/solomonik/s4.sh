#!/bin/bash
#SBATCH -N 4
#SBATCH -n 4
#SBATCH -c 40
#SBATCH -p gpu

# 8 GPU: 2 2 2
# 16 GPU: 4 1 4
# -lg:inorder

mpirun --bind-to none ../bin/solomonikMM-cuda -n 20000 -rpoc 4 -c 1 -rpoc3 4 -ll:gpu 4 -ll:fsize 14G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -level mapper=debug -logfile mapper44_taco%.log -lg:inorder # -lg:prof 2 -lg:prof_logfile prof_taco%.gz
mpirun --bind-to none ../bin/solomonikMM-cuda -n 20000 -rpoc 4 -c 1 -rpoc3 4 -ll:gpu 4 -ll:fsize 14G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -mapping mappings -dslmapper -level nsmapper=debug -level mapper=debug -logfile mapper44_dsl%.log -lg:inorder # -lg:prof 2 -lg:prof_logfile prof_dsl%.gz


# mpirun --bind-to none ../bin/solomonikMM-cuda -n 20000 -rpoc 2 -c 2 -rpoc3 2 -ll:gpu 4 -ll:fsize 14G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -level mapper=debug -logfile mapper24_taco%.log # -lg:prof 2 -lg:prof_logfile prof_taco%.gz
# mpirun --bind-to none ../bin/solomonikMM-cuda -n 20000 -rpoc 2 -c 2 -rpoc3 2 -ll:gpu 4 -ll:fsize 14G -ll:ocpu 1 -ll:othr 1 -ll:nsize 30G -ll:ncsize 0 -mapping mappings -dslmapper -level nsmapper=debug -level mapper=debug -logfile mapper24_dsl%.log # -lg:prof 2 -lg:prof_logfile prof_dsl%.gz
