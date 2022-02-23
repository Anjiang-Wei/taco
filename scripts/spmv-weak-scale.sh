#!/bin/bash

set -x
set -e

NODES="1 2 4 8 16 32 64"

echo "DISTAL CPU"
for n in $NODES; do
	jsrun -b none -c ALL_CPUS -g ALL_GPUS -r 1 -n $n /g/g15/yadav2/taco/build/bin/spmv -ll:ocpu 2 -ll:othr 18 -ll:onuma 1 -ll:nsize 100G -ll:ncsize 0 -ll:util 2 -tm:numa_aware_alloc -n 20 -warmup 10 -weak_scale
done

echo "DISTAL GPU"
# 1 and 2 GPUs.
jsrun -b none -c ALL_CPUS -g ALL_GPUS -r 1 -n 1 /g/g15/yadav2/taco/build/bin/spmv-cuda -ll:ocpu 2 -ll:othr 18 -ll:onuma 1 -ll:nsize 100G -ll:ncsize 0 -ll:util 2 -tm:numa_aware_alloc -n 20 -warmup 10 -weak_scale -ll:gpu 1 -ll:fsize 14.5G -tm:align128 -lg:eager_alloc_percentage 5
jsrun -b none -c ALL_CPUS -g ALL_GPUS -r 1 -n 1 /g/g15/yadav2/taco/build/bin/spmv-cuda -ll:ocpu 2 -ll:othr 18 -ll:onuma 1 -ll:nsize 100G -ll:ncsize 0 -ll:util 2 -tm:numa_aware_alloc -n 20 -warmup 10 -weak_scale -ll:gpu 2 -ll:fsize 14.5G -tm:align128 -lg:eager_alloc_percentage 5
for n in $NODES; do
    jsrun -b none -c ALL_CPUS -g ALL_GPUS -r 1 -n $n /g/g15/yadav2/taco/build/bin/spmv-cuda -ll:ocpu 2 -ll:othr 18 -ll:onuma 1 -ll:nsize 100G -ll:ncsize 0 -ll:util 2 -tm:numa_aware_alloc -n 20 -warmup 10 -weak_scale -ll:gpu 4 -ll:fsize 14.5G -tm:align128 -lg:eager_alloc_percentage 5
done

echo "PETSc CPU"
for n in $NODES; do
	jsrun -n $((40 * $n)) -r 40 -c 1 -b rs /g/g15/yadav2/taco/petsc/bin/benchmark -n 20 -warmup 10 -bench spmv-weak-scale
done

echo "PETSc GPU"
# 1 and 2 GPUs.
jsrun -n 1 -g 1 -r 1 -c 10 -b rs -M "-gpu" /g/g15/yadav2/taco/petsc/bin/benchmark -n 20 -warmup 10 -enable_gpu -vec_type cuda -bench spmv-weak-scale -mat_type mpiaij
jsrun -n 2 -g 1 -r 2 -c 10 -b rs -M "-gpu" /g/g15/yadav2/taco/petsc/bin/benchmark -n 20 -warmup 10 -enable_gpu -vec_type cuda -bench spmv-weak-scale -mat_type mpiaij
for n in $NODES; do
	jsrun -n $(($n * 4)) -g 1 -r 4 -c 10 -b rs -M "-gpu" /g/g15/yadav2/taco/petsc/bin/benchmark -n 20 -warmup 10 -enable_gpu -vec_type cuda -bench spmv-weak-scale -mat_type mpiaij
done
