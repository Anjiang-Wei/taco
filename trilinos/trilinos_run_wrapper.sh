#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1 
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 
eval "$@"
