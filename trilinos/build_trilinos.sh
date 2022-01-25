#!/bin/bash

export TRILINOS_DIR=/g/g15/yadav2/taco/trilinos/Trilinos/
export OMPI_CXX=$TRILINOS_DIR/packages/kokkos/bin/nvcc_wrapper
export LLNL_USE_OMPI_VARS=Y
# These supposedly are needed by the FAQ. I think they are only needed
# at runtime now?
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

cmake \
-DMPI_BASE_DIR=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1/ \
-DTPL_ENABLE_MPI=ON \
-DTPL_ENABLE_CUDA:BOOL=ON \
-DTrilinos_ENABLE_Tpetra=ON \
-DTrilinos_ENABLE_OpenMP=ON \
-DTrilinos_ENABLE_CUDA=ON \
-DKokkos_ENABLE_CUDA:BOOL=ON \
-DTPetra_ENABLE_CUDA:BOOL=ON \
-D Kokkos_ENABLE_Cuda_UVM:BOOL=ON \
-D Kokkos_ENABLE_Cuda_Lambda:BOOL=ON \
-DTpetra_INST_CUDA=ON \
-DTpetra_INST_OPENMP=ON \
-DCMAKE_INSTALL_PREFIX=../cmake-install/ \
..

make -j install

# Use when pointing to the installation of Trilinos.
# cmake ../ -DCMAKE_PREFIX_PATH=./Trilinos/cmake-install/lib/cmake/Trilinos/
