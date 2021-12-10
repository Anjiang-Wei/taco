#!/bin/bash

cmake \
-DTPL_ENABLE_MPI=ON \
-DMPI_BASE_DIR=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1/ \
-DTrilinos_ENABLE_Tpetra=ON \
-DTrilinos_ENABLE_Xpetra=ON \
-DTrilinos_ENABLE_OpenMP=ON \
-DTrilinos_ENABLE_CUDA=ON \
-DCMAKE_INSTALL_PREFIX=../cmake-install/ \
..

make -j install

# Use when pointing to the installation of Trilinos.
# cmake ../ -DCMAKE_PREFIX_PATH=./Trilinos/cmake-install/lib/cmake/Trilinos/
