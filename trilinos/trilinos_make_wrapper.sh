#!/bin/bash
OMPI_CXX=/g/g15/yadav2/taco/trilinos/Trilinos/packages/kokkos/bin/nvcc_wrapper LLNL_USE_OMPI_VARS=Y make "$@"
