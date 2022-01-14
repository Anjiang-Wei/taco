#!/usr/tce/packages/python/python-3.7.2/bin/python3.7
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-c2html=0',
    '--download-hwloc=0',
    '--download-sowing=0',
    '--prefix=./petsc-install/',
    '--with-64-bit-indices=0',
    '--with-blaslapack-lib=/usr/tcetmp/packages/lapack/lapack-3.9.0-gcc-7.3.1/lib/liblapack.so /usr/tcetmp/packages/lapack/lapack-3.9.0-gcc-7.3.1/lib/libblas.so',
    '--with-cc=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1/bin/mpigcc',
    '--with-clanguage=C',
    '--with-cxx-dialect=C++17',
    '--with-cxx=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1/bin/mpig++',
    '--with-cuda=1',
    '--with-debugging=0',
    '--with-fc=/usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1/bin/mpigfortran',
    '--with-fftw=0',
    '--with-hdf5-dir=/usr/tcetmp/packages/petsc/build/3.13.0/spack/opt/spack/linux-rhel7-power9le/xl_r-16.1/hdf5-1.10.6-e7e7urb5k7va3ib7j4uro56grvzmcmd4',
    '--with-hdf5=1',
    '--with-mumps=0',
    '--with-precision=double',
    '--with-scalapack=0',
    '--with-scalar-type=real',
    '--with-shared-libraries=1',
    '--with-ssl=0',
    '--with-suitesparse=0',
    '--with-trilinos=0',
    '--with-valgrind=0',
    '--with-x=0',
    '--with-zlib-include=/usr/include',
    '--with-zlib-lib=/usr/lib64/libz.so',
    '--with-zlib=1',
    'CFLAGS=-g -DNoChange',
    'COPTFLAGS=\"-O3\"',
    'CXXFLAGS=\"-O3\"',
    'CXXOPTFLAGS=\"-O3\"',
    'FFLAGS=-g',
    'CUDAFLAGS=-std=c++17',
    'FOPTFLAGS=',
    'PETSC_ARCH=arch-linux-c-opt',
  ]
  configure.petsc_configure(configure_options)
