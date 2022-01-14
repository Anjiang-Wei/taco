# Note that you must delete and redownload PETSc between build attempts.
# Also, PETSC_DIR must be unset before attempting to run this script.
set -e
set -x
cd petsc
git checkout v3.16.3
../configure_petsc.py
make PETSC_DIR=/g/g15/yadav2/taco/petsc/petsc PETSC_ARCH=arch-linux-c-opt all
make PETSC_DIR=/g/g15/yadav2/taco/petsc/petsc PETSC_ARCH=arch-linux-c-opt install
make PETSC_DIR=/g/g15/yadav2/taco/petsc/petsc/petsc-install PETSC_ARCH="" check
