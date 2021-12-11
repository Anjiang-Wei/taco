cd petsc
git checkout v3.13
../configure_petsc.py
make PETSC_DIR=/g/g15/yadav2/taco/petsc/petsc PETSC_ARCH=arch-linux-c-opt all
make PETSC_DIR=/g/g15/yadav2/taco/petsc/petsc PETSC_ARCH=arch-linux-c-opt install
make PETSC_DIR=/g/g15/yadav2/taco/petsc/petsc/petsc-install PETSC_ARCH="" check
