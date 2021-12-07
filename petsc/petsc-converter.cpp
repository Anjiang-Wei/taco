#include "mpi.h"
#include "petscmat.h"
#include "petscvec.h"
#include "petsc.h"
#include "petscsys.h"
#include <iostream>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include "taco.h"

static const char help[] = "Matrix Market file to PETSc binary converter";

using namespace taco;

int main(int argc, char** argv) {
  PetscErrorCode ierr;

  char matrixInputFile[PETSC_MAX_PATH_LEN];
  char matrixOutputFile[PETSC_MAX_PATH_LEN];

  PetscBool inputSet, outputSet;

  PetscInitialize(&argc, &argv, (char *)0, help);

  ierr=PetscOptionsGetString(NULL, PETSC_NULL, "-matrix", matrixInputFile, PETSC_MAX_PATH_LEN-1, &inputSet); CHKERRQ(ierr);
  ierr=PetscOptionsGetString(NULL, PETSC_NULL, "-o", matrixOutputFile, PETSC_MAX_PATH_LEN-1, &outputSet); CHKERRQ(ierr);
  assert(inputSet && outputSet);

  // Declare the matrix.
  Mat A;

  // PETSc is stupid and has extremely slow writes to matrices. The only sensible way to do this is
  // to already pack the data in CSR with TACO and then give PETSc the pointers to those arrays.
  auto tensor = read(std::string(matrixInputFile), {Dense, Sparse});
  tensor.pack();
  assert(sizeof(int32_t) == sizeof(PetscInt));
  assert(sizeof(double) == sizeof(PetscScalar));
  auto it = tensor.getTacoTensorT();
  auto row = it->indices[1][0];
  auto crd = it->indices[1][1];
  auto vals = it->vals;

  // Finally create the matrix and dump it out.
  MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, tensor.getDimension(0), tensor.getDimension(1), (PetscInt*)row, (PetscInt*)crd, (PetscScalar*)vals, &A);
  PetscPrintf(PETSC_COMM_WORLD, "Assembling matrix within PETSc.\n");
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  PetscPrintf(PETSC_COMM_WORLD, "Finished matrix assembly.\n");
  PetscPrintf(PETSC_COMM_WORLD, "Beginning dump to output file. \n", matrixOutputFile);
  PetscViewer output_viewer;
  PetscViewerBinaryOpen(PETSC_COMM_SELF, matrixOutputFile, FILE_MODE_WRITE, &output_viewer);
  MatView(A, output_viewer);
  PetscViewerDestroy(&output_viewer);
  PetscPrintf(PETSC_COMM_WORLD, "Finished dump to output file. \n", matrixOutputFile);
  PetscFinalize();
  return 0;
}
