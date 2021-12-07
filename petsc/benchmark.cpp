#include "mpi.h"
#include "petscmat.h"
#include "petscvec.h"
#include "petsc.h"
#include "petscsys.h"
#include "petsctime.h"

#include <iostream>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <functional>

static const char help[] = "Petsc benchmark utility";

void dump(Mat A) {
  MatView(A, PETSC_VIEWER_STDOUT_WORLD);
}
void dump(Vec v) {
  VecView(v, PETSC_VIEWER_STDOUT_WORLD);
}

double benchmarkWithWarmup(int warmup, int numIter, std::function<void(void)> f) {
  PetscLogDouble start, end;
  for (int i = 0; i < warmup; i++) {
    f();
  }
  PetscTime(&start);
  for (int i = 0; i < numIter; i++) {
    f();
  }
  PetscTime(&end);
  auto sec = end - start;
  return double(sec) / double(numIter);
}

void spmv(Mat A, int warmup, int niter) {
  Vec x, y;
  PetscScalar one = 1.0, zero = 0.0;
  PetscInt m, n;
  MatGetSize(A, &m, &n);
  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetFromOptions(x);
  VecSetSizes(x, PETSC_DECIDE, n);
  VecCreate(PETSC_COMM_WORLD, &y);
  VecSetFromOptions(y);
  VecSetSizes(y, PETSC_DECIDE, m);
  VecSet(x, one);
  VecSet(y, zero);
  auto avgTime = benchmarkWithWarmup(warmup, niter, [&]() {
    MatMult(A, x, y);
  });
  PetscPrintf(PETSC_COMM_WORLD, "Average time: %lf ms.\n", avgTime * 1000);
}

int main(int argc, char** argv) {
  PetscErrorCode ierr;
  
  char matrixInputFile[PETSC_MAX_PATH_LEN]; PetscBool matrixInputFileSet;
  PetscInt warmup = 5, nIter = 10; PetscBool warmupSet, nIterSet;

  PetscInitialize(&argc, &argv, (char *)0, help);
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-matrix", matrixInputFile, PETSC_MAX_PATH_LEN-1, &matrixInputFileSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-warmup", &warmup, &warmupSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-n", &nIter, &nIterSet); CHKERRQ(ierr);

  // Load the input matrix from the file.
  Mat A;
  PetscPrintf(PETSC_COMM_WORLD, "Before matrix load\n");
  // Turns out this is actually necessary...
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  PetscViewer viewer; 
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERBINARY);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, matrixInputFile);
  MatLoad(A, viewer);
  PetscPrintf(PETSC_COMM_WORLD, "After matrix load\n");

  // TODO (rohany): Add a branch here when we start benchmarking more computations.
  spmv(A, warmup, nIter);
  
  PetscFinalize();
  return 0;
}
