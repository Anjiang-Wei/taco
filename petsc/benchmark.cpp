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

void spmm(Mat B, int warmup, int niter, int jdim) {
  Mat A, C;
  PetscScalar one = 1.0, zero = 0.0;
  PetscInt i, j = jdim, k;
  MatGetSize(B, &i, &k);
  // Create the other matrices.
  MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, i, j, NULL, &A);
  MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, k, j, NULL, &C);
  // Initialize entries in the output.
  MatZeroEntries(A);
  for (int kk = 0; kk < k; kk++) {
    for (int jj = 0; jj < j; jj++) {
      MatSetValue(C, kk, jj, 1, INSERT_VALUES);
    }
  }
  MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

  // Finally, do the computation.
  auto avgTime = benchmarkWithWarmup(warmup, niter, [&]() {
    MatMatMult(B, C, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A);
  });
  PetscPrintf(PETSC_COMM_WORLD, "Average time: %lf ms.\n", avgTime * 1000);
}

int main(int argc, char** argv) {
  PetscErrorCode ierr;
  
  char matrixInputFile[PETSC_MAX_PATH_LEN]; PetscBool matrixInputFileSet;
  PetscInt warmup = 5, nIter = 10; PetscBool warmupSet, nIterSet;
  PetscInt spmmJdim = 32; PetscBool spmmJdimSet;
  const int BENCHMARK_NAME_MAX_LEN = 20;
  char benchmarkKindInput[BENCHMARK_NAME_MAX_LEN]; PetscBool benchmarkKindNameSet;

  PetscInitialize(&argc, &argv, (char *)0, help);
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-matrix", matrixInputFile, PETSC_MAX_PATH_LEN-1, &matrixInputFileSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-warmup", &warmup, &warmupSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-n", &nIter, &nIterSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-bench", benchmarkKindInput, BENCHMARK_NAME_MAX_LEN - 1, &benchmarkKindNameSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-spmmJdim", &spmmJdim, &spmmJdimSet); CHKERRQ(ierr);

  std::string benchmark = "spmv";
  if (benchmarkKindNameSet) {
    benchmark = benchmarkKindInput;
  }

  // Load the input matrix from the file.
  Mat A;
  ierr = MatCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  PetscViewer viewer; 
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERBINARY);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, matrixInputFile);
  MatLoad(A, viewer);

  if (benchmark == "spmv") {
    spmv(A, warmup, nIter);
  } else if (benchmark == "spmm") {
    spmm(A, warmup, nIter, spmmJdim);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Invalid benchmark name, choose one of spmv,spmm.\n");
    PetscFinalize();
    return -1;
  }

  PetscFinalize();
  return 0;
}
