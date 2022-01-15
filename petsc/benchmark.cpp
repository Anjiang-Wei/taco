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

PetscBool GPUENABLED = PETSC_FALSE; PetscBool GPUENABLED_SET = PETSC_FALSE;

void dump(Mat A) {
  MatView(A, PETSC_VIEWER_STDOUT_WORLD);
}
void dump(Vec v) {
  VecView(v, PETSC_VIEWER_STDOUT_WORLD);
}

void setMatToConstant(Mat mat, PetscScalar c) {
  PetscInt rStart, rEnd, m, n;
  MatGetSize(mat, &m, &n);
  MatGetOwnershipRange(mat, &rStart, &rEnd);
  for (int i = rStart; i < rEnd; i++) {
    for (int j = 0; j < n; j++) {
      MatSetValue(mat, i, j, c, INSERT_VALUES);
    }
  }
  MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
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

double benchmarkWithWarmupAndSetup(int warmup, int numIter, std::function<void(void)> work, std::function<void(void)> setup) {
  for (int i = 0; i < warmup; i++) {
    setup();
    work();
  }
  PetscLogDouble total = 0;
  PetscLogDouble start, end;
  for (int i = 0; i < numIter; i++) {
    setup();
    PetscTime(&start);
    work();
    PetscTime(&end);
    total += (end - start);
  }
  return double(total) / double(numIter);
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
  PetscInt i, j = jdim, k;
  MatGetSize(B, &i, &k);
  // Create the other matrices.
  if (GPUENABLED) {
    MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, i, j, NULL, &A);
    MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, k, j, NULL, &C);
  } else {
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, i, j, NULL, &A);
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, k, j, NULL, &C);
  }
  // Initialize entries in the output.
  MatZeroEntries(A);
  setMatToConstant(C, 1.0);
  
  // Finally, do the computation.
  auto avgTime = benchmarkWithWarmup(warmup, niter, [&]() {
    MatMatMult(B, C, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A);
  });
  PetscPrintf(PETSC_COMM_WORLD, "Average time: %lf ms.\n", avgTime * 1000);

  // Verification code below.
  // {
  //   Vec y;
  //   VecCreate(PETSC_COMM_WORLD, &y);
  //   VecSetFromOptions(y);
  //   VecSetSizes(y, PETSC_DECIDE, i);
  //   for (int ctr = 0; ctr < j; ctr++) {
  //     MatGetColumnVector(A, y, ctr);
  //     PetscReal norm;
  //     VecNorm(y, NORM_1, &norm);
  //     PetscPrintf(PETSC_COMM_WORLD, "%lf\n", norm);
  //   }
  // }
}

void spadd3(Mat B, Mat C, Mat D, int warmup, int niter) {
  Mat A;
  PetscInt i, j;
  MatGetSize(B, &i, &j);
  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, i, j);
  MatSetFromOptions(A);
  MatSetUp(A);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  auto avgTime = benchmarkWithWarmupAndSetup(warmup, niter, [&]() {
    // Since A is B, we just need to do A = C + A, A = D + A.
    MatAXPY(A, 1.0, C, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(A, 1.0, D, DIFFERENT_NONZERO_PATTERN);
  }, [&]() {
    // The setup is resetting A to be B.
    MatCopy(B, A, DIFFERENT_NONZERO_PATTERN);
  });

  PetscPrintf(PETSC_COMM_WORLD, "Average time: %lf ms.\n", avgTime * 1000);

  // Validation code.
  // {
  //   Vec norms;
  //   VecCreate(PETSC_COMM_WORLD, &norms);
  //   VecSetFromOptions(norms);
  //   VecSetSizes(norms, PETSC_DECIDE, i);
  //   VecSet(norms, 0.0);
  //   MatGetRowSum(A, norms);
  //   PetscScalar sum;
  //   VecSum(norms, &sum);
  //   PetscPrintf(PETSC_COMM_WORLD, "Element sum: %lf.\n", sum);
  // }
}

void sddmm(Mat B, int warmup, int niter, int jdim) {
  PetscInt i, j = jdim, k;
  MatGetSize(B, &i, &k);
  Mat A, C, D;
  // Create the dense C and D matrices.
  if (GPUENABLED) {
    MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, i, j, NULL, &C);
    MatCreateDenseCUDA(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, j, k, NULL, &D);
  } else {
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, i, j, NULL, &C);
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, j, k, NULL, &D);
  }
  // Create the output A matrix.
  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, i, k);
  MatSetFromOptions(A);
  MatSetUp(A);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  // Initialize entries in the inputs.
  setMatToConstant(C, 1.0);
  setMatToConstant(D, 1.0);
  // We have to use MAT_INITIAL_MATRIX on the first call, and after that
  // we can reuse the matrix since the non-zero pattern will be the same.
  bool isFirst = true;
  auto avgTime = benchmarkWithWarmup(warmup, niter, [&]() {
    if (isFirst) {
      MatMatMatMult(B, C, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A);
    } else {
      MatMatMatMult(B, C, D, MAT_REUSE_MATRIX, PETSC_DEFAULT, &A);
    }
  });
  PetscPrintf(PETSC_COMM_WORLD, "Average time: %lf ms.\n", avgTime * 1000);
}

int loadMatrixFromFile(Mat* A, char* filename) {
  auto ierr = MatCreate(PETSC_COMM_WORLD, A); CHKERRQ(ierr);
  MatSetFromOptions(*A);
  PetscViewer viewer;
  PetscViewerCreate(PETSC_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERBINARY);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, filename);
  MatLoad(*A, viewer);
  return 0;
}

int main(int argc, char** argv) {
  PetscErrorCode ierr;
  
  char matrixInputFile[PETSC_MAX_PATH_LEN]; PetscBool matrixInputFileSet;
  // Extra input matrices, set only for the SpAdd3 kernel.
  char add3MatrixInputFileC[PETSC_MAX_PATH_LEN]; PetscBool add3MatrixInputFileCSet;
  char add3MatrixInputFileD[PETSC_MAX_PATH_LEN]; PetscBool add3MatrixInputFileDSet;
  PetscInt warmup = 5, nIter = 10; PetscBool warmupSet, nIterSet;
  PetscInt spmmJdim = 32; PetscBool spmmJdimSet;
  PetscInt sddmmJdim = 128; PetscBool sddmmJdimSet;
  const int BENCHMARK_NAME_MAX_LEN = 20;
  char benchmarkKindInput[BENCHMARK_NAME_MAX_LEN]; PetscBool benchmarkKindNameSet;

  PetscInitialize(&argc, &argv, (char *)0, help);
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-matrix", matrixInputFile, PETSC_MAX_PATH_LEN-1, &matrixInputFileSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-add3MatrixC", add3MatrixInputFileC, PETSC_MAX_PATH_LEN-1, &add3MatrixInputFileCSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-add3MatrixD", add3MatrixInputFileD, PETSC_MAX_PATH_LEN-1, &add3MatrixInputFileDSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-warmup", &warmup, &warmupSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-n", &nIter, &nIterSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-bench", benchmarkKindInput, BENCHMARK_NAME_MAX_LEN - 1, &benchmarkKindNameSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-spmmJdim", &spmmJdim, &spmmJdimSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-sddmmJdim", &sddmmJdim, &sddmmJdimSet); CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL, NULL, "-enable_gpu", &GPUENABLED, &GPUENABLED_SET); CHKERRQ(ierr);

  std::string benchmark = "spmv";
  if (benchmarkKindNameSet) {
    benchmark = benchmarkKindInput;
  }

  // Load the input matrix from the file.
  Mat A;
  ierr = loadMatrixFromFile(&A, matrixInputFile); CHKERRQ(ierr);

  if (benchmark == "spmv") {
    spmv(A, warmup, nIter);
  } else if (benchmark == "spmm") {
    spmm(A, warmup, nIter, spmmJdim);
  } else if (benchmark == "spadd3") {
    if (!add3MatrixInputFileCSet || !add3MatrixInputFileDSet) {
      PetscPrintf(PETSC_COMM_WORLD, "Must provide C and D matrices with -add3Matrix{C,D}.\n");
      PetscFinalize();
      return -1;
    }
    // If we're doing SpAdd3, then we need to load in the other matrices too.
    Mat B = A;
    Mat C, D;
    ierr = loadMatrixFromFile(&C, add3MatrixInputFileC); CHKERRQ(ierr);
    ierr = loadMatrixFromFile(&D, add3MatrixInputFileD); CHKERRQ(ierr);
    spadd3(B, C, D, warmup, nIter);
  } else if (benchmark == "sddmm") {
    sddmm(A, warmup, nIter, sddmmJdim);
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Invalid benchmark name, choose one of spmv,spmm,spadd3,sddmm.\n");
    PetscFinalize();
    return -1;
  }

  PetscFinalize();
  return 0;
}
