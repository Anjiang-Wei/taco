#ifndef DISTAL_BLAS_H
#define DISTAL_BLAS_H

// Provide an indirection around CPU BLAS operations to allow for swapping in
// an out of different libraries. For convenience sake, we will omit the arguments
// that control whether the inputs are transposed / row major format etc.
template<typename T>
void BLAS_GEMM(size_t m,
               size_t n,
               size_t k,
               const T alpha,
               const T* A,
               const int lda,
               const T* B,
               const int ldb,
               const T beta,
               T* C,
               const int ldc);


// TODO (rohany): Gaurd this behind a set of defines so that we can swap the BLAS vendor that we are using. Ideally,
//   all that will change is that we just define including a different header set.
#include "cblas.h"

template<>
void BLAS_GEMM<double>(size_t m,
                       size_t n,
                       size_t k,
                       const double alpha,
                       const double* A,
                       const int lda,
                       const double* B,
                       const int ldb,
                       const double beta,
                       double* C,
                       const int ldc) {
  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    m,
    n,
    k,
    alpha,
    A,
    lda,
    B,
    ldb,
    beta,
    C,
    ldc
  );
}

#endif DISTAL_BLAS_H