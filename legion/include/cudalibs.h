#ifndef TACO_CUDALIBS_H
#define TACO_CUDALIBS_H

#include "legion.h"
#include "cublas_v2.h"
#include "cusparse.h"

// Macro for easily checking the result of CuBLAS calls.
#define CHECK_CUBLAS(expr)                    \
  {                                           \
    cublasStatus_t result = (expr);           \
    checkCuBLAS(result, __FILE__, __LINE__); \
  }

// Macro for easily checking the result of CuSparse calls.
#define CHECK_CUSPARSE(expr)                  \
  {                                           \
    cusparseStatus_t result = (expr);           \
    checkCuSparse(result, __FILE__, __LINE__);  \
  }

// CuBLAS error checker.
void checkCuBLAS(cublasStatus_t status, const char* file, int line);
// CuSparse error checker.
void checkCuSparse(cusparseStatus_t status, const char* file, int line);

// Get and potentially initialize the CuBLAS handle for this processor.
cublasHandle_t getCuBLAS();
void initCUDA();
// Get and potentially initialize the CuSparse handle for this processor.
cusparseHandle_t getCuSparse();
void initCuSparseAtStartup();
#endif // TACO_CUDALIBS_H
