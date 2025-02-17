#ifndef HANDWRITTEN_H
#define HANDWRITTEN_H
#include "legion.h"
#include <functional>
enum FieldIDs {
  FID_VALUE,
  FID_INDEX,
  FID_RECT_1,
  FID_COORD_X,
  FID_COORD_Y,
};

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_PARTITION_A2_CRD,
  TID_SPMV,
  TID_SPMV_POS_SPLIT,
  TID_PRINT_COORD_LIST,
  TID_PACK_A_CSR,
  TID_ATTACH_REGIONS,
  TID_SPMV_GPU,
  TID_DEPPART_POS_1D,
  TID_INIT_X,
};

struct spmvPosSplitArgs {
  int nnz;
  int pieces;
  int A1_dimension;
};

void benchmarkAsyncCall(Legion::Context ctx, Legion::Runtime* runtime, std::vector<size_t>& times, std::function<void(void)> f);
void runAsyncCall(Legion::Context ctx, Legion::Runtime* runtime, std::function<void(void)> f);

void registerSPMVGPU();
void spmvgpu(Legion::Context ctx,
             Legion::Runtime* runtime,
             int32_t n, size_t nnz,
             Legion::LogicalRegion y,
             Legion::LogicalRegion A2_pos,
             Legion::LogicalRegion A2_pos_par,
             Legion::LogicalRegion A2_crd,
             Legion::LogicalRegion A2_crd_par,
             Legion::LogicalRegion A_vals,
             Legion::LogicalRegion A_vals_par,
             Legion::LogicalRegion x_vals);

#endif // HANDWRITTEN_H
