#ifndef TACO_GENERATED_CUH
#define TACO_GENERATED_CUH
#include "legion.h"
#include "legion_tensor.h"

struct partitionPackForcomputeLegion {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, int32_t pieces);


void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, partitionPackForcomputeLegion* partitionPack, int32_t pieces);
void registerTacoTasks();
#endif // TACO_GENERATED_CUH
