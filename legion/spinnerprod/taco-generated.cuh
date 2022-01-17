#ifndef TACO_GENERATED_CUH
#define TACO_GENERATED_CUH
#include "legion.h"
#include "legion_tensor.h"

struct partitionPackForcomputeLegion {
  LegionTensorPartition BPartition;
  LegionTensorPartition CPartition;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, LegionTensor* C, int32_t pieces);


double computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t pieces);
void registerTacoTasks();
#endif // TACO_GENERATED_CUH
