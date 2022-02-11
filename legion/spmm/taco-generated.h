#ifndef TACO_GENERATED_H
#define TACO_GENERATED_H
#include "legion.h"
#include "legion_tensor.h"

struct partitionPackForcomputeLegion {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
};

struct partitionPackForcomputeLegionBatched {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
  LegionTensorPartition CPartition;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gx);


void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t gx);


partitionPackForcomputeLegionBatched partitionForcomputeLegionBatched(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gx);


void computeLegionBatched(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegionBatched* partitionPack, int32_t gx);
void registerTacoTasks();
#endif // TACO_GENERATED_H
