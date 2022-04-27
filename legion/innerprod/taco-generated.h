#ifndef TACO_GENERATED_H
#define TACO_GENERATED_H
#include "legion.h"
#include "legion_tensor.h"

struct partitionPackForcomputeLegion {
  LegionTensorPartition bPartition;
  LegionTensorPartition cPartition;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, LegionTensor* c, int32_t pieces);


double computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, LegionTensor* c, partitionPackForcomputeLegion* partitionPack, int32_t pieces);
void registerTacoTasks();
#endif // TACO_GENERATED_H
