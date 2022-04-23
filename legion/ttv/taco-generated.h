#ifndef TACO_GENERATED_H
#define TACO_GENERATED_H
#include "legion.h"
#include "legion_tensor.h"


LogicalPartition partitionLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX, int32_t gridY);

LogicalPartition partitionLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY);


LogicalPartition placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX, int32_t gridY);


LogicalPartition placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY);


LogicalPartition placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridX, int32_t gridY);


void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LogicalPartition BPartition, LogicalPartition APartition, LogicalPartition CPartition);
void registerTacoTasks();
#endif // TACO_GENERATED_H
