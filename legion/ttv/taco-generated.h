#ifndef TACO_GENERATED_H
#define TACO_GENERATED_H
#include "legion.h"
#include "legion_tensor.h"


Legion::LogicalPartition partitionLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX, int32_t gridY);

Legion::LogicalPartition partitionLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY);


Legion::LogicalPartition placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX, int32_t gridY);


Legion::LogicalPartition placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY);


Legion::LogicalPartition placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridX, int32_t gridY);


void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, Legion::LogicalPartition BPartition, Legion::LogicalPartition APartition, Legion::LogicalPartition CPartition);
void registerTacoTasks();
#endif // TACO_GENERATED_H
