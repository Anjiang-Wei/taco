#ifndef TACO_GENERATED_H
#define TACO_GENERATED_H
#include "legion.h"
#include "legion_tensor.h"

struct partitionPackForplaceLegionA {
  LegionTensorPartition APartition;
};

struct partitionPackForplaceLegionB {
  LegionTensorPartition BPartition;
};

struct partitionPackForplaceLegionC {
  LegionTensorPartition CPartition;
};

struct partitionPackForplaceLegionD {
  LegionTensorPartition DPartition;
};

struct partitionPackForcomputeLegion {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
  LegionTensorPartition CPartition;
  LegionTensorPartition DPartition;
};


Legion::LogicalPartition partitionLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX);

Legion::LogicalPartition partitionLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY, int32_t gridZ);

Legion::LogicalPartition partitionLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridY);

Legion::LogicalPartition partitionLegionD(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* D, int32_t gridZ);

partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX);


void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY, int32_t gridZ);

partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY, int32_t gridZ);


void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY, int32_t gridZ);

partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridY);


void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, partitionPackForplaceLegionC* partitionPack, int32_t gridY, int32_t gridX, int32_t gridZ);

partitionPackForplaceLegionD partitionForplaceLegionD(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* D, int32_t gridZ);


void placeLegionD(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* D, partitionPackForplaceLegionD* partitionPack, int32_t gridZ, int32_t gridX, int32_t gridY);

partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, int32_t gridX, int32_t gridY, int32_t gridZ);


void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, partitionPackForcomputeLegion* partitionPack, int32_t gridX, int32_t gridY, int32_t gridZ);
void registerTacoTasks();
#endif // TACO_GENERATED_H
