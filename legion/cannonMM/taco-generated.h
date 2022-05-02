#ifndef TACO_GENERATED_H
#define TACO_GENERATED_H
#include "legion.h"
#include "legion_tensor.h"

struct partitionPackForplaceLegionA {
  LegionTensorPartition aPartition;
};

struct partitionPackForplaceLegionB {
  LegionTensorPartition bPartition;
};

struct partitionPackForplaceLegionC {
  LegionTensorPartition cPartition;
};

struct partitionPackForcomputeLegion {
  LegionTensorPartition aPartition;
  LegionTensorPartition bPartition;
  LegionTensorPartition cPartition;
};


partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, int32_t gridX, int32_t gridY);


void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY);

partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, int32_t gridX, int32_t gridY);


void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY);

partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, int32_t gridX, int32_t gridY);


void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, partitionPackForplaceLegionC* partitionPack, int32_t gridX, int32_t gridY);


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, int32_t gridX, int32_t gridY);



void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, partitionPackForcomputeLegion* partitionPack, int32_t gridX, int32_t gridY);
void registerTacoTasks();
#endif // TACO_GENERATED_H
