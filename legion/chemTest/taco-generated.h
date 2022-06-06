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

struct partitionPackForcomputeLegionNestedOMP {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
  LegionTensorPartition CPartition;
};

struct partitionPackForcomputeLegionTblis {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
  LegionTensorPartition CPartition;
};


partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX, int32_t gridY);


void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY);

partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY);


void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY);

partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridX, int32_t gridY);


void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, partitionPackForplaceLegionC* partitionPack, int32_t gridX, int32_t gridY);


partitionPackForcomputeLegionNestedOMP partitionForcomputeLegionNestedOMP(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gridX, int32_t gridY);



void computeLegionNestedOMP(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegionNestedOMP* partitionPack, int32_t gridX, int32_t gridY);


partitionPackForcomputeLegionTblis partitionForcomputeLegionTblis(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gridX, int32_t gridY);



void computeLegionTblis(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegionTblis* partitionPack, int32_t gridX, int32_t gridY);
void registerTacoTasks();
#endif // TACO_GENERATED_H
