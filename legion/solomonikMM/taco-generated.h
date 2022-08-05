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

struct partitionPackForcomputeLegion {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
  LegionTensorPartition CPartition;
};


extern "C" partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, int32_t rpoc);


extern "C" void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY);

extern "C" partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, int32_t rpoc);


extern "C" void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY);

extern "C" partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, int32_t rpoc);


extern "C" void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, partitionPackForplaceLegionC* partitionPack, int32_t gridX, int32_t gridY);


extern "C" partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, int32_t gridX, int32_t gridY);



extern "C" void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, partitionPackForcomputeLegion* partitionPack, int32_t rpoc, int32_t c2, int32_t rpoc3);
extern "C" void registerTacoTasks();
extern "C" void dynamicallyRegisterDISTALTasks(void** args);
#endif // TACO_GENERATED_H
