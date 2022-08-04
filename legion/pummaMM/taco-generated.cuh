#ifndef TACO_GENERATED_CUH
#define TACO_GENERATED_CUH
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


extern "C" partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, int32_t gridX, int32_t gridY);


extern "C" void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY);

extern "C" partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, int32_t gridX, int32_t gridY);


extern "C" void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY);

extern "C" partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, int32_t gridX, int32_t gridY);


extern "C" void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, partitionPackForplaceLegionC* partitionPack, int32_t gridX, int32_t gridY);




extern "C" partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, int32_t gridX, int32_t gridY);





extern "C" void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, partitionPackForcomputeLegion* partitionPack, int32_t gridX, int32_t gridY);
extern "C" void registerTacoTasks();
extern "C" void dynamicallyRegisterDISTALTasks(void** args);
#endif // TACO_GENERATED_CUH
