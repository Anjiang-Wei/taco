#ifndef TACO_GENERATED_CUH
#define TACO_GENERATED_CUH
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


extern "C" partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t rpoc);


extern "C" void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, partitionPackForplaceLegionA* partitionPack, int32_t rpoc, int32_t c);

extern "C" partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t rpoc);


extern "C" void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, partitionPackForplaceLegionB* partitionPack, int32_t rpoc, int32_t c);

extern "C" partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t rpoc);


extern "C" void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, partitionPackForplaceLegionC* partitionPack, int32_t rpoc, int32_t c);


extern "C" partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t rpoc, int32_t c, int32_t rpoc3);



extern "C" void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t rpoc, int32_t c, int32_t rpoc3);
extern "C" void registerTacoTasks();
extern "C" void dynamicallyRegisterDISTALTasks(void** args);
#endif // TACO_GENERATED_CUH
