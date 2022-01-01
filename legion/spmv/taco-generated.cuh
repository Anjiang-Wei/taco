#ifndef TACO_GENERATED_CUH
#define TACO_GENERATED_CUH
#include "legion.h"
#include "legion_tensor.h"

struct partitionPackForcomputeLegionRowSplit {
  LegionTensorPartition aPartition;
  LegionTensorPartition BPartition;
};

struct partitionPackForcomputeLegionPosSplit {
  LegionTensorPartition aPartition;
  LegionTensorPartition BPartition;
};

struct partitionPackForcomputeLegionPosSplitDCSR {
  LegionTensorPartition BPartition;
};


partitionPackForcomputeLegionRowSplit partitionForcomputeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces);


void computeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionRowSplit* partitionPack, int32_t pieces);

partitionPackForcomputeLegionPosSplit partitionForcomputeLegionPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces);


void computeLegionPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplit* partitionPack, int32_t pieces);

partitionPackForcomputeLegionPosSplitDCSR partitionForcomputeLegionPosSplitDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces);


void computeLegionPosSplitDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplitDCSR* partitionPack, int32_t pieces);
void registerTacoTasks();
#endif // TACO_GENERATED_CUH
