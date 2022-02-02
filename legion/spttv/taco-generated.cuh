#ifndef TACO_GENERATED_CUH
#define TACO_GENERATED_CUH
#include "legion.h"
#include "legion_tensor.h"

struct partitionPackForcomputeLegionDSS {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
};

struct partitionPackForcomputeLegionDSSPosSplit {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
};

struct partitionPackForcomputeLegionDSSPartialPosSplit {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
};


partitionPackForcomputeLegionDSS partitionForcomputeLegionDSS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, int32_t pieces);


void computeLegionDSS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSS* partitionPack, int32_t pieces);

partitionPackForcomputeLegionDSSPosSplit partitionForcomputeLegionDSSPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, int32_t pieces);


void computeLegionDSSPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSSPosSplit* partitionPack, int32_t pieces);

partitionPackForcomputeLegionDSSPartialPosSplit partitionForcomputeLegionDSSPartialPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, int32_t pieces);


void computeLegionDSSPartialPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSSPartialPosSplit* partitionPack, int32_t pieces);
void registerTacoTasks();
#endif // TACO_GENERATED_CUH
