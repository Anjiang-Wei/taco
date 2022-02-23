#ifndef TACO_TEMP_H
#define TACO_TEMP_H

#include "legion.h"

struct partitionPackForcomputeLegionRowSplit {
  LegionTensorPartition aPartition;
  LegionTensorPartition BPartition;
};

// typedef Legion::Rect<1, int32_t> PosRect;
typedef Legion::Rect<1> PosRect;

#endif
