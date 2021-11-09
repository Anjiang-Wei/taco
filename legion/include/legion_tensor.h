#ifndef TACO_LEGION_TENSOR_H
#define TACO_LEGION_TENSOR_H

#include "legion.h"

// TODO (rohany): We might have to add a mirrored "physical tensor" that has all of this stuff
//  pulled out as the physical one when in a child task.
// LegionTensor is a representation of a taco_tensor_t for the Legion backend.
struct LegionTensor {
  // The number of modes the tensor has.
  int32_t order;
  // The size of each dimension in the tensor.
  std::vector<int32_t> dims;

  // Matching the taco_tensor_t, we'll have a vector of regions for each of the indices
  // in the representation of the tensor.
  std::vector<std::vector<Legion::LogicalRegion>> indices;
  // Unlike the taco_tensor_t, we have to maintain the parent regions for each region so
  // that we can pass the region to child tasks and partitioning operators that need to
  // read the regions.
  std::vector<std::vector<Legion::LogicalRegion>> indicesParents;

  // We have a vals region just like the taco_tensor_t, and also must maintain its parent.
  Legion::LogicalRegion vals;
  Legion::LogicalRegion valsParent;

  // denseLevelRuns is a set of index spaces corresponding to each run of dense
  // levels in the tensor. It is used to create partitions of the dense levels
  // for use in partitioning following levels of the tensor.
  std::vector<Legion::IndexSpace> denseLevelRuns;

  // TODO (rohany): There are some sort of serialization methods here, as well
  //  as potentially methods to put each region into a region requirement. Actually,
  //  I don't think that there is a method that can apriori put all of the regions
  //  into a region requirement as it requires knowing what regions are actually
  //  part of the pack, which the compiler does.
};


#endif //TACO_LEGION_TENSOR_H
