#ifndef TACO_LEGION_INCLUDES_H
#define TACO_LEGION_INCLUDES_H

#include "legion.h"
#include "mappers/default_mapper.h"

// Fields used by the generated TACO code.
enum TensorFields {
  FID_VAL,
  FID_RECT_1,
  FID_COORD,
};
const int TACO_TASK_BASE_ID = 10000;
const int TACO_SHARD_BASE_ID = 1000;

Legion::IndexSpace get_index_space(Legion::PhysicalRegion r);
Legion::IndexSpace get_index_space(Legion::LogicalRegion r);
Legion::LogicalRegion get_logical_region(Legion::PhysicalRegion r);
Legion::LogicalRegion get_logical_region(Legion::LogicalRegion r);
Legion::IndexPartition get_index_partition(Legion::IndexPartition i);
Legion::IndexPartition get_index_partition(Legion::LogicalPartition l);
int getIndexPoint(const Legion::Task* task, int index);
Legion::TaskID taskID(int offset);
Legion::ShardingID shardingID(int offset);

void registerPlacementShardingFunctor(Legion::Context ctx, Legion::Runtime* runtime, Legion::ShardingID funcID, std::vector<int>& dims);

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

// TODO (rohany): These might need to be templated on the dimension of the region.
// Functions for performing allocations on a region.
Legion::PhysicalRegion legionMalloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, size_t size, Legion::FieldID fid);
Legion::PhysicalRegion legionRealloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::PhysicalRegion old, size_t newSize, Legion::FieldID fid);

// Copy a partition onto a region with the same index space.
Legion::LogicalPartition copyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::IndexPartition toCopy, Legion::LogicalRegion toPartition);
Legion::LogicalPartition copyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalPartition toCopy, Legion::LogicalRegion toPartition);

#endif // TACO_LEGION_INCLUDES_H
