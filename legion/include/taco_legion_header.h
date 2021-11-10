#ifndef TACO_LEGION_INCLUDES_H
#define TACO_LEGION_INCLUDES_H

#include "legion.h"
#include "mappers/default_mapper.h"
#include "legion_tensor.h"

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

// TODO (rohany): These might need to be templated on the dimension of the region.
// Functions for performing allocations on a region.
// TODO (rohany): Do these need to all take in parents and target regions?
// Allocate the entirety of a region with a given parent.
Legion::PhysicalRegion legionMalloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::LogicalRegion parent, Legion::FieldID fid);
// Allocate a subregion from a region of the given size, i.e. [0, size).
Legion::PhysicalRegion legionMalloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, size_t size, Legion::FieldID fid);
// Allocate a subregion from a region of a given size extended from an old size.
Legion::PhysicalRegion legionRealloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::PhysicalRegion old, size_t newSize, Legion::FieldID fid);

// TODO (rohany): This probably needs to be templated.
// getSubRegion returns the LogicalRegion corresponding to the subregion of region with bounds.
Legion::LogicalRegion getSubRegion(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::Rect<1> bounds);

// Copy a partition onto a region with the same index space.
Legion::LogicalPartition copyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::IndexPartition toCopy, Legion::LogicalRegion toPartition);
Legion::LogicalPartition copyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalPartition toCopy, Legion::LogicalRegion toPartition);

#endif // TACO_LEGION_INCLUDES_H
