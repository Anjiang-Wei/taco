#include "taco_legion_header.h"
#include "shard.h"
#include <mutex>

using namespace Legion;

IndexSpace get_index_space(PhysicalRegion r) { return r.get_logical_region().get_index_space(); }
IndexSpace get_index_space(LogicalRegion r) { return r.get_index_space(); }

LogicalRegion get_logical_region(PhysicalRegion r) { return r.get_logical_region(); }
LogicalRegion get_logical_region(LogicalRegion r) { return r; }

IndexPartition get_index_partition(IndexPartition i) { return i; }
IndexPartition get_index_partition(LogicalPartition l) { return l.get_index_partition(); }

int getIndexPoint(const Legion::Task* task, int index) {
  return task->index_point[index];
}

TaskID taskID(int offset) {
  return TACO_TASK_BASE_ID + offset;
}

ShardingID shardingID(int offset) {
  return TACO_SHARD_BASE_ID + offset;
}

void registerPlacementShardingFunctor(Context ctx, Runtime* runtime, ShardingID funcID, std::vector<int>& dims) {
  // If we have multiple shards on the same node, they might all try to register sharding
  // functors at the same time. Put a lock here to make sure that only one actually does it.
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);

  auto func = Legion::Runtime::get_sharding_functor(funcID);
  // If the sharding functor already exists, return.
  if (func) { return; }
  // Otherwise, register the functor.
  auto functor = new TACOPlacementShardingFunctor(dims);
  runtime->register_sharding_functor(funcID, functor, true /* silence_warnings */);
}

// getSubRegion returns a subregion of the input region with the desired start and end coordinate.
LogicalRegion getSubRegion(Context ctx, Runtime* runtime, LogicalRegion region, Rect<1> bounds) {
  // TODO (rohany): Can I avoid creating this IndexSpace on each reallocation call?
  IndexSpaceT<1> colorSpace = runtime->create_index_space(ctx, Rect<1>(0, 0));
  Transform<1,1> transform;
  transform[0][0] = 0;
  // TODO (rohany): Is there a way to destroy the old partition of this region's index space?
  auto ip = runtime->create_partition_by_restriction(
      ctx,
      region.get_index_space(),
      colorSpace,
      transform,
      bounds,
      DISJOINT_KIND
  );
  // Get the subregion of the only point in the partition.
  return runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, region, ip), 0);
}

// mapRegion inline maps a region. It abstracts away a bit of cruft around creating RegionRequirements
// and making runtime calls.
Legion::PhysicalRegion mapRegion(Context ctx, Runtime* runtime, LogicalRegion region, FieldID fid) {
  // TODO (rohany): Do we need to pass in a privilege here, or if we are doing allocations
  //  we can assume that the region is going to be written to?
  RegionRequirement req(region, READ_WRITE, EXCLUSIVE, region, Mapping::DefaultMapper::EXACT_REGION);
  req.add_field(fid);
  return runtime->map_region(ctx, req);
}

Legion::PhysicalRegion legionMalloc(Context ctx, Runtime* runtime, LogicalRegion region, size_t size, FieldID fid) {
  auto subreg = getSubRegion(ctx, runtime, region, Rect<1>(0, size - 1));
  return mapRegion(ctx, runtime, subreg, fid);
}

Legion::PhysicalRegion legionRealloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::PhysicalRegion old, size_t newSize, Legion::FieldID fid) {
  // Get the bounds on the old region.
  assert(old.get_logical_region().get_dim() == 1);
  auto oldBounds = old.get_bounds<1, coord_t>();
  // Unmap the input existing region.
  runtime->unmap_region(ctx, old);
  // Just get between the old size and the new size to avoid having to copy the old data.
  auto subreg = getSubRegion(ctx, runtime, region, Rect<1>(oldBounds.bounds.hi[0] + 1, newSize - 1));
  return mapRegion(ctx, runtime, subreg, fid);
}
