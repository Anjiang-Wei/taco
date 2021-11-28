#include "taco_legion_header.h"
#include "shard.h"
#include <mutex>

using namespace Legion;

IndexSpace get_index_space(PhysicalRegion r) { return r.get_logical_region().get_index_space(); }
IndexSpace get_index_space(LogicalRegion r) { return r.get_index_space(); }
IndexSpace get_index_space(RegionWrapper r) { return r.get_index_space(); }

LogicalRegion get_logical_region(PhysicalRegion r) { return r.get_logical_region(); }
LogicalRegion get_logical_region(LogicalRegion r) { return r; }
LogicalRegion get_logical_region(RegionWrapper r) { return LogicalRegion(r); }

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
Legion::PhysicalRegion mapRegion(Context ctx, Runtime* runtime, LogicalRegion region, LogicalRegion parent, FieldID fid) {
  // TODO (rohany): Do we need to pass in a privilege here, or if we are doing allocations
  //  we can assume that the region is going to be written to?
  RegionRequirement req(region, READ_WRITE, EXCLUSIVE, parent, Mapping::DefaultMapper::EXACT_REGION);
  req.add_field(fid);
  auto result = runtime->map_region(ctx, req);
  result.wait_until_valid();
  return result;
}

Legion::PhysicalRegion legionMalloc(Context ctx, Runtime* runtime, LogicalRegion region, size_t size, FieldID fid) {
  auto subreg = getSubRegion(ctx, runtime, region, Rect<1>(0, size - 1));
  return mapRegion(ctx, runtime, subreg, region, fid);
}

Legion::PhysicalRegion legionMalloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::LogicalRegion parent, Legion::FieldID fid) {
  return mapRegion(ctx, runtime, region, parent, fid);
}

Legion::PhysicalRegion legionRealloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::PhysicalRegion old, size_t newSize, Legion::FieldID fid) {
  // Get the bounds on the old region.
  assert(old.get_logical_region().get_dim() == 1);
  auto oldBounds = old.get_bounds<1, coord_t>();
  // Unmap the input existing region.
  runtime->unmap_region(ctx, old);
  // Just get between the old size and the new size to avoid having to copy the old data.
  auto subreg = getSubRegion(ctx, runtime, region, Rect<1>(oldBounds.bounds.hi[0] + 1, newSize - 1));
  return mapRegion(ctx, runtime, subreg, region, fid);
}

LogicalPartition copyPartition(Context ctx, Runtime* runtime, IndexPartition toCopy, LogicalRegion toPartition, Color color) {
  std::map<DomainPoint, Domain> domains;
  auto colorSpace = runtime->get_index_partition_color_space(ctx, toCopy);
  auto colorSpaceName = runtime->get_index_partition_color_space_name(ctx, toCopy);
  switch (colorSpace.get_dim()) {
#define BLOCK(DIM) \
    case DIM:      \
      {            \
        for (PointInDomainIterator<DIM> itr(colorSpace); itr(); itr++) { \
          auto subspace = runtime->get_index_subspace(ctx, toCopy, DomainPoint(*itr)); \
          auto domain = runtime->get_index_space_domain(ctx, subspace);          \
          domains[*itr] = domain;                                        \
        }          \
        break;     \
      }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      assert(false);
  }
  // TODO (rohany): Is there a way to get the kind (i.e. disjoint, aliased etc) of the existing partition?
  auto toPartitionIndexSpace = toPartition.get_index_space();
  auto indexPart = runtime->create_partition_by_domain(ctx, toPartitionIndexSpace, domains, colorSpaceName, color);
  return runtime->get_logical_partition(ctx, toPartition, indexPart);
}

LogicalPartition copyPartition(Context ctx, Runtime* runtime, LogicalPartition toCopy, LogicalRegion toPartition, Color color) {
  return copyPartition(ctx, runtime, toCopy.get_index_partition(), toPartition, color);
}

IndexPartition densifyPartition(Context ctx, Runtime* runtime, IndexSpace ispace, IndexPartition part, Color color) {
  DomainPointColoring coloring;
  auto colorSpace = runtime->get_index_partition_color_space_name(ctx, part);
  auto colorSpaceDom = runtime->get_index_space_domain(ctx, colorSpace);
  assert(colorSpace.get_dim() == 1);
  for (PointInDomainIterator<1> itr(colorSpaceDom); itr(); itr++) {
    auto subreg = runtime->get_index_subspace(ctx, part, Color(*itr));
    auto subDom = runtime->get_index_space_domain(ctx, subreg);
    // TODO (rohany): Do I need to cast this into a DomainT and call tighten? Or will the domains
    //  computed by create_partition_by_preimage_range already be tightened?
    coloring[*itr] = {subDom.lo(), subDom.hi()};
  }
  return runtime->create_partition_by_domain(ctx, ispace, coloring, colorSpace, true /* perform_intersections */, LEGION_COMPUTE_KIND, color);
}

AffineProjection::AffineProjection(std::vector<int> projs) : projs(projs), outputDim(0) {
  for (auto it : this->projs) {
    if (it != AffineProjection::BOT) {
      this->outputDim++;
    }
  }
}

int AffineProjection::dim() const {
  return int(this->projs.size());
}

int AffineProjection::outDim() const {
  return this->outputDim;
}

int AffineProjection::operator[] (size_t i) const {
  assert(i >= 0 && int(i) < this->dim());
  return this->projs[i];
}

Legion::IndexPartition AffineProjection::apply(Legion::Context ctx, Runtime *runtime, Legion::IndexPartition part,
                                               Legion::IndexSpace ispace, Color color) {
  DomainPointColoring col;
  auto colorSpace = runtime->get_index_partition_color_space(ctx, part);
  auto colorSpaceName = runtime->get_index_partition_color_space_name(ctx, part);
  assert(colorSpace.get_dim() == 1);
  for (PointInDomainIterator<1> itr(colorSpace); itr(); itr++) {
    auto subspace = runtime->get_index_subspace(ctx, part, Color(*itr));
    auto subspaceDom = runtime->get_index_space_domain(ctx, subspace);
    assert(subspaceDom.dense());
    auto projected = Domain(this->apply(subspaceDom.lo()), this->apply(subspaceDom.hi()));
    col[*itr] = projected;
  }
  return runtime->create_partition_by_domain(ctx, ispace, col, colorSpaceName, true /* perform_intersections */, LEGION_COMPUTE_KIND, color);
}

Legion::DomainPoint AffineProjection::apply(Legion::DomainPoint point) {
  assert(point.get_dim() == this->dim());
  DomainPoint res;
  res.dim = this->outDim();
  for (int i = 0; i < this->dim(); i++) {
    auto mapTo = this->operator[](i);
    if (mapTo == AffineProjection::BOT) { continue; }
    res[mapTo] = point[i];
  }
  return res;
}
