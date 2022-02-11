#include "taco_legion_header.h"
#include "taco_mapper.h"
#include "shard.h"
#include "error.h"
#include "task_ids.h"
#include "legion/legion_utilities.h"
#include "pitches.h"
#include <mutex>
#include <legion_utils.h>

#ifdef REALM_USE_OPENMP
#include <omp.h>
#endif

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

FieldSpace createFieldSpaceWithSize(Context ctx, Runtime* runtime, FieldID id, size_t size) {
  auto fspace = runtime->create_field_space(ctx);
  {
    auto falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(size, id);
  }
  return fspace;
}

// getSubRegion returns a subregion of the input region with the desired start and end coordinate.
LogicalRegion getSubRegion(Context ctx, Runtime* runtime, LogicalRegion region, Domain bounds) {
  // TODO (rohany): Can I avoid creating this IndexSpace on each reallocation call?
  IndexSpaceT<1> colorSpace = runtime->create_index_space(ctx, Rect<1>(0, 0));
  DomainPointColoring coloring;
  // Construct the resulting domain to get the subregion of.
  Domain partBounds;
  // If the dimensions match, then there's nothing to do.
  if (bounds.dim == region.get_dim()) {
    partBounds = bounds;
  } else {
    // Otherwise, we'll take all of the dimensions from bounds, and then fill in
    // the remaining from the region's index space.
    taco_iassert(bounds.dim < region.get_dim());
    auto regionBounds = runtime->get_index_space_domain(ctx, region.get_index_space());
    // TODO (rohany): I've implemented this logic a few times, maybe it's time to pull it out
    //  into a helper method?
    DomainPoint lo, hi;
    lo.dim = region.get_dim();
    hi.dim = region.get_dim();
    for (int i = 0; i < bounds.dim; i++) {
      lo[i] = bounds.lo()[i];
      hi[i] = bounds.hi()[i];
    }
    for (int i = bounds.dim; i < region.get_dim(); i++) {
      lo[i] = regionBounds.lo()[i];
      hi[i] = regionBounds.hi()[i];
    }
    partBounds = Domain(lo, hi);
  }
  coloring[0] = partBounds;
  // TODO (rohany): Is there a way to destroy the old partition of this region's index space?
  auto ip = runtime->create_partition_by_domain(
    ctx,
    region.get_index_space(),
    coloring,
    colorSpace
  );
  // Get the subregion of the only point in the partition.
  return runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, region, ip), 0);
}

// mapRegion inline maps a region. It abstracts away a bit of cruft around creating RegionRequirements
// and making runtime calls.
Legion::PhysicalRegion mapRegion(Context ctx, Runtime* runtime, LogicalRegion region, LogicalRegion parent, FieldID fid, PrivilegeMode priv) {
  // TODO (rohany): Do we need to pass in a privilege here, or if we are doing allocations
  //  we can assume that the region is going to be written to?
  RegionRequirement req(region, priv, EXCLUSIVE, parent, Mapping::DefaultMapper::EXACT_REGION);
  req.add_field(fid);
  auto result = runtime->map_region(ctx, req);
  result.wait_until_valid();
  return result;
}

Legion::PhysicalRegion legionMalloc(Context ctx, Runtime* runtime, LogicalRegion region, size_t size, FieldID fid, PrivilegeMode priv) {
  auto subreg = getSubRegion(ctx, runtime, region, Rect<1>(0, size - 1));
  return mapRegion(ctx, runtime, subreg, region, fid, priv);
}

Legion::PhysicalRegion legionMalloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::LogicalRegion parent, Legion::FieldID fid, PrivilegeMode priv) {
  return mapRegion(ctx, runtime, region, parent, fid, priv);
}

Legion::PhysicalRegion legionMalloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::Domain domain, Legion::FieldID fid, Legion::PrivilegeMode priv) {
  taco_iassert(domain.dim == region.get_dim());
  auto subreg = getSubRegion(ctx, runtime, region, domain);
  return mapRegion(ctx, runtime, subreg, region, fid, priv);
}

Legion::PhysicalRegion legionRealloc(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion region, Legion::PhysicalRegion old, size_t newSize, Legion::FieldID fid, PrivilegeMode priv) {
  // Get the bounds on the old region.
  auto bounds = runtime->get_index_space_domain(ctx, old.get_logical_region().get_index_space());
  // Unmap the input existing region.
  runtime->unmap_region(ctx, old);
  // Just get between the old size and the new size to avoid having to copy the old data.
  auto subreg = getSubRegion(ctx, runtime, region, Rect<1>(bounds.hi()[0] + 1, newSize - 1));
  return mapRegion(ctx, runtime, subreg, region, fid, priv);
}

LogicalPartition copyPartition(Context ctx, Runtime* runtime, IndexPartition toCopy, LogicalRegion toPartition, Color color) {
  auto indexPart = copyPartition(ctx, runtime, toCopy, toPartition.get_index_space(), color);
  return runtime->get_logical_partition(ctx, toPartition, indexPart);
}

LogicalPartition copyPartition(Context ctx, Runtime* runtime, LogicalPartition toCopy, LogicalRegion toPartition, Color color) {
  return copyPartition(ctx, runtime, toCopy.get_index_partition(), toPartition, color);
}

Legion::IndexPartition copyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::IndexPartition toCopy, Legion::IndexSpace toPartition, Legion::Color color) {
  // Currently, copyPartition only supports the following cases:
  //  toCopy.index_space.dim() == toPartition.dim()
  //  toCopy.index_space.dim() == n && toPartition.dim() > n
  if (toCopy.get_dim() != toPartition.get_dim()) {
    taco_iassert(toPartition.get_dim() > toCopy.get_dim());
  }
  auto toPartitionBounds = runtime->get_index_space_domain(ctx, toPartition);

  auto copyDomain = [&](Domain d) {
    // If the dimensionality of the input and output index spaces are the same, then
    // we can copy the domain directly.
    if (toCopy.get_dim() == toPartition.get_dim()) {
      return d;
    }
    // In the case where dimensionalities are not equal, then we must extend the
    // input domain to the domain of the output region. Currently, we assume two things:
    // * The input domain is not sparse, as this requires doing a per-element operation.
    // * The output dimensionality is larger than the input dimensionality.
    taco_iassert(toPartition.get_dim() > toCopy.get_dim());
    taco_iassert(d.dense());
    // In the case where we must extend the dimensionality of the partition, this function
    // is specific to DISTAL's use cases. In particular, the first toCopy.get_dim() dimensions
    // of the domain are copied over, and the toPartition.get_dim() - toCopy.get_dim() dimensions
    // span their full extent. This code is slightly duplicated with AffineProjection.
    DomainPoint lo, hi;
    lo.dim = toPartition.get_dim();
    hi.dim = toPartition.get_dim();
    for (int i = 0; i < toCopy.get_dim(); i++) {
      lo[i] = d.lo()[i];
      hi[i] = d.hi()[i];
    }
    for (int i = toCopy.get_dim(); i < toPartition.get_dim(); i++) {
      lo[i] = toPartitionBounds.lo()[i];
      hi[i] = toPartitionBounds.hi()[i];
    }
    return Domain(lo, hi);
  };

  std::map<DomainPoint, Domain> domains;
  auto colorSpace = runtime->get_index_partition_color_space(ctx, toCopy);
  auto colorSpaceName = runtime->get_index_partition_color_space_name(ctx, toCopy);
  switch (colorSpace.get_dim()) {
#define BLOCK(DIM)                                                                     \
    case DIM:                                                                          \
      {                                                                                \
        for (PointInDomainIterator<DIM> itr(colorSpace); itr(); itr++) {               \
          auto subspace = runtime->get_index_subspace(ctx, toCopy, DomainPoint(*itr)); \
          auto domain = runtime->get_index_space_domain(ctx, subspace);                \
          domains[*itr] = copyDomain(domain);                                          \
        }                                                                              \
        break;                                                                         \
      }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }
  // TODO (rohany): Is there a way to get the kind (i.e. disjoint, aliased etc) of the existing partition?
  return runtime->create_partition_by_domain(ctx, toPartition, domains, colorSpaceName, true /* perform_intersections */, LEGION_COMPUTE_KIND, color);
}

Legion::IndexPartition copyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalPartition toCopy, Legion::IndexSpace toPartition, Legion::Color color) {
  return copyPartition(ctx, runtime, toCopy.get_index_partition(), toPartition, color);
}

IndexPartition densifyPartition(Context ctx, Runtime* runtime, IndexSpace ispace, IndexPartition part, Color color) {
  DomainPointColoring coloring;
  auto colorSpace = runtime->get_index_partition_color_space_name(ctx, part);
  auto colorSpaceDom = runtime->get_index_space_domain(ctx, colorSpace);
  taco_iassert(colorSpace.get_dim() == 1);
  for (PointInDomainIterator<1> itr(colorSpaceDom); itr(); itr++) {
    auto subreg = runtime->get_index_subspace(ctx, part, Color(*itr));
    auto subDom = runtime->get_index_space_domain(ctx, subreg);
    // TODO (rohany): Do I need to cast this into a DomainT and call tighten? Or will the domains
    //  computed by create_partition_by_preimage_range already be tightened?
    coloring[*itr] = {subDom.lo(), subDom.hi()};
  }
  return runtime->create_partition_by_domain(ctx, ispace, coloring, colorSpace, true /* perform_intersections */, LEGION_COMPUTE_KIND, color);
}

Legion::IndexPartition createSparseAliasingPartitions(Legion::Context ctx, Legion::Runtime* runtime, Legion::IndexSpace ispace, Legion::IndexPartition part) {
  DomainPointColoring coloring;
  auto colorSpace = runtime->get_index_partition_color_space_name(ctx, part);
  auto colorSpaceDom = runtime->get_index_space_domain(ctx, colorSpace);
  taco_iassert(colorSpaceDom.dim == 1);
  auto pieces = colorSpaceDom.get_volume();
  // Return an empty index partition if we only have a single piece.
  if (pieces == 1) {
    DomainPoint lo, hi;
    lo.dim = ispace.get_dim();
    hi.dim = ispace.get_dim();
    for (int i = 0; i < ispace.get_dim(); i++) {
      lo[i] = 1;
      hi[i] = 0;
    }
    coloring[0] = Domain(lo, hi);
    return runtime->create_index_partition(ctx, ispace, colorSpaceDom, coloring);
  }
  // Otherwise, compute the actual aliasing that we want.
  for (PointInDomainIterator<1> itr(colorSpaceDom); itr(); itr++) {
    size_t i = *itr;
    auto subreg = runtime->get_index_subspace(ctx, part, i);
    auto subregDomain = runtime->get_index_space_domain(ctx, subreg);
    Domain prevDomain, nextDomain;
    if (i > 0) {
      auto prev = runtime->get_index_subspace(ctx, part, i - 1);
      auto dom = runtime->get_index_space_domain(ctx, prev);
      prevDomain = dom.intersection(subregDomain);
    }
    if (i < pieces - 1) {
      auto next = runtime->get_index_subspace(ctx, part, i + 1);
      auto dom = runtime->get_index_space_domain(ctx, next);
      nextDomain = dom.intersection(subregDomain);
    }
    std::vector<IndexSpace> toUnion;
    if (prevDomain.exists()) {
      toUnion.push_back(runtime->create_index_space(ctx, prevDomain));
    }
    if (nextDomain.exists()) {
      toUnion.push_back(runtime->create_index_space(ctx, nextDomain));
    }
    auto result = runtime->union_index_spaces(ctx, toUnion);
    auto resultDomain = runtime->get_index_space_domain(ctx, result);
    coloring[i] = resultDomain;
  }
  return runtime->create_index_partition(ctx, ispace, colorSpaceDom, coloring);
}

// We have to include this separate definition out here to make the linker happy, as per
// https://stackoverflow.com/questions/4891067/weird-undefined-symbols-of-static-constants-inside-a-struct-class.
const int AffineProjection::BOT = -1;

AffineProjection::AffineProjection(std::vector<int> projs) : projs(projs) {}

int AffineProjection::dim() const {
  return int(this->projs.size());
}

int AffineProjection::operator[] (size_t i) const {
  taco_iassert(int(i) < this->dim());
  return this->projs[i];
}

Legion::IndexPartition AffineProjection::apply(Legion::Context ctx, Runtime *runtime, Legion::IndexPartition part,
                                               Legion::IndexSpace ispace, DomainPointColoring coloringOverride, Color color) {
  DomainPointColoring col;
  auto colorSpace = runtime->get_index_partition_color_space(ctx, part);
  auto colorSpaceName = runtime->get_index_partition_color_space_name(ctx, part);
  auto outputDomain = runtime->get_index_space_domain(ctx, ispace);
  taco_iassert(colorSpace.get_dim() == 1);
  for (PointInDomainIterator<1> itr(colorSpace); itr(); itr++) {
    auto subspace = runtime->get_index_subspace(ctx, part, Color(*itr));
    auto subspaceDom = runtime->get_index_space_domain(ctx, subspace);
    taco_iassert(subspaceDom.dense());
    auto lo = this->apply(subspaceDom.lo(), outputDomain.lo());
    auto hi = this->apply(subspaceDom.hi(), outputDomain.hi());

    // TODO (rohany): Comment this overriding business.
    auto it = coloringOverride.find(*itr);
    if (it != coloringOverride.end()) {
      for (auto idx : this->overrides) {
        lo[idx] = it->second.lo()[idx];
        hi[idx] = it->second.hi()[idx];
      }
    }

    auto projected = Domain(lo, hi);
    col[*itr] = projected;
  }
  return runtime->create_partition_by_domain(ctx, ispace, col, colorSpaceName, true /* perform_intersections */, LEGION_COMPUTE_KIND, color);
}

Legion::DomainPoint AffineProjection::apply(Legion::DomainPoint point, Legion::DomainPoint outputBounds) {
  taco_iassert(point.get_dim() == this->dim());
  DomainPoint res, setMask;
  res.dim = outputBounds.dim;
  setMask.dim = outputBounds.dim;

  // Initialize the output mask to all 0's.
  for (int i = 0; i < setMask.dim; i++) {
    setMask[i] = 0;
  }

  // Apply the projection.
  for (int i = 0; i < this->dim(); i++) {
    auto mapTo = this->operator[](i);
    if (mapTo == AffineProjection::BOT) { continue; }
    res[mapTo] = point[i];
    setMask[mapTo] = 1;
  }

  // For all dimensions that haven't been set, take the bounds.
  for (int i = 0; i < res.dim; i++) {
    if (setMask[i] == 0) {
      res[i] = outputBounds[i];
    }
  }

  return res;
}

const int RectCompressedPosPartitionDownwards::taskID = TID_RECT_COMPRESSED_POS_PARTITION_DOWNWARDS;

Legion::IndexPartition RectCompressedPosPartitionDownwards::apply(Legion::Context ctx, Legion::Runtime *runtime,
                                                                  Legion::IndexSpace ispace,
                                                                  Legion::LogicalPartition part,
                                                                  Legion::LogicalRegion parent, Legion::FieldID fid,
                                                                  Legion::Color color) {
  // If we are deriving a partition from a multi-dimensional region, then we
  // will just fall back to using the partition_by_image_range operation.
  auto colorSpace = runtime->get_index_partition_color_space_name(ctx, part.get_index_partition());
  auto colorSpaceDomain = runtime->get_index_space_domain(ctx, colorSpace);
  if (parent.get_index_space().get_dim() != 1) {
    return runtime->create_partition_by_image_range(ctx, ispace, part, parent, fid, colorSpace, LEGION_COMPUTE_KIND, color);
  }

  // If the partition contains any sparse index spaces, we can't apply the optimization either.
  bool containsSparse = false;
  switch (colorSpace.get_dim()) {
#define BLOCK(DIM) \
    case DIM: {    \
      for (PointInDomainIterator<DIM> itr(colorSpaceDomain); itr(); itr++) { \
        auto subreg = runtime->get_index_subspace(ctx, part.get_index_partition(), DomainPoint(*itr)); \
        if (!runtime->get_index_space_domain(ctx, subreg).dense()) {         \
          containsSparse = true;                                             \
          break;\
        }\
      }             \
      break;       \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }
  if (containsSparse) {
    return runtime->create_partition_by_image_range(ctx, ispace, part, parent, fid, colorSpace, LEGION_COMPUTE_KIND, color);
  }

  // Otherwise, we can be a bit smarter and calculate the partitions symbolically, rather than
  // doing operations per element.
  IndexLauncher launcher(RectCompressedPosPartitionDownwards::taskID, colorSpaceDomain, TaskArgument(&fid, sizeof(FieldID)), ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(part, 0, READ_ONLY, EXCLUSIVE, parent).add_field(fid));
  launcher.tag |= TACOMapper::UNTRACK_VALID_REGIONS;
  auto domains = runtime->execute_index_space(ctx, launcher);
  return runtime->create_partition_by_domain(ctx, ispace, domains, colorSpace, true /* perform_intersections */, LEGION_COMPUTE_KIND, color);
}

Domain RectCompressedPosPartitionDownwards::task(const Task *task, const std::vector<Legion::PhysicalRegion> &regions,
                                                 Legion::Context ctx, Runtime *runtime) {
  FieldID field = *(FieldID*)(task->args);
  Accessor acc(regions[0], field);
  auto dom = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
  taco_iassert(dom.dense());
  if (dom.empty()) {
    return Rect<1>::make_empty();
  }
  auto lo = acc[dom.lo()].lo;
  auto hi = acc[dom.hi()].hi;
  return Rect<1>{lo, hi};
}

void RectCompressedPosPartitionDownwards::registerTasks() {
  {
    TaskVariantRegistrar registrar(RectCompressedPosPartitionDownwards::taskID, "rectCompressedPosPartitionDownwards");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Domain, RectCompressedPosPartitionDownwards::task>(registrar, "rectCompressedPosPartitionDownwards");
  }
  {
    TaskVariantRegistrar registrar(RectCompressedPosPartitionDownwards::taskID, "rectCompressedPosPartitionDownwards");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Domain, RectCompressedPosPartitionDownwards::task>(registrar, "rectCompressedPosPartitionDownwards");
  }
#ifdef TACO_USE_CUDA
  {
    TaskVariantRegistrar registrar(RectCompressedPosPartitionDownwards::taskID, "rectCompressedPosPartitionDownwards");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<Domain, RectCompressedPosPartitionDownwards::gputask>(registrar, "rectCompressedPosPartitionDownwards");
  }
#endif
}

Legion::LogicalPartition RectCompressedCoordinatePartition::apply(Legion::Context ctx, Legion::Runtime *runtime,
                                                                  Legion::LogicalRegion region,
                                                                  Legion::LogicalRegion parent, Legion::FieldID fid,
                                                                  Legion::DomainPointColoring buckets,
                                                                  Legion::IndexSpace colorSpace, Legion::Color color) {
  taco_iassert(colorSpace.get_dim() == 1);
  taco_iassert(region.get_index_space().get_dim() == 1);
  auto colorSpaceDomain = runtime->get_index_space_domain(ctx, colorSpace);
  taco_iassert(colorSpaceDomain.lo()[0] == 0 && colorSpaceDomain.hi()[0] == coord_t(colorSpaceDomain.get_volume() - 1));

  // Create a temporary region to use as the target for a partition_by_field operation.
  auto fspace = runtime->create_field_space(ctx);
  {
    auto falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(sizeof(Point<1>), FID_POINT_1);
  }
  auto tempReg = runtime->create_logical_region(ctx, region.get_index_space(), fspace);

  // Create an equal partition of the input region and temporary region to perform the
  // bucketing operation on.
  auto ipart = runtime->create_equal_partition(ctx, region.get_index_space(), colorSpace);
  auto regLpart = runtime->get_logical_partition(ctx, region, ipart);
  auto tempLpart = runtime->get_logical_partition(ctx, tempReg, ipart);

  Serializer ser;
  ser.serialize(fid);
  ser.serialize(colorSpace);
  // Serialize the buckets into arguments for the tasks.
  for (PointInDomainIterator<1> itr(colorSpaceDomain); itr(); itr++) {
    ser.serialize(Rect<1>(buckets[*itr]));
  }

  IndexTaskLauncher launcher(RectCompressedCoordinatePartition::taskID, colorSpaceDomain,
                             TaskArgument(ser.get_buffer(), ser.get_used_bytes()), ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(regLpart, 0, READ_ONLY, EXCLUSIVE, parent).add_field(fid));
  launcher.add_region_requirement(RegionRequirement(tempLpart, 0, WRITE_ONLY, EXCLUSIVE, tempReg).add_field(FID_POINT_1));
  runtime->execute_index_space(ctx, launcher);

  // Perform the partition op.
  auto result = runtime->create_partition_by_field(ctx, tempReg, tempReg, FID_POINT_1, colorSpace, color);

  // Clean up after ourselves.
  runtime->destroy_logical_region(ctx, tempReg);
  runtime->destroy_field_space(ctx, fspace);

  return runtime->get_logical_partition(ctx, parent, result);
}

const int RectCompressedCoordinatePartition::taskID = TID_RECT_COMPRESSED_COORDINATE_PARTITION;
void RectCompressedCoordinatePartition::task(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx, Legion::Runtime* runtime) {
  FieldID fid;
  IndexSpace colorSpace;
  std::vector<Rect<1>> buckets;
  Deserializer derez(task->args, task->arglen);
  derez.deserialize(fid);
  derez.deserialize(colorSpace);
  for (PointInDomainIterator<1> itr(runtime->get_index_space_domain(ctx, colorSpace)); itr(); itr++) {
    Rect<1> bucket;
    derez.deserialize(bucket);
    buckets.push_back(bucket);
  }

  // TODO (rohany): This assumes that coordinates are int32_t's.
  Accessor<int32_t, READ_ONLY> input(regions[0], fid);
  Accessor<Point<1>, WRITE_ONLY> output(regions[1], FID_POINT_1);

  auto domain = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
  #pragma omp parallel for schedule(static)
  for (coord_t i = domain.lo()[0]; i <= domain.hi()[0]; i++) {
    bool found = false;
    for (size_t j = 0; j < buckets.size(); j++) {
      if (buckets[j].contains(input[i])) {
        output[i] = j;
        found = true;
        break;
      }
    }
    taco_iassert(found);
  }
}

void RectCompressedCoordinatePartition::registerTasks() {
  {
    TaskVariantRegistrar registrar(RectCompressedCoordinatePartition::taskID, "rectCompressedCoordinatePartition");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RectCompressedCoordinatePartition::task>(registrar, "rectCompressedCoordinatePartition");
  }
  {
    TaskVariantRegistrar registrar(RectCompressedCoordinatePartition::taskID, "rectCompressedCoordinatePartition");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RectCompressedCoordinatePartition::task>(registrar, "rectCompressedCoordinatePartition");
  }
}

SparseGatherProjection::SparseGatherProjection(int mapTo) : mapTo(mapTo) {}

Legion::IndexPartition SparseGatherProjection::apply(Legion::Context ctx, Legion::Runtime *runtime,
                                                     Legion::LogicalRegion reg, Legion::LogicalPartition part,
                                                     Legion::FieldID fieldID, Legion::IndexSpace toPartition,
                                                     Legion::Color color) {
  // Create a temporary region partitioned in the same way as the input.
  auto newFieldSpace = runtime->create_field_space(ctx);
  auto falloc = runtime->create_field_allocator(ctx, newFieldSpace);
  size_t fieldSize = 0;
  switch (toPartition.get_dim()) {
#define BLOCK(DIM) \
    case DIM: { \
      fieldSize = sizeof(Rect<DIM>); \
      break;       \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }
  falloc.allocate_field(fieldSize, fieldID);
  auto regPtrs = runtime->create_logical_region(ctx, reg.get_index_space(), newFieldSpace);
  auto indexPart = part.get_index_partition();
  auto regPtrsPart = runtime->get_logical_partition(ctx, regPtrs, indexPart);
  auto colorSpace = runtime->get_index_partition_color_space(ctx, part.get_index_partition());

  // Launch a task to project the coordinates into Rect's.
  auto args = taskArgs {
    .mapTo = this->mapTo,
    .target = toPartition,
    .fieldID = fieldID,
  };
  IndexLauncher launcher(SparseGatherProjection::taskID, colorSpace, TaskArgument(&args, sizeof(taskArgs)), ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(part, 0, READ_ONLY, EXCLUSIVE, reg).add_field(fieldID));
  launcher.add_region_requirement(RegionRequirement(regPtrsPart, 0, WRITE_ONLY, EXCLUSIVE, regPtrs).add_field(fieldID));
  runtime->execute_index_space(ctx, launcher);

  // Use the temporary region to perform the partitioning operation.
  auto result = runtime->create_partition_by_image_range(ctx, toPartition, regPtrsPart, regPtrs, fieldID,
                                                         runtime->get_index_partition_color_space_name(ctx, indexPart),
                                                         LEGION_COMPUTE_KIND, color);

  // Cleanup the temporary resources.
  runtime->destroy_field_space(ctx, newFieldSpace);
  runtime->destroy_logical_region(ctx, regPtrs);

  // Return the result.
  return result;
}

template<int DIM>
void SparseGatherProjection::taskBody(Legion::Context ctx, Legion::Runtime* runtime,
                                      taskArgs args, Legion::PhysicalRegion input, Legion::PhysicalRegion output) {

  // Here dim is the dimension of the target index space to partition.
  auto ispace = IndexSpaceT<DIM>(args.target);
  auto ispaceBounds = runtime->get_index_space_domain(ispace);
  taco_iassert(input.get_logical_region().get_dim() == 1);
  auto inputBounds = runtime->get_index_space_domain(IndexSpaceT<1>(input.get_logical_region().get_index_space()));
  typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorInput;
  typedef FieldAccessor<WRITE_ONLY,Rect<DIM>,1,coord_t,Realm::AffineAccessor<Rect<DIM>,1,coord_t>> AccessorOutput;
  AccessorInput ai(input, args.fieldID);
  AccessorOutput ao(output, args.fieldID);

  #pragma omp parallel for schedule(static)
  for (int i = inputBounds.bounds.lo; i <= inputBounds.bounds.hi; i++) {
    Rect<DIM> res;
    for (int j = 0; j < DIM; j++) {
      if (j == args.mapTo) {
        res.lo[j] = ai[i];
        res.hi[j] = ai[i];
      } else {
        res.lo[j] = ispaceBounds.bounds.lo[j];
        res.hi[j] = ispaceBounds.bounds.hi[j];
      }
    }
    ao[i] = res;
  }
}

void SparseGatherProjection::task(const Legion::Task *task, const std::vector<Legion::PhysicalRegion> &regions,
                                  Legion::Context ctx, Legion::Runtime *runtime) {
  auto args = *(taskArgs*)(task->args);
  switch (args.target.get_dim()) {
#define BLOCK(DIM) \
    case DIM: { \
      SparseGatherProjection::taskBody<DIM>(ctx, runtime, args, regions[0], regions[1]); \
      break;       \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }
}

// Separate declaration of taskID, similar to AffineProjection::BOT.
const int SparseGatherProjection::taskID = TID_SPARSE_GATHER_PARTITION;
void SparseGatherProjection::registerTasks() {
  {
    TaskVariantRegistrar registrar(SparseGatherProjection::taskID, "sparseGatherWrite");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SparseGatherProjection::task>(registrar, "sparseGatherWrite");
  }
  {
    TaskVariantRegistrar registrar(SparseGatherProjection::taskID, "sparseGatherWrite");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<SparseGatherProjection::task>(registrar, "sparseGatherWrite");
  }
}

void RectCompressedFinalizeYieldPositions::compute(Context ctx, Runtime *runtime, LogicalRegion region,
                                                   LogicalPartition part, FieldID fid) {
  auto launchDomain = runtime->get_index_partition_color_space(ctx, part.get_index_partition());
  auto launchSpace = runtime->get_index_partition_color_space_name(ctx, part.get_index_partition());

  // The input partition must be complete and disjoint.
  taco_iassert(runtime->is_index_partition_complete(ctx, part.get_index_partition()));
  taco_iassert(runtime->is_index_partition_disjoint(ctx, part.get_index_partition()));

  // TODO (rohany): For now, let's just worry about one-dimensional color spaces.
  taco_iassert(launchDomain.dim == 1);
  // Create a ghost partition.
  // TODO (rohany): I think that in the general case with this access pattern
  //  only one of the dimensions can be partitioned. This makes sense because
  //  the access order that we have is i - 1 in a linearized space, so it can
  //  only be partitioned in one dimension.
  DomainPointColoring coloring;
  switch (region.get_dim()) {
#define BLOCK(DIM) \
    case DIM: {    \
      DomainT<DIM> regionBounds = runtime->get_index_space_domain(ctx, region.get_index_space());             \
      for (PointInDomainIterator<1> itr(launchDomain); itr(); itr++) { \
        auto subreg = runtime->get_logical_subregion_by_color(ctx, part, Color(*itr)); \
        DomainT<DIM> bounds = runtime->get_index_space_domain(ctx, subreg.get_index_space());                 \
        taco_iassert(bounds.dense());           \
        for (int i = 1; i < DIM; i++) {                                                                       \
          taco_iassert(bounds.bounds.lo[i] == regionBounds.bounds.lo[i]);           \
          taco_iassert(bounds.bounds.hi[i] == regionBounds.bounds.hi[i]);           \
        }          \
        /* We might need to access the previous "i" dimension. */           \
        Point<DIM> newLo = bounds.bounds.lo;                                                                  \
        if (newLo[0] != 0) newLo[0]--;           \
        coloring[*itr] = {newLo, bounds.bounds.hi};           \
      }             \
      break;       \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }

  // We can't launch a task on two partitions of the same region and same field
  // to get a "update in one step" kind of behavior. Instead, we'll create a
  // temporary region that will hold the data before the finalization process.
  // We'll then use this temporary region as the old version of the data.
  auto tempReg = runtime->create_logical_region(ctx, region.get_index_space(), region.get_field_space());
  auto tempPart = runtime->get_logical_partition(ctx, tempReg, part.get_index_partition());
  {
    IndexCopyLauncher cl(launchDomain);
    cl.add_copy_requirements(
        RegionRequirement(part, 0, READ_ONLY, EXCLUSIVE, region).add_field(fid),
        RegionRequirement(tempPart, 0, WRITE_ONLY, EXCLUSIVE, tempReg).add_field(fid)
    );
    runtime->issue_copy_operation(ctx, cl);
  }

  auto ghostip = runtime->create_partition_by_domain(ctx, tempReg.get_index_space(), coloring, launchSpace,
                                                     false /* perform_intersections */,
                                                     LEGION_ALIASED_COMPLETE_KIND);
  auto ghostlp = runtime->get_logical_partition(ctx, tempReg, ghostip);

  // Add the regions arguments.
  {
    IndexLauncher launcher(RectCompressedFinalizeYieldPositions::taskID, launchDomain, TaskArgument(), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(part, 0, READ_WRITE, EXCLUSIVE, region).add_field(fid));
    launcher.add_region_requirement(RegionRequirement(ghostlp, 0, READ_ONLY, EXCLUSIVE, tempReg).add_field(fid));
    launcher.tag |= TACOMapper::UNTRACK_VALID_REGIONS;
    runtime->execute_index_space(ctx, launcher);
  }

  // Clean up after ourselves.
  runtime->destroy_logical_region(ctx, tempReg);
}

template<>
void RectCompressedFinalizeYieldPositions::body<1>(Context ctx, Runtime* runtime,
                                                   Rect<1> fullBounds, Rect<1> iterBounds,
                                                   Accessor<1, READ_WRITE> output, Accessor<1, READ_ONLY> ghost) {
  #pragma omp parallel for schedule(static)
  for (coord_t i = iterBounds.lo.x; i <= iterBounds.hi.x; i++) {
    if (i == 0) {
      output[0].lo = 0;
    } else {
      output[i].lo = ghost[i - 1].lo;
    }
  }
}

template<>
void RectCompressedFinalizeYieldPositions::body<2>(Context ctx, Runtime* runtime,
                                                   Rect<2> fullBounds, Rect<2> iterBounds,
                                                   Accessor<2, READ_WRITE> output, Accessor<2, READ_ONLY> ghost) {
  #pragma omp parallel for schedule(static) collapse(2)
  for (coord_t i = iterBounds.lo.x; i <= iterBounds.hi.x; i++) {
    for (coord_t j = iterBounds.lo.y; j <= iterBounds.hi.y; j++) {
      Point<2> point(i, j);
      if (point == Point<2>::ZEROES()) {
        output[point].lo = 0;
      } else {
        output[point].lo = ghost[getPreviousPoint(point, fullBounds)].lo;
      }
    }
  }
}

template<>
void RectCompressedFinalizeYieldPositions::body<3>(Context ctx, Runtime* runtime,
                                                   Rect<3> fullBounds, Rect<3> iterBounds,
                                                   Accessor<3, READ_WRITE> output, Accessor<3, READ_ONLY> ghost) {
  #pragma omp parallel for schedule(static) collapse(3)
  for (coord_t i = iterBounds.lo.x; i <= iterBounds.hi.x; i++) {
    for (coord_t j = iterBounds.lo.y; j <= iterBounds.hi.y; j++) {
      for (coord_t k = iterBounds.lo.z; k <= iterBounds.hi.z; k++) {
        Point<3> point(i, j, k);
        if (point == Point<3>::ZEROES()) {
          output[point].lo = 0;
        } else {
          output[point].lo = ghost[getPreviousPoint(point, fullBounds)].lo;
        }
      }
    }
  }
}

void RectCompressedFinalizeYieldPositions::task(const Legion::Task *task,
                                                const std::vector<Legion::PhysicalRegion> &regions, Legion::Context ctx,
                                                Legion::Runtime *runtime) {
  auto output = regions[0];
  auto outputlr = output.get_logical_region();
  auto ghost = regions[1];
  std::vector<FieldID> fields;
  output.get_fields(fields);
  taco_iassert(runtime->has_parent_logical_partition(ctx, outputlr));
  auto outputPart = runtime->get_parent_logical_partition(ctx, outputlr);
  auto outputParent = runtime->get_parent_logical_region(ctx, outputPart);
  taco_iassert(fields.size() == 1);
  switch (outputlr.get_dim()) {
#define BLOCK(DIM) \
    case DIM: {    \
      Rect<DIM> fullBounds = runtime->get_index_space_domain(ctx, outputParent.get_index_space()).bounds<DIM, coord_t>(); \
      Rect<DIM> iterBounds = runtime->get_index_space_domain(ctx, outputlr.get_index_space()).bounds<DIM, coord_t>();     \
      Accessor<DIM, READ_WRITE> outAcc(output, fields[0]); \
      Accessor<DIM, READ_ONLY> ghostAcc(ghost, fields[0]); \
      RectCompressedFinalizeYieldPositions::body<DIM>(ctx, runtime, fullBounds, iterBounds, outAcc, ghostAcc); \
      break;       \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }
}

const int RectCompressedFinalizeYieldPositions::taskID = TID_RECT_COMPRESSED_FINALIZE_YIELD_POSITIONS;
void RectCompressedFinalizeYieldPositions::registerTasks() {
  {
    TaskVariantRegistrar registrar(RectCompressedFinalizeYieldPositions::taskID, "rectCompressedFinalizeYieldPositions");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RectCompressedFinalizeYieldPositions::task>(registrar, "rectCompressedFinalizeYieldPositions");
  }
  {
    TaskVariantRegistrar registrar(RectCompressedFinalizeYieldPositions::taskID, "rectCompressedFinalizeYieldPositions");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RectCompressedFinalizeYieldPositions::task>(registrar, "rectCompressedFinalizeYieldPositions");
  }
#ifdef TACO_USE_CUDA
  {
    TaskVariantRegistrar registrar(RectCompressedFinalizeYieldPositions::taskID, "rectCompressedFinalizeYieldPositions");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RectCompressedFinalizeYieldPositions::gputask>(registrar, "rectCompressedFinalizeYieldPositions");
  }
#endif
}

RectCompressedGetSeqInsertEdges::ResultValuePack
RectCompressedGetSeqInsertEdges::compute(Legion::Context ctx, Legion::Runtime *runtime,
                                         Legion::IndexSpace colorSpace,
                                         Legion::LogicalRegion pos, Legion::FieldID posFid,
                                         Legion::LogicalRegion nnz, Legion::FieldID nnzFid) {
  // We'll create a partition of the input regions using create_equal_partition, as this
  // will only partition the first dimension of each region. We take in the color space to use
  // to decide how many pieces to create. However, we only can do this for a 1-dimensional
  // color space.
  taco_iassert(colorSpace.get_dim() == 1);
  taco_iassert(pos.get_dim() == nnz.get_dim());
  auto colorSpaceDomain = runtime->get_index_space_domain(ctx, colorSpace);
  taco_iassert(colorSpaceDomain.dense());
  auto posDomain = runtime->get_index_space_domain(ctx, pos.get_index_space());
  auto nnzDomain = runtime->get_index_space_domain(ctx, nnz.get_index_space());
  taco_iassert(posDomain == nnzDomain);
  auto posipart = runtime->create_equal_partition(ctx, pos.get_index_space(), colorSpace);
  auto nnzipart = runtime->create_equal_partition(ctx, nnz.get_index_space(), colorSpace);
  auto pospart = runtime->get_logical_partition(ctx, pos, posipart);
  auto nnzpart = runtime->get_logical_partition(ctx, nnz, nnzipart);

  // However, just double check that the partitions are indeed as we expect.
#ifndef NDEBUG
  for (PointInDomainIterator<1> itr(colorSpaceDomain); itr(); itr++) {
    auto posSubSpace = runtime->get_index_subspace(ctx, posipart, Color(*itr));
    auto nnzSubSpace = runtime->get_index_subspace(ctx, nnzipart, Color(*itr));
    auto posSubSpaceDomain = runtime->get_index_space_domain(ctx, posSubSpace);
    auto nnzSubSpaceDomain = runtime->get_index_space_domain(ctx, nnzSubSpace);
    for (int i = 1; i < pos.get_dim(); i++) {
      taco_iassert(posSubSpaceDomain.lo()[i] == 0);
      taco_iassert(nnzSubSpaceDomain.lo()[i] == 0);
      taco_iassert(posSubSpaceDomain.hi()[i] == posDomain.hi()[i]);
      taco_iassert(nnzSubSpaceDomain.hi()[i] == nnzDomain.hi()[i]);
    }
  }
#endif

  Serializer localScanSer;
  localScanSer.serialize(posFid);
  localScanSer.serialize(nnzFid);
  IndexLauncher localScanLauncher(RectCompressedGetSeqInsertEdges::scanTaskID, colorSpace,
                                  TaskArgument(localScanSer.get_buffer(), localScanSer.get_used_bytes()),
                                  ArgumentMap());
  localScanLauncher.add_region_requirement(RegionRequirement(pospart, 0, WRITE_ONLY, EXCLUSIVE, pos).add_field(posFid));
  localScanLauncher.add_region_requirement(RegionRequirement(nnzpart, 0, READ_ONLY, EXCLUSIVE, nnz).add_field(nnzFid));
  localScanLauncher.tag |= TACOMapper::UNTRACK_VALID_REGIONS;
  auto localScanResults = runtime->execute_index_space(ctx, localScanLauncher);
  // TODO (rohany): See if there is a more efficient way to not have to
  //  wait on all of these. I think that this is unavoidable though.
  localScanResults.wait_all_results();

  // Perform a local exclusive scan on the per-processor results to construct
  // an ArgumentMap for the finalization launch.
  int64_t scanVal = 0;
  ArgumentMap scanMap;
  std::vector<int64_t> scanData(colorSpaceDomain.get_volume());
  for (PointInDomainIterator<1> itr(colorSpaceDomain); itr(); itr++) {
    scanData[int(*itr)] = scanVal;
    scanMap.set_point(*itr, UntypedBuffer(&scanData[int(*itr)], sizeof(int64_t)));
    scanVal += localScanResults.get_result<int64_t>(*itr);
  }

  // Now, we just need to apply the partial results to each subregion of pos.
  Serializer applyPartialResultsSer;
  applyPartialResultsSer.serialize(posFid);
  IndexLauncher applyPartialResultsLauncher(RectCompressedGetSeqInsertEdges::applyPartialResultsTaskID, colorSpace,
                                            TaskArgument(applyPartialResultsSer.get_buffer(),
                                                         applyPartialResultsSer.get_used_bytes()), scanMap);
  applyPartialResultsLauncher.add_region_requirement(
      RegionRequirement(pospart, 0, READ_WRITE, EXCLUSIVE, pos).add_field(posFid));
  runtime->execute_index_space(ctx, applyPartialResultsLauncher);

  // At this point, the final result of the scan is in scanVal.
  ResultValuePack result;
  result.scanResult = scanVal;
  result.partition = pospart;
  return result;
}

template<typename T, int DIM, typename ACC>
T inclusiveScanPlus(
    DeferredBuffer<T, DIM> output,
    ACC input,
    const Rect<DIM> &bounds,
    const Pitches<DIM - 1> pitches,
    size_t volume
) {
#ifdef REALM_USE_OPENMP
  // The strategy here will be to create an array with an entry for each
  // OpenMP thread. Each thread will perform a local scan, and then we'll
  // aggregate the results back, just like in the distributed version.
  const auto numThreads = omp_get_max_threads();
  T intermediateResults[numThreads];
  // Thread-local scan.
  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    T val = 0;
    #pragma omp for schedule(static)
    for (size_t i = 0; i < volume; i++) {
      auto point = pitches.unflatten(i, bounds.lo);
      val += input[point];
      output[point] = val;
    }
    intermediateResults[tid] = val;
  }
  // Now do an exclusive scan over the intermediate results.
  T scanRes[numThreads];
  T scanVal = 0;
  for (int i = 0; i < numThreads; i++) {
    scanRes[i] = scanVal;
    scanVal += intermediateResults[i];
  }
  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    #pragma omp for schedule(static)
    for (size_t i = 0; i < volume; i++) {
      auto point = pitches.unflatten(i, bounds.lo);
      output[point] += scanRes[tid];
    }
  }
  return scanVal;
#else
  // If we don't have OpenMP, then just do a sequential scan.
  T val = 0;
  for (size_t i = 0; i < volume; i++) {
    auto point = pitches.unflatten(i, bounds.lo);
    val += input[point];
    output[point] = val;
  }
  return val;
#endif
}

template<int DIM>
int64_t RectCompressedGetSeqInsertEdges::scanBody(Context ctx, Runtime *runtime, Rect<DIM> iterBounds,
                                                  Accessor<Rect<1>, DIM, WRITE_ONLY> output,
                                                  Accessor<int64_t, DIM, READ_ONLY> input,
                                                  Memory::Kind tmpMemKind) {
  int64_t initVal = 0;
  DeferredBuffer<int64_t, DIM> scanBuf(iterBounds, tmpMemKind, &initVal);

  Pitches<DIM - 1> pitches;
  auto volume = pitches.flatten(iterBounds);

  // First, compute the scan over input into a temporary buffer.
  int64_t scanVal = inclusiveScanPlus<int64_t>(scanBuf, input, iterBounds, pitches, volume);

  // Next, use the result of the scan to compute the rectangle
  // bounds for the output pos array.
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < volume; i++) {
    auto point = pitches.unflatten(i, iterBounds.lo);
    auto lo = (i == 0) ? 0 : scanBuf[pitches.unflatten(i - 1, iterBounds.lo)];
    auto hi = scanBuf[point] - 1;
    output[point] = {lo, hi};
  }

  return scanVal;
}

int64_t RectCompressedGetSeqInsertEdges::scanTask(const Legion::Task *task,
                                                  const std::vector<Legion::PhysicalRegion> &regions,
                                                  Legion::Context ctx,
                                                  Legion::Runtime *runtime) {

  // Unpack arguments for the task.
  FieldID outputField, inputField;
  std::tie(outputField, inputField) = RectCompressedGetSeqInsertEdges::unpackScanTaskArgs(task);

  // Figure out what kind of memory body should allocate its temporary within.
  Memory::Kind tmpMemKind;
  switch (task->current_proc.kind()) {
    case Realm::Processor::LOC_PROC:
      tmpMemKind = Realm::Memory::SYSTEM_MEM;
      break;
    case Realm::Processor::OMP_PROC:
      tmpMemKind = Realm::Memory::SOCKET_MEM;
      break;
    default:
      taco_iassert(false);
  }

  auto output = regions[0];
  auto input = regions[1];
  auto outputlr = output.get_logical_region();
  switch (outputlr.get_dim()) {
#define BLOCK(DIM) \
    case DIM: {    \
      Rect<DIM> iterBounds = runtime->get_index_space_domain(ctx, outputlr.get_index_space()).bounds<DIM, coord_t>();     \
      Accessor<Rect<1>, DIM, WRITE_ONLY> outAcc(output, outputField); \
      Accessor<int64_t, DIM, READ_ONLY> inAcc(input, inputField); \
      return RectCompressedGetSeqInsertEdges::scanBody<DIM>(ctx, runtime, iterBounds, outAcc, inAcc, tmpMemKind); \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
      return 0; // Keep the compiler happy.
  }
}

std::pair<Legion::FieldID, Legion::FieldID> RectCompressedGetSeqInsertEdges::unpackScanTaskArgs(
    const Legion::Task *task) {
  FieldID outputField, inputField;
  Deserializer derez(task->args, task->arglen);
  derez.deserialize(outputField);
  derez.deserialize(inputField);
  return {outputField, inputField};
}

template<int DIM>
void RectCompressedGetSeqInsertEdges::applyPartialResultsBody(Legion::Context ctx, Legion::Runtime *runtime,
                                                              Legion::Rect<DIM> iterBounds,
                                                              Accessor<Legion::Rect<1>, DIM, READ_WRITE> output,
                                                              int64_t value) {
  Pitches<DIM - 1> pitches;
  auto volume = pitches.flatten(iterBounds);
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < volume; i++) {
    output[pitches.unflatten(i, iterBounds.lo)] += Point<1>(value);
  }
}

void RectCompressedGetSeqInsertEdges::applyPartialResultsTask(const Legion::Task *task,
                                                              const std::vector<Legion::PhysicalRegion> &regions,
                                                              Legion::Context ctx, Legion::Runtime *runtime) {
  FieldID outputField;
  int64_t value;
  std::tie(outputField, value) = RectCompressedGetSeqInsertEdges::unpackApplyPartialResultsTaskArgs(task);

  auto output = regions[0];
  auto outputlr = output.get_logical_region();
  switch (outputlr.get_dim()) {
#define BLOCK(DIM) \
    case DIM: {    \
      Rect<DIM> iterBounds = runtime->get_index_space_domain(ctx, outputlr.get_index_space()).bounds<DIM, coord_t>();     \
      Accessor<Rect<1>, DIM, READ_WRITE> outAcc(output, outputField); \
      RectCompressedGetSeqInsertEdges::applyPartialResultsBody<DIM>(ctx, runtime, iterBounds, outAcc, value);                \
      break;             \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }
}

std::pair<Legion::FieldID, int64_t> RectCompressedGetSeqInsertEdges::unpackApplyPartialResultsTaskArgs(
    const Legion::Task *task) {
  FieldID outputField;
  {
    Deserializer derez(task->args, task->arglen);
    derez.deserialize(outputField);
  }
  int64_t value;
  {
    Deserializer derez(task->local_args, task->local_arglen);
    derez.deserialize(value);
  }
  return {outputField, value};
}

const int RectCompressedGetSeqInsertEdges::scanTaskID = TID_RECT_COMPRESSED_GET_SEQ_INSERT_EDGES_LOCAL_SCAN;
const int RectCompressedGetSeqInsertEdges::applyPartialResultsTaskID = TID_RECT_COMPRESSED_GET_SEQ_INSERT_EDGES_APPLY_PARTIAL_RESULTS;
void RectCompressedGetSeqInsertEdges::registerTasks() {
  {
    TaskVariantRegistrar registrar(RectCompressedGetSeqInsertEdges::scanTaskID, "rectCompressedGetSeqInsertEdgesLocalScan");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int64_t, RectCompressedGetSeqInsertEdges::scanTask>(registrar, "rectCompressedGetSeqInsertEdgesLocalScan");
  }
  {
    TaskVariantRegistrar registrar(RectCompressedGetSeqInsertEdges::scanTaskID, "rectCompressedGetSeqInsertEdgesLocalScan");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int64_t, RectCompressedGetSeqInsertEdges::scanTask>(registrar, "rectCompressedGetSeqInsertEdgesLocalScan");
  }
#ifdef TACO_USE_CUDA
  {
    TaskVariantRegistrar registrar(RectCompressedGetSeqInsertEdges::scanTaskID, "rectCompressedGetSeqInsertEdgesLocalScan");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int64_t, RectCompressedGetSeqInsertEdges::scanTaskGPU>(registrar, "rectCompressedGetSeqInsertEdgesLocalScan");
  }
#endif
  {
    TaskVariantRegistrar registrar(RectCompressedGetSeqInsertEdges::applyPartialResultsTaskID, "rectCompressedGetSeqInsertEdgesApplyPartialResults");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RectCompressedGetSeqInsertEdges::applyPartialResultsTask>(registrar, "rectCompressedGetSeqInsertEdgesApplyPartialResults");
  }
  {
    TaskVariantRegistrar registrar(RectCompressedGetSeqInsertEdges::applyPartialResultsTaskID, "rectCompressedGetSeqInsertEdgesApplyPartialResults");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RectCompressedGetSeqInsertEdges::applyPartialResultsTask>(registrar, "rectCompressedGetSeqInsertEdgesApplyPartialResults");
  }
#ifdef TACO_USE_CUDA
  {
    TaskVariantRegistrar registrar(RectCompressedGetSeqInsertEdges::applyPartialResultsTaskID, "rectCompressedGetSeqInsertEdgesApplyPartialResults");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<RectCompressedGetSeqInsertEdges::applyPartialResultsTaskGPU>(registrar, "rectCompressedGetSeqInsertEdgesApplyPartialResults");
  }
#endif
}

void registerTacoRuntimeLibTasks() {
  SparseGatherProjection::registerTasks();
  RectCompressedCoordinatePartition::registerTasks();
  RectCompressedFinalizeYieldPositions::registerTasks();
  RectCompressedGetSeqInsertEdges::registerTasks();
  RectCompressedPosPartitionDownwards::registerTasks();
}
