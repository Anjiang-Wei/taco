#include "taco_legion_header.h"
#include "shard.h"
#include "error.h"
#include "task_ids.h"
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
                                               Legion::IndexSpace ispace, Color color) {
  DomainPointColoring col;
  auto colorSpace = runtime->get_index_partition_color_space(ctx, part);
  auto colorSpaceName = runtime->get_index_partition_color_space_name(ctx, part);
  auto outputDomain = runtime->get_index_space_domain(ctx, ispace);
  taco_iassert(colorSpace.get_dim() == 1);
  for (PointInDomainIterator<1> itr(colorSpace); itr(); itr++) {
    auto subspace = runtime->get_index_subspace(ctx, part, Color(*itr));
    auto subspaceDom = runtime->get_index_space_domain(ctx, subspace);
    taco_iassert(subspaceDom.dense());
    auto projected = Domain(this->apply(subspaceDom.lo(), outputDomain.lo()), this->apply(subspaceDom.hi(), outputDomain.hi()));
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
    Runtime::preregister_task_variant<SparseGatherProjection::task>(registrar, "sparseGatherWrite");
  }
  {
    TaskVariantRegistrar registrar(SparseGatherProjection::taskID, "sparseGatherWrite");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    Runtime::preregister_task_variant<SparseGatherProjection::task>(registrar, "sparseGatherWrite");
  }
}

void registerTacoRuntimeLibTasks() {
  SparseGatherProjection::registerTasks();
}
