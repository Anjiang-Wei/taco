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

// A helper widget to treat LogicalRegions and PhysicalRegions the same.
struct RegionWrapper {
  Legion::PhysicalRegion physReg;
  Legion::LogicalRegion logReg;
  enum RegionKind {
    PHYSICAL,
    LOGICAL,
  } regionKind;

  RegionWrapper(Legion::LogicalRegion logReg) : logReg(logReg), regionKind(LOGICAL) {}
  RegionWrapper(Legion::PhysicalRegion physReg) : physReg(physReg), regionKind(PHYSICAL) {}
  Legion::IndexSpace get_index_space() {
    switch (this->regionKind) {
      case PHYSICAL:
        return this->physReg.get_logical_region().get_index_space();
      case LOGICAL:
        return this->logReg.get_index_space();
      default:
        assert(false);
    }
  }

  // We purposely don't make these `explicit` so that we don't have to generate code
  // that performs the desired casts. For now, the overload resolution has been enough.
  operator Legion::PhysicalRegion() {
    // If we aren't a physical region yet, then return an invalid PhysicalRegion
    // to be explicit that we are sending invalid data.
    if (this->regionKind == LOGICAL) {
      return Legion::PhysicalRegion();
    }
    return this->physReg;
  }
  operator Legion::LogicalRegion() {
    // If we're a PhysicalRegion, then we can easily return the corresponding LogicalRegion.
    if (this->regionKind == PHYSICAL) {
      return this->physReg.get_logical_region();
    }
    return this->logReg;
  }
};

Legion::IndexSpace get_index_space(Legion::PhysicalRegion r);
Legion::IndexSpace get_index_space(Legion::LogicalRegion r);
Legion::IndexSpace get_index_space(RegionWrapper r);
Legion::LogicalRegion get_logical_region(Legion::PhysicalRegion r);
Legion::LogicalRegion get_logical_region(Legion::LogicalRegion r);
Legion::LogicalRegion get_logical_region(RegionWrapper r);
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
Legion::LogicalPartition copyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::IndexPartition toCopy, Legion::LogicalRegion toPartition, Legion::Color color = LEGION_AUTO_GENERATE_ID);
Legion::LogicalPartition copyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalPartition toCopy, Legion::LogicalRegion toPartition, Legion::Color color = LEGION_AUTO_GENERATE_ID);

// densifyPartition creates a partition where the domain of each subregion in part is converted
// to the bounding box of the partition. This function is currently only implemented for
// 1-dimensional color spaces.
Legion::IndexPartition densifyPartition(Legion::Context ctx, Legion::Runtime* runtime, Legion::IndexSpace ispace, Legion::IndexPartition part, Legion::Color color = LEGION_AUTO_GENERATE_ID);

// Templated helper functions to potentially create accessors. These allow us to generate
// accessors when we don't have valid PhysicalRegions without running into problems.
template<typename T>
T createAccessor(Legion::PhysicalRegion& r, Legion::FieldID fid) {
  if (r.is_valid()) {
    return T(r, fid);
  }
  return T();
}
template<typename T>
T createAccessor(Legion::PhysicalRegion&& r, Legion::FieldID fid) {
  if (r.is_valid()) {
    return T(r, fid);
  }
  return T();
}

#endif // TACO_LEGION_INCLUDES_H
