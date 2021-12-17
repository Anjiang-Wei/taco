#ifndef TACO_LEGION_TENSOR_H
#define TACO_LEGION_TENSOR_H

#include "legion.h"
#include "error.h"

// LegionTensorLevelFormat is an enum used to perform introspection on LegionTensor
// objects to understand the construction of the components within it.
enum LegionTensorLevelFormat {
  Dense,
  Sparse,
  Singleton,
};
typedef std::vector<LegionTensorLevelFormat> LegionTensorFormat;

// LegionTensor is a representation of a taco_tensor_t for the Legion backend.
// IMPORTANT: There must be no duplicate IndexSpace's within a LegionTensor. Specifically,
//  no IndexSpace may appear twice within a LegionTensor (as a Region's IndexSpace or within
//  a denseLevelRun. This is so that the partitioning color marking operators in generated
//  code do not attempt to create multiple partitions of the same index space with the same
//  color value. Future work in improving code generation may lift this restriction.
struct LegionTensor {
  // Construct an invalid initial LegionTensor.
  // TODO (rohany): Do I want to have a static invalid initial value?
  LegionTensor() {};
  // Construct a blank LegionTensor in a state that other methods can fill
  // in all of the necessary data structures within it.
  LegionTensor(LegionTensorFormat format, std::vector<int32_t> dims);
  // Construct for a LegionTensor that explicitly sets all necessary fields.
  LegionTensor(LegionTensorFormat format, int32_t order, std::vector<int32_t> dims,
               std::vector<std::vector<Legion::LogicalRegion>> indices,
               std::vector<std::vector<Legion::LogicalRegion>> indicesParents, Legion::LogicalRegion vals,
               Legion::LogicalRegion valsParent, std::vector<Legion::IndexSpace> denseLevelRuns);

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

  // Maintain at what the format of this tensor is.
  LegionTensorFormat format;

  // toString returns a human-readable string containing the data structure layout of
  // a LegionTensor.
  std::string toString(Legion::Context ctx, Legion::Runtime* runtime);

  // indicesEqPartitions is a cache of LogicalPartition objects associated with
  // indices regions in a LegionTensor. This is _not_ used by generated code, but
  // is instead used by runtime and utility codes when filling and copying
  // to and from regions in a distributed manner. It is to avoid repeatedly creating
  // many equal partitions and confusing the runtime.
  std::vector<std::vector<Legion::LogicalPartition>> indicesEqPartitions;
  // valsEqPartition is the same as indicesEqPartitions but for the vals region.
  Legion::LogicalPartition valsEqPartition;

  // TODO (rohany): Consider adding a validation method on a LegionTensor.
};

// LegionTensorPartition is a representation of a partition of a LegionTensor that is
// specific to the execution of a kernel. A LegionTensorPartition should be associated
// with a LegionTensor for full access to metadata about the tensor. This means that
// a LegionTensorPartition is not a standalone object.
struct LegionTensorPartition {
  // Partitions of the indices of a tensor.
  std::vector<std::vector<Legion::LogicalPartition>> indicesPartitions;

  // Partitions of the vals array of a tensor.
  Legion::LogicalPartition valsPartition;

  // Partitions of the dense format runs in the tensor. Note that these partitions
  // are IndexPartitions instead of LogicalPartitions as the dense format runs are
  // not associated with a particular region.
  std::vector<Legion::IndexPartition> denseLevelRunPartitions;
};

// ExternalHDF5LegionTensor represents metadata corresponding to a LegionTensor
// loaded from an HDF5 file.
struct ExternalHDF5LegionTensor {
  // Add an external allocation to this tensor.
  void addExternalAllocation(Legion::PhysicalRegion);
  // Free all resources attached to this tensor.
  void destroy(Legion::Context, Legion::Runtime*);

  // attachedRegions is a list of regions containing all external allocations of
  // a LegionTensor.
  std::vector<Legion::PhysicalRegion> attachedRegions;
};

// Utility method to create a dense tensor with the given dimensions.
template<int DIM, typename T>
LegionTensor createDenseTensor(Legion::Context ctx, Legion::Runtime* runtime, std::vector<int32_t> dims, Legion::FieldID valsField) {
  taco_iassert(dims.size() == DIM);
  Legion::Point<DIM> lo, hi;
  for (int i = 0; i < DIM; i++) {
    lo[i] = 0;
    hi[i] = dims[i] - 1;
  }
  auto ispace = runtime->create_index_space(ctx, Legion::Domain(lo, hi));
  // Importantly, we make a copy of the ispace here so that the DenseFormatRun
  // is not an index space of any regions in the tensor. The reasoning for this
  // is detailed in the LegionTensor struct.
  auto ispaceCopy = runtime->create_index_space(ctx, Legion::Domain(lo, hi));
  auto fspace = runtime->create_field_space(ctx);
  Legion::FieldAllocator fa = runtime->create_field_allocator(ctx, fspace);
  fa.allocate_field(sizeof(T), valsField);
  auto reg = runtime->create_logical_region(ctx, ispace, fspace);
  runtime->fill_field(ctx, reg, reg, valsField, T(0));

  LegionTensor result(LegionTensorFormat(DIM, Dense), dims);
  result.vals = reg;
  result.valsParent = reg;
  result.denseLevelRuns = {ispaceCopy};
  return result;
}

// createSparseTensorForPack creates an empty sparse tensor that can be used as the target of
// a pack operation from a COO matrix. This tensor is initialized to have regions of "infinite"
// size that are mapped into variable size chunks depending on what the actual sizes of the
// compressed regions are.
// TODO (rohany): This code assumes that modes == levels, which isn't a correct assumption.
//  It will have to be updated as I write more benchmark codes.
// TODO (rohany): I can't hardcode the fields here because of an import cycle with taco_legion_header.h.
//  decoupling those header files will allow for hardcoding the fields here.
template<typename T>
LegionTensor createSparseTensorForPack(Legion::Context ctx, Legion::Runtime* runtime, LegionTensorFormat format, std::vector<int32_t> dims,
                                       Legion::FieldID posField, Legion::FieldID crdField, Legion::FieldID valsField) {
  taco_iassert(dims.size() == format.size());
  LegionTensor result(format, dims);

  // Field spaces for the values, pos and crd arrays.
  auto valFspace = runtime->create_field_space(ctx);
  auto posFspace = runtime->create_field_space(ctx);
  auto crdFspace = runtime->create_field_space(ctx);
  {
    Legion::FieldAllocator fa = runtime->create_field_allocator(ctx, valFspace);
    fa.allocate_field(sizeof(T), valsField);
  }
  {
    Legion::FieldAllocator fa = runtime->create_field_allocator(ctx, posFspace);
    fa.allocate_field(sizeof(Legion::Rect<1>), posField);
  }
  {
    Legion::FieldAllocator fa = runtime->create_field_allocator(ctx, crdFspace);
    // TODO (rohany): Should the crd array have int32_t's?
    fa.allocate_field(sizeof(int32_t), crdField);
  }

  // An "infinitely" sized region will just be a really big one. I'm not sure how
  // big we actually need this to be, but 1 << 30 wasn't enough. However, 1 << 63
  // caused some other problems (bounds check failure) that I don't understand.
  auto infty = Legion::coord_t(1) << 48;

  bool inDenseRun = false;
  std::vector<int> levelsInDenseRun;
  auto getIspaceDomainPoints = [&](Legion::DomainPoint& lo, Legion::DomainPoint& hi) {
    if (!inDenseRun) {
      // If we weren't in a dense run, then the next region is one-dimensional
      // of unknown size.
      lo.dim = 1;
      hi.dim = 1;
      lo[0] = 0;
      hi[0] = infty - 1;
    } else {
      // Otherwise, we were in a dense run. If the first level is contained in the dense run,
      // then the dimensionality of the array is |levelsInDenseRun|. Otherwise, it's
      // 1 + |levelsInDenseRun|, where the first dimension has size infty.
      if (std::find(levelsInDenseRun.begin(), levelsInDenseRun.end(), 0) != levelsInDenseRun.end()) {
        auto dim = levelsInDenseRun.size();
        lo.dim = dim;
        hi.dim = dim;
        for (size_t i = 0; i < dim; i++) {
          lo[i] = 0;
          hi[i] = dims[levelsInDenseRun[i]] - 1;
        }
      } else {
        auto dim = levelsInDenseRun.size() + 1;
        lo.dim = dim;
        hi.dim = dim;
        lo[0] = 0;
        hi[0] = infty;
        for (size_t i = 1; i < dim; i++) {
          lo[i] = 0;
          hi[i] = dims[levelsInDenseRun[i - 1]] - 1;
        }
      }
    }
  };

  for (size_t level = 0; level < format.size(); level++) {
    auto levelFormat = format[level];
    switch (levelFormat) {
      case Dense: {
        // If we aren't in a dense run already, then start a new run.
        if (!inDenseRun) {
          inDenseRun = true;
          levelsInDenseRun = {};
        }
        levelsInDenseRun.push_back(level);
        break;
      }
      case Sparse: {
        // Create the regions necessary for the sparse level.
        auto lo = Legion::DomainPoint();
        auto hi = Legion::DomainPoint();
        getIspaceDomainPoints(lo, hi);
        // Create the index space using the information about the dense runs.
        auto posIspace = runtime->create_index_space(ctx, Legion::Domain(lo, hi));
        auto posReg = runtime->create_logical_region(ctx, posIspace, posFspace);
        auto crdIspace = runtime->create_index_space(ctx, Legion::Rect<1>(0, infty));
        auto crdReg = runtime->create_logical_region(ctx, crdIspace, crdFspace);
        result.indices[level] = {posReg, crdReg};
        result.indicesParents[level] = {posReg, crdReg};

        // Fill the regions as well.
        runtime->fill_field(ctx, posReg, posReg, posField, Legion::Rect<1>(0, 0));
        // TODO (rohany): Should crd hold int64_t's?
        runtime->fill_field(ctx, crdReg, crdReg, crdField, int32_t(0));

        // If we just ended a dense run, add the index space corresponding to the dense run to
        // the tensor as well.
        if (inDenseRun) {
          result.denseLevelRuns.push_back(posIspace);
        }

        // Finally mark the dense run as ended.
        inDenseRun = false;
        break;
      }
      default:
        taco_iassert(false);
    }
  }

  // Perform the same analysis for the values of the tensor.
  auto lo = Legion::DomainPoint();
  auto hi = Legion::DomainPoint();
  getIspaceDomainPoints(lo, hi);
  auto valsIspace = runtime->create_index_space(ctx, Legion::Domain(lo, hi));
  auto vals = runtime->create_logical_region(ctx, valsIspace, valFspace);
  result.vals = vals;
  result.valsParent = vals;
  runtime->fill_field(ctx, vals, vals, valsField, T(0));

  // Make sure the last dense run is added to the tensor.
  if (inDenseRun) {
    result.denseLevelRuns.push_back(valsIspace);
  }

  return result;
}

// copyNonZeroStructure copies the non-zero structure of the src tensor into a new
// tensor, but does not copy the values. This method is intended to be used in the case
// where the result tensor of a computation has a sparse output with non-zero structure
// identical to an input tensor's non-zero structure.
template<typename T>
LegionTensor copyNonZeroStructure(Legion::Context ctx, Legion::Runtime* runtime, LegionTensorFormat format, LegionTensor src) {
  using namespace Legion;
  // Double check that the result format is a prefix of the source format.
  taco_uassert(format.size() <= src.format.size());
  std::vector<int32_t> dims;
  for (size_t i = 0; i < format.size(); i++) {
    dims.push_back(src.dims[i]);
    taco_uassert(format[i] == src.format[i]);
  }

  auto createLogicalPart = [&](LogicalRegion part, IndexSpace domain) {
    auto ipart = runtime->create_equal_partition(ctx, part.get_index_space(), domain);
    return runtime->get_logical_partition(ctx, part, ipart);
  };

  auto getField = [&](LogicalRegion reg) {
    std::vector<FieldID> fields;
    runtime->get_field_space_fields(ctx, reg.get_field_space(), fields);
    taco_uassert(fields.size() == 1);
    return fields[0];
  };

  // TODO (rohany): We'll assume for now that once we hit a sparse level,
  //  we can't run into any more dense levels. This is just an implementation
  //  limitation because I'm too lazy to implement some of the logic needed
  //  for sparse-dense style formats.

  bool seenSparse = false;
  Domain currentValsDomain;
  LegionTensor result(format, dims);
  // TODO (rohany): This is probably not correct, but I'll run with it for now.
  result.denseLevelRuns = src.denseLevelRuns;
  for (size_t level = 0; level < format.size(); level++) {
    switch (format[level]) {
      case Dense: {
        taco_uassert(!seenSparse) << "currently not supporting sparse-dense formats";
        break; // Nothing to do here.
      }
      case Sparse: {
        seenSparse = true;
        // Copy the pos and crd arrays over from the source tensor.
        auto srcPosReg = src.indices[level][0];
        auto srcPosParent = src.indicesParents[level][0];
        auto srcCrdReg = src.indices[level][1];
        auto srcCrdParent = src.indicesParents[level][1];
        auto posField = getField(srcPosReg);
        auto crdField = getField(srcCrdReg);
        auto posReg = runtime->create_logical_region(ctx, srcPosReg.get_index_space(), srcPosReg.get_field_space());
        auto crdReg = runtime->create_logical_region(ctx, srcCrdReg.get_index_space(), srcCrdReg.get_field_space());

        // Add the regions to the result tensor.
        result.indices[level].push_back(posReg);
        result.indicesParents[level].push_back(posReg);
        result.indices[level].push_back(crdReg);
        result.indicesParents[level].push_back(crdReg);

        // Now copy the regions over from the source into the destination.
        // For simplicity, we'll assert that some partitions of the source tensor
        // have already been created.
        taco_uassert(src.indicesEqPartitions[level].size() == 2);
        auto srcPosPart = src.indicesEqPartitions[level][0];
        auto srcCrdPart = src.indicesEqPartitions[level][1];
        taco_uassert(srcPosPart.exists());
        taco_uassert(srcCrdPart.exists());

        // Create some equal partitions of the destination arrays as well.
        auto posPart = createLogicalPart(posReg, runtime->get_index_partition_color_space_name(srcPosPart.get_index_partition()));
        auto crdPart = createLogicalPart(crdReg, runtime->get_index_partition_color_space_name(srcCrdPart.get_index_partition()));
        result.indicesEqPartitions[level].push_back(posPart);
        result.indicesEqPartitions[level].push_back(crdPart);

        // Launch an IndexCopy over these partitions.
        IndexCopyLauncher launcher(runtime->get_index_partition_color_space_name(srcPosPart.get_index_partition()));
        launcher.add_copy_requirements(
            RegionRequirement(srcPosPart, 0, READ_ONLY, EXCLUSIVE, srcPosParent).add_field(posField),
            RegionRequirement(posPart, 0, WRITE_ONLY, EXCLUSIVE, posReg).add_field(posField)
        );
        launcher.add_copy_requirements(
            RegionRequirement(srcCrdPart, 0, READ_ONLY, EXCLUSIVE, srcCrdParent).add_field(crdField),
            RegionRequirement(crdPart, 0, WRITE_ONLY, EXCLUSIVE, crdReg).add_field(crdField)
        );
        runtime->issue_copy_operation(ctx, launcher);

        // Finally, remember that this is the dimensionality of the vals region.
        currentValsDomain = runtime->get_index_space_domain(srcCrdReg.get_index_space());
        break;
      }
      case Singleton: {
        taco_iassert(false) << "not handling the Singleton case here yet";
        break;
      }
    }
  }

  // Perform a similar operation as above but for the values array.
  // However, we only need to construct the values, not copy anything into them.
  taco_uassert(currentValsDomain.exists()) << "cannot copy prefix to dense tensor";
  auto valsIspace = runtime->create_index_space(ctx, currentValsDomain);
  auto vals = runtime->create_logical_region(ctx, valsIspace, src.vals.get_field_space());
  result.vals = vals;
  result.valsParent = vals;
  runtime->fill_field(ctx, vals, vals, getField(vals), T(0));
  return result;
}

#endif //TACO_LEGION_TENSOR_H
