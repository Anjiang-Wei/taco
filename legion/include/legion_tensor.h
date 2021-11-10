#ifndef TACO_LEGION_TENSOR_H
#define TACO_LEGION_TENSOR_H

#include "legion.h"

// TODO (rohany): We might have to add a mirrored "physical tensor" that has all of this stuff
//  pulled out as the physical one when in a child task.
// LegionTensor is a representation of a taco_tensor_t for the Legion backend.
struct LegionTensor {
  // TODO (rohany): I want to maybe turn this into a class or at least remove the default
  //  construct for it.

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

  // toString returns a human-readable string containing the data structure layout of
  // a LegionTensor.
  std::string toString(Legion::Context ctx, Legion::Runtime* runtime);

  // TODO (rohany): Consider adding a validation method on a LegionTensor.

  // TODO (rohany): There are some sort of serialization methods here, as well
  //  as potentially methods to put each region into a region requirement. Actually,
  //  I don't think that there is a method that can apriori put all of the regions
  //  into a region requirement as it requires knowing what regions are actually
  //  part of the pack, which the compiler does.
};

// LegionTensorLevelFormat is an enum used to perform introspection on LegionTensor
// objects to understand the construction of the components within it.
enum LegionTensorLevelFormat {
  Dense,
  Sparse,
};

// Utility method to create a dense tensor with the given dimensions.
template<int DIM, typename T>
LegionTensor createDenseTensor(Legion::Context ctx, Legion::Runtime* runtime, std::vector<int32_t> dims, Legion::FieldID valsField) {
  assert(dims.size() == DIM);
  Legion::Point<DIM> lo, hi;
  for (int i = 0; i < DIM; i++) {
    lo[i] = 0;
    hi[i] = dims[i] - 1;
  }
  auto ispace = runtime->create_index_space(ctx, Legion::Domain(lo, hi));
  auto fspace = runtime->create_field_space(ctx);
  Legion::FieldAllocator fa = runtime->create_field_allocator(ctx, fspace);
  fa.allocate_field(sizeof(T), valsField);
  auto reg = runtime->create_logical_region(ctx, ispace, fspace);
  // TODO (rohany): Do I want to issue this fill?
  runtime->fill_field(ctx, reg, reg, valsField, T(0));
  return LegionTensor {
    .order = int32_t(dims.size()),
    .dims = dims,
    .indices = {},
    .indicesParents = {},
    .vals = reg,
    .valsParent = reg,
    .denseLevelRuns = {reg.get_index_space()},
  };
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
LegionTensor createSparseTensorForPack(Legion::Context ctx, Legion::Runtime* runtime, std::vector<LegionTensorLevelFormat> formats, std::vector<int32_t> dims,
                                       Legion::FieldID posField, Legion::FieldID crdField, Legion::FieldID valsField) {
  assert(dims.size() == formats.size());
  auto result = LegionTensor{};
  result.order = formats.size();
  result.dims = dims;
  // Initialize the indices vectors for the tensor.
  result.indices = std::vector<std::vector<Legion::LogicalRegion>>(result.order);
  result.indicesParents = std::vector<std::vector<Legion::LogicalRegion>>(result.order);

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

  // An "infinitely" sized region will just be a really big one.
  auto infty = 1 << 30;

  bool inDenseRun = false;
  std::vector<int> levelsInDenseRun;
  auto getIspaceDomainPoints = [&](Legion::DomainPoint& lo, Legion::DomainPoint& hi) {
    if (!inDenseRun) {
      // If we weren't in a dense run, then the next region is one-dimensional
      // of unknown size.
      lo.dim = 1;
      hi.dim = 1;
      lo[0] = 0;
      hi[0] = infty;
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
          hi[i] = dims[levelsInDenseRun[i]];
        }
      } else {
        auto dim = levelsInDenseRun.size() + 1;
        lo.dim = dim;
        hi.dim = dim;
        lo[0] = 0;
        hi[0] = infty;
        for (size_t i = 1; i < dim; i++) {
          lo[i] = 0;
          hi[i] = dims[levelsInDenseRun[i - 1]];
        }
      }
    }
  };

  for (size_t level = 0; level < formats.size(); level++) {
    auto format = formats[level];
    switch (format) {
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
        assert(false);
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


#endif //TACO_LEGION_TENSOR_H
