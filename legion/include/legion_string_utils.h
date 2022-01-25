#ifndef TACO_LEGION_STRING_UTILS_H
#define TACO_LEGION_STRING_UTILS_H

#include "legion.h"
#include "legion_tensor.h"
#include "taco_legion_header.h"
#include "error.h"

// Generic function to print out region data. We'll declare this in a header and instantiate
// a few versions of the function that have pretty output. We have to wrap this in a helper
// widget to allow for partial specialization.
template <typename T, int DIM>
struct PhysicalRegionPrinter {
  void printPhysicalRegion(Legion::Context ctx, Legion::Runtime* runtime, Legion::PhysicalRegion reg, Legion::FieldID fid);
};

// We abstract direct use of PhysicalRegionPrinter through this helper method that performs
// the dispatch of the dimensionality of the region.
template <typename T>
void printPhysicalRegion(Legion::Context ctx, Legion::Runtime* runtime, Legion::PhysicalRegion reg, Legion::FieldID fid) {
  switch (reg.get_logical_region().get_dim()) {
    case 1: PhysicalRegionPrinter<T, 1>().printPhysicalRegion(ctx, runtime, reg, fid); break;
    case 2: PhysicalRegionPrinter<T, 2>().printPhysicalRegion(ctx, runtime, reg, fid); break;
    case 3: PhysicalRegionPrinter<T, 3>().printPhysicalRegion(ctx, runtime, reg, fid); break;
    default:
      taco_iassert(false);
  }
}

template <typename T>
struct PhysicalRegionPrinter<T, 1> {
  void printPhysicalRegion(Legion::Context ctx, Legion::Runtime* runtime, Legion::PhysicalRegion reg, Legion::FieldID fid) {
    using namespace Legion;
    FieldAccessor<READ_ONLY,T,1,coord_t, Realm::AffineAccessor<T, 1, coord_t>> acc(reg, fid);
    auto dom = runtime->get_index_space_domain<1>(IndexSpaceT<1>(reg.get_logical_region().get_index_space()));
    for (int i = dom.bounds.lo; i <= dom.bounds.hi; i++) {
      std::cout << acc[i] << " ";
    }
    std::cout << std::endl;
  }
};

template <typename T>
struct PhysicalRegionPrinter<T, 2> {
  void printPhysicalRegion(Legion::Context ctx, Legion::Runtime* runtime, Legion::PhysicalRegion reg, Legion::FieldID fid) {
    using namespace Legion;
    FieldAccessor<READ_ONLY,T,2,coord_t, Realm::AffineAccessor<T, 2, coord_t>> acc(reg, fid);
    auto dom = runtime->get_index_space_domain<2>(IndexSpaceT<2>(reg.get_logical_region().get_index_space()));
    for (int i = dom.bounds.lo[0]; i <= dom.bounds.hi[0]; i++) {
      for (int j = dom.bounds.lo[1]; j <= dom.bounds.hi[1]; j++) {
        std::cout << acc[Point<2>(i, j)] << " ";
      }
      std::cout << std::endl;
    }
  }
};

template <typename T>
struct PhysicalRegionPrinter<T, 3> {
  void printPhysicalRegion(Legion::Context ctx, Legion::Runtime* runtime, Legion::PhysicalRegion reg, Legion::FieldID fid) {
    using namespace Legion;
    FieldAccessor<READ_ONLY,T,3,coord_t, Realm::AffineAccessor<T, 3, coord_t>> acc(reg, fid);
    auto dom = runtime->get_index_space_domain<3>(IndexSpaceT<3>(reg.get_logical_region().get_index_space()));
    for (int i = dom.bounds.lo[0]; i <= dom.bounds.hi[0]; i++) {
      std::cout << "Slice: " << i << std::endl;
      for (int j = dom.bounds.lo[1]; j <= dom.bounds.hi[1]; j++) {
        for (int k = dom.bounds.lo[2]; k <= dom.bounds.hi[2]; k++) {
          std::cout << acc[Point<3>(i, j, k)] << " ";
        }
        std::cout << std::endl;
      }
    }
  }
};

template<typename T>
void printLegionTensor(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor& tensor) {
  auto format = tensor.format;
  // For each level in the tensor, map the regions and print them.
  for (size_t i = 0; i < format.size(); i++) {
    switch (format[i]) {
      case Dense:
        break; // Nothing to do for dense levels.
      case Sparse: {
        // Print out the crd and pos arrays here.
        auto pos = legionMalloc(ctx, runtime, tensor.indices[i][0], tensor.indicesParents[i][0], FID_RECT_1, READ_ONLY);
        auto crd = legionMalloc(ctx, runtime, tensor.indices[i][1], tensor.indicesParents[i][1], FID_COORD, READ_ONLY);
        std::cout << "pos " << i << ":" << std::endl;
        printPhysicalRegion<Legion::Rect<1>>(ctx, runtime, pos, FID_RECT_1);
        std::cout << "crd " << i << ":" << std::endl;
        printPhysicalRegion<int32_t>(ctx, runtime, crd, FID_COORD);
        runtime->unmap_region(ctx, pos);
        runtime->unmap_region(ctx, crd);
        break;
      }
      case Singleton: {
        // Print out the crd array.
        auto crd = legionMalloc(ctx, runtime, tensor.indices[i][0], tensor.indicesParents[i][0], FID_COORD, READ_ONLY);
        std::cout << "crd " << i << ":" << std::endl;
        printPhysicalRegion<int32_t>(ctx, runtime, crd, FID_COORD);
        runtime->unmap_region(ctx, crd);
        break;
      }
      default:
        taco_iassert(false);
        break;
    }
  }
  // Finally map the values.
  auto vals = legionMalloc(ctx, runtime, tensor.vals, tensor.valsParent, FID_VAL, READ_ONLY);
  std::cout << "vals " << ":" << std::endl;
  printPhysicalRegion<T>(ctx, runtime, vals, FID_VAL);
  runtime->unmap_region(ctx, vals);
}
#endif
