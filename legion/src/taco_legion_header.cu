#include "taco_legion_header.h"
#include "pitches.h"
#include "cub/cub.cuh"

using namespace Legion;

const int THREADS_PER_BLOCK = 256;

Domain RectCompressedPosPartitionDownwards::gputask(const Task *task, const std::vector<Legion::PhysicalRegion> &regions,
                                                    Legion::Context ctx, Runtime *runtime) {
  FieldID field = *(FieldID*)(task->args);
  Accessor acc(regions[0], field);
  auto dom = runtime->get_index_space_domain(ctx, regions[0].get_logical_region().get_index_space());
  taco_iassert(dom.dense());
  if (dom.empty()) {
    return Rect<1>::make_empty();
  }
  Rect<1> lo, hi;
  cudaMemcpy(&lo, acc.ptr(dom.lo()), sizeof(Rect<1>), cudaMemcpyHostToDevice);
  cudaMemcpy(&hi, acc.ptr(dom.hi()), sizeof(Rect<1>), cudaMemcpyHostToDevice);
  return Rect<1>{lo.lo, hi.hi};
}

template<int DIM, Legion::PrivilegeMode MODE>
using RCFYPAccessor = Legion::FieldAccessor<MODE, Legion::Rect<1>, DIM, Legion::coord_t, Realm::AffineAccessor<Legion::Rect<1>, DIM, Legion::coord_t>>;
template<int DIM>
__global__
void rectCompressedFinalizeYieldPositionsKernel(const Rect <DIM> fullBounds, const Rect <DIM> iterBounds,
                                                const Pitches<DIM - 1> pitches, size_t volume,
                                                RCFYPAccessor <DIM, READ_WRITE> output,
                                                RCFYPAccessor <DIM, READ_ONLY> ghost) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, iterBounds.lo);
  if (point == Point<DIM>::ZEROES()) {
    output[point].lo = 0;
  } else {
    output[point].lo = ghost[getPreviousPoint(point, fullBounds)].lo;
  }
}

void RectCompressedFinalizeYieldPositions::gputask(const Legion::Task *task,
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
      Pitches<DIM - 1> pitches;                                                                                           \
      auto volume = pitches.flatten(iterBounds);                                                                          \
      auto blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;                                                 \
      if (blocks != 0) {                                                                                                  \
        Accessor<DIM, READ_WRITE> outAcc(output, fields[0]); \
        Accessor<DIM, READ_ONLY> ghostAcc(ghost, fields[0]); \
        rectCompressedFinalizeYieldPositionsKernel<DIM><<<blocks, THREADS_PER_BLOCK>>>(fullBounds, iterBounds, pitches, volume, outAcc, ghostAcc); \
      }            \
      break;       \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }
}

template<typename T, int DIM, Legion::PrivilegeMode MODE>
using RCGSEIAccessor = Legion::FieldAccessor<MODE, T, DIM, Legion::coord_t, Realm::AffineAccessor<T, DIM, Legion::coord_t>>;

template<int DIM>
__global__
void rectCompressedGetSeqInsertEdgesConstructFinalKernel(const Rect<DIM> iterBounds,
                                                         const Pitches<DIM - 1> pitches,
                                                         const size_t volume,
                                                         RCGSEIAccessor<Rect<1>, DIM, WRITE_ONLY> output,
                                                         DeferredBuffer<int64_t, DIM> scanBuf) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, iterBounds.lo);
  auto lo = (idx == 0) ? 0 : scanBuf[pitches.unflatten(idx - 1, iterBounds.lo)];
  auto hi = scanBuf[point] - 1;
  output[point] = {lo, hi};
}

template<int DIM>
int64_t RectCompressedGetSeqInsertEdges::scanBodyGPU(Context ctx, Runtime *runtime, Rect<DIM> iterBounds,
                                                     Accessor<Rect<1>, DIM, WRITE_ONLY> output,
                                                     Accessor<int64_t, DIM, READ_ONLY> input,
                                                     Memory::Kind tmpMemKind) {
  Pitches<DIM - 1> pitches;
  auto volume = pitches.flatten(iterBounds);
  if (volume == 0) {
    return 0;
  }

  int64_t initVal = 0;
  DeferredBuffer<int64_t, DIM> scanBuf(iterBounds, tmpMemKind, &initVal);

  auto inputPtrBot = input.ptr(iterBounds.lo);
  auto bufPtrBot = scanBuf.ptr(iterBounds.lo);

  // Perform the scan.
  void* tmpStorage = NULL;
  size_t tmpStorageBytes = 0;
  cub::DeviceScan::InclusiveSum(tmpStorage, tmpStorageBytes, inputPtrBot, bufPtrBot, volume);
  DeferredBuffer<char, 1> cubTmpStorage(Rect<1>(0, tmpStorageBytes - 1), tmpMemKind);
  cub::DeviceScan::InclusiveSum(cubTmpStorage.ptr(0), tmpStorageBytes, inputPtrBot, bufPtrBot, volume);

  // Construct the final result.
  auto blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  if (blocks != 0) {
    rectCompressedGetSeqInsertEdgesConstructFinalKernel<DIM><<<blocks, THREADS_PER_BLOCK>>>(iterBounds, pitches, volume, output, scanBuf);
  }

  // Return the result of the scan.
  int64_t scanVal;
  cudaMemcpy(&scanVal, scanBuf.ptr(iterBounds.hi), sizeof(int64_t), cudaMemcpyHostToDevice);
  return scanVal;
}

int64_t RectCompressedGetSeqInsertEdges::scanTaskGPU(const Legion::Task *task,
                                                     const std::vector<Legion::PhysicalRegion> &regions,
                                                     Legion::Context ctx,
                                                     Legion::Runtime *runtime) {
  // Unpack arguments for the task.
  FieldID outputField, inputField;
  std::tie(outputField, inputField) = RectCompressedGetSeqInsertEdges::unpackScanTaskArgs(task);

  // Figure out what kind of memory body should allocate its temporary within.
  Memory::Kind tmpMemKind = Realm::Memory::GPU_FB_MEM;

  auto output = regions[0];
  auto input = regions[1];
  auto outputlr = output.get_logical_region();
  switch (outputlr.get_dim()) {
#define BLOCK(DIM) \
    case DIM: {    \
      Rect<DIM> iterBounds = runtime->get_index_space_domain(ctx, outputlr.get_index_space()).bounds<DIM, coord_t>();     \
      Accessor<Rect<1>, DIM, WRITE_ONLY> outAcc(output, outputField); \
      Accessor<int64_t, DIM, READ_ONLY> inAcc(input, inputField); \
      return RectCompressedGetSeqInsertEdges::scanBodyGPU<DIM>(ctx, runtime, iterBounds, outAcc, inAcc, tmpMemKind); \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
      return 0; // Keep the compiler happy.
  }
}

template<int DIM>
__global__
void rectCompressedGetSeqInsertEdgesApplyPartialResultsKernel(
    const Rect<DIM> iterBounds,
    const Pitches<DIM - 1> pitches,
    const size_t volume,
    RCGSEIAccessor<Rect<1>, DIM, READ_WRITE> output,
    int64_t value
) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  output[pitches.unflatten(idx, iterBounds.lo)] += Point<1>(value);
}

void RectCompressedGetSeqInsertEdges::applyPartialResultsTaskGPU(const Legion::Task *task,
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
      Accessor<Rect<1>, DIM, READ_WRITE> outAcc(output, outputField);                                                     \
      Pitches<DIM - 1> pitches;                                                                                           \
      auto volume = pitches.flatten(iterBounds);                                                                          \
      auto blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;                                                 \
      if (blocks != 0) {                                                                                                  \
        rectCompressedGetSeqInsertEdgesApplyPartialResultsKernel<DIM><<<blocks, THREADS_PER_BLOCK>>>(iterBounds, pitches, volume, outAcc, value);             \
      }             \
      break;             \
    }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      taco_iassert(false);
  }
}
