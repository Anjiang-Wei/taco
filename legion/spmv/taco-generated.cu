#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.cuh"
#include "cublas_v2.h"
#include "cusparse.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef ReductionAccessor<SumReduction<double>,false,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorReduceNonExcldouble1;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int64_t B2_dimension;
  int32_t pieces;
};

struct task_2Args {
  int64_t B2Size;
  int32_t pieces;
};

struct task_3Args {
  int64_t B2Size;
  int32_t pieces;
};

struct task_4Args {
  int32_t pieces;
};


partitionPackForcomputeLegionRowSplit partitionForcomputeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];
  int B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  IndexSpace c_dense_run_0 = c->denseLevelRuns[0];


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t io = (*itr)[0];
    Point<1> aStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> aEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)),aDomain.hi()[0]));
    Rect<1> aRect = Rect<1>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<1> BStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)),BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto a_dense_run_0_Partition = runtime->create_index_partition(ctx, a_dense_run_0, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto a_vals_partition = copyPartition(ctx, runtime, a_dense_run_0_Partition, get_logical_region(a_vals));
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition, B2_pos);
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(
    ctx,
    runtime,
    B2_crd.get_index_space(),
    posPartB2,
    B2_pos_parent,
    FID_RECT_1
  ));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  auto computePartitions = partitionPackForcomputeLegionRowSplit();
  computePartitions.aPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(1);
  computePartitions.aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.aPartition.valsPartition = a_vals_partition;
  computePartitions.aPartition.denseLevelRunPartitions[0] = a_dense_run_0_Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition;

  return computePartitions;
}

__global__
void constructValidPos(size_t elems, const Rect<1>* posGlobal, int32_t* posLocal) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= elems) {
    return;
  }
  posLocal[idx] = posGlobal[idx].lo - posGlobal[0].lo;
  if (idx == 0) {
    posLocal[elems] = posGlobal[elems - 1].hi + 1;
  }
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B2_pos = regions[1];
  LogicalRegion B2_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B2_crd = regions[2];
  LogicalRegion B2_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B_vals = regions[3];
  LogicalRegion B_vals_parent = regions[3].get_logical_region();
  PhysicalRegion c_vals = regions[4];
  LogicalRegion c_vals_parent = regions[4].get_logical_region();

  int64_t io = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int64_t B2_dimension = args->B2_dimension;
  int32_t pieces = args->pieces;

  cusparseHandle_t handle = getCuSparse();
  cudaStream_t taskStream = cudaStream_t();
  cudaStreamCreate(&(taskStream));

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble1>(a_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  auto posDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  auto crdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  auto aValsDomain = runtime->get_index_space_domain(ctx, get_index_space(a_vals));
  auto BValsDomain = runtime->get_index_space_domain(ctx, get_index_space(B_vals));
  auto cValsDomain = runtime->get_index_space_domain(ctx, get_index_space(c_vals));
  double alpha = 1.0000000000000000;
  CHECK_CUSPARSE(cusparseSetStream(handle, taskStream));
  cusparseMatDescr_t mat = cusparseMatDescr_t();
  CHECK_CUSPARSE(cusparseCreateMatDescr(&(mat)));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(mat, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparseSetMatType(mat, CUSPARSE_MATRIX_TYPE_GENERAL));

  // We have to convert this globally addressed pos array into a local pos array.
  uint64_t bufferSize = 0;
  Legion::DeferredBuffer<int32_t, 1> posBuf(Rect<1>(0, posDomain.get_volume()), Memory::Kind::GPU_FB_MEM);
  const int THREADS_PER_BLOCK = 256;
  auto blocks = (posDomain.get_volume() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  taco_iassert(blocks != 0);
  constructValidPos<<<blocks, THREADS_PER_BLOCK, 0, taskStream>>>(posDomain.get_volume(), B2_pos_accessor.ptr(posDomain.lo()), posBuf.ptr(0));

  CHECK_CUSPARSE(cusparseCsrmvEx_bufferSize(
    handle,
    CUSPARSE_ALG_MERGE_PATH,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    posDomain.get_volume(),
    B2_dimension,
    crdDomain.get_volume(),
    &(alpha),
    CUDA_R_64F,
    mat,
    B_vals_ro_accessor.ptr(BValsDomain.lo()),
    CUDA_R_64F,
    posBuf.ptr(0),
    B2_crd_accessor.ptr(crdDomain.lo()),
    c_vals_ro_accessor.ptr(cValsDomain.lo()),
    CUDA_R_64F,
    &(alpha),
    CUDA_R_64F,
    a_vals_rw_accessor.ptr(aValsDomain.lo()),
    CUDA_R_64F,
    CUDA_R_64F,
    &(bufferSize)
  ));
  void* workspacePtr = NULL;
  if (bufferSize > 0) {
    Legion::DeferredBuffer<char, 1> buf = Legion::DeferredBuffer<char, 1>(Rect<1>(0, (bufferSize - 1)), Memory::Kind::GPU_FB_MEM);
    workspacePtr = buf.ptr(0);
  }
  CHECK_CUSPARSE(cusparseCsrmvEx(
    handle,
    CUSPARSE_ALG_MERGE_PATH,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    posDomain.get_volume(),
    B2_dimension,
    crdDomain.get_volume(),
    &(alpha),
    CUDA_R_64F,
    mat,
    B_vals_ro_accessor.ptr(BValsDomain.lo()),
    CUDA_R_64F,
    posBuf.ptr(0),
    B2_crd_accessor.ptr(crdDomain.lo()),
    c_vals_ro_accessor.ptr(cValsDomain.lo()),
    CUDA_R_64F,
    &(alpha),
    CUDA_R_64F,
    a_vals_rw_accessor.ptr(aValsDomain.lo()),
    CUDA_R_64F,
    CUDA_R_64F,
    workspacePtr
  ));
}

void computeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionRowSplit* partitionPack, int32_t pieces) {
  auto a_vals_parent = a->valsParent;
  int B2_dimension = B->dims[1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.B2_dimension = B2_dimension;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, a_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionPosSplit partitionForcomputeLegionPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  DomainT<1> B2_crd_domain = runtime->get_index_space_domain(ctx, B2_crd.get_index_space());
  DomainPointColoring B2_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t fposo = (*itr)[0];
    Point<1> B2CrdStart = Point<1>((fposo * ((B2Size + (pieces - 1)) / pieces)));
    Point<1> B2CrdEnd = Point<1>(TACO_MIN((fposo * ((B2Size + (pieces - 1)) / pieces) + ((B2Size + (pieces - 1)) / pieces - 1)),B2_crd_domain.bounds.hi[0]));
    Rect<1> B2CrdRect = Rect<1>(B2CrdStart, B2CrdEnd);
    if (!B2_crd_domain.contains(B2CrdRect.lo) || !B2_crd_domain.contains(B2CrdRect.hi)) {
      B2CrdRect = B2CrdRect.make_empty();
    }
    B2_crd_coloring[(*itr)] = B2CrdRect;
  }
  IndexPartition B2_crd_index_part = runtime->create_index_partition(ctx, B2_crd.get_index_space(), domain, B2_crd_coloring, LEGION_COMPUTE_KIND);
  LogicalPartition B2_crd_part = runtime->get_logical_partition(ctx, B2_crd, B2_crd_index_part);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(
    ctx,
    B2_crd_index_part,
    B2_pos,
    B2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, B2_crd_index_part)
  );
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B2_crd_part, B_vals);
  IndexPartition BDenseRun0Partition = copyPartition(ctx, runtime, posPartB2, B_dense_run_0);
  IndexPartition aDenseRun0Partition = AffineProjection(0).apply(ctx, runtime, BDenseRun0Partition, a_dense_run_0);
  auto a_vals_partition = copyPartition(ctx, runtime, aDenseRun0Partition, get_logical_region(a_vals));
  auto computePartitions = partitionPackForcomputeLegionPosSplit();
  computePartitions.aPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(1);
  computePartitions.aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.aPartition.valsPartition = a_vals_partition;
  computePartitions.aPartition.denseLevelRunPartitions[0] = aDenseRun0Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(B2_crd_part);
  computePartitions.BPartition.valsPartition = BValsLogicalPart;
  computePartitions.BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;

  return computePartitions;
}

__global__
void task_2DeviceKernel0(int64_t B2Size, int64_t* i_blockStarts, int32_t pieces, AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble1 c_vals_ro_accessor, AccessorReduceNonExcldouble1 a_vals_red_accessor_non_excl, int64_t fposo) {

  int64_t block = blockIdx.x;
  int64_t thread = (threadIdx.x % (32));
  int64_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int64_t pointID2 = fposo * (((B2Size + (pieces - 1)) / pieces + 2047) / 2048) + block;
  int64_t pointID3 = pointID2 * 8 + warp;
  int64_t pointID4 = pointID3 * 32 + thread;
  double* precomputed = 0;
  double precomputed__codegen_cuda_tmp[8];
  precomputed = precomputed__codegen_cuda_tmp;
  for (int64_t pprecomputed = 0; pprecomputed < 8; pprecomputed++) {
    precomputed[pprecomputed] = 0.0;
  }
  for (int64_t thread_nz_pre = 0; thread_nz_pre < 8; thread_nz_pre++) {
    int64_t thread_nz = thread_nz_pre;
    int64_t fpos2 = thread * 8 + thread_nz;
    int64_t fpos1 = warp * 256 + fpos2;
    int64_t fposi = block * 2048 + fpos1;
    int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B2Size)
      break;

    int64_t f = B2_crd_accessor[fposB];
    precomputed[thread_nz_pre] = B_vals_ro_accessor[Point<1>(fposB)] * c_vals_ro_accessor[Point<1>(f)];
  }
  int64_t pB2_begin = i_blockStarts[block];
  int64_t pB2_end = i_blockStarts[(block + 1)];
  int64_t thread_nz = 0;
  int64_t fpos2 = thread * 8 + thread_nz;
  int64_t fpos1 = warp * 256 + fpos2;
  int64_t fposi = block * 2048 + fpos1;
  int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
  int64_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
  int64_t i = i_pos;
  for (int64_t thread_nz = 0; thread_nz < 8; thread_nz++) {
    int64_t fpos2 = thread * 8 + thread_nz;
    int64_t fpos1 = warp * 256 + fpos2;
    int64_t fposi = block * 2048 + fpos1;
    int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B2Size)
      break;

    int64_t f = B2_crd_accessor[fposB];
    while (!(B2_pos_accessor[i_pos].contains(fposB))) {
      i_pos = i_pos + 1;
      i = i_pos;
    }
    atomicAddWarp(a_vals_red_accessor_non_excl.ptr(Point<1>(i)), flattenPoint(a_vals_red_accessor_non_excl, Point<1>(i)), precomputed[thread_nz]);
  }
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B2_pos = regions[1];
  LogicalRegion B2_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B2_crd = regions[2];
  LogicalRegion B2_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B_vals = regions[3];
  LogicalRegion B_vals_parent = regions[3].get_logical_region();
  PhysicalRegion c_vals = regions[4];
  LogicalRegion c_vals_parent = regions[4].get_logical_region();

  int64_t fposo = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int64_t B2Size = args->B2Size;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(a_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  Legion::DeferredBuffer<int64_t, 1> buf = Legion::DeferredBuffer<int64_t, 1>(Rect<1>(0, (((B2Size + (pieces - 1)) / pieces + 2047) / 2048)), Legion::Memory::Kind::GPU_FB_MEM);
  int64_t* i_blockStarts = buf.ptr(0);
  taco_binarySearchBeforeBlockLaunch(
    B2_pos_accessor,
    i_blockStarts,
    B2PosDomain.bounds.lo,
    B2PosDomain.bounds.hi,
    2048,
    256,
    (((B2Size + (pieces - 1)) / pieces + 2047) / 2048),
    B2CrdDomain.bounds.lo
  );
  if (((((B2Size + (pieces - 1)) / pieces + 2047) / 2048)) > 0) {
    task_2DeviceKernel0<<<(((B2Size + (pieces - 1)) / pieces + 2047) / 2048), (32 * 8)>>>(B2Size, i_blockStarts, pieces, B2_pos_accessor, B2_crd_accessor, B_vals_ro_accessor, c_vals_ro_accessor, a_vals_red_accessor_non_excl, fposo);
  }
}

void computeLegionPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplit* partitionPack, int32_t pieces) {
  auto a_vals_parent = a->valsParent;
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.B2Size = B2Size;
  taskArgsRaw2.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, a_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionPosSplitDCSR partitionForcomputeLegionPosSplitDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper B1_pos = B->indices[0][0];
  RegionWrapper B1_crd = B->indices[0][1];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  DomainT<1> B2_crd_domain = runtime->get_index_space_domain(ctx, B2_crd.get_index_space());
  DomainPointColoring B2_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t fposo = (*itr)[0];
    Point<1> B2CrdStart = Point<1>((fposo * ((B2Size + (pieces - 1)) / pieces)));
    Point<1> B2CrdEnd = Point<1>(TACO_MIN((fposo * ((B2Size + (pieces - 1)) / pieces) + ((B2Size + (pieces - 1)) / pieces - 1)),B2_crd_domain.bounds.hi[0]));
    Rect<1> B2CrdRect = Rect<1>(B2CrdStart, B2CrdEnd);
    if (!B2_crd_domain.contains(B2CrdRect.lo) || !B2_crd_domain.contains(B2CrdRect.hi)) {
      B2CrdRect = B2CrdRect.make_empty();
    }
    B2_crd_coloring[(*itr)] = B2CrdRect;
  }
  IndexPartition B2_crd_index_part = runtime->create_index_partition(ctx, B2_crd.get_index_space(), domain, B2_crd_coloring, LEGION_COMPUTE_KIND);
  LogicalPartition B2_crd_part = runtime->get_logical_partition(ctx, B2_crd, B2_crd_index_part);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(
    ctx,
    B2_crd_index_part,
    B2_pos,
    B2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, B2_crd_index_part)
  );
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B2_crd_part, B_vals);
  LogicalPartition crdPartB1 = copyPartition(ctx, runtime, posPartB2, B1_crd);
  IndexPartition posSparsePartB1 = runtime->create_partition_by_preimage_range(
    ctx,
    crdPartB1.get_index_partition(),
    B1_pos,
    B1_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, crdPartB1.get_index_partition())
  );
  IndexPartition posIndexPartB1 = densifyPartition(ctx, runtime, get_index_space(B1_pos), posSparsePartB1);
  LogicalPartition posPartB1 = runtime->get_logical_partition(ctx, B1_pos, posIndexPartB1);
  auto computePartitions = partitionPackForcomputeLegionPosSplitDCSR();
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[0].push_back(posPartB1);
  computePartitions.BPartition.indicesPartitions[0].push_back(crdPartB1);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(B2_crd_part);
  computePartitions.BPartition.valsPartition = BValsLogicalPart;

  return computePartitions;
}

__global__
void task_3DeviceKernel0(int64_t B2Size, int64_t* i_blockStarts, int32_t pieces, AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble1 c_vals_ro_accessor, AccessorROint32_t1 B1_crd_accessor, AccessorReduceNonExcldouble1 a_vals_red_accessor_non_excl, int64_t fposo) {

  int64_t block = blockIdx.x;
  int64_t thread = (threadIdx.x % (32));
  int64_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int64_t pointID2 = fposo * (((B2Size + (pieces - 1)) / pieces + 2047) / 2048) + block;
  int64_t pointID3 = pointID2 * 8 + warp;
  int64_t pointID4 = pointID3 * 32 + thread;
  double* precomputed = 0;
  double precomputed__codegen_cuda_tmp[8];
  precomputed = precomputed__codegen_cuda_tmp;
  for (int64_t pprecomputed = 0; pprecomputed < 8; pprecomputed++) {
    precomputed[pprecomputed] = 0.0;
  }
  for (int64_t thread_nz_pre = 0; thread_nz_pre < 8; thread_nz_pre++) {
    int64_t thread_nz = thread_nz_pre;
    int64_t fpos2 = thread * 8 + thread_nz;
    int64_t fpos1 = warp * 256 + fpos2;
    int64_t fposi = block * 2048 + fpos1;
    int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B2Size)
      break;

    int64_t f = B2_crd_accessor[fposB];
    precomputed[thread_nz_pre] = B_vals_ro_accessor[Point<1>(fposB)] * c_vals_ro_accessor[Point<1>(f)];
  }
  int64_t pB2_begin = i_blockStarts[block];
  int64_t pB2_end = i_blockStarts[(block + 1)];
  int64_t thread_nz = 0;
  int64_t fpos2 = thread * 8 + thread_nz;
  int64_t fpos1 = warp * 256 + fpos2;
  int64_t fposi = block * 2048 + fpos1;
  int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
  int64_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
  int64_t i = B1_crd_accessor[i_pos];
  for (int64_t thread_nz = 0; thread_nz < 8; thread_nz++) {
    int64_t fpos2 = thread * 8 + thread_nz;
    int64_t fpos1 = warp * 256 + fpos2;
    int64_t fposi = block * 2048 + fpos1;
    int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B2Size)
      break;

    int64_t f = B2_crd_accessor[fposB];
    if (!(B2_pos_accessor[i_pos].contains(fposB))) {
      i_pos = i_pos + 1;
      i = B1_crd_accessor[i_pos];
    }
    atomicAddWarp(a_vals_red_accessor_non_excl.ptr(Point<1>(i)), flattenPoint(a_vals_red_accessor_non_excl, Point<1>(i)), precomputed[thread_nz]);
  }
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B1_pos = regions[1];
  LogicalRegion B1_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B1_crd = regions[2];
  LogicalRegion B1_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B2_pos = regions[3];
  LogicalRegion B2_pos_parent = regions[3].get_logical_region();
  PhysicalRegion B2_crd = regions[4];
  LogicalRegion B2_crd_parent = regions[4].get_logical_region();
  PhysicalRegion B_vals = regions[5];
  LogicalRegion B_vals_parent = regions[5].get_logical_region();
  PhysicalRegion c_vals = regions[6];
  LogicalRegion c_vals_parent = regions[6].get_logical_region();

  int64_t fposo = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int64_t B2Size = args->B2Size;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(a_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B1_crd_accessor = createAccessor<AccessorROint32_t1>(B1_crd, FID_COORD);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  Legion::DeferredBuffer<int64_t, 1> buf = Legion::DeferredBuffer<int64_t, 1>(Rect<1>(0, (((B2Size + (pieces - 1)) / pieces + 2047) / 2048)), Legion::Memory::Kind::GPU_FB_MEM);
  int64_t* i_blockStarts = buf.ptr(0);
  taco_binarySearchBeforeBlockLaunch(
    B2_pos_accessor,
    i_blockStarts,
    B2PosDomain.bounds.lo,
    B2PosDomain.bounds.hi,
    2048,
    256,
    (((B2Size + (pieces - 1)) / pieces + 2047) / 2048),
    B2CrdDomain.bounds.lo
  );
  if (((((B2Size + (pieces - 1)) / pieces + 2047) / 2048)) > 0) {
    task_3DeviceKernel0<<<(((B2Size + (pieces - 1)) / pieces + 2047) / 2048), (32 * 8)>>>(B2Size, i_blockStarts, pieces, B2_pos_accessor, B2_crd_accessor, B_vals_ro_accessor, c_vals_ro_accessor, B1_crd_accessor, a_vals_red_accessor_non_excl, fposo);
  }
}

void computeLegionPosSplitDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplitDCSR* partitionPack, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  auto a_vals_parent = a->valsParent;
  RegionWrapper B2_crd = B->indices[1][1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B1_crd_parent = B->indicesParents[0][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_3Args taskArgsRaw3;
  taskArgsRaw3.B2Size = B2Size;
  taskArgsRaw3.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, a_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(
    partitionPack->BPartition.indicesPartitions[0][0],
    0,
    READ_ONLY,
    EXCLUSIVE,
    get_logical_region(B1_pos_parent),
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[0][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B1_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionCSCMSpV partitionForcomputeLegionCSCMSpV(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  RegionWrapper c1_pos = c->indices[0][0];
  RegionWrapper c1_crd = c->indices[0][1];
  auto c1_pos_parent = c->indicesParents[0][0];
  auto c1_crd_parent = c->indicesParents[0][1];
  RegionWrapper c_vals = c->vals;
  auto c1_pos_accessor = createAccessor<AccessorRORect_1_1>(c1_pos, FID_RECT_1);

  c1_pos = legionMalloc(
    ctx,
    runtime,
    c1_pos,
    c1_pos_parent,
    FID_RECT_1,
    READ_ONLY
  );
  c1_pos_accessor = createAccessor<AccessorRORect_1_1>(c1_pos, FID_RECT_1);


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto jposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(jposoIndexSpace));
  DomainT<1> c1_crd_domain = runtime->get_index_space_domain(ctx, c1_crd.get_index_space());
  DomainPointColoring c1_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t jposo = (*itr)[0];
    Point<1> c1CrdStart = Point<1>((jposo * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + c1_pos_accessor[Point<1>(0)].lo));
    Point<1> c1CrdEnd = Point<1>(TACO_MIN(((jposo * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces - 1)) + c1_pos_accessor[Point<1>(0)].lo),c1_crd_domain.bounds.hi[0]));
    Rect<1> c1CrdRect = Rect<1>(c1CrdStart, c1CrdEnd);
    if (!c1_crd_domain.contains(c1CrdRect.lo) || !c1_crd_domain.contains(c1CrdRect.hi)) {
      c1CrdRect = c1CrdRect.make_empty();
    }
    c1_crd_coloring[(*itr)] = c1CrdRect;
  }
  IndexPartition c1_crd_index_part = runtime->create_index_partition(ctx, c1_crd.get_index_space(), domain, c1_crd_coloring, LEGION_COMPUTE_KIND);
  LogicalPartition c1_crd_part = runtime->get_logical_partition(ctx, c1_crd, c1_crd_index_part);
  IndexPartition posSparsePartc1 = runtime->create_partition_by_preimage_range(
    ctx,
    c1_crd_index_part,
    c1_pos,
    c1_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, c1_crd_index_part)
  );
  IndexPartition posIndexPartc1 = densifyPartition(ctx, runtime, get_index_space(c1_pos), posSparsePartc1);
  LogicalPartition posPartc1 = runtime->get_logical_partition(ctx, c1_pos, posIndexPartc1);
  LogicalPartition cValsLogicalPart = copyPartition(ctx, runtime, c1_crd_part, c_vals);
  IndexPartition BDenseRun0Partition = SparseGatherProjection(0).apply(ctx, runtime, c1_crd_parent, c1_crd_part, FID_COORD, B_dense_run_0);
  LogicalPartition posPartB2 = copyPartition(ctx, runtime, BDenseRun0Partition, B2_pos);
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(
    ctx,
    runtime,
    B2_crd.get_index_space(),
    posPartB2,
    B2_pos_parent,
    FID_RECT_1
  ));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  auto computePartitions = partitionPackForcomputeLegionCSCMSpV();
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;
  computePartitions.cPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(1);
  computePartitions.cPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.cPartition.indicesPartitions[0].push_back(posPartc1);
  computePartitions.cPartition.indicesPartitions[0].push_back(c1_crd_part);
  computePartitions.cPartition.valsPartition = cValsLogicalPart;

  runtime->unmap_region(ctx, c1_pos);

  return computePartitions;
}

__global__
void task_4DeviceKernel0(AccessorRORect_1_1 c1_pos_accessor, AccessorROint32_t1 c1_crd_accessor, AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorReduceNonExcldouble1 a_vals_red_accessor_non_excl, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble1 c_vals_ro_accessor, int32_t pieces, int64_t jposo) {

  int64_t block = blockIdx.x;
  int64_t thread = (threadIdx.x % (512));
  if (threadIdx.x >= 512) {
    return;
  }

  int64_t pointID2 = jposo * (((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces + 511) / 512) + block;
  int64_t jposi = block * 512 + thread;
  int64_t jposc = (jposo * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + jposi) + c1_pos_accessor[Point<1>(0)].lo;
  if (jposc >= (jposo + 1) * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + c1_pos_accessor[Point<1>(0)].lo)
    return;

  if (jposc < c1_pos_accessor[Point<1>(0)].lo || jposc >= c1_pos_accessor[Point<1>(0)].hi + 1)
    return;

  int64_t j = c1_crd_accessor[jposc];
  for (int64_t iB = B2_pos_accessor[Point<1>(j)].lo; iB < (B2_pos_accessor[Point<1>(j)].hi + 1); iB++) {
    int64_t i = B2_crd_accessor[iB];
    atomicAddWarp(a_vals_red_accessor_non_excl.ptr(Point<1>(i)), flattenPoint(a_vals_red_accessor_non_excl, Point<1>(i)), (B_vals_ro_accessor[Point<1>(iB)] * c_vals_ro_accessor[Point<1>(jposc)]));
  }
}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B2_pos = regions[1];
  LogicalRegion B2_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B2_crd = regions[2];
  LogicalRegion B2_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B_vals = regions[3];
  LogicalRegion B_vals_parent = regions[3].get_logical_region();
  PhysicalRegion c1_pos = regions[4];
  LogicalRegion c1_pos_parent = regions[4].get_logical_region();
  PhysicalRegion c1_crd = regions[5];
  LogicalRegion c1_crd_parent = regions[5].get_logical_region();
  PhysicalRegion c_vals = regions[6];
  LogicalRegion c_vals_parent = regions[6].get_logical_region();

  int64_t jposo = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(a_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto c1_pos_accessor = createAccessor<AccessorRORect_1_1>(c1_pos, FID_RECT_1);
  auto c1_crd_accessor = createAccessor<AccessorROint32_t1>(c1_crd, FID_COORD);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(c1_crd)).empty())
    return ;

  if (((((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces + 511) / 512)) > 0) {
    task_4DeviceKernel0<<<(((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces + 511) / 512), 512>>>(c1_pos_accessor, c1_crd_accessor, B2_pos_accessor, B2_crd_accessor, a_vals_red_accessor_non_excl, B_vals_ro_accessor, c_vals_ro_accessor, pieces, jposo);
  }
}

void computeLegionCSCMSpV(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionCSCMSpV* partitionPack, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  auto a_vals_parent = a->valsParent;
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  auto c1_pos_parent = c->indicesParents[0][0];
  auto c1_crd_parent = c->indicesParents[0][1];
  auto c_vals_parent = c->valsParent;


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto jposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(jposoIndexSpace));
  task_4Args taskArgsRaw4;
  taskArgsRaw4.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw4, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, a_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.indicesPartitions[0][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(c1_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.indicesPartitions[0][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(c1_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher);

}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_3>(registrar, "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(4), "task_4");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
}
