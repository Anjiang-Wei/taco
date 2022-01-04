#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.cuh"
#include "cublas_v2.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef ReductionAccessor<SumReduction<double>,false,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorReduceNonExcldouble1;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int32_t B1_dimension;
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
    int32_t io = (*itr)[0];
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
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, runtime->create_partition_by_image_range(
    ctx,
    B2_crd.get_index_space(),
    posPartB2,
    B2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, posPartB2.get_index_partition())
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
void task_1DeviceKernel0(AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorRWdouble1 a_vals_rw_accessor, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble1 c_vals_ro_accessor, int32_t B1_dimension, int32_t pieces, int32_t io) {

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int64_t pointID2 = io * (((B1_dimension + (pieces - 1)) / pieces + 255) / 256) + block;
  int64_t pointID3 = pointID2 * 8 + warp;
  for (int32_t warp_row = 0; warp_row < 32; warp_row++) {
    int32_t block_row = warp_row * 8 + warp;
    int32_t ii = block * 256 + block_row;
    int32_t i = io * ((B1_dimension + (pieces - 1)) / pieces) + ii;
    if (i >= B1_dimension)
      break;

    if (i >= (io + 1) * ((B1_dimension + (pieces - 1)) / pieces))
      break;

    int64_t pointID4 = pointID3 * 32 + warp_row;
    int64_t pointID5 = pointID4 * 32 + thread;
    for (int32_t thread_nz = 0; thread_nz < ((((B2_pos_accessor[Point<1>(i)].hi + 1) - B2_pos_accessor[Point<1>(i)].lo) + 31) / 32); thread_nz++) {
      int32_t jposB = (thread_nz * 32 + thread) + B2_pos_accessor[Point<1>(i)].lo;
      if (jposB < B2_pos_accessor[Point<1>(i)].lo || jposB >= B2_pos_accessor[Point<1>(i)].hi + 1)
        break;

      int32_t j = B2_crd_accessor[jposB];
      atomicAddWarp(a_vals_rw_accessor.ptr(Point<1>(i)), flattenPoint(a_vals_rw_accessor, Point<1>(i)), (B_vals_ro_accessor[Point<1>(jposB)] * c_vals_ro_accessor[Point<1>(j)]));
    }
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

  int32_t io = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t B1_dimension = args->B1_dimension;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble1>(a_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).empty())
    return ;

  if (((((B1_dimension + (pieces - 1)) / pieces + 255) / 256)) > 0) {
    task_1DeviceKernel0<<<(((B1_dimension + (pieces - 1)) / pieces + 255) / 256), (32 * 8)>>>(B2_pos_accessor, B2_crd_accessor, a_vals_rw_accessor, B_vals_ro_accessor, c_vals_ro_accessor, B1_dimension, pieces, io);
  }
}

void computeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionRowSplit* partitionPack, int32_t pieces) {
  auto a_vals_parent = a->valsParent;
  int B1_dimension = B->dims[0];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
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
    int32_t fposo = (*itr)[0];
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
void task_2DeviceKernel0(int64_t B2Size, int32_t* i_blockStarts, int32_t pieces, AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble1 c_vals_ro_accessor, AccessorReduceNonExcldouble1 a_vals_red_accessor_non_excl, int32_t fposo) {

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int64_t pointID2 = fposo * (((B2Size + (pieces - 1)) / pieces + 2047) / 2048) + block;
  int64_t pointID3 = pointID2 * 8 + warp;
  int64_t pointID4 = pointID3 * 32 + thread;
  double* precomputed = 0;
  double precomputed__codegen_cuda_tmp[8];
  precomputed = precomputed__codegen_cuda_tmp;
  for (int32_t pprecomputed = 0; pprecomputed < 8; pprecomputed++) {
    precomputed[pprecomputed] = 0.0;
  }
  for (int32_t thread_nz_pre = 0; thread_nz_pre < 8; thread_nz_pre++) {
    int32_t thread_nz = thread_nz_pre;
    int32_t fpos2 = thread * 8 + thread_nz;
    int32_t fpos1 = warp * 256 + fpos2;
    int32_t fposi = block * 2048 + fpos1;
    int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B2Size)
      break;

    int32_t f = B2_crd_accessor[fposB];
    precomputed[thread_nz_pre] = B_vals_ro_accessor[Point<1>(fposB)] * c_vals_ro_accessor[Point<1>(f)];
  }
  int32_t pB2_begin = i_blockStarts[block];
  int32_t pB2_end = i_blockStarts[(block + 1)];
  int32_t thread_nz = 0;
  int32_t fpos2 = thread * 8 + thread_nz;
  int32_t fpos1 = warp * 256 + fpos2;
  int32_t fposi = block * 2048 + fpos1;
  int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
  int32_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
  int32_t i = i_pos;
  for (int32_t thread_nz = 0; thread_nz < 8; thread_nz++) {
    int32_t fpos2 = thread * 8 + thread_nz;
    int32_t fpos1 = warp * 256 + fpos2;
    int32_t fposi = block * 2048 + fpos1;
    int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B2Size)
      break;

    int32_t f = B2_crd_accessor[fposB];
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

  int32_t fposo = task->index_point[0];
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
  Legion::DeferredBuffer<int32_t, 1> buf = Legion::DeferredBuffer<int32_t, 1>(Rect<1>(0, (((B2Size + (pieces - 1)) / pieces + 2047) / 2048)), Legion::Memory::Kind::GPU_FB_MEM);
  int32_t* i_blockStarts = buf.ptr(0);
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
  task_2Args taskArgsRaw;
  taskArgsRaw.B2Size = B2Size;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
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
    int32_t fposo = (*itr)[0];
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
void task_3DeviceKernel0(int64_t B2Size, int32_t* i_blockStarts, int32_t pieces, AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble1 c_vals_ro_accessor, AccessorROint32_t1 B1_crd_accessor, AccessorReduceNonExcldouble1 a_vals_red_accessor_non_excl, int32_t fposo) {

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int64_t pointID2 = fposo * (((B2Size + (pieces - 1)) / pieces + 2047) / 2048) + block;
  int64_t pointID3 = pointID2 * 8 + warp;
  int64_t pointID4 = pointID3 * 32 + thread;
  double* precomputed = 0;
  double precomputed__codegen_cuda_tmp[8];
  precomputed = precomputed__codegen_cuda_tmp;
  for (int32_t pprecomputed = 0; pprecomputed < 8; pprecomputed++) {
    precomputed[pprecomputed] = 0.0;
  }
  for (int32_t thread_nz_pre = 0; thread_nz_pre < 8; thread_nz_pre++) {
    int32_t thread_nz = thread_nz_pre;
    int32_t fpos2 = thread * 8 + thread_nz;
    int32_t fpos1 = warp * 256 + fpos2;
    int32_t fposi = block * 2048 + fpos1;
    int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B2Size)
      break;

    int32_t f = B2_crd_accessor[fposB];
    precomputed[thread_nz_pre] = B_vals_ro_accessor[Point<1>(fposB)] * c_vals_ro_accessor[Point<1>(f)];
  }
  int32_t pB2_begin = i_blockStarts[block];
  int32_t pB2_end = i_blockStarts[(block + 1)];
  int32_t thread_nz = 0;
  int32_t fpos2 = thread * 8 + thread_nz;
  int32_t fpos1 = warp * 256 + fpos2;
  int32_t fposi = block * 2048 + fpos1;
  int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
  int32_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
  int32_t i = B1_crd_accessor[i_pos];
  for (int32_t thread_nz = 0; thread_nz < 8; thread_nz++) {
    int32_t fpos2 = thread * 8 + thread_nz;
    int32_t fpos1 = warp * 256 + fpos2;
    int32_t fposi = block * 2048 + fpos1;
    int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B2Size)
      break;

    int32_t f = B2_crd_accessor[fposB];
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

  int32_t fposo = task->index_point[0];
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
  Legion::DeferredBuffer<int32_t, 1> buf = Legion::DeferredBuffer<int32_t, 1>(Rect<1>(0, (((B2Size + (pieces - 1)) / pieces + 2047) / 2048)), Legion::Memory::Kind::GPU_FB_MEM);
  int32_t* i_blockStarts = buf.ptr(0);
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
  task_3Args taskArgsRaw;
  taskArgsRaw.B2Size = B2Size;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
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
    int32_t jposo = (*itr)[0];
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
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, runtime->create_partition_by_image_range(
    ctx,
    B2_crd.get_index_space(),
    posPartB2,
    B2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, posPartB2.get_index_partition())
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
void task_4DeviceKernel0(AccessorRORect_1_1 c1_pos_accessor, AccessorROint32_t1 c1_crd_accessor, AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorReduceNonExcldouble1 a_vals_red_accessor_non_excl, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble1 c_vals_ro_accessor, int32_t pieces, int32_t jposo) {

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (512));
  if (threadIdx.x >= 512) {
    return;
  }

  int64_t pointID2 = jposo * (((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces + 511) / 512) + block;
  int32_t jposi = block * 512 + thread;
  int32_t jposc = (jposo * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + jposi) + c1_pos_accessor[Point<1>(0)].lo;
  if (jposc >= (jposo + 1) * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + c1_pos_accessor[Point<1>(0)].lo)
    return;

  if (jposc < c1_pos_accessor[Point<1>(0)].lo || jposc >= c1_pos_accessor[Point<1>(0)].hi + 1)
    return;

  int32_t j = c1_crd_accessor[jposc];
  for (int32_t iB = B2_pos_accessor[Point<1>(j)].lo; iB < (B2_pos_accessor[Point<1>(j)].hi + 1); iB++) {
    int32_t i = B2_crd_accessor[iB];
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

  int32_t jposo = task->index_point[0];
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
  task_4Args taskArgsRaw;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
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
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
}
