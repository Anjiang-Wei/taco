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
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,false,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReduceNonExcldouble2;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int64_t B3Size;
  int32_t C2_dimension;
  int32_t pieces;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, int32_t pieces) {
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_pos = B->indices[2][0];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B3_pos_parent = B->indicesParents[2][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  DomainT<1> B3_crd_domain = runtime->get_index_space_domain(ctx, B3_crd.get_index_space());
  DomainPointColoring B3_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t fposo = (*itr)[0];
    Point<1> B3CrdStart = Point<1>((fposo * ((B3Size + (pieces - 1)) / pieces)));
    Point<1> B3CrdEnd = Point<1>(TACO_MIN((fposo * ((B3Size + (pieces - 1)) / pieces) + ((B3Size + (pieces - 1)) / pieces - 1)),B3_crd_domain.bounds.hi[0]));
    Rect<1> B3CrdRect = Rect<1>(B3CrdStart, B3CrdEnd);
    if (!B3_crd_domain.contains(B3CrdRect.lo) || !B3_crd_domain.contains(B3CrdRect.hi)) {
      B3CrdRect = B3CrdRect.make_empty();
    }
    B3_crd_coloring[(*itr)] = B3CrdRect;
  }
  IndexPartition B3_crd_index_part = runtime->create_index_partition(ctx, B3_crd.get_index_space(), domain, B3_crd_coloring, LEGION_COMPUTE_KIND);
  LogicalPartition B3_crd_part = runtime->get_logical_partition(ctx, B3_crd, B3_crd_index_part);
  IndexPartition posSparsePartB3 = runtime->create_partition_by_preimage_range(
    ctx,
    B3_crd_index_part,
    B3_pos,
    B3_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, B3_crd_index_part)
  );
  IndexPartition posIndexPartB3 = densifyPartition(ctx, runtime, get_index_space(B3_pos), posSparsePartB3);
  LogicalPartition posPartB3 = runtime->get_logical_partition(ctx, B3_pos, posIndexPartB3);
  LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B3_crd_part, B_vals);
  LogicalPartition crdPartB2 = copyPartition(ctx, runtime, posPartB3, B2_crd);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(
    ctx,
    crdPartB2.get_index_partition(),
    B2_pos,
    B2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, crdPartB2.get_index_partition())
  );
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  IndexPartition BDenseRun0Partition = copyPartition(ctx, runtime, posPartB2, B_dense_run_0);
  IndexPartition ADenseRun0Partition = AffineProjection(0).apply(ctx, runtime, BDenseRun0Partition, A_dense_run_0);
  auto A_vals_partition = copyPartition(ctx, runtime, ADenseRun0Partition, get_logical_region(A_vals));
  auto computePartitions = partitionPackForcomputeLegion();
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = ADenseRun0Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(3);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.indicesPartitions[2].push_back(posPartB3);
  computePartitions.BPartition.indicesPartitions[2].push_back(B3_crd_part);
  computePartitions.BPartition.valsPartition = BValsLogicalPart;
  computePartitions.BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;

  return computePartitions;
}

__global__
void task_1DeviceKernel0(int64_t B3Size, int32_t* i_blockStarts, int32_t* j_blockStarts, int32_t pieces, AccessorRORect_1_1 B3_pos_accessor, AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorROint32_t1 B3_crd_accessor, AccessorReduceNonExcldouble2 A_vals_red_accessor_non_excl, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble2 C_vals_ro_accessor, AccessorROdouble2 D_vals_ro_accessor, int32_t C2_dimension, int32_t fposo) {

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int64_t pointID2 = fposo * (((B3Size + (pieces - 1)) / pieces + 2047) / 2048) + block;
  int64_t pointID3 = pointID2 * 8 + warp;
  int64_t pointID4 = pointID3 * 32 + thread;
  int32_t pB3_begin = j_blockStarts[block];
  int32_t pB3_end = j_blockStarts[(block + 1)];
  int32_t fposi2 = thread * 8;
  int32_t fposi1 = warp * 256 + fposi2;
  int32_t fposi = block * 2048 + fposi1;
  int32_t fposB = fposo * ((B3Size + (pieces - 1)) / pieces) + fposi;
  int32_t j_pos = taco_binarySearchBefore(B3_pos_accessor, pB3_begin, pB3_end, fposB);
  int32_t j = B2_crd_accessor[j_pos];
  int32_t pB2_begin = i_blockStarts[block];
  int32_t pB2_end = i_blockStarts[(block + 1)];
  int32_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, j_pos);
  int32_t i = i_pos;
  for (int32_t thread_nz = 0; thread_nz < 8; thread_nz++) {
    int32_t fposi2 = thread * 8 + thread_nz;
    int32_t fposi1 = warp * 256 + fposi2;
    int32_t fposi = block * 2048 + fposi1;
    int32_t fposB = fposo * ((B3Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B3Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B3Size)
      break;

    int32_t f2 = B3_crd_accessor[fposB];
    if (!(B3_pos_accessor[j_pos].contains(fposB))) {
      j_pos = j_pos + 1;
      j = B2_crd_accessor[j_pos];
      while (!(B2_pos_accessor[i_pos].contains(j_pos))) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
    }
    for (int32_t l = 0; l < C2_dimension; l++) {
      atomicAddWarp(A_vals_red_accessor_non_excl.ptr(Point<2>(i, l)), flattenPoint(A_vals_red_accessor_non_excl, Point<2>(i, l)), ((B_vals_ro_accessor[Point<1>(fposB)] * C_vals_ro_accessor[Point<2>(j, l)]) * D_vals_ro_accessor[Point<2>(f2, l)]));
    }
  }
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B2_pos = regions[1];
  LogicalRegion B2_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B2_crd = regions[2];
  LogicalRegion B2_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B3_pos = regions[3];
  LogicalRegion B3_pos_parent = regions[3].get_logical_region();
  PhysicalRegion B3_crd = regions[4];
  LogicalRegion B3_crd_parent = regions[4].get_logical_region();
  PhysicalRegion B_vals = regions[5];
  LogicalRegion B_vals_parent = regions[5].get_logical_region();
  PhysicalRegion C_vals = regions[6];
  LogicalRegion C_vals_parent = regions[6].get_logical_region();
  PhysicalRegion D_vals = regions[7];
  LogicalRegion D_vals_parent = regions[7].get_logical_region();

  int32_t fposo = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int64_t B3Size = args->B3Size;
  int32_t C2_dimension = args->C2_dimension;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, FID_VAL);
  auto D_vals_ro_accessor = createAccessor<AccessorROdouble2>(D_vals, FID_VAL);
  auto A_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble2>(A_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);
  auto B3_pos_accessor = createAccessor<AccessorRORect_1_1>(B3_pos, FID_RECT_1);
  auto B3_crd_accessor = createAccessor<AccessorROint32_t1>(B3_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).empty())
    return ;

  DomainT<1> B3PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B3_pos));
  DomainT<1> B3CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B3_crd));
  Legion::DeferredBuffer<int32_t, 1> buf = Legion::DeferredBuffer<int32_t, 1>(Rect<1>(0, (((B3Size + (pieces - 1)) / pieces + 2047) / 2048)), Legion::Memory::Kind::GPU_FB_MEM);
  int32_t* j_blockStarts = buf.ptr(0);
  taco_binarySearchBeforeBlockLaunch(
    B3_pos_accessor,
    j_blockStarts,
    B3PosDomain.bounds.lo,
    B3PosDomain.bounds.hi,
    2048,
    256,
    (((B3Size + (pieces - 1)) / pieces + 2047) / 2048),
    B3CrdDomain.bounds.lo
  );
  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  Legion::DeferredBuffer<int32_t, 1> buf0 = Legion::DeferredBuffer<int32_t, 1>(Rect<1>(0, (((B3Size + (pieces - 1)) / pieces + 2047) / 2048)), Legion::Memory::Kind::GPU_FB_MEM);
  int32_t* i_blockStarts = buf0.ptr(0);
  taco_binarySearchIndirectBeforeBlockLaunch(
    B2_pos_accessor,
    i_blockStarts,
    B2PosDomain.bounds.lo,
    B2PosDomain.bounds.hi,
    j_blockStarts,
    256,
    (((B3Size + (pieces - 1)) / pieces + 2047) / 2048)
  );
  if (((((B3Size + (pieces - 1)) / pieces + 2047) / 2048)) > 0) {
    task_1DeviceKernel0<<<(((B3Size + (pieces - 1)) / pieces + 2047) / 2048), (32 * 8)>>>(B3Size, i_blockStarts, j_blockStarts, pieces, B3_pos_accessor, B2_pos_accessor, B2_crd_accessor, B3_crd_accessor, A_vals_red_accessor_non_excl, B_vals_ro_accessor, C_vals_ro_accessor, D_vals_ro_accessor, C2_dimension, fposo);
  }
}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, partitionPackForcomputeLegion* partitionPack, int32_t pieces) {
  auto A_vals_parent = A->valsParent;
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B3_pos_parent = B->indicesParents[2][0];
  auto B3_crd_parent = B->indicesParents[2][1];
  auto B_vals_parent = B->valsParent;
  int C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  RegionWrapper D_vals = D->vals;
  auto D_vals_parent = D->valsParent;

  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.B3Size = B3Size;
  taskArgsRaw1.C2_dimension = C2_dimension;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(C_vals), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(D_vals), READ_ONLY, EXCLUSIVE, D_vals_parent).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher);

}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
}
