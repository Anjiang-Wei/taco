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
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,false,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReduceNonExcldouble2;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  Legion::FieldID A_vals_field_id;
  Legion::FieldID B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1;
  int64_t B3Size;
  Legion::FieldID B3_indices_field_id_2_0;
  Legion::FieldID B3_indices_field_id_2_1;
  Legion::FieldID B_vals_field_id;
  int64_t C2_dimension;
  Legion::FieldID C_vals_field_id;
  Legion::FieldID D_vals_field_id;
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
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B3_indices_field_id_2_0 = B->indicesFieldIDs[2][0];

  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  DomainT<1> B3_crd_domain = runtime->get_index_space_domain(ctx, B3_crd.get_index_space());
  DomainPointColoring B3_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t fposo = (*itr)[0];
    Point<1> B3CrdStart = Point<1>((fposo * ((B3Size + (pieces - 1)) / pieces)));
    Point<1> B3CrdEnd = Point<1>(TACO_MIN((fposo * ((B3Size + (pieces - 1)) / pieces) + ((B3Size + (pieces - 1)) / pieces - 1)),B3_crd_domain.bounds.hi[0]));
    Rect<1> B3CrdRect = Rect<1>(B3CrdStart, B3CrdEnd);
    if (!B3_crd_domain.contains(B3CrdRect.lo) || !B3_crd_domain.contains(B3CrdRect.hi)) {
      B3CrdRect = B3CrdRect.make_empty();
    }
    B3_crd_coloring[(*itr)] = B3CrdRect;
  }
  IndexPartition B3_crd_index_part = runtime->create_index_partition(ctx, B3_crd.get_index_space(), domain, B3_crd_coloring, LEGION_COMPUTE_KIND);
  Legion::LogicalPartition B3_crd_part = runtime->get_logical_partition(ctx, B3_crd, B3_crd_index_part);
  IndexPartition posSparsePartB3 = runtime->create_partition_by_preimage_range(
    ctx,
    B3_crd_index_part,
    B3_pos,
    B3_pos_parent,
    B3_indices_field_id_2_0,
    runtime->get_index_partition_color_space_name(ctx, B3_crd_index_part)
  );
  IndexPartition posIndexPartB3 = densifyPartition(ctx, runtime, get_index_space(B3_pos), posSparsePartB3);
  Legion::LogicalPartition posPartB3 = runtime->get_logical_partition(ctx, B3_pos, posIndexPartB3);
  Legion::LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B3_crd_part, B_vals);
  Legion::LogicalPartition crdPartB2 = copyPartition(ctx, runtime, posPartB3, B2_crd);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(
    ctx,
    crdPartB2.get_index_partition(),
    B2_pos,
    B2_pos_parent,
    B2_indices_field_id_1_0,
    runtime->get_index_partition_color_space_name(ctx, crdPartB2.get_index_partition())
  );
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  Legion::LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  IndexPartition BDenseRun0Partition = copyPartition(ctx, runtime, posPartB2, B_dense_run_0);
  IndexPartition ADenseRun0Partition = AffineProjection(0).apply(ctx, runtime, BDenseRun0Partition, A_dense_run_0);
  auto A_vals_partition = copyPartition(ctx, runtime, ADenseRun0Partition, get_logical_region(A_vals));
  auto computePartitions = partitionPackForcomputeLegion();
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = ADenseRun0Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(3);
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
void task_1DeviceKernel0(int64_t B3Size, int64_t* i_blockStarts, int64_t* j_blockStarts, int32_t pieces, AccessorRORect_1_1 B3_pos_accessor, AccessorRORect_1_1 B2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorROint32_t1 B3_crd_accessor, AccessorReduceNonExcldouble2 A_vals_red_accessor_non_excl, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble2 C_vals_ro_accessor, AccessorROdouble2 D_vals_ro_accessor, Legion::FieldID A_vals_field_id, Legion::FieldID B2_indices_field_id_1_0, Legion::FieldID B2_indices_field_id_1_1, Legion::FieldID B3_indices_field_id_2_0, Legion::FieldID B3_indices_field_id_2_1, Legion::FieldID B_vals_field_id, int64_t C2_dimension, Legion::FieldID C_vals_field_id, Legion::FieldID D_vals_field_id, int64_t fposo) {

  int64_t block = blockIdx.x;
  int64_t thread = (threadIdx.x % (32));
  int64_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int64_t pointID2 = fposo * (((B3Size + (pieces - 1)) / pieces + 2047) / 2048) + block;
  int64_t pointID3 = pointID2 * 8 + warp;
  int64_t pointID4 = pointID3 * 32 + thread;
  int64_t pB3_begin = j_blockStarts[block];
  int64_t pB3_end = j_blockStarts[(block + 1)];
  int64_t fposi2 = thread * 8;
  int64_t fposi1 = warp * 256 + fposi2;
  int64_t fposi = block * 2048 + fposi1;
  int64_t fposB = fposo * ((B3Size + (pieces - 1)) / pieces) + fposi;
  int64_t j_pos = taco_binarySearchBefore(B3_pos_accessor, pB3_begin, pB3_end, fposB);
  int64_t j = B2_crd_accessor[j_pos];
  int64_t pB2_begin = i_blockStarts[block];
  int64_t pB2_end = i_blockStarts[(block + 1)];
  int64_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, j_pos);
  int64_t i = i_pos;
  for (int64_t thread_nz = 0; thread_nz < 8; thread_nz++) {
    int64_t fposi2 = thread * 8 + thread_nz;
    int64_t fposi1 = warp * 256 + fposi2;
    int64_t fposi = block * 2048 + fposi1;
    int64_t fposB = fposo * ((B3Size + (pieces - 1)) / pieces) + fposi;
    if (fposB >= (fposo + 1) * ((B3Size + (pieces - 1)) / pieces))
      break;

    if (fposB >= B3Size)
      break;

    int64_t f2 = B3_crd_accessor[fposB];
    if (!(B3_pos_accessor[j_pos].contains(fposB))) {
      j_pos = j_pos + 1;
      j = B2_crd_accessor[j_pos];
      while (!(B2_pos_accessor[i_pos].contains(j_pos))) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
    }
    for (int64_t l = 0; l < C2_dimension; l++) {
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

  int64_t fposo = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  Legion::FieldID B2_indices_field_id_1_0 = args->B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1 = args->B2_indices_field_id_1_1;
  int64_t B3Size = args->B3Size;
  Legion::FieldID B3_indices_field_id_2_0 = args->B3_indices_field_id_2_0;
  Legion::FieldID B3_indices_field_id_2_1 = args->B3_indices_field_id_2_1;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  int64_t C2_dimension = args->C2_dimension;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  Legion::FieldID D_vals_field_id = args->D_vals_field_id;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, B_vals_field_id);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, C_vals_field_id);
  auto D_vals_ro_accessor = createAccessor<AccessorROdouble2>(D_vals, D_vals_field_id);
  auto A_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble2>(A_vals, A_vals_field_id, LEGION_REDOP_SUM_FLOAT64);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, B2_indices_field_id_1_0);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, B2_indices_field_id_1_1);
  auto B3_pos_accessor = createAccessor<AccessorRORect_1_1>(B3_pos, B3_indices_field_id_2_0);
  auto B3_crd_accessor = createAccessor<AccessorROint32_t1>(B3_crd, B3_indices_field_id_2_1);

  if (runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).empty())
    return ;

  DomainT<1> B3PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B3_pos));
  DomainT<1> B3CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B3_crd));
  Legion::DeferredBuffer<int64_t, 1> buf = Legion::DeferredBuffer<int64_t, 1>(Rect<1>(0, (((B3Size + (pieces - 1)) / pieces + 2047) / 2048)), Legion::Memory::Kind::GPU_FB_MEM);
  int64_t* j_blockStarts = buf.ptr(0);
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
  Legion::DeferredBuffer<int64_t, 1> buf0 = Legion::DeferredBuffer<int64_t, 1>(Rect<1>(0, (((B3Size + (pieces - 1)) / pieces + 2047) / 2048)), Legion::Memory::Kind::GPU_FB_MEM);
  int64_t* i_blockStarts = buf0.ptr(0);
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
    task_1DeviceKernel0<<<(((B3Size + (pieces - 1)) / pieces + 2047) / 2048), (32 * 8)>>>(B3Size, i_blockStarts, j_blockStarts, pieces, B3_pos_accessor, B2_pos_accessor, B2_crd_accessor, B3_crd_accessor, A_vals_red_accessor_non_excl, B_vals_ro_accessor, C_vals_ro_accessor, D_vals_ro_accessor, A_vals_field_id, B2_indices_field_id_1_0, B2_indices_field_id_1_1, B3_indices_field_id_2_0, B3_indices_field_id_2_1, B_vals_field_id, C2_dimension, C_vals_field_id, D_vals_field_id, fposo);
  }
}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, partitionPackForcomputeLegion* partitionPack, int32_t pieces) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B3_pos_parent = B->indicesParents[2][0];
  auto B3_crd_parent = B->indicesParents[2][1];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B2_indices_field_id_1_1 = B->indicesFieldIDs[1][1];
  auto B3_indices_field_id_2_0 = B->indicesFieldIDs[2][0];
  auto B3_indices_field_id_2_1 = B->indicesFieldIDs[2][1];
  int C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;
  RegionWrapper D_vals = D->vals;
  auto D_vals_parent = D->valsParent;
  auto D_vals_field_id = D->valsFieldID;

  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.A_vals_field_id = A_vals_field_id;
  taskArgsRaw1.B2_indices_field_id_1_0 = B2_indices_field_id_1_0;
  taskArgsRaw1.B2_indices_field_id_1_1 = B2_indices_field_id_1_1;
  taskArgsRaw1.B3Size = B3Size;
  taskArgsRaw1.B3_indices_field_id_2_0 = B3_indices_field_id_2_0;
  taskArgsRaw1.B3_indices_field_id_2_1 = B3_indices_field_id_2_1;
  taskArgsRaw1.B_vals_field_id = B_vals_field_id;
  taskArgsRaw1.C2_dimension = C2_dimension;
  taskArgsRaw1.C_vals_field_id = C_vals_field_id;
  taskArgsRaw1.D_vals_field_id = D_vals_field_id;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, A_vals_parent).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(B2_indices_field_id_1_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(B2_indices_field_id_1_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_pos_parent)).add_field(B3_indices_field_id_2_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_crd_parent)).add_field(B3_indices_field_id_2_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(C_vals), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(C_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(D_vals), READ_ONLY, EXCLUSIVE, D_vals_parent).add_field(D_vals_field_id));
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
