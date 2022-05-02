#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef ReductionAccessor<SumReduction<double>,false,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorReduceNonExcldouble1;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int64_t B1_dimension;
  Legion::FieldID B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID a_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t pieces;
};

struct task_2Args {
  int64_t B2Size;
  Legion::FieldID B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID a_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t pieces;
};

struct task_3Args {
  Legion::FieldID B1_indices_field_id_0_1;
  int64_t B2Size;
  Legion::FieldID B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID a_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t pieces;
};

struct task_4Args {
  Legion::FieldID B1_indices_field_id_0_0;
  Legion::FieldID B1_indices_field_id_0_1;
  int64_t B2_dimension;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID a_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t pieces;
};

struct task_5Args {
  Legion::FieldID B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID a_vals_field_id;
  Legion::FieldID c1_indices_field_id_0_0;
  Legion::FieldID c1_indices_field_id_0_1;
  Legion::FieldID c_vals_field_id;
  int32_t pieces;
};


partitionPackForcomputeLegionRowSplit partitionForcomputeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];

  auto computePartitions = partitionPackForcomputeLegionRowSplit();


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t io = (*itr)[0];
    Point<1> BStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<1> aStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> aEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), aDomain.hi()[0]));
    Rect<1> aRect = Rect<1>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  IndexPartition B_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  Legion::LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition_0, B2_pos);
  Legion::LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B2_crd.get_index_space(), posPartB2, B2_pos_parent, B2_indices_field_id_1_0));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  IndexPartition a_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, a_dense_run_0, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto a_vals_partition = copyPartition(ctx, runtime, a_dense_run_0_Partition_0, get_logical_region(a_vals));
  computePartitions.aPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(1);
  computePartitions.aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.aPartition.valsPartition = a_vals_partition;
  computePartitions.aPartition.denseLevelRunPartitions[0] = a_dense_run_0_Partition_0;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_0;

  return computePartitions;
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
  int64_t B1_dimension = args->B1_dimension;
  Legion::FieldID B2_indices_field_id_1_0 = args->B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1 = args->B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, B_vals_field_id);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, c_vals_field_id);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble1>(a_vals, a_vals_field_id);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, B2_indices_field_id_1_0);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, B2_indices_field_id_1_1);

  #pragma omp parallel for schedule(dynamic, 128)
  for (int64_t ii = 0; ii < ((B1_dimension + (pieces - 1)) / pieces); ii++) {
    int64_t i = io * ((B1_dimension + (pieces - 1)) / pieces) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (pieces - 1)) / pieces))
      continue;

    for (int64_t jB = B2_pos_accessor[Point<1>(i)].lo; jB < (B2_pos_accessor[Point<1>(i)].hi + 1); jB++) {
      int64_t j = B2_crd_accessor[jB];
      a_vals_rw_accessor[Point<1>(i)] = a_vals_rw_accessor[Point<1>(i)] + B_vals_ro_accessor[Point<1>(jB)] * c_vals_ro_accessor[Point<1>(j)];
    }
  }
}

void computeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionRowSplit* partitionPack, int32_t pieces) {
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;
  size_t B1_dimension = B->dims[0];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B2_indices_field_id_1_1 = B->indicesFieldIDs[1][1];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.B1_dimension = B1_dimension;
  taskArgsRaw1.B2_indices_field_id_1_0 = B2_indices_field_id_1_0;
  taskArgsRaw1.B2_indices_field_id_1_1 = B2_indices_field_id_1_1;
  taskArgsRaw1.B_vals_field_id = B_vals_field_id;
  taskArgsRaw1.a_vals_field_id = a_vals_field_id;
  taskArgsRaw1.c_vals_field_id = c_vals_field_id;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, a_vals_parent).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(B2_indices_field_id_1_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(B2_indices_field_id_1_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(c_vals_field_id));
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
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];

  auto computePartitions = partitionPackForcomputeLegionPosSplit();

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
    Point<1> B2CrdEnd = Point<1>(TACO_MIN((fposo * ((B2Size + (pieces - 1)) / pieces) + ((B2Size + (pieces - 1)) / pieces - 1)), B2_crd_domain.bounds.hi[0]));
    Rect<1> B2CrdRect = Rect<1>(B2CrdStart, B2CrdEnd);
    if (!B2_crd_domain.contains(B2CrdRect.lo) || !B2_crd_domain.contains(B2CrdRect.hi)) {
      B2CrdRect = B2CrdRect.make_empty();
    }
    B2_crd_coloring[(*itr)] = B2CrdRect;
  }
  IndexPartition B2_crd_index_part = runtime->create_index_partition(ctx, B2_crd.get_index_space(), domain, B2_crd_coloring, LEGION_COMPUTE_KIND);
  Legion::LogicalPartition B2_crd_part = runtime->get_logical_partition(ctx, B2_crd, B2_crd_index_part);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(
    ctx,
    B2_crd_index_part,
    B2_pos,
    B2_pos_parent,
    B2_indices_field_id_1_0,
    runtime->get_index_partition_color_space_name(ctx, B2_crd_index_part),
    LEGION_ALIASED_INCOMPLETE_KIND
  );
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  Legion::LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  Legion::LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B2_crd_part, B_vals);
  IndexPartition BDenseRun0Partition = copyPartition(ctx, runtime, posPartB2, B_dense_run_0);
  IndexPartition aDenseRun0Partition = AffineProjection(0).apply(ctx, runtime, BDenseRun0Partition, a_dense_run_0);
  auto a_vals_partition = copyPartition(ctx, runtime, aDenseRun0Partition, get_logical_region(a_vals));
  computePartitions.aPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(1);
  computePartitions.aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.aPartition.valsPartition = a_vals_partition;
  computePartitions.aPartition.denseLevelRunPartitions[0] = aDenseRun0Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(B2_crd_part);
  computePartitions.BPartition.valsPartition = BValsLogicalPart;
  computePartitions.BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;

  return computePartitions;
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
  Legion::FieldID B2_indices_field_id_1_0 = args->B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1 = args->B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, B_vals_field_id);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, c_vals_field_id);
  auto a_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(a_vals, a_vals_field_id, LEGION_REDOP_SUM_FLOAT64);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, B2_indices_field_id_1_0);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, B2_indices_field_id_1_1);

  if (runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  int64_t pB2_begin = B2PosDomain.bounds.lo;
  int64_t pB2_end = B2PosDomain.bounds.hi;
  #pragma omp parallel for schedule(static)
  for (int64_t fposio = 0; fposio < (((B2Size + (pieces - 1)) / pieces + 2047) / 2048); fposio++) {
    int64_t fposi = fposio * 2048;
    int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    int64_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
    int64_t i = i_pos;
    for (int64_t fposii = 0; fposii < 2048; fposii++) {
      int64_t fposi = fposio * 2048 + fposii;
      int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
      if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
        continue;

      if (fposB >= B2Size)
        continue;

      int64_t f = B2_crd_accessor[fposB];
      while (!(B2_pos_accessor[i_pos].contains(fposB))) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      a_vals_red_accessor_non_excl[Point<1>(i)] <<= B_vals_ro_accessor[Point<1>(fposB)] * c_vals_ro_accessor[Point<1>(f)];
    }
  }
}

void computeLegionPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplit* partitionPack, int32_t pieces) {
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B2_indices_field_id_1_1 = B->indicesFieldIDs[1][1];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.B2Size = B2Size;
  taskArgsRaw2.B2_indices_field_id_1_0 = B2_indices_field_id_1_0;
  taskArgsRaw2.B2_indices_field_id_1_1 = B2_indices_field_id_1_1;
  taskArgsRaw2.B_vals_field_id = B_vals_field_id;
  taskArgsRaw2.a_vals_field_id = a_vals_field_id;
  taskArgsRaw2.c_vals_field_id = c_vals_field_id;
  taskArgsRaw2.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, a_vals_parent).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(B2_indices_field_id_1_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(B2_indices_field_id_1_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(c_vals_field_id));
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
  auto B1_indices_field_id_0_0 = B->indicesFieldIDs[0][0];
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];

  auto computePartitions = partitionPackForcomputeLegionPosSplitDCSR();

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
    Point<1> B2CrdEnd = Point<1>(TACO_MIN((fposo * ((B2Size + (pieces - 1)) / pieces) + ((B2Size + (pieces - 1)) / pieces - 1)), B2_crd_domain.bounds.hi[0]));
    Rect<1> B2CrdRect = Rect<1>(B2CrdStart, B2CrdEnd);
    if (!B2_crd_domain.contains(B2CrdRect.lo) || !B2_crd_domain.contains(B2CrdRect.hi)) {
      B2CrdRect = B2CrdRect.make_empty();
    }
    B2_crd_coloring[(*itr)] = B2CrdRect;
  }
  IndexPartition B2_crd_index_part = runtime->create_index_partition(ctx, B2_crd.get_index_space(), domain, B2_crd_coloring, LEGION_COMPUTE_KIND);
  Legion::LogicalPartition B2_crd_part = runtime->get_logical_partition(ctx, B2_crd, B2_crd_index_part);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(
    ctx,
    B2_crd_index_part,
    B2_pos,
    B2_pos_parent,
    B2_indices_field_id_1_0,
    runtime->get_index_partition_color_space_name(ctx, B2_crd_index_part),
    LEGION_ALIASED_INCOMPLETE_KIND
  );
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  Legion::LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  Legion::LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B2_crd_part, B_vals);
  Legion::LogicalPartition crdPartB1 = copyPartition(ctx, runtime, posPartB2, B1_crd);
  IndexPartition posSparsePartB1 = runtime->create_partition_by_preimage_range(
    ctx,
    crdPartB1.get_index_partition(),
    B1_pos,
    B1_pos_parent,
    B1_indices_field_id_0_0,
    runtime->get_index_partition_color_space_name(ctx, crdPartB1.get_index_partition()),
    LEGION_ALIASED_INCOMPLETE_KIND
  );
  IndexPartition posIndexPartB1 = densifyPartition(ctx, runtime, get_index_space(B1_pos), posSparsePartB1);
  Legion::LogicalPartition posPartB1 = runtime->get_logical_partition(ctx, B1_pos, posIndexPartB1);
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[0].push_back(posPartB1);
  computePartitions.BPartition.indicesPartitions[0].push_back(crdPartB1);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(B2_crd_part);
  computePartitions.BPartition.valsPartition = BValsLogicalPart;

  return computePartitions;
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
  Legion::FieldID B1_indices_field_id_0_1 = args->B1_indices_field_id_0_1;
  int64_t B2Size = args->B2Size;
  Legion::FieldID B2_indices_field_id_1_0 = args->B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1 = args->B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, B_vals_field_id);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, c_vals_field_id);
  auto a_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(a_vals, a_vals_field_id, LEGION_REDOP_SUM_FLOAT64);
  auto B1_crd_accessor = createAccessor<AccessorROint32_t1>(B1_crd, B1_indices_field_id_0_1);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, B2_indices_field_id_1_0);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, B2_indices_field_id_1_1);

  if (runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  int64_t pB2_begin = B2PosDomain.bounds.lo;
  int64_t pB2_end = B2PosDomain.bounds.hi;
  #pragma omp parallel for schedule(static)
  for (int64_t fposio = 0; fposio < (((B2Size + (pieces - 1)) / pieces + 2047) / 2048); fposio++) {
    int64_t fposi = fposio * 2048;
    int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    int64_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
    int64_t i = B1_crd_accessor[i_pos];
    for (int64_t fposii = 0; fposii < 2048; fposii++) {
      int64_t fposi = fposio * 2048 + fposii;
      int64_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
      if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
        continue;

      if (fposB >= B2Size)
        continue;

      int64_t f = B2_crd_accessor[fposB];
      if (!(B2_pos_accessor[i_pos].contains(fposB))) {
        i_pos = i_pos + 1;
        i = B1_crd_accessor[i_pos];
      }
      a_vals_red_accessor_non_excl[Point<1>(i)] <<= B_vals_ro_accessor[Point<1>(fposB)] * c_vals_ro_accessor[Point<1>(f)];
    }
  }
}

void computeLegionPosSplitDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplitDCSR* partitionPack, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;
  RegionWrapper B2_crd = B->indices[1][1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B1_crd_parent = B->indicesParents[0][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B1_indices_field_id_0_0 = B->indicesFieldIDs[0][0];
  auto B1_indices_field_id_0_1 = B->indicesFieldIDs[0][1];
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B2_indices_field_id_1_1 = B->indicesFieldIDs[1][1];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_3Args taskArgsRaw3;
  taskArgsRaw3.B1_indices_field_id_0_1 = B1_indices_field_id_0_1;
  taskArgsRaw3.B2Size = B2Size;
  taskArgsRaw3.B2_indices_field_id_1_0 = B2_indices_field_id_1_0;
  taskArgsRaw3.B2_indices_field_id_1_1 = B2_indices_field_id_1_1;
  taskArgsRaw3.B_vals_field_id = B_vals_field_id;
  taskArgsRaw3.a_vals_field_id = a_vals_field_id;
  taskArgsRaw3.c_vals_field_id = c_vals_field_id;
  taskArgsRaw3.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, a_vals_parent).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[0][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B1_pos_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(B1_indices_field_id_0_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[0][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B1_crd_parent)).add_field(B1_indices_field_id_0_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(B2_indices_field_id_1_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(B2_indices_field_id_1_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(c_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionSparseDensePosParallelize partitionForcomputeLegionSparseDensePosParallelize(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];
  RegionWrapper B1_pos = B->indices[0][0];
  RegionWrapper B1_crd = B->indices[0][1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B1_crd_parent = B->indicesParents[0][1];
  RegionWrapper B_vals = B->vals;
  auto B1_indices_field_id_0_0 = B->indicesFieldIDs[0][0];
  auto B1_indices_field_id_0_1 = B->indicesFieldIDs[0][1];
  auto B1_pos_accessor = createAccessor<AccessorRORect_1_1>(B1_pos, B1_indices_field_id_0_0);

  B1_pos = legionMalloc(ctx, runtime, B1_pos, B1_pos_parent, B1_indices_field_id_0_0, READ_ONLY);
  B1_pos_accessor = createAccessor<AccessorRORect_1_1>(B1_pos, B1_indices_field_id_0_0);

  auto computePartitions = partitionPackForcomputeLegionSparseDensePosParallelize();


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  DomainT<1> B1_crd_domain = runtime->get_index_space_domain(ctx, B1_crd.get_index_space());
  DomainPointColoring B1_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t fposo = (*itr)[0];
    Point<1> B1CrdStart = Point<1>((fposo * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + B1_pos_accessor[Point<1>(0)].lo));
    Point<1> B1CrdEnd = Point<1>(TACO_MIN(((fposo * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces - 1)) + B1_pos_accessor[Point<1>(0)].lo), B1_crd_domain.bounds.hi[0]));
    Rect<1> B1CrdRect = Rect<1>(B1CrdStart, B1CrdEnd);
    if (!B1_crd_domain.contains(B1CrdRect.lo) || !B1_crd_domain.contains(B1CrdRect.hi)) {
      B1CrdRect = B1CrdRect.make_empty();
    }
    B1_crd_coloring[(*itr)] = B1CrdRect;
  }
  IndexPartition B1_crd_index_part = runtime->create_index_partition(ctx, B1_crd.get_index_space(), domain, B1_crd_coloring, LEGION_COMPUTE_KIND);
  Legion::LogicalPartition B1_crd_part = runtime->get_logical_partition(ctx, B1_crd, B1_crd_index_part);
  IndexPartition posSparsePartB1 = runtime->create_partition_by_preimage_range(
    ctx,
    B1_crd_index_part,
    B1_pos,
    B1_pos_parent,
    B1_indices_field_id_0_0,
    runtime->get_index_partition_color_space_name(ctx, B1_crd_index_part),
    LEGION_ALIASED_INCOMPLETE_KIND
  );
  IndexPartition posIndexPartB1 = densifyPartition(ctx, runtime, get_index_space(B1_pos), posSparsePartB1);
  Legion::LogicalPartition posPartB1 = runtime->get_logical_partition(ctx, B1_pos, posIndexPartB1);
  Legion::LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B1_crd_part, B_vals);
  IndexPartition aDenseRun0Partition = SparseGatherProjection(0).apply(ctx, runtime, B1_crd_parent, B1_crd_part, B1_indices_field_id_0_1, a_dense_run_0);
  auto a_vals_partition = copyPartition(ctx, runtime, aDenseRun0Partition, get_logical_region(a_vals));
  computePartitions.aPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(1);
  computePartitions.aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.aPartition.valsPartition = a_vals_partition;
  computePartitions.aPartition.denseLevelRunPartitions[0] = aDenseRun0Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[0].push_back(posPartB1);
  computePartitions.BPartition.indicesPartitions[0].push_back(B1_crd_part);
  computePartitions.BPartition.valsPartition = BValsLogicalPart;

  runtime->unmap_region(ctx, B1_pos);

  return computePartitions;
}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B1_pos = regions[1];
  LogicalRegion B1_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B1_crd = regions[2];
  LogicalRegion B1_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B_vals = regions[3];
  LogicalRegion B_vals_parent = regions[3].get_logical_region();
  PhysicalRegion c_vals = regions[4];
  LogicalRegion c_vals_parent = regions[4].get_logical_region();

  int64_t fposo = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  Legion::FieldID B1_indices_field_id_0_0 = args->B1_indices_field_id_0_0;
  Legion::FieldID B1_indices_field_id_0_1 = args->B1_indices_field_id_0_1;
  int64_t B2_dimension = args->B2_dimension;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t pieces = args->pieces;

  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, c_vals_field_id);
  auto B_vals_ro_accessor = createAccessor<AccessorROdouble2>(B_vals, B_vals_field_id);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble1>(a_vals, a_vals_field_id);
  auto B1_pos_accessor = createAccessor<AccessorRORect_1_1>(B1_pos, B1_indices_field_id_0_0);
  auto B1_crd_accessor = createAccessor<AccessorROint32_t1>(B1_crd, B1_indices_field_id_0_1);

  if (runtime->get_index_space_domain(ctx, get_index_space(B1_crd)).empty())
    return ;

  for (int64_t fposi = 0; fposi < ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces); fposi++) {
    int64_t fposB = (fposo * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + fposi) + B1_pos_accessor[Point<1>(0)].lo;
    if (fposB >= (fposo + 1) * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + B1_pos_accessor[Point<1>(0)].lo)
      continue;

    if (fposB < B1_pos_accessor[Point<1>(0)].lo || fposB >= B1_pos_accessor[Point<1>(0)].hi + 1)
      continue;

    int64_t i = B1_crd_accessor[fposB];
    for (int64_t j = 0; j < B2_dimension; j++) {
      a_vals_rw_accessor[Point<1>(i)] = a_vals_rw_accessor[Point<1>(i)] + B_vals_ro_accessor[Point<2>(fposB, j)] * c_vals_ro_accessor[Point<1>(j)];
    }
  }
}

void computeLegionSparseDensePosParallelize(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionSparseDensePosParallelize* partitionPack, int32_t pieces) {
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;
  size_t B2_dimension = B->dims[1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B1_crd_parent = B->indicesParents[0][1];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B1_indices_field_id_0_0 = B->indicesFieldIDs[0][0];
  auto B1_indices_field_id_0_1 = B->indicesFieldIDs[0][1];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_4Args taskArgsRaw4;
  taskArgsRaw4.B1_indices_field_id_0_0 = B1_indices_field_id_0_0;
  taskArgsRaw4.B1_indices_field_id_0_1 = B1_indices_field_id_0_1;
  taskArgsRaw4.B2_dimension = B2_dimension;
  taskArgsRaw4.B_vals_field_id = B_vals_field_id;
  taskArgsRaw4.a_vals_field_id = a_vals_field_id;
  taskArgsRaw4.c_vals_field_id = c_vals_field_id;
  taskArgsRaw4.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw4, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, a_vals_parent).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[0][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B1_pos_parent)).add_field(B1_indices_field_id_0_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[0][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B1_crd_parent)).add_field(B1_indices_field_id_0_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(c_vals_field_id));
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionCSCMSpV partitionForcomputeLegionCSCMSpV(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  RegionWrapper c1_pos = c->indices[0][0];
  RegionWrapper c1_crd = c->indices[0][1];
  auto c1_pos_parent = c->indicesParents[0][0];
  auto c1_crd_parent = c->indicesParents[0][1];
  RegionWrapper c_vals = c->vals;
  auto c1_indices_field_id_0_0 = c->indicesFieldIDs[0][0];
  auto c1_indices_field_id_0_1 = c->indicesFieldIDs[0][1];
  auto c1_pos_accessor = createAccessor<AccessorRORect_1_1>(c1_pos, c1_indices_field_id_0_0);

  c1_pos = legionMalloc(ctx, runtime, c1_pos, c1_pos_parent, c1_indices_field_id_0_0, READ_ONLY);
  c1_pos_accessor = createAccessor<AccessorRORect_1_1>(c1_pos, c1_indices_field_id_0_0);

  auto computePartitions = partitionPackForcomputeLegionCSCMSpV();


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto jposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(jposoIndexSpace));
  DomainT<1> c1_crd_domain = runtime->get_index_space_domain(ctx, c1_crd.get_index_space());
  DomainPointColoring c1_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t jposo = (*itr)[0];
    Point<1> c1CrdStart = Point<1>((jposo * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + c1_pos_accessor[Point<1>(0)].lo));
    Point<1> c1CrdEnd = Point<1>(TACO_MIN(((jposo * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces - 1)) + c1_pos_accessor[Point<1>(0)].lo), c1_crd_domain.bounds.hi[0]));
    Rect<1> c1CrdRect = Rect<1>(c1CrdStart, c1CrdEnd);
    if (!c1_crd_domain.contains(c1CrdRect.lo) || !c1_crd_domain.contains(c1CrdRect.hi)) {
      c1CrdRect = c1CrdRect.make_empty();
    }
    c1_crd_coloring[(*itr)] = c1CrdRect;
  }
  IndexPartition c1_crd_index_part = runtime->create_index_partition(ctx, c1_crd.get_index_space(), domain, c1_crd_coloring, LEGION_COMPUTE_KIND);
  Legion::LogicalPartition c1_crd_part = runtime->get_logical_partition(ctx, c1_crd, c1_crd_index_part);
  IndexPartition posSparsePartc1 = runtime->create_partition_by_preimage_range(
    ctx,
    c1_crd_index_part,
    c1_pos,
    c1_pos_parent,
    c1_indices_field_id_0_0,
    runtime->get_index_partition_color_space_name(ctx, c1_crd_index_part),
    LEGION_ALIASED_INCOMPLETE_KIND
  );
  IndexPartition posIndexPartc1 = densifyPartition(ctx, runtime, get_index_space(c1_pos), posSparsePartc1);
  Legion::LogicalPartition posPartc1 = runtime->get_logical_partition(ctx, c1_pos, posIndexPartc1);
  Legion::LogicalPartition cValsLogicalPart = copyPartition(ctx, runtime, c1_crd_part, c_vals);
  IndexPartition BDenseRun0Partition = SparseGatherProjection(0).apply(ctx, runtime, c1_crd_parent, c1_crd_part, c1_indices_field_id_0_1, B_dense_run_0);
  Legion::LogicalPartition posPartB2 = copyPartition(ctx, runtime, BDenseRun0Partition, B2_pos);
  Legion::LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B2_crd.get_index_space(), posPartB2, B2_pos_parent, B2_indices_field_id_1_0));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;
  computePartitions.cPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(1);
  computePartitions.cPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.cPartition.indicesPartitions[0].push_back(posPartc1);
  computePartitions.cPartition.indicesPartitions[0].push_back(c1_crd_part);
  computePartitions.cPartition.valsPartition = cValsLogicalPart;

  runtime->unmap_region(ctx, c1_pos);

  return computePartitions;
}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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
  task_5Args* args = (task_5Args*)(task->args);
  Legion::FieldID B2_indices_field_id_1_0 = args->B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1 = args->B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID c1_indices_field_id_0_0 = args->c1_indices_field_id_0_0;
  Legion::FieldID c1_indices_field_id_0_1 = args->c1_indices_field_id_0_1;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, B_vals_field_id);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, c_vals_field_id);
  auto a_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(a_vals, a_vals_field_id, LEGION_REDOP_SUM_FLOAT64);
  auto c1_pos_accessor = createAccessor<AccessorRORect_1_1>(c1_pos, c1_indices_field_id_0_0);
  auto c1_crd_accessor = createAccessor<AccessorROint32_t1>(c1_crd, c1_indices_field_id_0_1);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, B2_indices_field_id_1_0);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, B2_indices_field_id_1_1);

  if (runtime->get_index_space_domain(ctx, get_index_space(c1_crd)).empty())
    return ;

  #pragma omp parallel for schedule(dynamic, 128)
  for (int64_t jposi = 0; jposi < ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces); jposi++) {
    int64_t jposc = (jposo * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + jposi) + c1_pos_accessor[Point<1>(0)].lo;
    if (jposc >= (jposo + 1) * ((((c1_pos_accessor[Point<1>(0)].hi + 1) - c1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + c1_pos_accessor[Point<1>(0)].lo)
      continue;

    if (jposc < c1_pos_accessor[Point<1>(0)].lo || jposc >= c1_pos_accessor[Point<1>(0)].hi + 1)
      continue;

    int64_t j = c1_crd_accessor[jposc];
    for (int64_t iB = B2_pos_accessor[Point<1>(j)].lo; iB < (B2_pos_accessor[Point<1>(j)].hi + 1); iB++) {
      int64_t i = B2_crd_accessor[iB];
      a_vals_red_accessor_non_excl[Point<1>(i)] <<= B_vals_ro_accessor[Point<1>(iB)] * c_vals_ro_accessor[Point<1>(jposc)];
    }
  }
}

void computeLegionCSCMSpV(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionCSCMSpV* partitionPack, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B2_indices_field_id_1_1 = B->indicesFieldIDs[1][1];
  auto c1_pos_parent = c->indicesParents[0][0];
  auto c1_crd_parent = c->indicesParents[0][1];
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;
  auto c1_indices_field_id_0_0 = c->indicesFieldIDs[0][0];
  auto c1_indices_field_id_0_1 = c->indicesFieldIDs[0][1];


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto jposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(jposoIndexSpace));
  task_5Args taskArgsRaw5;
  taskArgsRaw5.B2_indices_field_id_1_0 = B2_indices_field_id_1_0;
  taskArgsRaw5.B2_indices_field_id_1_1 = B2_indices_field_id_1_1;
  taskArgsRaw5.B_vals_field_id = B_vals_field_id;
  taskArgsRaw5.a_vals_field_id = a_vals_field_id;
  taskArgsRaw5.c1_indices_field_id_0_0 = c1_indices_field_id_0_0;
  taskArgsRaw5.c1_indices_field_id_0_1 = c1_indices_field_id_0_1;
  taskArgsRaw5.c_vals_field_id = c_vals_field_id;
  taskArgsRaw5.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw5, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, a_vals_parent).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(B2_indices_field_id_1_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(B2_indices_field_id_1_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.indicesPartitions[0][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(c1_pos_parent)).add_field(c1_indices_field_id_0_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.indicesPartitions[0][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(c1_crd_parent)).add_field(c1_indices_field_id_0_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(c_vals_field_id));
  runtime->execute_index_space(ctx, launcher);

}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_3>(registrar, "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(4), "task_4");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
  {
    TaskVariantRegistrar registrar(taskID(5), "task_5");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_5>(registrar, "task_5");
  }
}
