#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef ReductionAccessor<SumReduction<double>,true,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorReducedouble1;
typedef ReductionAccessor<SumReduction<double>,false,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorReduceNonExcldouble1;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int32_t B1_dimension;
  int32_t a1_dimension;
  int32_t c1_dimension;
  int32_t pieces;
};

struct task_2Args {
  int64_t B2Size;
  int32_t a1_dimension;
  int32_t c1_dimension;
  int32_t pieces;
};

struct task_3Args {
  int64_t B2Size;
  int32_t a1_dimension;
  int32_t c1_dimension;
  int32_t pieces;
};

struct task_4Args {
  int32_t B2_dimension;
  int32_t a1_dimension;
  int32_t c1_dimension;
  int32_t pieces;
};


partitionPackForcomputeLegionRowSplit* partitionForcomputeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];
  int B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  IndexSpace c_dense_run_0 = c->denseLevelRuns[0];

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t io = (*itr)[0];
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
  auto a_dense_run_0_Partition = runtime->create_index_partition(ctx, a_dense_run_0, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto a_vals_partition = copyPartition(ctx, runtime, a_dense_run_0_Partition, get_logical_region(a_vals));
  auto computePartitions = new(partitionPackForcomputeLegionRowSplit);
  computePartitions->aPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(1);
  computePartitions->aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions->aPartition.valsPartition = a_vals_partition;
  computePartitions->aPartition.denseLevelRunPartitions[0] = a_dense_run_0_Partition;
  computePartitions->BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions->BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions->BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions->BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions->BPartition.valsPartition = B_vals_partition;
  computePartitions->BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition;
  return computePartitions;
}

void task_1(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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
  int32_t a1_dimension = args->a1_dimension;
  int32_t c1_dimension = args->c1_dimension;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble1>(a_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  int64_t pointID1 = io;
  #pragma omp parallel for schedule(static)
  for (int32_t ii = 0; ii < ((B1_dimension + (pieces - 1)) / pieces); ii++) {
    int32_t i = io * ((B1_dimension + (pieces - 1)) / pieces) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (pieces - 1)) / pieces))
      continue;

    int64_t pointID2 = pointID1 * ((B1_dimension + (pieces - 1)) / pieces) + ii;
    int32_t ia = 0 * a1_dimension + i;
    int32_t iB = 0 * B1_dimension + i;
    for (int32_t jB = B2_pos_accessor[Point<1>(i)].lo; jB < (B2_pos_accessor[Point<1>(i)].hi + 1); jB++) {
      int32_t j = B2_crd_accessor[(jB * 1)];
      int32_t jc = 0 * c1_dimension + j;
      a_vals_rw_accessor[Point<1>(i)] = a_vals_rw_accessor[Point<1>(i)] + B_vals_ro_accessor[Point<1>(jB)] * c_vals_ro_accessor[Point<1>(j)];
    }
  }
}

void computeLegionRowSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionRowSplit* partitionPack, int32_t pieces) {
  int a1_dimension = a->dims[0];
  auto a_vals_parent = a->valsParent;
  int B1_dimension = B->dims[0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  int c1_dimension = c->dims[0];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.a1_dimension = a1_dimension;
  taskArgsRaw.c1_dimension = c1_dimension;
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

partitionPackForcomputeLegionPosSplit* partitionForcomputeLegionPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
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
    Point<1> B2CrdEnd = Point<1>(TACO_MIN((fposo * ((B2Size + (pieces - 1)) / pieces) + ((B2Size + (pieces - 1)) / pieces - 1)), B2_crd_domain.bounds.hi[0]));
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
  auto computePartitions = new(partitionPackForcomputeLegionPosSplit);
  computePartitions->aPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(1);
  computePartitions->aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions->aPartition.valsPartition = a_vals_partition;
  computePartitions->aPartition.denseLevelRunPartitions[0] = aDenseRun0Partition;
  computePartitions->BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions->BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions->BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions->BPartition.indicesPartitions[1].push_back(B2_crd_part);
  computePartitions->BPartition.valsPartition = BValsLogicalPart;
  computePartitions->BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;
  return computePartitions;
}

void task_2(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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
  int32_t a1_dimension = args->a1_dimension;
  int32_t c1_dimension = args->c1_dimension;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(a_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  DomainT<1> BValsDomain = runtime->get_index_space_domain(ctx, get_index_space(B_vals));
  if (BValsDomain.empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  int32_t pB2_begin = B2PosDomain.bounds.lo;
  int32_t pB2_end = B2PosDomain.bounds.hi;
  int64_t pointID1 = fposo;
  #pragma omp parallel for schedule(static)
  for (int32_t fposio = 0; fposio < (((B2Size + (pieces - 1)) / pieces + 2047) / 2048); fposio++) {
    int64_t pointID2 = pointID1 * (((B2Size + (pieces - 1)) / pieces + 2047) / 2048) + fposio;
    int32_t fposi = fposio * 2048;
    int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    int32_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
    int32_t i = i_pos;
    for (int32_t fposii = 0; fposii < 2048; fposii++) {
      int32_t fposi = fposio * 2048 + fposii;
      int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
      if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
        continue;

      if (fposB >= B2Size)
        continue;

      int32_t f = B2_crd_accessor[fposB];
      int32_t j = f;
      while (!(B2_pos_accessor[i_pos].contains(fposB))) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t ia = 0 * a1_dimension + i;
      int32_t jc = 0 * c1_dimension + j;
      a_vals_red_accessor_non_excl[Point<1>(i)] <<= B_vals_ro_accessor[Point<1>(fposB)] * c_vals_ro_accessor[Point<1>(j)];
    }
  }
}

void computeLegionPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplit* partitionPack, int32_t pieces) {
  int a1_dimension = a->dims[0];
  auto a_vals_parent = a->valsParent;
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  int c1_dimension = c->dims[0];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_2Args taskArgsRaw;
  taskArgsRaw.B2Size = B2Size;
  taskArgsRaw.a1_dimension = a1_dimension;
  taskArgsRaw.c1_dimension = c1_dimension;
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

partitionPackForcomputeLegionPosSplitDCSR* partitionForcomputeLegionPosSplitDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper B1_pos = B->indices[0][0];
  RegionWrapper B1_crd = B->indices[0][1];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;

  int64_t B1Size = runtime->get_index_space_domain(ctx, get_index_space(B1_crd)).hi()[0] + 1;
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
    Point<1> B2CrdEnd = Point<1>(TACO_MIN((fposo * ((B2Size + (pieces - 1)) / pieces) + ((B2Size + (pieces - 1)) / pieces - 1)), B2_crd_domain.bounds.hi[0]));
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
  auto computePartitions = new(partitionPackForcomputeLegionPosSplitDCSR);
  computePartitions->BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions->BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions->BPartition.indicesPartitions[0].push_back(posPartB1);
  computePartitions->BPartition.indicesPartitions[0].push_back(crdPartB1);
  computePartitions->BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions->BPartition.indicesPartitions[1].push_back(B2_crd_part);
  computePartitions->BPartition.valsPartition = BValsLogicalPart;
  return computePartitions;
}

void task_3(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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
  int32_t a1_dimension = args->a1_dimension;
  int32_t c1_dimension = args->c1_dimension;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(a_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B1_crd_accessor = createAccessor<AccessorROint32_t1>(B1_crd, FID_COORD);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  DomainT<1> BValsDomain = runtime->get_index_space_domain(ctx, get_index_space(B_vals));
  if (BValsDomain.empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  int32_t pB2_begin = B2PosDomain.bounds.lo;
  int32_t pB2_end = B2PosDomain.bounds.hi;
  int64_t pointID1 = fposo;
  #pragma omp parallel for schedule(static)
  for (int32_t fposio = 0; fposio < (((B2Size + (pieces - 1)) / pieces + 2047) / 2048); fposio++) {
    int64_t pointID2 = pointID1 * (((B2Size + (pieces - 1)) / pieces + 2047) / 2048) + fposio;
    int32_t fposi = fposio * 2048;
    int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
    int32_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
    int32_t i = B1_crd_accessor[i_pos];
    for (int32_t fposii = 0; fposii < 2048; fposii++) {
      int32_t fposi = fposio * 2048 + fposii;
      int32_t fposB = fposo * ((B2Size + (pieces - 1)) / pieces) + fposi;
      if (fposB >= (fposo + 1) * ((B2Size + (pieces - 1)) / pieces))
        continue;

      if (fposB >= B2Size)
        continue;

      int32_t f = B2_crd_accessor[fposB];
      int32_t j = f;
      if (!(B2_pos_accessor[i_pos].contains(fposB))) {
        i_pos = i_pos + 1;
        i = B1_crd_accessor[i_pos];
      }
      int32_t iB = i_pos;
      int32_t ia = 0 * a1_dimension + i;
      int32_t jc = 0 * c1_dimension + j;
      a_vals_red_accessor_non_excl[Point<1>(i)] <<= B_vals_ro_accessor[Point<1>(fposB)] * c_vals_ro_accessor[Point<1>(j)];
    }
  }
}

void computeLegionPosSplitDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplitDCSR* partitionPack, int32_t pieces) {
  int a1_dimension = a->dims[0];
  RegionWrapper a_vals = a->vals;
  auto a_vals_parent = a->valsParent;
  RegionWrapper B1_crd = B->indices[0][1];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B1_crd_parent = B->indicesParents[0][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  int c1_dimension = c->dims[0];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t B1Size = runtime->get_index_space_domain(ctx, get_index_space(B1_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_3Args taskArgsRaw;
  taskArgsRaw.B2Size = B2Size;
  taskArgsRaw.a1_dimension = a1_dimension;
  taskArgsRaw.c1_dimension = c1_dimension;
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

partitionPackForcomputeLegionSparseDensePosParallelize* partitionForcomputeLegionSparseDensePosParallelize(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper B1_pos = B->indices[0][0];
  RegionWrapper B1_crd = B->indices[0][1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B1_crd_parent = B->indicesParents[0][1];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  auto B_vals_ro_accessor = createAccessor<AccessorROdouble2>(B_vals, FID_VAL);
  auto B1_pos_accessor = createAccessor<AccessorRORect_1_1>(B1_pos, FID_RECT_1);
  auto B1_crd_accessor = createAccessor<AccessorROint32_t1>(B1_crd, FID_COORD);
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);

  B1_pos = legionMalloc(ctx, runtime, B1_pos, B1_pos_parent, FID_RECT_1);
  B1_pos_accessor = createAccessor<AccessorRORect_1_1>(B1_pos, FID_RECT_1);
  B1_crd = legionMalloc(ctx, runtime, B1_crd, B1_crd_parent, FID_COORD);
  B1_crd_accessor = createAccessor<AccessorROint32_t1>(B1_crd, FID_COORD);
  B_vals = legionMalloc(ctx, runtime, B_vals, B_vals_parent, FID_VAL);
  B_vals_ro_accessor = createAccessor<AccessorROdouble2>(B_vals, FID_VAL);
  c_vals = legionMalloc(ctx, runtime, c_vals, c_vals_parent, FID_VAL);
  c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);

  int64_t B1Size = runtime->get_index_space_domain(ctx, get_index_space(B1_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  DomainT<1> B1_crd_domain = runtime->get_index_space_domain(ctx, B1_crd.get_index_space());
  DomainPointColoring B1_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t fposo = (*itr)[0];
    Point<1> B1CrdStart = Point<1>((fposo * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + B1_pos_accessor[Point<1>(0)].lo));
    Point<1> B1CrdEnd = Point<1>(TACO_MIN(((fposo * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces - 1)) + B1_pos_accessor[Point<1>(0)].lo), B1_crd_domain.bounds.hi[0]));
    Rect<1> B1CrdRect = Rect<1>(B1CrdStart, B1CrdEnd);
    if (!B1_crd_domain.contains(B1CrdRect.lo) || !B1_crd_domain.contains(B1CrdRect.hi)) {
      B1CrdRect = B1CrdRect.make_empty();
    }
    B1_crd_coloring[(*itr)] = B1CrdRect;
  }
  IndexPartition B1_crd_index_part = runtime->create_index_partition(ctx, B1_crd.get_index_space(), domain, B1_crd_coloring, LEGION_COMPUTE_KIND);
  LogicalPartition B1_crd_part = runtime->get_logical_partition(ctx, B1_crd, B1_crd_index_part);
  IndexPartition posSparsePartB1 = runtime->create_partition_by_preimage_range(
    ctx,
    B1_crd_index_part,
    B1_pos,
    B1_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, B1_crd_index_part)
  );
  IndexPartition posIndexPartB1 = densifyPartition(ctx, runtime, get_index_space(B1_pos), posSparsePartB1);
  LogicalPartition posPartB1 = runtime->get_logical_partition(ctx, B1_pos, posIndexPartB1);
  LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B1_crd_part, B_vals);
  auto computePartitions = new(partitionPackForcomputeLegionSparseDensePosParallelize);
  computePartitions->BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions->BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions->BPartition.indicesPartitions[0].push_back(posPartB1);
  computePartitions->BPartition.indicesPartitions[0].push_back(B1_crd_part);
  computePartitions->BPartition.valsPartition = BValsLogicalPart;
  return computePartitions;

  runtime->unmap_region(ctx, B1_crd);
  runtime->unmap_region(ctx, B1_pos);
  runtime->unmap_region(ctx, B_vals);
  runtime->unmap_region(ctx, c_vals);
}

void task_4(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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

  int32_t fposo = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t B2_dimension = args->B2_dimension;
  int32_t a1_dimension = args->a1_dimension;
  int32_t c1_dimension = args->c1_dimension;
  int32_t pieces = args->pieces;

  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto B_vals_ro_accessor = createAccessor<AccessorROdouble2>(B_vals, FID_VAL);
  auto a_vals_red_accessor = createAccessor<AccessorReducedouble1>(a_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B1_pos_accessor = createAccessor<AccessorRORect_1_1>(B1_pos, FID_RECT_1);
  auto B1_crd_accessor = createAccessor<AccessorROint32_t1>(B1_crd, FID_COORD);

  int64_t pointID1 = fposo;
  for (int32_t fposi = 0; fposi < ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces); fposi++) {
    int32_t fposB = (fposo * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + fposi) + B1_pos_accessor[Point<1>(0)].lo;
    if (fposB >= (fposo + 1) * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces))
      continue;

    if (fposB < B1_pos_accessor[Point<1>(0)].lo || fposB >= B1_pos_accessor[Point<1>(0)].hi + 1)
      continue;

    int32_t i = B1_crd_accessor[fposB];
    int64_t pointID2 = pointID1 * ((((B1_pos_accessor[Point<1>(0)].hi + 1) - B1_pos_accessor[Point<1>(0)].lo) + (pieces - 1)) / pieces) + fposi;
    int32_t ia = 0 * a1_dimension + i;
    for (int32_t j = 0; j < B2_dimension; j++) {
      int64_t pointID3 = pointID2 * B2_dimension + j;
      int32_t jB = fposB * B2_dimension + j;
      int32_t jc = 0 * c1_dimension + j;
      a_vals_red_accessor[Point<1>(i)] <<= B_vals_ro_accessor[Point<2>(fposB, j)] * c_vals_ro_accessor[Point<1>(j)];
    }
  }
}

void computeLegionSparseDensePosParallelize(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionSparseDensePosParallelize* partitionPack, int32_t pieces) {
  int a1_dimension = a->dims[0];
  RegionWrapper a_vals = a->vals;
  auto a_vals_parent = a->valsParent;
  int B2_dimension = B->dims[1];
  RegionWrapper B1_crd = B->indices[0][1];
  auto B1_pos_parent = B->indicesParents[0][0];
  auto B1_crd_parent = B->indicesParents[0][1];
  auto B_vals_parent = B->valsParent;
  int c1_dimension = c->dims[0];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t B1Size = runtime->get_index_space_domain(ctx, get_index_space(B1_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_4Args taskArgsRaw;
  taskArgsRaw.B2_dimension = B2_dimension;
  taskArgsRaw.a1_dimension = a1_dimension;
  taskArgsRaw.c1_dimension = c1_dimension;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, a_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[0][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B1_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[0][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B1_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
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
}
