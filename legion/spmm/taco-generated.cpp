#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int64_t A2_dimension;
  int64_t B1_dimension;
  int64_t C2_dimension;
  int32_t gx;
};

struct task_2Args {
  int64_t B2Size;
  int32_t gx;
  int64_t pointID1;
};

struct task_3Args {
  int64_t A2_dimension;
  int64_t B2Size;
  int64_t C2_dimension;
  int32_t gx;
  int64_t jo;
  int64_t pointID1;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gx) {
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  size_t C2_dimension = C->dims[1];
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegion();

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gx - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t io = (*itr)[0];
    Point<2> AStart = Point<2>((io * ((B1_dimension + (gx - 1)) / gx)), 0);
    Point<2> AEnd = Point<2>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), ADomain.hi()[0]), TACO_MIN(C2_dimension, ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<1> BStart = Point<1>((io * ((B1_dimension + (gx - 1)) / gx)));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto A_dense_run_0_Partition = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_COMPUTE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition, get_logical_region(A_vals));
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition, B2_pos);
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B2_crd.get_index_space(), posPartB2, B2_pos_parent, FID_RECT_1));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition;

  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B2_pos = regions[1];
  LogicalRegion B2_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B2_crd = regions[2];
  LogicalRegion B2_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B_vals = regions[3];
  LogicalRegion B_vals_parent = regions[3].get_logical_region();
  PhysicalRegion C_vals = regions[4];
  LogicalRegion C_vals_parent = regions[4].get_logical_region();

  int64_t io = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int64_t A2_dimension = args->A2_dimension;
  int64_t B1_dimension = args->B1_dimension;
  int64_t C2_dimension = args->C2_dimension;
  int32_t gx = args->gx;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, FID_VAL);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble2>(A_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  int64_t pointID1 = io + TACO_PARTITION_COLOR_OFFSET;
  #pragma omp parallel for schedule(dynamic, 128)
  for (int64_t ii = 0; ii < ((B1_dimension + (gx - 1)) / gx); ii++) {
    int64_t i = io * ((B1_dimension + (gx - 1)) / gx) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (gx - 1)) / gx))
      continue;

    int64_t pointID2 = pointID1 * ((B1_dimension + (gx - 1)) / gx) + ii;
    int64_t iA = i;
    int64_t iB = i;
    for (int64_t kB = B2_pos_accessor[Point<1>(i)].lo; kB < (B2_pos_accessor[Point<1>(i)].hi + 1); kB++) {
      int64_t k = B2_crd_accessor[(kB * 1)];
      int64_t kC = k;
      for (int64_t j = 0; j < C2_dimension; j++) {
        int64_t pointID3 = pointID2 * C2_dimension + j;
        int64_t jA = iA * A2_dimension + j;
        int64_t jC = kC * C2_dimension + j;
        A_vals_rw_accessor[Point<2>(i, j)] = A_vals_rw_accessor[Point<2>(i, j)] + B_vals_ro_accessor[Point<1>(kB)] * C_vals_ro_accessor[Point<2>(k, j)];
      }
    }
  }
}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t gx) {
  size_t A2_dimension = A->dims[1];
  auto A_vals_parent = A->valsParent;
  size_t B1_dimension = B->dims[0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gx - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.A2_dimension = A2_dimension;
  taskArgsRaw1.B1_dimension = B1_dimension;
  taskArgsRaw1.C2_dimension = C2_dimension;
  taskArgsRaw1.gx = gx;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(C_vals), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);

}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B2_pos = regions[1];
  LogicalRegion B2_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B2_crd = regions[2];
  LogicalRegion B2_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B_vals = regions[3];
  LogicalRegion B_vals_parent = regions[3].get_logical_region();
  PhysicalRegion C_vals = regions[4];
  LogicalRegion C_vals_parent = regions[4].get_logical_region();

  int64_t fposo = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int64_t B2Size = args->B2Size;
  int32_t gx = args->gx;
  int64_t pointID1 = args->pointID1;

  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  int64_t pB2_begin = B2PosDomain.bounds.lo;
  int64_t pB2_end = B2PosDomain.bounds.hi;
  int64_t pointID2 = pointID1 * gx + fposo;
  int64_t fposB = fposo * ((B2Size + (gx - 1)) / gx);
  int64_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
  int64_t i = i_pos;
  for (int64_t fposi = 0; fposi < ((B2Size + (gx - 1)) / gx); fposi++) {
    int64_t fposB = fposo * ((B2Size + (gx - 1)) / gx) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (gx - 1)) / gx))
      continue;

    if (fposB >= B2Size)
      continue;

    int64_t f = B2_crd_accessor[fposB];
    int64_t k = f;
    while (!(B2_pos_accessor[i_pos].contains(fposB))) {
      i_pos = i_pos + 1;
      i = i_pos;
    }
    int64_t iA = i;
    int64_t kC = k;
  }
}

partitionPackForcomputeLegionBatched partitionForcomputeLegionBatched(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gx) {
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegionBatched();

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  for (int64_t jo = 0; jo < C2_dimension; jo++) {
    int64_t pointID1 = jo + TACO_PARTITION_COLOR_OFFSET;
    Point<1> lowerBound = Point<1>(0);
    Point<1> upperBound = Point<1>((gx - 1));
    auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
    DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
    DomainT<1> B2_crd_domain = runtime->get_index_space_domain(ctx, B2_crd.get_index_space());
    DomainPointColoring B2_crd_coloring = DomainPointColoring();
    auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
    auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
    auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
    DomainPointColoring AColoring = DomainPointColoring();
    DomainPointColoring CColoring = DomainPointColoring();
    for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
      int64_t fposo = (*itr)[0];
      Point<1> B2CrdStart = Point<1>((fposo * ((B2Size + (gx - 1)) / gx)));
      Point<1> B2CrdEnd = Point<1>(TACO_MIN((fposo * ((B2Size + (gx - 1)) / gx) + ((B2Size + (gx - 1)) / gx - 1)), B2_crd_domain.bounds.hi[0]));
      Rect<1> B2CrdRect = Rect<1>(B2CrdStart, B2CrdEnd);
      if (!B2_crd_domain.contains(B2CrdRect.lo) || !B2_crd_domain.contains(B2CrdRect.hi)) {
        B2CrdRect = B2CrdRect.make_empty();
      }
      B2_crd_coloring[(*itr)] = B2CrdRect;
      Point<2> AStart = Point<2>((0 / B2_dimension), jo);
      Point<2> AEnd = Point<2>(TACO_MIN(((B1_dimension * B2_dimension - 1) / B2_dimension), ADomain.hi()[0]), TACO_MIN(jo, ADomain.hi()[1]));
      Rect<2> ARect = Rect<2>(AStart, AEnd);
      if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
        ARect = ARect.make_empty();
      }
      AColoring[(*itr)] = ARect;
      Point<2> CStart = Point<2>(0, jo);
      Point<2> CEnd = Point<2>(TACO_MIN((B1_dimension * B2_dimension - 1), CDomain.hi()[0]), TACO_MIN(jo, CDomain.hi()[1]));
      Rect<2> CRect = Rect<2>(CStart, CEnd);
      if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
        CRect = CRect.make_empty();
      }
      CColoring[(*itr)] = CRect;
    }
    IndexPartition B2_crd_index_part = runtime->create_index_partition(ctx, B2_crd.get_index_space(), domain, B2_crd_coloring, LEGION_COMPUTE_KIND, pointID1);
    LogicalPartition B2_crd_part = runtime->get_logical_partition(ctx, B2_crd, B2_crd_index_part);
    IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(
      ctx,
      B2_crd_index_part,
      B2_pos,
      B2_pos_parent,
      FID_RECT_1,
      runtime->get_index_partition_color_space_name(ctx, B2_crd_index_part),
      LEGION_ALIASED_INCOMPLETE_KIND
    );
    IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2, pointID1);
    LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
    LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B2_crd_part, B_vals, pointID1);
    IndexPartition BDenseRun0Partition = copyPartition(ctx, runtime, posPartB2, B_dense_run_0);
    IndexPartition ADenseRun0Partition = AffineProjection(0).addOverrides(1).apply(ctx, runtime, BDenseRun0Partition, A_dense_run_0, AColoring);
    auto A_vals_partition = copyPartition(ctx, runtime, ADenseRun0Partition, get_logical_region(A_vals), pointID1);
    IndexPartition CDenseRun0Partition = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
    auto C_vals_partition = copyPartition(ctx, runtime, CDenseRun0Partition, get_logical_region(C_vals), pointID1);
    task_2Args taskArgsRaw2;
    taskArgsRaw2.B2Size = B2Size;
    taskArgsRaw2.gx = gx;
    taskArgsRaw2.pointID1 = pointID1;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
    IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(A_vals_partition, 0, READ_ONLY, EXCLUSIVE, A_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_VAL));
    launcher.add_region_requirement(RegionRequirement(posPartB2, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
    launcher.add_region_requirement(RegionRequirement(B2_crd_part, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
    launcher.add_region_requirement(RegionRequirement(BValsLogicalPart, 0, READ_ONLY, EXCLUSIVE, B_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_VAL));
    launcher.add_region_requirement(RegionRequirement(C_vals_partition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_VAL));
    launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
    runtime->execute_index_space(ctx, launcher);


    computePartitions.APartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
    computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
    computePartitions.APartition.valsPartition = A_vals_partition;
    computePartitions.APartition.denseLevelRunPartitions[0] = ADenseRun0Partition;
    computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
    computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
    computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
    computePartitions.BPartition.indicesPartitions[1].push_back(B2_crd_part);
    computePartitions.BPartition.valsPartition = BValsLogicalPart;
    computePartitions.BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;
    computePartitions.CPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
    computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
    computePartitions.CPartition.valsPartition = C_vals_partition;
    computePartitions.CPartition.denseLevelRunPartitions[0] = CDenseRun0Partition;
  }

  return computePartitions;
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B2_pos = regions[1];
  LogicalRegion B2_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B2_crd = regions[2];
  LogicalRegion B2_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B_vals = regions[3];
  LogicalRegion B_vals_parent = regions[3].get_logical_region();
  PhysicalRegion C_vals = regions[4];
  LogicalRegion C_vals_parent = regions[4].get_logical_region();

  int64_t fposo = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int64_t A2_dimension = args->A2_dimension;
  int64_t B2Size = args->B2Size;
  int64_t C2_dimension = args->C2_dimension;
  int32_t gx = args->gx;
  int64_t jo = args->jo;
  int64_t pointID1 = args->pointID1;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, FID_VAL);
  auto A_vals_red_accessor = createAccessor<AccessorReducedouble2>(A_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  int64_t pB2_begin = B2PosDomain.bounds.lo;
  int64_t pB2_end = B2PosDomain.bounds.hi;
  int64_t pointID2 = pointID1 * gx + fposo;
  int64_t fposB = fposo * ((B2Size + (gx - 1)) / gx);
  int64_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, fposB);
  int64_t i = i_pos;
  for (int64_t fposi = 0; fposi < ((B2Size + (gx - 1)) / gx); fposi++) {
    int64_t fposB = fposo * ((B2Size + (gx - 1)) / gx) + fposi;
    if (fposB >= (fposo + 1) * ((B2Size + (gx - 1)) / gx))
      continue;

    if (fposB >= B2Size)
      continue;

    int64_t f = B2_crd_accessor[fposB];
    int64_t k = f;
    while (!(B2_pos_accessor[i_pos].contains(fposB))) {
      i_pos = i_pos + 1;
      i = i_pos;
    }
    int64_t iA = i;
    int64_t kC = k;
    for (int64_t ji = 0; ji < 1; ji++) {
      int64_t j = jo + ji;
      if (j >= C2_dimension)
        continue;

      int64_t pointID3 = pointID2 * 1 + ji;
      int64_t jA = iA * A2_dimension + j;
      int64_t jC = kC * C2_dimension + j;
      A_vals_red_accessor[Point<2>(i, j)] <<= B_vals_ro_accessor[Point<1>(fposB)] * C_vals_ro_accessor[Point<2>(k, j)];
    }
  }
}

void computeLegionBatched(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegionBatched* partitionPack, int32_t gx) {
  size_t A2_dimension = A->dims[1];
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;

  for (int64_t jo = 0; jo < C2_dimension; jo++) {
    int64_t pointID1 = jo + TACO_PARTITION_COLOR_OFFSET;
    Point<1> lowerBound = Point<1>(0);
    Point<1> upperBound = Point<1>((gx - 1));
    auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
    DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
    task_3Args taskArgsRaw3;
    taskArgsRaw3.A2_dimension = A2_dimension;
    taskArgsRaw3.B2Size = B2Size;
    taskArgsRaw3.C2_dimension = C2_dimension;
    taskArgsRaw3.gx = gx;
    taskArgsRaw3.jo = jo;
    taskArgsRaw3.pointID1 = pointID1;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
    IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(A_vals), pointID1), 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, A_vals_parent).add_field(FID_VAL));
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(B2_pos), pointID1), 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(B2_crd), pointID1), 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(B_vals), pointID1), 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(C_vals), pointID1), 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
    launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
    runtime->execute_index_space(ctx, launcher);

  }
}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_3>(registrar, "task_3");
  }
}
