#include "cblas.h"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;

struct task_1Args {
  int32_t sfID;
  int32_t gridDim;
};
struct task_2Args {
  int32_t sfID;
  int32_t gridDim;
};
struct task_3Args {
  int32_t sfID;
  int32_t gridDim;
};
struct task_4Args {
};
struct task_5Args {
  int32_t b1_dimension;
  int32_t b2_dimension;
  int32_t c2_dimension;
  int32_t gridDim;
};

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridDim) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), (gridDim - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((a2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridDim - 1)) / gridDim) + ((a1_dimension + (gridDim - 1)) / gridDim - 1)), aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gridDim - 1)) / gridDim) + ((a2_dimension + (gridDim - 1)) / gridDim - 1)), aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(a), aPartition));
  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];

  int32_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridDim = args->gridDim;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, LogicalPartition aPartition, int32_t gridDim, Legion::PrivilegeMode priv) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), (gridDim - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement aReq = RegionRequirement(aPartition, 0, priv, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridDim);
  dims.push_back(gridDim);
  dims.push_back(gridDim);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(1), dims);
  task_1Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(1);
  taskArgsRaw.gridDim = gridDim;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridDim) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), 0, (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[2];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((b2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + ((b1_dimension + (gridDim - 1)) / gridDim - 1)), bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + (gridDim - 1)) / gridDim) + ((b2_dimension + (gridDim - 1)) / gridDim - 1)), bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(b), bPartition));
  return computePartitions;
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b = regions[0];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridDim = args->gridDim;


  int32_t in = getIndexPoint(task, 0);
  int32_t kn = getIndexPoint(task, 1);
  int32_t jn = getIndexPoint(task, 2);
}

void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, LogicalPartition bPartition, int32_t gridDim, Legion::PrivilegeMode priv) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), 0, (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement bReq = RegionRequirement(bPartition, 0, priv, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridDim);
  dims.push_back(gridDim);
  dims.push_back(gridDim);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(3), dims);
  task_2Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(3);
  taskArgsRaw.gridDim = gridDim;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridDim) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, (gridDim - 1), (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t jn = (*itr)[1];
    int32_t in = (*itr)[2];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gridDim - 1)) / gridDim) + ((c1_dimension + (gridDim - 1)) / gridDim - 1)), cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(c), cPartition));
  return computePartitions;
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c = regions[0];

  int32_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridDim = args->gridDim;


  int32_t kn = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t in = getIndexPoint(task, 2);
}

void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, LogicalPartition cPartition, int32_t gridDim, Legion::PrivilegeMode priv) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, (gridDim - 1), (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement cReq = RegionRequirement(cPartition, 0, priv, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridDim);
  dims.push_back(gridDim);
  dims.push_back(gridDim);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(5), dims);
  task_3Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(5);
  taskArgsRaw.gridDim = gridDim;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(cReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int32_t gridDim) {
  auto a_index_space = get_index_space(a);
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), (gridDim - 1), (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int32_t kn = (*itr)[2];
    Point<2> aStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + ((b1_dimension + (gridDim - 1)) / gridDim - 1)), aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (kn * ((b2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + ((b1_dimension + (gridDim - 1)) / gridDim - 1)), bDomain.hi()[0]), TACO_MIN((kn * ((b2_dimension + (gridDim - 1)) / gridDim) + ((b2_dimension + (gridDim - 1)) / gridDim - 1)), bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((kn * ((b2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> cEnd = Point<2>(TACO_MIN((kn * ((b2_dimension + (gridDim - 1)) / gridDim) + ((b2_dimension + (gridDim - 1)) / gridDim - 1)), cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_ALIASED_COMPLETE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(a), aPartition));
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(b), bPartition));
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(c), cPartition));
  return computePartitions;
}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t iln = task->index_point[0];
  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);
  AccessorROdouble2 b_vals(b, FID_VAL);
  AccessorROdouble2 c_vals(c, FID_VAL);
  AccessorReducedouble2 a_vals(a, FID_VAL, LEGION_REDOP_SUM_FLOAT64);

  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  if (bDomain.get_volume() == 0 || cDomain.get_volume() == 0)
    return ;

  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    (1 + (bDomain.hi()[0] - bDomain.lo()[0])),
    (1 + (cDomain.hi()[1] - cDomain.lo()[1])),
    (1 + (cDomain.hi()[0] - cDomain.lo()[0])),
    1.00000000,
    b_vals.ptr(bDomain.lo()),
    (b_vals.accessor.strides[0] / sizeof(double)),
    c_vals.ptr(cDomain.lo()),
    (c_vals.accessor.strides[0] / sizeof(double)),
    1.00000000,
    a_vals.ptr(aDomain.lo()),
    (a_vals.accessor.strides[0] / sizeof(double))
  );
}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused = task->index_point[0];
  task_5Args* args = (task_5Args*)(task->args);
  int32_t b1_dimension = args->b1_dimension;
  int32_t b2_dimension = args->b2_dimension;
  int32_t c2_dimension = args->c2_dimension;
  int32_t gridDim = args->gridDim;

  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(1);
  auto ilnIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ilnIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t iln = (*itr)[0];
    Point<2> aStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + (iln * ((((b1_dimension + (gridDim - 1)) / gridDim - 0 / gridDim) + 1) / 2) + (0 / gridDim) / 2)), (jn * ((c2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + (iln * ((((b1_dimension + (gridDim - 1)) / gridDim - 0 / gridDim) + 1) / 2) + (((b1_dimension + (gridDim - 1)) / gridDim + 1) / 2 - 1))), aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridDim - 1)) / gridDim) + ((c2_dimension + (gridDim - 1)) / gridDim - 1)), aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridDim - 1)) / gridDim) + (iln * ((((b1_dimension + (gridDim - 1)) / gridDim - 0 / gridDim) + 1) / 2) + (0 / gridDim) / 2)), (kn * ((b2_dimension + (gridDim - 1)) / gridDim) + 0 / gridDim));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridDim - 1)) / gridDim) + (iln * ((((b1_dimension + (gridDim - 1)) / gridDim - 0 / gridDim) + 1) / 2) + (((b1_dimension + (gridDim - 1)) / gridDim + 1) / 2 - 1))), bDomain.hi()[0]), TACO_MIN((kn * ((b2_dimension + (gridDim - 1)) / gridDim) + ((b2_dimension + (gridDim - 1)) / gridDim - 1)), bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(a));
  aReq.add_field(FID_VAL);
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  RegionRequirement cReq = RegionRequirement(get_logical_region(c), READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  task_4Args taskArgsRaw;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
  launcher.tag |= TACOMapper::UNTRACK_VALID_REGIONS;
  launcher.tag |= Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  runtime->execute_index_space(ctx, launcher);

}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition, LogicalPartition bPartition, LogicalPartition cPartition, int32_t gridDim) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridDim - 1), (gridDim - 1), (gridDim - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement aReq = RegionRequirement(aPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(a));
  aReq.add_field(FID_VAL);
  aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement bReq = RegionRequirement(bPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement cReq = RegionRequirement(cPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_5Args taskArgsRaw;
  taskArgsRaw.b1_dimension = b1_dimension;
  taskArgsRaw.b2_dimension = b2_dimension;
  taskArgsRaw.c2_dimension = c2_dimension;
  taskArgsRaw.gridDim = gridDim;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
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
    registrar.set_inner();
    Runtime::preregister_task_variant<task_5>(registrar, "task_5");
  }
}
