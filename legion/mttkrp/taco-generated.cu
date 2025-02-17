#include "cublas_v2.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;
typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorROdouble3;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;

struct task_1Args {
  int32_t sfID;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};
struct task_2Args {
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};
struct task_3Args {
  int32_t sfID;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};
struct task_4Args {
  int32_t sfID;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};
struct task_5Args {
  int32_t gridX;
  int32_t lo;
};
struct task_6Args {
  int32_t B1_dimension;
  int32_t B2_dimension;
  int32_t B3_dimension;
  int32_t C2_dimension;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};

LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (gridX - 1)) / gridX) + 0 / gridX), 0);
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)),ADomain.hi()[0]), TACO_MIN(A2_dimension,ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
}

LogicalPartition partitionLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY, int32_t gridZ) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int32_t kn = (*itr)[2];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), (kn * ((B3_dimension + (gridZ - 1)) / gridZ) + 0 / gridZ));
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)),BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)),BDomain.hi()[1]), TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)),BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);
}

LogicalPartition partitionLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t gridY) {
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridY - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<2> CStart = Point<2>((in * ((C1_dimension + (gridY - 1)) / gridY) + 0 / gridY), 0);
    Point<2> CEnd = Point<2>(TACO_MIN((in * ((C1_dimension + (gridY - 1)) / gridY) + ((C1_dimension + (gridY - 1)) / gridY - 1)),CDomain.hi()[0]), TACO_MIN(C2_dimension,CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);
}

LogicalPartition partitionLegionD(Context ctx, Runtime* runtime, LogicalRegion D, int32_t gridZ) {
  int D1_dimension = runtime->get_index_space_domain(get_index_space(D)).hi()[0] + 1;
  int D2_dimension = runtime->get_index_space_domain(get_index_space(D)).hi()[1] + 1;
  auto D_index_space = get_index_space(D);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridZ - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto DDomain = runtime->get_index_space_domain(ctx, D_index_space);
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<2> DStart = Point<2>((in * ((D1_dimension + (gridZ - 1)) / gridZ) + 0 / gridZ), 0);
    Point<2> DEnd = Point<2>(TACO_MIN((in * ((D1_dimension + (gridZ - 1)) / gridZ) + ((D1_dimension + (gridZ - 1)) / gridZ - 1)),DDomain.hi()[0]), TACO_MIN(D2_dimension,DDomain.hi()[1]));
    Rect<2> DRect = Rect<2>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  auto DPartition = runtime->create_index_partition(ctx, D_index_space, domain, DColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(D), DPartition);
}

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), 0, 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (gridX - 1)) / gridX) + 0 / gridX), 0);
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)),ADomain.hi()[0]), TACO_MIN(A2_dimension,ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(A), APartition));
  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];

  int32_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


  int32_t in = getIndexPoint(task, 0);
  int32_t kn = getIndexPoint(task, 1);
  int32_t ln = getIndexPoint(task, 2);
}

void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, LogicalPartition APartition, int32_t gridX, int32_t gridY, int32_t gridZ) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), 0, 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement AReq = RegionRequirement(APartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(A));
  AReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridX);
  dims.push_back(gridY);
  dims.push_back(gridZ);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(1), dims);
  task_1Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(1);
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  taskArgsRaw.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY, int32_t gridZ) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int32_t kn = (*itr)[2];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), (kn * ((B3_dimension + (gridZ - 1)) / gridZ) + 0 / gridZ));
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)),BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)),BDomain.hi()[1]), TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)),BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(B), BPartition));
  return computePartitions;
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B = regions[0];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, LogicalPartition BPartition, int32_t gridX, int32_t gridY, int32_t gridZ) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement BReq = RegionRequirement(BPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  taskArgsRaw.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(BReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t gridY) {
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, (gridY - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[1];
    Point<2> CStart = Point<2>((in * ((C1_dimension + (gridY - 1)) / gridY) + 0 / gridY), 0);
    Point<2> CEnd = Point<2>(TACO_MIN((in * ((C1_dimension + (gridY - 1)) / gridY) + ((C1_dimension + (gridY - 1)) / gridY - 1)),CDomain.hi()[0]), TACO_MIN(C2_dimension,CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(C), CPartition));
  return computePartitions;
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion C = regions[0];

  int32_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


  int32_t kn = getIndexPoint(task, 0);
  int32_t in = getIndexPoint(task, 1);
  int32_t ln = getIndexPoint(task, 2);
}

void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, LogicalPartition CPartition, int32_t gridY, int32_t gridX, int32_t gridZ) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, (gridY - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement CReq = RegionRequirement(CPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridX);
  dims.push_back(gridY);
  dims.push_back(gridZ);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(3), dims);
  task_3Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(3);
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  taskArgsRaw.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(CReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionD(Context ctx, Runtime* runtime, LogicalRegion D, int32_t gridZ) {
  int D1_dimension = runtime->get_index_space_domain(get_index_space(D)).hi()[0] + 1;
  int D2_dimension = runtime->get_index_space_domain(get_index_space(D)).hi()[1] + 1;
  auto D_index_space = get_index_space(D);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, 0, (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto DDomain = runtime->get_index_space_domain(ctx, D_index_space);
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[2];
    Point<2> DStart = Point<2>((in * ((D1_dimension + (gridZ - 1)) / gridZ) + 0 / gridZ), 0);
    Point<2> DEnd = Point<2>(TACO_MIN((in * ((D1_dimension + (gridZ - 1)) / gridZ) + ((D1_dimension + (gridZ - 1)) / gridZ - 1)),DDomain.hi()[0]), TACO_MIN(D2_dimension,DDomain.hi()[1]));
    Rect<2> DRect = Rect<2>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  auto DPartition = runtime->create_index_partition(ctx, D_index_space, domain, DColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(D), DPartition));
  return computePartitions;
}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion D = regions[0];

  int32_t distFused = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


  int32_t kn = getIndexPoint(task, 0);
  int32_t ln = getIndexPoint(task, 1);
  int32_t in = getIndexPoint(task, 2);
}

void placeLegionD(Context ctx, Runtime* runtime, LogicalRegion D, LogicalPartition DPartition, int32_t gridZ, int32_t gridX, int32_t gridY) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, 0, (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement DReq = RegionRequirement(DPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(D));
  DReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridX);
  dims.push_back(gridY);
  dims.push_back(gridZ);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(5), dims);
  task_4Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(5);
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  taskArgsRaw.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(DReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalRegion D, int32_t gridX, int32_t gridY, int32_t gridZ) {
  auto A_index_space = get_index_space(A);
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);
  auto D_index_space = get_index_space(D);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  auto DDomain = runtime->get_index_space_domain(ctx, D_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int32_t kn = (*itr)[2];
    Point<2> AStart = Point<2>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (0 / gridX));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)),ADomain.hi()[0]), TACO_MIN(((gridX - 1) * ((C2_dimension + (gridX - 1)) / gridX) + ((C2_dimension + (gridX - 1)) / gridX - 1)),ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), (kn * ((B3_dimension + (gridZ - 1)) / gridZ) + 0 / gridZ));
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)),BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)),BDomain.hi()[1]), TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)),BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> CStart = Point<2>((jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), (0 / gridX));
    Point<2> CEnd = Point<2>(TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)),CDomain.hi()[0]), TACO_MIN(((gridX - 1) * ((C2_dimension + (gridX - 1)) / gridX) + ((C2_dimension + (gridX - 1)) / gridX - 1)),CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
    Point<2> DStart = Point<2>((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + 0 / gridZ), (0 / gridX));
    Point<2> DEnd = Point<2>(TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)),DDomain.hi()[0]), TACO_MIN(((gridX - 1) * ((C2_dimension + (gridX - 1)) / gridX) + ((C2_dimension + (gridX - 1)) / gridX - 1)),DDomain.hi()[1]));
    Rect<2> DRect = Rect<2>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto DPartition = runtime->create_index_partition(ctx, D_index_space, domain, DColoring, LEGION_ALIASED_COMPLETE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(A), APartition));
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(B), BPartition));
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(C), CPartition));
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(D), DPartition));
  return computePartitions;
}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];
  PhysicalRegion D = regions[3];

  task_5Args* args = (task_5Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t lo = args->lo;

  auto A_index_space = get_index_space(A);
  auto B_index_space = get_index_space(B);
  auto C_index_space = get_index_space(C);
  auto D_index_space = get_index_space(D);
  AccessorROdouble2 C_vals(C, FID_VAL);
  AccessorROdouble2 D_vals(D, FID_VAL);
  AccessorROdouble3 B_vals(B, FID_VAL);
  AccessorReducedouble2 A_vals(A, FID_VAL, LEGION_REDOP_SUM_FLOAT64);

  auto aDomain = runtime->get_index_space_domain(ctx, A_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, B_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, C_index_space);
  auto dDomain = runtime->get_index_space_domain(ctx, D_index_space);
  if ((bDomain.get_volume() == 0 || cDomain.get_volume() == 0) || dDomain.get_volume() == 0)
    return ;

  MTTKRPPack pack = MTTKRPPack();
  pack.iDim = 1 + (bDomain.hi()[0] - bDomain.lo()[0]);
  pack.jDim = 1 + (bDomain.hi()[1] - bDomain.lo()[1]);
  pack.kDim = 1 + (bDomain.hi()[2] - bDomain.lo()[2]);
  pack.lDim = 1 + (aDomain.hi()[1] - aDomain.lo()[1]);
  pack.ldA = A_vals.accessor.strides[0] / sizeof(double);
  pack.ldC = C_vals.accessor.strides[0] / sizeof(double);
  pack.ldD = D_vals.accessor.strides[0] / sizeof(double);
  pack.ldB1 = B_vals.accessor.strides[0] / sizeof(double);
  pack.ldB2 = (B_vals.accessor.strides[0] / sizeof(double)) / (B_vals.accessor.strides[1] / sizeof(double));
  pack.ldB3 = B_vals.accessor.strides[1] / sizeof(double);
  cu_mttkrp<double>(pack, A_vals.ptr(aDomain.lo()), B_vals.ptr(bDomain.lo()), C_vals.ptr(cDomain.lo()), D_vals.ptr(dDomain.lo()));
}

void task_6(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];
  PhysicalRegion D = regions[3];

  int32_t distFused = task->index_point[0];
  task_6Args* args = (task_6Args*)(task->args);
  int32_t B1_dimension = args->B1_dimension;
  int32_t B2_dimension = args->B2_dimension;
  int32_t B3_dimension = args->B3_dimension;
  int32_t C2_dimension = args->C2_dimension;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;

  auto A_index_space = get_index_space(A);
  auto C_index_space = get_index_space(C);
  auto D_index_space = get_index_space(D);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto loIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(loIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  auto DDomain = runtime->get_index_space_domain(ctx, D_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t lo = (*itr)[0];
    Point<2> AStart = Point<2>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (lo * ((C2_dimension + (gridX - 1)) / gridX) + 0 / gridX));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)),ADomain.hi()[0]), TACO_MIN((lo * ((C2_dimension + (gridX - 1)) / gridX) + ((C2_dimension + (gridX - 1)) / gridX - 1)),ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<2> CStart = Point<2>((jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), (lo * ((C2_dimension + (gridX - 1)) / gridX) + 0 / gridX));
    Point<2> CEnd = Point<2>(TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)),CDomain.hi()[0]), TACO_MIN((lo * ((C2_dimension + (gridX - 1)) / gridX) + ((C2_dimension + (gridX - 1)) / gridX - 1)),CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
    Point<2> DStart = Point<2>((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + 0 / gridZ), (lo * ((C2_dimension + (gridX - 1)) / gridX) + 0 / gridX));
    Point<2> DEnd = Point<2>(TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)),DDomain.hi()[0]), TACO_MIN((lo * ((C2_dimension + (gridX - 1)) / gridX) + ((C2_dimension + (gridX - 1)) / gridX - 1)),DDomain.hi()[1]));
    Rect<2> DRect = Rect<2>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto DPartition = runtime->create_index_partition(ctx, D_index_space, domain, DColoring, LEGION_DISJOINT_COMPLETE_KIND);
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t lo = (*itr);
    auto AsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(A), APartition), lo);
    RegionRequirement AReq = RegionRequirement(AsubReg, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(A));
    AReq.add_field(FID_VAL);
    RegionRequirement BReq = RegionRequirement(get_logical_region(B), READ_ONLY, EXCLUSIVE, get_logical_region(B));
    BReq.add_field(FID_VAL);
    auto CsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(C), CPartition), lo);
    RegionRequirement CReq = RegionRequirement(CsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(C));
    CReq.add_field(FID_VAL);
    auto DsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(D), DPartition), lo);
    RegionRequirement DReq = RegionRequirement(DsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(D));
    DReq.add_field(FID_VAL);
    task_5Args taskArgsRaw;
    taskArgsRaw.gridX = gridX;
    taskArgsRaw.lo = lo;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_5Args));
    TaskLauncher launcher = TaskLauncher(taskID(5), taskArgs);
    launcher.add_region_requirement(AReq);
    launcher.add_region_requirement(BReq);
    launcher.add_region_requirement(CReq);
    launcher.add_region_requirement(DReq);
    launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalRegion D, LogicalPartition APartition, LogicalPartition BPartition, LogicalPartition CPartition, LogicalPartition DPartition, int32_t gridX, int32_t gridY, int32_t gridZ) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement AReq = RegionRequirement(APartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(A));
  AReq.add_field(FID_VAL);
  AReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement BReq = RegionRequirement(BPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  BReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement CReq = RegionRequirement(CPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  CReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement DReq = RegionRequirement(DPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(D));
  DReq.add_field(FID_VAL);
  DReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_6Args taskArgsRaw;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.B2_dimension = B2_dimension;
  taskArgsRaw.B3_dimension = B3_dimension;
  taskArgsRaw.C2_dimension = C2_dimension;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  taskArgsRaw.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_6Args));
  IndexLauncher launcher = IndexLauncher(taskID(6), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  launcher.add_region_requirement(DReq);
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
  {
    TaskVariantRegistrar registrar(taskID(5), "task_5");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_5>(registrar, "task_5");
  }
  {
    TaskVariantRegistrar registrar(taskID(6), "task_6");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_6>(registrar, "task_6");
  }
}
