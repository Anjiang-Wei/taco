#include "cublas_v2.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;

struct task_1Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_2Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_3Args {
  int32_t gridX;
  int32_t gridY;
};
struct task_4Args {
  int32_t kios;
};
struct task_5Args {
  int32_t b1_dimension;
  int32_t b2_dimension;
  int32_t c2_dimension;
  int32_t gridX;
  int32_t gridY;
  int32_t in;
  int32_t jn;
  int32_t kos;
};
struct task_6Args {
  int32_t b1_dimension;
  int32_t b2_dimension;
  int32_t c2_dimension;
  int32_t gridX;
  int32_t gridY;
  int32_t in;
  int32_t jn;
  int32_t kos;
};
struct task_7Args {
  int32_t b1_dimension;
  int32_t b2_dimension;
  int32_t c2_dimension;
  int32_t gridX;
  int32_t gridY;
};

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridX, int32_t gridY) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + 0 / (gridX * 2)), (jn * ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + 0 / (gridY * 2)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2) - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2) - 1)),aDomain.hi()[1]));
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
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, LogicalPartition aPartition, int32_t gridX, int32_t gridY) {

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  RegionRequirement aReq = RegionRequirement(aPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  task_1Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridX, int32_t gridY) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + 0 / (gridX * 2)), (jn * ((b2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + 0 / (gridY * 2)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + ((b1_dimension + (gridX * 2 - 1)) / (gridX * 2) - 1)),bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + ((b2_dimension + (gridY * 2 - 1)) / (gridY * 2) - 1)),bDomain.hi()[1]));
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
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, LogicalPartition bPartition, int32_t gridX, int32_t gridY) {

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  RegionRequirement bReq = RegionRequirement(bPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridX, int32_t gridY) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + 0 / (gridX * 2)), (jn * ((c2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + 0 / (gridY * 2)));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + ((c1_dimension + (gridX * 2 - 1)) / (gridX * 2) - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + ((c2_dimension + (gridY * 2 - 1)) / (gridY * 2) - 1)),cDomain.hi()[1]));
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
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, LogicalPartition cPartition, int32_t gridX, int32_t gridY) {

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  RegionRequirement cReq = RegionRequirement(cPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  task_3Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(cReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int32_t gridX, int32_t gridY) {
  auto a_index_space = get_index_space(a);
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((c2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (0 / gridX));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[0]), TACO_MIN(((gridX - 1) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((0 / gridX), (jn * ((c2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> cEnd = Point<2>(TACO_MIN(((gridX - 1) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
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

  task_4Args* args = (task_4Args*)(task->args);
  int32_t kios = args->kios;

  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);
  AccessorROdouble2 b_vals(b, FID_VAL);
  AccessorROdouble2 c_vals(c, FID_VAL);
  AccessorRWdouble2 a_vals(a, FID_VAL);

  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  if (bDomain.get_volume() == 0 || cDomain.get_volume() == 0)
    return ;

  double alpha = 1.0000000000000000;
  cublasHandle_t handle = getCuBLAS();
  cudaStream_t taskStream = cudaStream_t();
  cudaStreamCreate(&(taskStream));
  CHECK_CUBLAS(cublasSetStream(handle, taskStream));
  CHECK_CUBLAS(cublasDgemm(
    handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    (1 + (cDomain.hi()[1] - cDomain.lo()[1])),
    (1 + (bDomain.hi()[0] - bDomain.lo()[0])),
    (1 + (cDomain.hi()[0] - cDomain.lo()[0])),
    &(alpha),
    c_vals.ptr(cDomain.lo()),
    (c_vals.accessor.strides[0] / sizeof(double)),
    b_vals.ptr(bDomain.lo()),
    (b_vals.accessor.strides[0] / sizeof(double)),
    &(alpha),
    a_vals.ptr(aDomain.lo()),
    (a_vals.accessor.strides[0] / sizeof(double))
  ));
}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused1 = task->index_point[0];
  task_5Args* args = (task_5Args*)(task->args);
  int32_t b1_dimension = args->b1_dimension;
  int32_t b2_dimension = args->b2_dimension;
  int32_t c2_dimension = args->c2_dimension;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t in = args->in;
  int32_t jn = args->jn;
  int32_t kos = args->kos;

  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);

  int32_t iln = getIndexPoint(task, 0);
  int32_t jln = getIndexPoint(task, 1);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(1);
  auto kiosIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(kiosIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t kios = (*itr)[0];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * ((((b1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)), (((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + (((jln + (iln + kios)) % 2) * ((((b2_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * ((((b1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((b1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),bDomain.hi()[0]), TACO_MIN((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + (((jln + (iln + kios)) % 2) * ((((b2_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((b2_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + (((jln + (iln + kios)) % 2) * ((((b2_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)), (jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * ((((c2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (0 / gridY) / 2)));
    Point<2> cEnd = Point<2>(TACO_MIN((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + (((jln + (iln + kios)) % 2) * ((((b2_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((b2_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * ((((c2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (((c2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t kios = (*itr);
    RegionRequirement aReq = RegionRequirement(get_logical_region(a), READ_WRITE, EXCLUSIVE, get_logical_region(a));
    aReq.add_field(FID_VAL);
    auto bsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(b), bPartition), kios);
    RegionRequirement bReq = RegionRequirement(bsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(b));
    bReq.add_field(FID_VAL);
    auto csubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(c), cPartition), kios);
    RegionRequirement cReq = RegionRequirement(csubReg, READ_ONLY, EXCLUSIVE, get_logical_region(c));
    cReq.add_field(FID_VAL);
    task_4Args taskArgsRaw;
    taskArgsRaw.kios = kios;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
    TaskLauncher launcher = TaskLauncher(taskID(4), taskArgs);
    launcher.add_region_requirement(aReq);
    launcher.add_region_requirement(bReq);
    launcher.add_region_requirement(cReq);
    launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

void task_6(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  task_6Args* args = (task_6Args*)(task->args);
  int32_t b1_dimension = args->b1_dimension;
  int32_t b2_dimension = args->b2_dimension;
  int32_t c2_dimension = args->c2_dimension;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t in = args->in;
  int32_t jn = args->jn;
  int32_t kos = args->kos;

  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFused1IndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFused1IndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t iln = (*itr)[0];
    int32_t jln = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * ((((b1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)), (((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + 0 / gridX));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * ((((b1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((b1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),bDomain.hi()[0]), TACO_MIN((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * ((((c2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (0 / gridY) / 2)));
    Point<2> cEnd = Point<2>(TACO_MIN((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * ((((c2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (((c2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
    Point<2> aStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * ((((b1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)), (jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * ((((c2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (0 / gridY) / 2)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * ((((b1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((b1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * ((((c2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (((c2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_5Args taskArgsRaw;
  taskArgsRaw.b1_dimension = b1_dimension;
  taskArgsRaw.b2_dimension = b2_dimension;
  taskArgsRaw.c2_dimension = c2_dimension;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  taskArgsRaw.in = in;
  taskArgsRaw.jn = jn;
  taskArgsRaw.kos = kos;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
  launcher.tag = launcher.tag | Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  runtime->execute_index_space(ctx, launcher);

}

void task_7(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused = task->index_point[0];
  task_7Args* args = (task_7Args*)(task->args);
  int32_t b1_dimension = args->b1_dimension;
  int32_t b2_dimension = args->b2_dimension;
  int32_t c2_dimension = args->c2_dimension;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;

  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto kosIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(kosIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t kos = (*itr)[0];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + 0 / gridX));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[0]), TACO_MIN((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((c2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> cEnd = Point<2>(TACO_MIN((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t kos = (*itr);
    RegionRequirement aReq = RegionRequirement(get_logical_region(a), READ_WRITE, EXCLUSIVE, get_logical_region(a));
    aReq.add_field(FID_VAL);
    aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
    auto bsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(b), bPartition), kos);
    RegionRequirement bReq = RegionRequirement(bsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(b));
    bReq.add_field(FID_VAL);
    bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
    auto csubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(c), cPartition), kos);
    RegionRequirement cReq = RegionRequirement(csubReg, READ_ONLY, EXCLUSIVE, get_logical_region(c));
    cReq.add_field(FID_VAL);
    cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
    task_6Args taskArgsRaw;
    taskArgsRaw.b1_dimension = b1_dimension;
    taskArgsRaw.b2_dimension = b2_dimension;
    taskArgsRaw.c2_dimension = c2_dimension;
    taskArgsRaw.gridX = gridX;
    taskArgsRaw.gridY = gridY;
    taskArgsRaw.in = in;
    taskArgsRaw.jn = jn;
    taskArgsRaw.kos = kos;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_6Args));
    TaskLauncher launcher = TaskLauncher(taskID(6), taskArgs);
    launcher.add_region_requirement(aReq);
    launcher.add_region_requirement(bReq);
    launcher.add_region_requirement(cReq);
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition, LogicalPartition bPartition, LogicalPartition cPartition, int32_t gridX, int32_t gridY) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  RegionRequirement aReq = RegionRequirement(aPartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement bReq = RegionRequirement(bPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement cReq = RegionRequirement(cPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_7Args taskArgsRaw;
  taskArgsRaw.b1_dimension = b1_dimension;
  taskArgsRaw.b2_dimension = b2_dimension;
  taskArgsRaw.c2_dimension = c2_dimension;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_7Args));
  IndexLauncher launcher = IndexLauncher(taskID(7), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

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
    registrar.set_inner();
    Runtime::preregister_task_variant<task_5>(registrar, "task_5");
  }
  {
    TaskVariantRegistrar registrar(taskID(6), "task_6");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_6>(registrar, "task_6");
  }
  {
    TaskVariantRegistrar registrar(taskID(7), "task_7");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_7>(registrar, "task_7");
  }
}
