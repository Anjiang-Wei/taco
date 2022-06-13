#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
#include "cblas.h"
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

struct task_5Args {
  IndexPartition a_dense_run_0_Partition_0;
  int64_t b1_dimension;
  int64_t b2_dimension;
  IndexSpace b_dense_run_0;
  IndexPartition b_dense_run_0_Partition_0;
  int64_t c2_dimension;
  IndexSpace c_dense_run_0;
  IndexPartition c_dense_run_0_Partition_0;
  int32_t gridX;
  int32_t gridY;
};

struct task_6Args {
  Legion::FieldID a_vals_field_id;
  Legion::FieldID b_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t gridX;
  int64_t kos;
};

struct task_7Args {
  Legion::FieldID a_vals_field_id;
  Legion::FieldID b_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t gridX;
  int32_t gridY;
};


partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, int32_t gridX, int32_t gridY) {
  size_t a1_dimension = a->dims[0];
  size_t a2_dimension = a->dims[1];
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionA();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridX - 1)) / gridX)), (jn * ((a2_dimension + (gridY - 1)) / gridY)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridX - 1)) / gridX) + ((a1_dimension + (gridX - 1)) / gridX - 1)), aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gridY - 1)) / gridY) + ((a2_dimension + (gridY - 1)) / gridY - 1)), aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  IndexPartition a_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, a_dense_run_0, domain, aColoring, LEGION_COMPUTE_KIND);
  auto a_vals_partition = copyPartition(ctx, runtime, a_dense_run_0_Partition_0, get_logical_region(a_vals));
  computePartitions.aPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.aPartition.valsPartition = a_vals_partition;
  computePartitions.aPartition.denseLevelRunPartitions[0] = a_dense_run_0_Partition_0;

  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY) {
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.gridX = gridX;
  taskArgsRaw1.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, a_vals_parent).add_field(a_vals_field_id));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, int32_t gridX, int32_t gridY) {
  size_t b1_dimension = b->dims[0];
  size_t b2_dimension = b->dims[1];
  RegionWrapper b_vals = b->vals;
  IndexSpace b_dense_run_0 = b->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionB();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_dense_run_0);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX)), (jn * ((b2_dimension + (gridY - 1)) / gridY)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)), bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + (gridY - 1)) / gridY) + ((b2_dimension + (gridY - 1)) / gridY - 1)), bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
  }
  IndexPartition b_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, b_dense_run_0, domain, bColoring, LEGION_COMPUTE_KIND);
  auto b_vals_partition = copyPartition(ctx, runtime, b_dense_run_0_Partition_0, get_logical_region(b_vals));
  computePartitions.bPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.bPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.bPartition.valsPartition = b_vals_partition;
  computePartitions.bPartition.denseLevelRunPartitions[0] = b_dense_run_0_Partition_0;

  return computePartitions;
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b_vals = regions[0];
  LogicalRegion b_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY) {
  auto b_vals_parent = b->valsParent;
  auto b_vals_field_id = b->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.gridX = gridX;
  taskArgsRaw2.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->bPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, b_vals_parent).add_field(b_vals_field_id));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, int32_t gridX, int32_t gridY) {
  size_t c1_dimension = c->dims[0];
  size_t c2_dimension = c->dims[1];
  RegionWrapper c_vals = c->vals;
  IndexSpace c_dense_run_0 = c->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionC();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gridX - 1)) / gridX)), (jn * ((c2_dimension + (gridY - 1)) / gridY)));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)), cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)), cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  IndexPartition c_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, c_dense_run_0, domain, cColoring, LEGION_COMPUTE_KIND);
  auto c_vals_partition = copyPartition(ctx, runtime, c_dense_run_0_Partition_0, get_logical_region(c_vals));
  computePartitions.cPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.cPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.cPartition.valsPartition = c_vals_partition;
  computePartitions.cPartition.denseLevelRunPartitions[0] = c_dense_run_0_Partition_0;

  return computePartitions;
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c_vals = regions[0];
  LogicalRegion c_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, partitionPackForplaceLegionC* partitionPack, int32_t gridX, int32_t gridY) {
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_3Args taskArgsRaw3;
  taskArgsRaw3.gridX = gridX;
  taskArgsRaw3.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(c_vals_field_id));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_5Args* args = (task_5Args*)(task->args);
  IndexPartition a_dense_run_0_Partition_0 = args->a_dense_run_0_Partition_0;
  int64_t b1_dimension = args->b1_dimension;
  int64_t b2_dimension = args->b2_dimension;
  IndexSpace b_dense_run_0 = args->b_dense_run_0;
  IndexPartition b_dense_run_0_Partition_0 = args->b_dense_run_0_Partition_0;
  int64_t c2_dimension = args->c2_dimension;
  IndexSpace c_dense_run_0 = args->c_dense_run_0;
  IndexPartition c_dense_run_0_Partition_0 = args->c_dense_run_0_Partition_0;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * gridY + jn;
  b_dense_run_0 = runtime->get_index_subspace(ctx, b_dense_run_0_Partition_0, task->index_point);
  c_dense_run_0 = runtime->get_index_subspace(ctx, c_dense_run_0_Partition_0, task->index_point);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto kosIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(kosIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t kos = (*itr)[0];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX)), (((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)), bDomain.hi()[0]), TACO_MIN((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)), bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX)), (jn * ((c2_dimension + (gridY - 1)) / gridY)));
    Point<2> cEnd = Point<2>(TACO_MIN((((jn + (in + kos)) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)), cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)), cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  IndexPartition b_dense_run_0_Partition_2 = runtime->create_index_partition(ctx, b_dense_run_0, domain, bColoring, LEGION_COMPUTE_KIND);
  auto b_vals_partition = copyPartition(ctx, runtime, b_dense_run_0_Partition_2, get_logical_region(b_vals), pointID2);
  IndexPartition c_dense_run_0_Partition_2 = runtime->create_index_partition(ctx, c_dense_run_0, domain, cColoring, LEGION_COMPUTE_KIND);
  auto c_vals_partition = copyPartition(ctx, runtime, c_dense_run_0_Partition_2, get_logical_region(c_vals), pointID2);
}

partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, int32_t gridX, int32_t gridY) {
  RegionWrapper a_vals = a->vals;
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];
  size_t b1_dimension = b->dims[0];
  size_t b2_dimension = b->dims[1];
  RegionWrapper b_vals = b->vals;
  auto b_vals_parent = b->valsParent;
  auto b_vals_field_id = b->valsFieldID;
  IndexSpace b_dense_run_0 = b->denseLevelRuns[0];
  size_t c2_dimension = c->dims[1];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;
  IndexSpace c_dense_run_0 = c->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegion();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  auto bDomain = runtime->get_index_space_domain(ctx, b_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX)), (jn * ((c2_dimension + (gridY - 1)) / gridY)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)), aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)), aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX)), 0);
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)), bDomain.hi()[0]), TACO_MIN(((gridX - 1) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)), bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>(0, (jn * ((c2_dimension + (gridY - 1)) / gridY)));
    Point<2> cEnd = Point<2>(TACO_MIN(((gridX - 1) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)), cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)), cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  IndexPartition a_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, a_dense_run_0, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto a_vals_partition = copyPartition(ctx, runtime, a_dense_run_0_Partition_0, get_logical_region(a_vals));
  IndexPartition b_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, b_dense_run_0, domain, bColoring, LEGION_COMPUTE_KIND);
  auto b_vals_partition = copyPartition(ctx, runtime, b_dense_run_0_Partition_0, get_logical_region(b_vals));
  IndexPartition c_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, c_dense_run_0, domain, cColoring, LEGION_COMPUTE_KIND);
  auto c_vals_partition = copyPartition(ctx, runtime, c_dense_run_0_Partition_0, get_logical_region(c_vals));
  task_5Args taskArgsRaw5;
  taskArgsRaw5.a_dense_run_0_Partition_0 = a_dense_run_0_Partition_0;
  taskArgsRaw5.b1_dimension = b1_dimension;
  taskArgsRaw5.b2_dimension = b2_dimension;
  taskArgsRaw5.b_dense_run_0 = b_dense_run_0;
  taskArgsRaw5.b_dense_run_0_Partition_0 = b_dense_run_0_Partition_0;
  taskArgsRaw5.c2_dimension = c2_dimension;
  taskArgsRaw5.c_dense_run_0 = c_dense_run_0;
  taskArgsRaw5.c_dense_run_0_Partition_0 = c_dense_run_0_Partition_0;
  taskArgsRaw5.gridX = gridX;
  taskArgsRaw5.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw5, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(a_vals_partition, 0, READ_ONLY, EXCLUSIVE, a_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(b_vals_partition, 0, READ_ONLY, EXCLUSIVE, b_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(b_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(c_vals_partition, 0, READ_ONLY, EXCLUSIVE, c_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(c_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();


  computePartitions.aPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.aPartition.valsPartition = a_vals_partition;
  computePartitions.aPartition.denseLevelRunPartitions[0] = a_dense_run_0_Partition_0;
  computePartitions.bPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.bPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.bPartition.valsPartition = b_vals_partition;
  computePartitions.bPartition.denseLevelRunPartitions[0] = b_dense_run_0_Partition_0;
  computePartitions.cPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.cPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.cPartition.valsPartition = c_vals_partition;
  computePartitions.cPartition.denseLevelRunPartitions[0] = c_dense_run_0_Partition_0;

  return computePartitions;
}

void task_6(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  task_6Args* args = (task_6Args*)(task->args);
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t gridX = args->gridX;
  int64_t kos = args->kos;

  auto b_vals_ro_accessor = createAccessor<AccessorROdouble2>(b_vals, b_vals_field_id);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble2>(c_vals, c_vals_field_id);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble2>(a_vals, a_vals_field_id);

  auto aDomain = runtime->get_index_space_domain(ctx, a_vals.get_logical_region().get_index_space());
  auto bDomain = runtime->get_index_space_domain(ctx, b_vals.get_logical_region().get_index_space());
  auto cDomain = runtime->get_index_space_domain(ctx, c_vals.get_logical_region().get_index_space());
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
    b_vals_ro_accessor.ptr(bDomain.lo()),
    (b_vals_ro_accessor.accessor.strides[0] / sizeof(double)),
    c_vals_ro_accessor.ptr(cDomain.lo()),
    (c_vals_ro_accessor.accessor.strides[0] / sizeof(double)),
    1.00000000,
    a_vals_rw_accessor.ptr(aDomain.lo()),
    (a_vals_rw_accessor.accessor.strides[0] / sizeof(double))
  );
}

void task_7(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_7Args* args = (task_7Args*)(task->args);
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * gridY + jn;
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto kosIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(kosIndexSpace));
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t kos = (*itr);
    task_6Args taskArgsRaw6;
    taskArgsRaw6.a_vals_field_id = a_vals_field_id;
    taskArgsRaw6.b_vals_field_id = b_vals_field_id;
    taskArgsRaw6.c_vals_field_id = c_vals_field_id;
    taskArgsRaw6.gridX = gridX;
    taskArgsRaw6.kos = kos;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw6, sizeof(task_6Args));
    TaskLauncher launcher = TaskLauncher(taskID(6), taskArgs);
    launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), READ_WRITE, EXCLUSIVE, a_vals_parent, 0).add_field(a_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(b_vals), pointID2), kos)), READ_ONLY, EXCLUSIVE, b_vals_parent, 0).add_field(b_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(c_vals), pointID2), kos)), READ_ONLY, EXCLUSIVE, c_vals_parent, 0).add_field(c_vals_field_id));
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, partitionPackForcomputeLegion* partitionPack, int32_t gridX, int32_t gridY) {
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;
  auto b_vals_parent = b->valsParent;
  auto b_vals_field_id = b->valsFieldID;
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_7Args taskArgsRaw7;
  taskArgsRaw7.a_vals_field_id = a_vals_field_id;
  taskArgsRaw7.b_vals_field_id = b_vals_field_id;
  taskArgsRaw7.c_vals_field_id = c_vals_field_id;
  taskArgsRaw7.gridX = gridX;
  taskArgsRaw7.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw7, sizeof(task_7Args));
  IndexLauncher launcher = IndexLauncher(taskID(7), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, a_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->bPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, b_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(b_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, c_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(c_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

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
    TaskVariantRegistrar registrar(taskID(5), "task_5");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_5>(registrar, "task_5");
  }
  {
    TaskVariantRegistrar registrar(taskID(6), "task_6");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_6>(registrar, "task_6");
  }
  {
    TaskVariantRegistrar registrar(taskID(7), "task_7");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_7>(registrar, "task_7");
  }
}
void dynamicallyRegisterDISTALTasks(Legion::Context ctx, Legion::Runtime* runtime) {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_1>(registrar);
    runtime->attach_name(taskID(1), "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_2>(registrar);
    runtime->attach_name(taskID(2), "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_3>(registrar);
    runtime->attach_name(taskID(3), "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(5), "task_5");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_5>(registrar);
    runtime->attach_name(taskID(5), "task_5");
  }
  {
    TaskVariantRegistrar registrar(taskID(6), "task_6");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_6>(registrar);
    runtime->attach_name(taskID(6), "task_6");
  }
  {
    TaskVariantRegistrar registrar(taskID(7), "task_7");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_7>(registrar);
    runtime->attach_name(taskID(7), "task_7");
  }
}
