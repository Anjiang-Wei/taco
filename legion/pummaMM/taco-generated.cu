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
  IndexPartition b_dense_run_0_Partition_3;
  int64_t c2_dimension;
  IndexSpace c_dense_run_0;
  IndexPartition c_dense_run_0_Partition_0;
  IndexPartition c_dense_run_0_Partition_3;
  int32_t gridX;
  int32_t gridY;
  int64_t in;
  int64_t jn;
  int64_t kos;
  int64_t pointID3;
};

struct task_6Args {
  IndexSpace a_dense_run_0;
  IndexPartition a_dense_run_0_Partition_0;
  Legion::FieldID a_vals_field_id;
  int64_t b1_dimension;
  int64_t b2_dimension;
  IndexSpace b_dense_run_0;
  IndexPartition b_dense_run_0_Partition_0;
  IndexPartition b_dense_run_0_Partition_2;
  Legion::FieldID b_vals_field_id;
  int64_t c2_dimension;
  IndexSpace c_dense_run_0;
  IndexPartition c_dense_run_0_Partition_0;
  IndexPartition c_dense_run_0_Partition_2;
  Legion::FieldID c_vals_field_id;
  int32_t gridX;
  int32_t gridY;
  int64_t in;
  int64_t jn;
  int64_t kos;
  int64_t pointID2;
};

struct task_7Args {
  IndexSpace a_dense_run_0;
  IndexPartition a_dense_run_0_Partition_0;
  Legion::FieldID a_vals_field_id;
  int64_t b1_dimension;
  int64_t b2_dimension;
  IndexSpace b_dense_run_0;
  IndexPartition b_dense_run_0_Partition_0;
  Legion::FieldID b_vals_field_id;
  int64_t c2_dimension;
  IndexSpace c_dense_run_0;
  IndexPartition c_dense_run_0_Partition_0;
  Legion::FieldID c_vals_field_id;
  int32_t gridX;
  int32_t gridY;
};

struct task_8Args {
  Legion::FieldID a_vals_field_id;
  Legion::FieldID b_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int64_t kio;
};

struct task_9Args {
  Legion::FieldID a_vals_field_id;
  Legion::FieldID b_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int64_t pointID3;
};

struct task_10Args {
  Legion::FieldID a_vals_field_id;
  Legion::FieldID b_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t gridX;
  int64_t kos;
  int64_t pointID2;
};

struct task_11Args {
  Legion::FieldID a_vals_field_id;
  Legion::FieldID b_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t gridX;
  int32_t gridY;
};


extern "C" partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, int32_t gridX, int32_t gridY) {
  size_t a1_dimension = a->dims[0];
  size_t a2_dimension = a->dims[1];
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionA();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2))), (jn * ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2))));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2) - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2) - 1)),aDomain.hi()[1]));
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

extern "C" void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
}

extern "C" void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY) {
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
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

extern "C" partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, int32_t gridX, int32_t gridY) {
  size_t b1_dimension = b->dims[0];
  size_t b2_dimension = b->dims[1];
  RegionWrapper b_vals = b->vals;
  IndexSpace b_dense_run_0 = b->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionB();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_dense_run_0);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX * 2 - 1)) / (gridX * 2))), (jn * ((b2_dimension + (gridY * 2 - 1)) / (gridY * 2))));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + ((b1_dimension + (gridX * 2 - 1)) / (gridX * 2) - 1)),bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + ((b2_dimension + (gridY * 2 - 1)) / (gridY * 2) - 1)),bDomain.hi()[1]));
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

extern "C" void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b_vals = regions[0];
  LogicalRegion b_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
}

extern "C" void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY) {
  auto b_vals_parent = b->valsParent;
  auto b_vals_field_id = b->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
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

extern "C" partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, int32_t gridX, int32_t gridY) {
  size_t c1_dimension = c->dims[0];
  size_t c2_dimension = c->dims[1];
  RegionWrapper c_vals = c->vals;
  IndexSpace c_dense_run_0 = c->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionC();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gridX * 2 - 1)) / (gridX * 2))), (jn * ((c2_dimension + (gridY * 2 - 1)) / (gridY * 2))));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + ((c1_dimension + (gridX * 2 - 1)) / (gridX * 2) - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + ((c2_dimension + (gridY * 2 - 1)) / (gridY * 2) - 1)),cDomain.hi()[1]));
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

extern "C" void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c_vals = regions[0];
  LogicalRegion c_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
}

extern "C" void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* c, partitionPackForplaceLegionC* partitionPack, int32_t gridX, int32_t gridY) {
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
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

extern "C" void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  int64_t distFused1 = task->index_point[0];
  task_5Args* args = (task_5Args*)(task->args);
  IndexPartition a_dense_run_0_Partition_0 = args->a_dense_run_0_Partition_0;
  int64_t b1_dimension = args->b1_dimension;
  int64_t b2_dimension = args->b2_dimension;
  IndexSpace b_dense_run_0 = args->b_dense_run_0;
  IndexPartition b_dense_run_0_Partition_0 = args->b_dense_run_0_Partition_0;
  IndexPartition b_dense_run_0_Partition_3 = args->b_dense_run_0_Partition_3;
  int64_t c2_dimension = args->c2_dimension;
  IndexSpace c_dense_run_0 = args->c_dense_run_0;
  IndexPartition c_dense_run_0_Partition_0 = args->c_dense_run_0_Partition_0;
  IndexPartition c_dense_run_0_Partition_3 = args->c_dense_run_0_Partition_3;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int64_t in = args->in;
  int64_t jn = args->jn;
  int64_t kos = args->kos;
  int64_t pointID3 = args->pointID3;


  int64_t iln = getIndexPoint(task, 0);
  int64_t jln = getIndexPoint(task, 1);
  int64_t pointID4 = pointID3 * 2 + iln;
  int64_t pointID5 = pointID4 * 2 + jln;
  b_dense_run_0 = runtime->get_index_subspace(ctx, b_dense_run_0_Partition_3, task->index_point);
  c_dense_run_0 = runtime->get_index_subspace(ctx, c_dense_run_0_Partition_3, task->index_point);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(1);
  auto kioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(kioIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t kio = (*itr)[0];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + iln * (((b1_dimension + (gridX - 1)) / gridX + 1) / 2)), (((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + kio * (((b2_dimension + (gridX - 1)) / gridX + 1) / 2)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * (((b1_dimension + (gridX - 1)) / gridX + 1) / 2) + (((b1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),bDomain.hi()[0]), TACO_MIN((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + (kio * (((b2_dimension + (gridX - 1)) / gridX + 1) / 2) + (((b2_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + kio * (((b2_dimension + (gridX - 1)) / gridX + 1) / 2)), (jn * ((c2_dimension + (gridY - 1)) / gridY) + jln * (((c2_dimension + (gridY - 1)) / gridY + 1) / 2)));
    Point<2> cEnd = Point<2>(TACO_MIN((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + (kio * (((b2_dimension + (gridX - 1)) / gridX + 1) / 2) + (((b2_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * (((c2_dimension + (gridY - 1)) / gridY + 1) / 2) + (((c2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  IndexPartition b_dense_run_0_Partition_5 = runtime->create_index_partition(ctx, b_dense_run_0, domain, bColoring, LEGION_COMPUTE_KIND);
  auto b_vals_partition = copyPartition(ctx, runtime, b_dense_run_0_Partition_5, get_logical_region(b_vals), pointID5);
  IndexPartition c_dense_run_0_Partition_5 = runtime->create_index_partition(ctx, c_dense_run_0, domain, cColoring, LEGION_COMPUTE_KIND);
  auto c_vals_partition = copyPartition(ctx, runtime, c_dense_run_0_Partition_5, get_logical_region(c_vals), pointID5);
}

extern "C" void task_6(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  task_6Args* args = (task_6Args*)(task->args);
  IndexSpace a_dense_run_0 = args->a_dense_run_0;
  IndexPartition a_dense_run_0_Partition_0 = args->a_dense_run_0_Partition_0;
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  int64_t b1_dimension = args->b1_dimension;
  int64_t b2_dimension = args->b2_dimension;
  IndexSpace b_dense_run_0 = args->b_dense_run_0;
  IndexPartition b_dense_run_0_Partition_0 = args->b_dense_run_0_Partition_0;
  IndexPartition b_dense_run_0_Partition_2 = args->b_dense_run_0_Partition_2;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  int64_t c2_dimension = args->c2_dimension;
  IndexSpace c_dense_run_0 = args->c_dense_run_0;
  IndexPartition c_dense_run_0_Partition_0 = args->c_dense_run_0_Partition_0;
  IndexPartition c_dense_run_0_Partition_2 = args->c_dense_run_0_Partition_2;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int64_t in = args->in;
  int64_t jn = args->jn;
  int64_t kos = args->kos;
  int64_t pointID2 = args->pointID2;


  int64_t pointID3 = pointID2 * gridX + kos;
  b_dense_run_0 = runtime->get_index_subspace(ctx, b_dense_run_0_Partition_2, kos);
  c_dense_run_0 = runtime->get_index_subspace(ctx, c_dense_run_0_Partition_2, kos);
  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFused1IndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFused1IndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t iln = (*itr)[0];
    int64_t jln = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + iln * (((b1_dimension + (gridX - 1)) / gridX + 1) / 2)), (((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * (((b1_dimension + (gridX - 1)) / gridX + 1) / 2) + (((b1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),bDomain.hi()[0]), TACO_MIN((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX)), (jn * ((c2_dimension + (gridY - 1)) / gridY) + jln * (((c2_dimension + (gridY - 1)) / gridY + 1) / 2)));
    Point<2> cEnd = Point<2>(TACO_MIN((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * (((c2_dimension + (gridY - 1)) / gridY + 1) / 2) + (((c2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
    Point<2> aStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + iln * (((b1_dimension + (gridX - 1)) / gridX + 1) / 2)), (jn * ((c2_dimension + (gridY - 1)) / gridY) + jln * (((c2_dimension + (gridY - 1)) / gridY + 1) / 2)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * (((b1_dimension + (gridX - 1)) / gridX + 1) / 2) + (((b1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * (((c2_dimension + (gridY - 1)) / gridY + 1) / 2) + (((c2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  IndexPartition b_dense_run_0_Partition_3 = runtime->create_index_partition(ctx, b_dense_run_0, domain, bColoring, LEGION_COMPUTE_KIND);
  auto b_vals_partition = copyPartition(ctx, runtime, b_dense_run_0_Partition_3, get_logical_region(b_vals), pointID3);
  IndexPartition c_dense_run_0_Partition_3 = runtime->create_index_partition(ctx, c_dense_run_0, domain, cColoring, LEGION_COMPUTE_KIND);
  auto c_vals_partition = copyPartition(ctx, runtime, c_dense_run_0_Partition_3, get_logical_region(c_vals), pointID3);
  IndexPartition a_dense_run_0_Partition_3 = runtime->create_index_partition(ctx, a_dense_run_0, domain, aColoring, LEGION_COMPUTE_KIND);
  auto a_vals_partition = copyPartition(ctx, runtime, a_dense_run_0_Partition_3, get_logical_region(a_vals), pointID3);
  task_5Args taskArgsRaw5;
  taskArgsRaw5.a_dense_run_0_Partition_0 = a_dense_run_0_Partition_0;
  taskArgsRaw5.b1_dimension = b1_dimension;
  taskArgsRaw5.b2_dimension = b2_dimension;
  taskArgsRaw5.b_dense_run_0 = b_dense_run_0;
  taskArgsRaw5.b_dense_run_0_Partition_0 = b_dense_run_0_Partition_0;
  taskArgsRaw5.b_dense_run_0_Partition_3 = b_dense_run_0_Partition_3;
  taskArgsRaw5.c2_dimension = c2_dimension;
  taskArgsRaw5.c_dense_run_0 = c_dense_run_0;
  taskArgsRaw5.c_dense_run_0_Partition_0 = c_dense_run_0_Partition_0;
  taskArgsRaw5.c_dense_run_0_Partition_3 = c_dense_run_0_Partition_3;
  taskArgsRaw5.gridX = gridX;
  taskArgsRaw5.gridY = gridY;
  taskArgsRaw5.in = in;
  taskArgsRaw5.jn = jn;
  taskArgsRaw5.kos = kos;
  taskArgsRaw5.pointID3 = pointID3;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw5, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(
    a_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    a_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    b_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    b_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(b_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    c_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    c_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(c_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  launcher.tag = launcher.tag | Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  runtime->execute_index_space(ctx, launcher);

}

extern "C" void task_7(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_7Args* args = (task_7Args*)(task->args);
  IndexSpace a_dense_run_0 = args->a_dense_run_0;
  IndexPartition a_dense_run_0_Partition_0 = args->a_dense_run_0_Partition_0;
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  int64_t b1_dimension = args->b1_dimension;
  int64_t b2_dimension = args->b2_dimension;
  IndexSpace b_dense_run_0 = args->b_dense_run_0;
  IndexPartition b_dense_run_0_Partition_0 = args->b_dense_run_0_Partition_0;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  int64_t c2_dimension = args->c2_dimension;
  IndexSpace c_dense_run_0 = args->c_dense_run_0;
  IndexPartition c_dense_run_0_Partition_0 = args->c_dense_run_0_Partition_0;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * gridY + jn;
  b_dense_run_0 = runtime->get_index_subspace(ctx, b_dense_run_0_Partition_0, task->index_point);
  c_dense_run_0 = runtime->get_index_subspace(ctx, c_dense_run_0_Partition_0, task->index_point);
  a_dense_run_0 = runtime->get_index_subspace(ctx, a_dense_run_0_Partition_0, task->index_point);
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
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX)), (((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[0]), TACO_MIN((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX)), (jn * ((c2_dimension + (gridY - 1)) / gridY)));
    Point<2> cEnd = Point<2>(TACO_MIN((((in + kos) % gridX) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)),cDomain.hi()[1]));
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
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t kos = (*itr);
    task_6Args taskArgsRaw6;
    taskArgsRaw6.a_dense_run_0 = a_dense_run_0;
    taskArgsRaw6.a_dense_run_0_Partition_0 = a_dense_run_0_Partition_0;
    taskArgsRaw6.a_vals_field_id = a_vals_field_id;
    taskArgsRaw6.b1_dimension = b1_dimension;
    taskArgsRaw6.b2_dimension = b2_dimension;
    taskArgsRaw6.b_dense_run_0 = b_dense_run_0;
    taskArgsRaw6.b_dense_run_0_Partition_0 = b_dense_run_0_Partition_0;
    taskArgsRaw6.b_dense_run_0_Partition_2 = b_dense_run_0_Partition_2;
    taskArgsRaw6.b_vals_field_id = b_vals_field_id;
    taskArgsRaw6.c2_dimension = c2_dimension;
    taskArgsRaw6.c_dense_run_0 = c_dense_run_0;
    taskArgsRaw6.c_dense_run_0_Partition_0 = c_dense_run_0_Partition_0;
    taskArgsRaw6.c_dense_run_0_Partition_2 = c_dense_run_0_Partition_2;
    taskArgsRaw6.c_vals_field_id = c_vals_field_id;
    taskArgsRaw6.gridX = gridX;
    taskArgsRaw6.gridY = gridY;
    taskArgsRaw6.in = in;
    taskArgsRaw6.jn = jn;
    taskArgsRaw6.kos = kos;
    taskArgsRaw6.pointID2 = pointID2;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw6, sizeof(task_6Args));
    TaskLauncher launcher = TaskLauncher(taskID(6), taskArgs);
    launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), READ_ONLY, EXCLUSIVE, a_vals_parent, (0 | Mapping::DefaultMapper::VIRTUAL_MAP)).add_field(a_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, b_vals_partition, kos)), READ_ONLY, EXCLUSIVE, b_vals_parent, (0 | Mapping::DefaultMapper::VIRTUAL_MAP)).add_field(b_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, c_vals_partition, kos)), READ_ONLY, EXCLUSIVE, c_vals_parent, (0 | Mapping::DefaultMapper::VIRTUAL_MAP)).add_field(c_vals_field_id));
    launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

extern "C" partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, int32_t gridX, int32_t gridY) {
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
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX)), 0);
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[0]), TACO_MIN(((gridX - 1) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>(0, (jn * ((c2_dimension + (gridY - 1)) / gridY)));
    Point<2> cEnd = Point<2>(TACO_MIN(((gridX - 1) * ((b2_dimension + (gridX - 1)) / gridX) + ((b2_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)),cDomain.hi()[1]));
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
  task_7Args taskArgsRaw7;
  taskArgsRaw7.a_dense_run_0 = a_dense_run_0;
  taskArgsRaw7.a_dense_run_0_Partition_0 = a_dense_run_0_Partition_0;
  taskArgsRaw7.a_vals_field_id = a_vals_field_id;
  taskArgsRaw7.b1_dimension = b1_dimension;
  taskArgsRaw7.b2_dimension = b2_dimension;
  taskArgsRaw7.b_dense_run_0 = b_dense_run_0;
  taskArgsRaw7.b_dense_run_0_Partition_0 = b_dense_run_0_Partition_0;
  taskArgsRaw7.b_vals_field_id = b_vals_field_id;
  taskArgsRaw7.c2_dimension = c2_dimension;
  taskArgsRaw7.c_dense_run_0 = c_dense_run_0;
  taskArgsRaw7.c_dense_run_0_Partition_0 = c_dense_run_0_Partition_0;
  taskArgsRaw7.c_vals_field_id = c_vals_field_id;
  taskArgsRaw7.gridX = gridX;
  taskArgsRaw7.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw7, sizeof(task_7Args));
  IndexLauncher launcher = IndexLauncher(taskID(7), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(
    a_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    a_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    b_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    b_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(b_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    c_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    c_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(c_vals_field_id));
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

extern "C" void task_8(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  task_8Args* args = (task_8Args*)(task->args);
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int64_t kio = args->kio;

  auto b_vals_ro_accessor = createAccessor<AccessorROdouble2>(b_vals, b_vals_field_id);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble2>(c_vals, c_vals_field_id);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble2>(a_vals, a_vals_field_id);

  auto aDomain = runtime->get_index_space_domain(ctx, a_vals.get_logical_region().get_index_space());
  auto bDomain = runtime->get_index_space_domain(ctx, b_vals.get_logical_region().get_index_space());
  auto cDomain = runtime->get_index_space_domain(ctx, c_vals.get_logical_region().get_index_space());
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
    c_vals_ro_accessor.ptr(cDomain.lo()),
    (c_vals_ro_accessor.accessor.strides[0] / sizeof(double)),
    b_vals_ro_accessor.ptr(bDomain.lo()),
    (b_vals_ro_accessor.accessor.strides[0] / sizeof(double)),
    &(alpha),
    a_vals_rw_accessor.ptr(aDomain.lo()),
    (a_vals_rw_accessor.accessor.strides[0] / sizeof(double))
  ));
}

extern "C" void task_9(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  int64_t distFused1 = task->index_point[0];
  task_9Args* args = (task_9Args*)(task->args);
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int64_t pointID3 = args->pointID3;


  int64_t iln = getIndexPoint(task, 0);
  int64_t jln = getIndexPoint(task, 1);
  int64_t pointID4 = pointID3 * 2 + iln;
  int64_t pointID5 = pointID4 * 2 + jln;
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(1);
  auto kioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(kioIndexSpace));
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t kio = (*itr);
    task_8Args taskArgsRaw8;
    taskArgsRaw8.a_vals_field_id = a_vals_field_id;
    taskArgsRaw8.b_vals_field_id = b_vals_field_id;
    taskArgsRaw8.c_vals_field_id = c_vals_field_id;
    taskArgsRaw8.kio = kio;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw8, sizeof(task_8Args));
    TaskLauncher launcher = TaskLauncher(taskID(8), taskArgs);
    launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), READ_WRITE, EXCLUSIVE, a_vals_parent, 0).add_field(a_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(b_vals), pointID5), kio)), READ_ONLY, EXCLUSIVE, b_vals_parent, 0).add_field(b_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(c_vals), pointID5), kio)), READ_ONLY, EXCLUSIVE, c_vals_parent, 0).add_field(c_vals_field_id));
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

extern "C" void task_10(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  task_10Args* args = (task_10Args*)(task->args);
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t gridX = args->gridX;
  int64_t kos = args->kos;
  int64_t pointID2 = args->pointID2;


  int64_t pointID3 = pointID2 * gridX + kos;
  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFused1IndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFused1IndexSpace));
  task_9Args taskArgsRaw9;
  taskArgsRaw9.a_vals_field_id = a_vals_field_id;
  taskArgsRaw9.b_vals_field_id = b_vals_field_id;
  taskArgsRaw9.c_vals_field_id = c_vals_field_id;
  taskArgsRaw9.pointID3 = pointID3;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw9, sizeof(task_9Args));
  IndexLauncher launcher = IndexLauncher(taskID(9), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(
    runtime->get_logical_partition_by_color(ctx, get_logical_region(a_vals), pointID3),
    0,
    READ_WRITE,
    EXCLUSIVE,
    a_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    runtime->get_logical_partition_by_color(ctx, get_logical_region(b_vals), pointID3),
    0,
    READ_ONLY,
    EXCLUSIVE,
    b_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(b_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    runtime->get_logical_partition_by_color(ctx, get_logical_region(c_vals), pointID3),
    0,
    READ_ONLY,
    EXCLUSIVE,
    c_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(c_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  launcher.tag = launcher.tag | Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  runtime->execute_index_space(ctx, launcher);

}

extern "C" void task_11(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b_vals = regions[1];
  LogicalRegion b_vals_parent = regions[1].get_logical_region();
  PhysicalRegion c_vals = regions[2];
  LogicalRegion c_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_11Args* args = (task_11Args*)(task->args);
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
    task_10Args taskArgsRaw10;
    taskArgsRaw10.a_vals_field_id = a_vals_field_id;
    taskArgsRaw10.b_vals_field_id = b_vals_field_id;
    taskArgsRaw10.c_vals_field_id = c_vals_field_id;
    taskArgsRaw10.gridX = gridX;
    taskArgsRaw10.kos = kos;
    taskArgsRaw10.pointID2 = pointID2;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw10, sizeof(task_10Args));
    TaskLauncher launcher = TaskLauncher(taskID(10), taskArgs);
    launcher.add_region_requirement(RegionRequirement(get_logical_region(a_vals), READ_WRITE, EXCLUSIVE, a_vals_parent, (0 | Mapping::DefaultMapper::VIRTUAL_MAP)).add_field(a_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(b_vals), pointID2), kos)), READ_ONLY, EXCLUSIVE, b_vals_parent, (0 | Mapping::DefaultMapper::VIRTUAL_MAP)).add_field(b_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(c_vals), pointID2), kos)), READ_ONLY, EXCLUSIVE, c_vals_parent, (0 | Mapping::DefaultMapper::VIRTUAL_MAP)).add_field(c_vals_field_id));
    launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

extern "C" void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, partitionPackForcomputeLegion* partitionPack, int32_t gridX, int32_t gridY) {
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
  task_11Args taskArgsRaw11;
  taskArgsRaw11.a_vals_field_id = a_vals_field_id;
  taskArgsRaw11.b_vals_field_id = b_vals_field_id;
  taskArgsRaw11.c_vals_field_id = c_vals_field_id;
  taskArgsRaw11.gridX = gridX;
  taskArgsRaw11.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw11, sizeof(task_11Args));
  IndexLauncher launcher = IndexLauncher(taskID(11), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(
    partitionPack->aPartition.valsPartition,
    0,
    READ_WRITE,
    EXCLUSIVE,
    a_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    partitionPack->bPartition.valsPartition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    b_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(b_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    partitionPack->cPartition.valsPartition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    c_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(c_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}
extern "C" void registerTacoTasks() {
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
  {
    TaskVariantRegistrar registrar(taskID(8), "task_8");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_8>(registrar, "task_8");
  }
  {
    TaskVariantRegistrar registrar(taskID(9), "task_9");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_9>(registrar, "task_9");
  }
  {
    TaskVariantRegistrar registrar(taskID(10), "task_10");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_10>(registrar, "task_10");
  }
  {
    TaskVariantRegistrar registrar(taskID(11), "task_11");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_11>(registrar, "task_11");
  }
}
extern "C" void dynamicallyRegisterDISTALTasks(void** args) {
  Legion::Context ctx = (Legion::Context)args[0];
  Legion::Runtime* runtime = (Legion::Runtime*)args[1];
  Legion::Processor::enable_scheduler_lock();
  auto barrier = runtime->create_phase_barrier(ctx, runtime->get_num_shards(ctx, true));
  barrier.arrive();
  barrier = runtime->advance_phase_barrier(ctx, barrier);
  barrier.wait();
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_1>(registrar);
    runtime->attach_name(taskID(1), "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_2>(registrar);
    runtime->attach_name(taskID(2), "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_3>(registrar);
    runtime->attach_name(taskID(3), "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(5), "task_5", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_5>(registrar);
    runtime->attach_name(taskID(5), "task_5");
  }
  {
    TaskVariantRegistrar registrar(taskID(6), "task_6", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_6>(registrar);
    runtime->attach_name(taskID(6), "task_6");
  }
  {
    TaskVariantRegistrar registrar(taskID(7), "task_7", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_7>(registrar);
    runtime->attach_name(taskID(7), "task_7");
  }
  {
    TaskVariantRegistrar registrar(taskID(8), "task_8", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_8>(registrar);
    runtime->attach_name(taskID(8), "task_8");
  }
  {
    TaskVariantRegistrar registrar(taskID(9), "task_9", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_9>(registrar);
    runtime->attach_name(taskID(9), "task_9");
  }
  {
    TaskVariantRegistrar registrar(taskID(10), "task_10", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_10>(registrar);
    runtime->attach_name(taskID(10), "task_10");
  }
  {
    TaskVariantRegistrar registrar(taskID(11), "task_11", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_11>(registrar);
    runtime->attach_name(taskID(11), "task_11");
  }
  barrier.arrive();
  barrier = runtime->advance_phase_barrier(ctx, barrier);
  barrier.wait();
  runtime->destroy_phase_barrier(ctx, barrier);
  Legion::Processor::disable_scheduler_lock();
}
