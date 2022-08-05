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
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;

struct task_1Args {
  int64_t sfID;
  int32_t c;
  int32_t rpoc;
};

struct task_2Args {
  int64_t sfID;
  int32_t c;
  int32_t rpoc;
};

struct task_3Args {
  int64_t sfID;
  int32_t c;
  int32_t rpoc;
};

struct task_5Args {
  IndexPartition A_dense_run_0_Partition_0;
  int64_t B1_dimension;
  int64_t B2_dimension;
  IndexSpace B_dense_run_0;
  IndexPartition B_dense_run_0_Partition_0;
  int64_t C2_dimension;
  IndexSpace C_dense_run_0;
  IndexPartition C_dense_run_0_Partition_0;
  int32_t c;
  int32_t rpoc;
  int32_t rpoc3;
};

struct task_6Args {
  Legion::FieldID A_vals_field_id;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID C_vals_field_id;
  int64_t k1s;
  int32_t rpoc3;
};

struct task_7Args {
  Legion::FieldID A_vals_field_id;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID C_vals_field_id;
  int32_t c;
  int32_t rpoc;
  int32_t rpoc3;
};


extern "C" partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t rpoc) {
  size_t A1_dimension = A->dims[0];
  size_t A2_dimension = A->dims[1];
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionA();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (rpoc - 1)) / rpoc)), (jn * ((A2_dimension + (rpoc - 1)) / rpoc)));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (rpoc - 1)) / rpoc) + ((A1_dimension + (rpoc - 1)) / rpoc - 1)),ADomain.hi()[0]), TACO_MIN((jn * ((A2_dimension + (rpoc - 1)) / rpoc) + ((A2_dimension + (rpoc - 1)) / rpoc - 1)),ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  IndexPartition A_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_COMPUTE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition_0, get_logical_region(A_vals));
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition_0;

  return computePartitions;
}

extern "C" void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t kn = getIndexPoint(task, 2);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * rpoc + jn;
}

extern "C" void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, partitionPackForplaceLegionA* partitionPack, int32_t rpoc, int32_t c) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  std::vector<int> dims = std::vector<int>();
  dims.push_back(rpoc);
  dims.push_back(rpoc);
  dims.push_back(c);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(1), dims);
  task_1Args taskArgsRaw1;
  taskArgsRaw1.sfID = shardingID(1);
  taskArgsRaw1.c = c;
  taskArgsRaw1.rpoc = rpoc;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, A_vals_parent).add_field(A_vals_field_id));
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

extern "C" partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t rpoc) {
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionB();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> BStart = Point<2>((in * ((B1_dimension + (rpoc - 1)) / rpoc)), (jn * ((B2_dimension + (rpoc - 1)) / rpoc)));
    Point<2> BEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (rpoc - 1)) / rpoc) + ((B1_dimension + (rpoc - 1)) / rpoc - 1)),BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (rpoc - 1)) / rpoc) + ((B2_dimension + (rpoc - 1)) / rpoc - 1)),BDomain.hi()[1]));
    Rect<2> BRect = Rect<2>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  IndexPartition B_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition_0, get_logical_region(B_vals));
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_0;

  return computePartitions;
}

extern "C" void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B_vals = regions[0];
  LogicalRegion B_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t kn = getIndexPoint(task, 2);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * rpoc + jn;
}

extern "C" void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, partitionPackForplaceLegionB* partitionPack, int32_t rpoc, int32_t c) {
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  std::vector<int> dims = std::vector<int>();
  dims.push_back(rpoc);
  dims.push_back(rpoc);
  dims.push_back(c);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(3), dims);
  task_2Args taskArgsRaw2;
  taskArgsRaw2.sfID = shardingID(3);
  taskArgsRaw2.c = c;
  taskArgsRaw2.rpoc = rpoc;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

extern "C" partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t rpoc) {
  size_t C1_dimension = C->dims[0];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionC();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> CStart = Point<2>((in * ((C1_dimension + (rpoc - 1)) / rpoc)), (jn * ((C2_dimension + (rpoc - 1)) / rpoc)));
    Point<2> CEnd = Point<2>(TACO_MIN((in * ((C1_dimension + (rpoc - 1)) / rpoc) + ((C1_dimension + (rpoc - 1)) / rpoc - 1)),CDomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (rpoc - 1)) / rpoc) + ((C2_dimension + (rpoc - 1)) / rpoc - 1)),CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition C_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_0, get_logical_region(C_vals));
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition_0;

  return computePartitions;
}

extern "C" void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion C_vals = regions[0];
  LogicalRegion C_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t kn = getIndexPoint(task, 2);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * rpoc + jn;
}

extern "C" void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, partitionPackForplaceLegionC* partitionPack, int32_t rpoc, int32_t c) {
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  std::vector<int> dims = std::vector<int>();
  dims.push_back(rpoc);
  dims.push_back(rpoc);
  dims.push_back(c);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(5), dims);
  task_3Args taskArgsRaw3;
  taskArgsRaw3.sfID = shardingID(5);
  taskArgsRaw3.c = c;
  taskArgsRaw3.rpoc = rpoc;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(C_vals_field_id));
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

extern "C" void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_5Args* args = (task_5Args*)(task->args);
  IndexPartition A_dense_run_0_Partition_0 = args->A_dense_run_0_Partition_0;
  int64_t B1_dimension = args->B1_dimension;
  int64_t B2_dimension = args->B2_dimension;
  IndexSpace B_dense_run_0 = args->B_dense_run_0;
  IndexPartition B_dense_run_0_Partition_0 = args->B_dense_run_0_Partition_0;
  int64_t C2_dimension = args->C2_dimension;
  IndexSpace C_dense_run_0 = args->C_dense_run_0;
  IndexPartition C_dense_run_0_Partition_0 = args->C_dense_run_0_Partition_0;
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;
  int32_t rpoc3 = args->rpoc3;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t kn = getIndexPoint(task, 2);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * rpoc + jn;
  int64_t pointID3 = pointID2 * c + kn;
  B_dense_run_0 = runtime->get_index_subspace(ctx, B_dense_run_0_Partition_0, task->index_point);
  C_dense_run_0 = runtime->get_index_subspace(ctx, C_dense_run_0_Partition_0, task->index_point);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((rpoc3 - 1));
  auto k1sIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(k1sIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t k1s = (*itr)[0];
    Point<2> BStart = Point<2>((in * ((B1_dimension + (rpoc - 1)) / rpoc)), (kn * ((B2_dimension + (c - 1)) / c) + ((jn + (in + k1s)) % rpoc3) * (((B2_dimension + (c - 1)) / c + (rpoc3 - 1)) / rpoc3)));
    Point<2> BEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (rpoc - 1)) / rpoc) + ((B1_dimension + (rpoc - 1)) / rpoc - 1)),BDomain.hi()[0]), TACO_MIN((kn * ((B2_dimension + (c - 1)) / c) + (((jn + (in + k1s)) % rpoc3) * (((B2_dimension + (c - 1)) / c + (rpoc3 - 1)) / rpoc3) + (((B2_dimension + (c - 1)) / c + (rpoc3 - 1)) / rpoc3 - 1))),BDomain.hi()[1]));
    Rect<2> BRect = Rect<2>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> CStart = Point<2>((kn * ((B2_dimension + (c - 1)) / c) + ((jn + (in + k1s)) % rpoc3) * (((B2_dimension + (c - 1)) / c + (rpoc3 - 1)) / rpoc3)), (jn * ((C2_dimension + (rpoc - 1)) / rpoc)));
    Point<2> CEnd = Point<2>(TACO_MIN((kn * ((B2_dimension + (c - 1)) / c) + (((jn + (in + k1s)) % rpoc3) * (((B2_dimension + (c - 1)) / c + (rpoc3 - 1)) / rpoc3) + (((B2_dimension + (c - 1)) / c + (rpoc3 - 1)) / rpoc3 - 1))),CDomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (rpoc - 1)) / rpoc) + ((C2_dimension + (rpoc - 1)) / rpoc - 1)),CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition B_dense_run_0_Partition_3 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition_3, get_logical_region(B_vals), pointID3);
  IndexPartition C_dense_run_0_Partition_3 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_3, get_logical_region(C_vals), pointID3);
}

extern "C" partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t rpoc, int32_t c, int32_t rpoc3) {
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegion();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), (c - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    int64_t kn = (*itr)[2];
    Point<2> AStart = Point<2>((in * ((B1_dimension + (rpoc - 1)) / rpoc)), (jn * ((C2_dimension + (rpoc - 1)) / rpoc)));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (rpoc - 1)) / rpoc) + ((B1_dimension + (rpoc - 1)) / rpoc - 1)),ADomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (rpoc - 1)) / rpoc) + ((C2_dimension + (rpoc - 1)) / rpoc - 1)),ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<2> BStart = Point<2>((in * ((B1_dimension + (rpoc - 1)) / rpoc)), (kn * ((B2_dimension + (c - 1)) / c)));
    Point<2> BEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (rpoc - 1)) / rpoc) + ((B1_dimension + (rpoc - 1)) / rpoc - 1)),BDomain.hi()[0]), TACO_MIN((kn * ((B2_dimension + (c - 1)) / c) + ((B2_dimension + (c - 1)) / c - 1)),BDomain.hi()[1]));
    Rect<2> BRect = Rect<2>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> CStart = Point<2>((kn * ((B2_dimension + (c - 1)) / c)), (jn * ((C2_dimension + (rpoc - 1)) / rpoc)));
    Point<2> CEnd = Point<2>(TACO_MIN((kn * ((B2_dimension + (c - 1)) / c) + ((B2_dimension + (c - 1)) / c - 1)),CDomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (rpoc - 1)) / rpoc) + ((C2_dimension + (rpoc - 1)) / rpoc - 1)),CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition A_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_COMPUTE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition_0, get_logical_region(A_vals));
  IndexPartition B_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition_0, get_logical_region(B_vals));
  IndexPartition C_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_0, get_logical_region(C_vals));
  task_5Args taskArgsRaw5;
  taskArgsRaw5.A_dense_run_0_Partition_0 = A_dense_run_0_Partition_0;
  taskArgsRaw5.B1_dimension = B1_dimension;
  taskArgsRaw5.B2_dimension = B2_dimension;
  taskArgsRaw5.B_dense_run_0 = B_dense_run_0;
  taskArgsRaw5.B_dense_run_0_Partition_0 = B_dense_run_0_Partition_0;
  taskArgsRaw5.C2_dimension = C2_dimension;
  taskArgsRaw5.C_dense_run_0 = C_dense_run_0;
  taskArgsRaw5.C_dense_run_0_Partition_0 = C_dense_run_0_Partition_0;
  taskArgsRaw5.c = c;
  taskArgsRaw5.rpoc = rpoc;
  taskArgsRaw5.rpoc3 = rpoc3;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw5, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(
    A_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    A_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    B_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    B_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    C_vals_partition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    C_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(C_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);


  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition_0;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_0;
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition_0;

  return computePartitions;
}

extern "C" void task_6(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  task_6Args* args = (task_6Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int64_t k1s = args->k1s;
  int32_t rpoc3 = args->rpoc3;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble2>(B_vals, B_vals_field_id);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, C_vals_field_id);
  auto A_vals_red_accessor = createAccessor<AccessorReducedouble2>(A_vals, A_vals_field_id, LEGION_REDOP_SUM_FLOAT64);

  auto aDomain = runtime->get_index_space_domain(ctx, A_vals.get_logical_region().get_index_space());
  auto bDomain = runtime->get_index_space_domain(ctx, B_vals.get_logical_region().get_index_space());
  auto cDomain = runtime->get_index_space_domain(ctx, C_vals.get_logical_region().get_index_space());
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
    C_vals_ro_accessor.ptr(cDomain.lo()),
    (C_vals_ro_accessor.accessor.strides[0] / sizeof(double)),
    B_vals_ro_accessor.ptr(bDomain.lo()),
    (B_vals_ro_accessor.accessor.strides[0] / sizeof(double)),
    &(alpha),
    A_vals_red_accessor.ptr(aDomain.lo()),
    (A_vals_red_accessor.accessor.strides[0] / sizeof(double))
  ));
}

extern "C" void task_7(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_7Args* args = (task_7Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;
  int32_t rpoc3 = args->rpoc3;


  int64_t in = getIndexPoint(task, 0);
  int64_t jn = getIndexPoint(task, 1);
  int64_t kn = getIndexPoint(task, 2);
  int64_t pointID1 = in + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * rpoc + jn;
  int64_t pointID3 = pointID2 * c + kn;
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((rpoc3 - 1));
  auto k1sIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(k1sIndexSpace));
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t k1s = (*itr);
    task_6Args taskArgsRaw6;
    taskArgsRaw6.A_vals_field_id = A_vals_field_id;
    taskArgsRaw6.B_vals_field_id = B_vals_field_id;
    taskArgsRaw6.C_vals_field_id = C_vals_field_id;
    taskArgsRaw6.k1s = k1s;
    taskArgsRaw6.rpoc3 = rpoc3;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw6, sizeof(task_6Args));
    TaskLauncher launcher = TaskLauncher(taskID(6), taskArgs);
    launcher.add_region_requirement(RegionRequirement(get_logical_region(A_vals), LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, A_vals_parent, 0).add_field(A_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(B_vals), pointID3), k1s)), READ_ONLY, EXCLUSIVE, B_vals_parent, 0).add_field(B_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(C_vals), pointID3), k1s)), READ_ONLY, EXCLUSIVE, C_vals_parent, 0).add_field(C_vals_field_id));
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

extern "C" void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t rpoc, int32_t c, int32_t rpoc3) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), (c - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  task_7Args taskArgsRaw7;
  taskArgsRaw7.A_vals_field_id = A_vals_field_id;
  taskArgsRaw7.B_vals_field_id = B_vals_field_id;
  taskArgsRaw7.C_vals_field_id = C_vals_field_id;
  taskArgsRaw7.c = c;
  taskArgsRaw7.rpoc = rpoc;
  taskArgsRaw7.rpoc3 = rpoc3;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw7, sizeof(task_7Args));
  IndexLauncher launcher = IndexLauncher(taskID(7), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(
    partitionPack->APartition.valsPartition,
    0,
    LEGION_REDOP_SUM_FLOAT64,
    LEGION_SIMULTANEOUS,
    A_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    partitionPack->BPartition.valsPartition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    B_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(
    partitionPack->CPartition.valsPartition,
    0,
    READ_ONLY,
    EXCLUSIVE,
    C_vals_parent,
    Mapping::DefaultMapper::VIRTUAL_MAP
  ).add_field(C_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);

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
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_6>(registrar, "task_6");
  }
  {
    TaskVariantRegistrar registrar(taskID(7), "task_7");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_7>(registrar, "task_7");
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
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    runtime->register_task_variant<task_6>(registrar);
    runtime->attach_name(taskID(6), "task_6");
  }
  {
    TaskVariantRegistrar registrar(taskID(7), "task_7", false /* global */);
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_inner();
    runtime->register_task_variant<task_7>(registrar);
    runtime->attach_name(taskID(7), "task_7");
  }
  barrier.arrive();
  barrier = runtime->advance_phase_barrier(ctx, barrier);
  barrier.wait();
  runtime->destroy_phase_barrier(ctx, barrier);
  Legion::Processor::disable_scheduler_lock();
}
