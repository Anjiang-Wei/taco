#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
#include "tblis/tblis.h"
#include "leaf_kernels.h"
typedef FieldAccessor<READ_ONLY,double,4,coord_t,Realm::AffineAccessor<double,4,coord_t>> AccessorROdouble4;
typedef FieldAccessor<READ_WRITE,double,4,coord_t,Realm::AffineAccessor<double,4,coord_t>> AccessorRWdouble4;

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
  IndexPartition A_dense_run_0_Partition_0;
  int64_t B1_dimension;
  int64_t B2_dimension;
  int64_t B3_dimension;
  int64_t B4_dimension;
  IndexSpace B_dense_run_0;
  IndexPartition B_dense_run_0_Partition_0;
  int64_t C2_dimension;
  int64_t C4_dimension;
  IndexSpace C_dense_run_0;
  IndexPartition C_dense_run_0_Partition_0;
  int32_t gridX;
  int32_t gridY;
};

struct task_6Args {
  Legion::FieldID A_vals_field_id;
  int64_t B1_dimension;
  int64_t B2_dimension;
  int64_t B3_dimension;
  int64_t B4_dimension;
  Legion::FieldID B_vals_field_id;
  int64_t C2_dimension;
  int64_t C4_dimension;
  Legion::FieldID C_vals_field_id;
  int64_t an;
  int64_t bn;
  int64_t cos;
  int32_t gridX;
  int32_t gridY;
};

struct task_7Args {
  Legion::FieldID A_vals_field_id;
  int64_t B1_dimension;
  int64_t B2_dimension;
  int64_t B3_dimension;
  int64_t B4_dimension;
  Legion::FieldID B_vals_field_id;
  int64_t C2_dimension;
  int64_t C4_dimension;
  Legion::FieldID C_vals_field_id;
  int32_t gridX;
  int32_t gridY;
};

struct task_9Args {
  IndexPartition A_dense_run_0_Partition_0;
  int64_t B1_dimension;
  int64_t B2_dimension;
  int64_t B3_dimension;
  int64_t B4_dimension;
  IndexSpace B_dense_run_0;
  IndexPartition B_dense_run_0_Partition_0;
  int64_t C2_dimension;
  int64_t C4_dimension;
  IndexSpace C_dense_run_0;
  IndexPartition C_dense_run_0_Partition_0;
  int32_t gridX;
  int32_t gridY;
};

struct task_10Args {
  Legion::FieldID A_vals_field_id;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID C_vals_field_id;
  int64_t cos;
  int32_t gridX;
};

struct task_11Args {
  Legion::FieldID A_vals_field_id;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID C_vals_field_id;
  int32_t gridX;
  int32_t gridY;
};


partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX, int32_t gridY) {
  size_t A1_dimension = A->dims[0];
  size_t A2_dimension = A->dims[1];
  size_t A3_dimension = A->dims[2];
  size_t A4_dimension = A->dims[3];
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionA();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<4> AStart = Point<4>((in * ((A1_dimension + (gridX - 1)) / gridX)), (jn * ((A2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> AEnd = Point<4>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN((jn * ((A2_dimension + (gridY - 1)) / gridY) + ((A2_dimension + (gridY - 1)) / gridY - 1)), ADomain.hi()[1]), TACO_MIN(A3_dimension, ADomain.hi()[2]), TACO_MIN(A4_dimension, ADomain.hi()[3]));
    Rect<4> ARect = Rect<4>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  IndexPartition A_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_COMPUTE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition_0, get_logical_region(A_vals));
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition_0;

  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.gridX = gridX;
  taskArgsRaw1.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, A_vals_parent).add_field(A_vals_field_id));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY) {
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  size_t B3_dimension = B->dims[2];
  size_t B4_dimension = B->dims[3];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionB();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<4> BStart = Point<4>((in * ((B1_dimension + (gridX - 1)) / gridX)), (jn * ((B2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> BEnd = Point<4>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]), TACO_MIN(B4_dimension, BDomain.hi()[3]));
    Rect<4> BRect = Rect<4>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  IndexPartition B_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition_0, get_logical_region(B_vals));
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_0;

  return computePartitions;
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B_vals = regions[0];
  LogicalRegion B_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY) {
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.gridX = gridX;
  taskArgsRaw2.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridX, int32_t gridY) {
  size_t C1_dimension = C->dims[0];
  size_t C2_dimension = C->dims[1];
  size_t C3_dimension = C->dims[2];
  size_t C4_dimension = C->dims[3];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionC();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<4> CStart = Point<4>((in * ((C1_dimension + (gridX - 1)) / gridX)), (jn * ((C2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> CEnd = Point<4>(TACO_MIN((in * ((C1_dimension + (gridX - 1)) / gridX) + ((C1_dimension + (gridX - 1)) / gridX - 1)), CDomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (gridY - 1)) / gridY) + ((C2_dimension + (gridY - 1)) / gridY - 1)), CDomain.hi()[1]), TACO_MIN(C3_dimension, CDomain.hi()[2]), TACO_MIN(C4_dimension, CDomain.hi()[3]));
    Rect<4> CRect = Rect<4>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition C_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_0, get_logical_region(C_vals));
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition_0;

  return computePartitions;
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion C_vals = regions[0];
  LogicalRegion C_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, partitionPackForplaceLegionC* partitionPack, int32_t gridX, int32_t gridY) {
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_3Args taskArgsRaw3;
  taskArgsRaw3.gridX = gridX;
  taskArgsRaw3.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(C_vals_field_id));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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
  int64_t B3_dimension = args->B3_dimension;
  int64_t B4_dimension = args->B4_dimension;
  IndexSpace B_dense_run_0 = args->B_dense_run_0;
  IndexPartition B_dense_run_0_Partition_0 = args->B_dense_run_0_Partition_0;
  int64_t C2_dimension = args->C2_dimension;
  int64_t C4_dimension = args->C4_dimension;
  IndexSpace C_dense_run_0 = args->C_dense_run_0;
  IndexPartition C_dense_run_0_Partition_0 = args->C_dense_run_0_Partition_0;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t an = getIndexPoint(task, 0);
  int64_t bn = getIndexPoint(task, 1);
  int64_t pointID1 = an + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * gridY + bn;
  B_dense_run_0 = runtime->get_index_subspace(ctx, B_dense_run_0_Partition_0, task->index_point);
  C_dense_run_0 = runtime->get_index_subspace(ctx, C_dense_run_0_Partition_0, task->index_point);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto cosIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(cosIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t cos = (*itr)[0];
    Point<4> BStart = Point<4>((an * ((B1_dimension + (gridX - 1)) / gridX)), (((bn + (an + cos)) % gridX) * ((B2_dimension + (gridX - 1)) / gridX)), 0, 0);
    Point<4> BEnd = Point<4>(TACO_MIN((an * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((((bn + (an + cos)) % gridX) * ((B2_dimension + (gridX - 1)) / gridX) + ((B2_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]), TACO_MIN(B4_dimension, BDomain.hi()[3]));
    Rect<4> BRect = Rect<4>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<4> CStart = Point<4>((((bn + (an + cos)) % gridX) * ((B2_dimension + (gridX - 1)) / gridX)), (bn * ((C2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> CEnd = Point<4>(TACO_MIN((((bn + (an + cos)) % gridX) * ((B2_dimension + (gridX - 1)) / gridX) + ((B2_dimension + (gridX - 1)) / gridX - 1)), CDomain.hi()[0]), TACO_MIN((bn * ((C2_dimension + (gridY - 1)) / gridY) + ((C2_dimension + (gridY - 1)) / gridY - 1)), CDomain.hi()[1]), TACO_MIN(B4_dimension, CDomain.hi()[2]), TACO_MIN(C4_dimension, CDomain.hi()[3]));
    Rect<4> CRect = Rect<4>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition B_dense_run_0_Partition_2 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition_2, get_logical_region(B_vals), pointID2);
  IndexPartition C_dense_run_0_Partition_2 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_2, get_logical_region(C_vals), pointID2);
}

partitionPackForcomputeLegionNestedOMP partitionForcomputeLegionNestedOMP(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gridX, int32_t gridY) {
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  size_t B3_dimension = B->dims[2];
  size_t B4_dimension = B->dims[3];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  size_t C2_dimension = C->dims[1];
  size_t C4_dimension = C->dims[3];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegionNestedOMP();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t an = (*itr)[0];
    int64_t bn = (*itr)[1];
    Point<4> AStart = Point<4>((an * ((B1_dimension + (gridX - 1)) / gridX)), (bn * ((C2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> AEnd = Point<4>(TACO_MIN((an * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN((bn * ((C2_dimension + (gridY - 1)) / gridY) + ((C2_dimension + (gridY - 1)) / gridY - 1)), ADomain.hi()[1]), TACO_MIN(B3_dimension, ADomain.hi()[2]), TACO_MIN(C4_dimension, ADomain.hi()[3]));
    Rect<4> ARect = Rect<4>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<4> BStart = Point<4>((an * ((B1_dimension + (gridX - 1)) / gridX)), 0, 0, 0);
    Point<4> BEnd = Point<4>(TACO_MIN((an * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN(((gridX - 1) * ((B2_dimension + (gridX - 1)) / gridX) + ((B2_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]), TACO_MIN(B4_dimension, BDomain.hi()[3]));
    Rect<4> BRect = Rect<4>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<4> CStart = Point<4>(0, (bn * ((C2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> CEnd = Point<4>(TACO_MIN(((gridX - 1) * ((B2_dimension + (gridX - 1)) / gridX) + ((B2_dimension + (gridX - 1)) / gridX - 1)), CDomain.hi()[0]), TACO_MIN((bn * ((C2_dimension + (gridY - 1)) / gridY) + ((C2_dimension + (gridY - 1)) / gridY - 1)), CDomain.hi()[1]), TACO_MIN(B4_dimension, CDomain.hi()[2]), TACO_MIN(C4_dimension, CDomain.hi()[3]));
    Rect<4> CRect = Rect<4>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition A_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition_0, get_logical_region(A_vals));
  IndexPartition B_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition_0, get_logical_region(B_vals));
  IndexPartition C_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_0, get_logical_region(C_vals));
  task_5Args taskArgsRaw5;
  taskArgsRaw5.A_dense_run_0_Partition_0 = A_dense_run_0_Partition_0;
  taskArgsRaw5.B1_dimension = B1_dimension;
  taskArgsRaw5.B2_dimension = B2_dimension;
  taskArgsRaw5.B3_dimension = B3_dimension;
  taskArgsRaw5.B4_dimension = B4_dimension;
  taskArgsRaw5.B_dense_run_0 = B_dense_run_0;
  taskArgsRaw5.B_dense_run_0_Partition_0 = B_dense_run_0_Partition_0;
  taskArgsRaw5.C2_dimension = C2_dimension;
  taskArgsRaw5.C4_dimension = C4_dimension;
  taskArgsRaw5.C_dense_run_0 = C_dense_run_0;
  taskArgsRaw5.C_dense_run_0_Partition_0 = C_dense_run_0_Partition_0;
  taskArgsRaw5.gridX = gridX;
  taskArgsRaw5.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw5, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(A_vals_partition, 0, READ_ONLY, EXCLUSIVE, A_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(B_vals_partition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(C_vals_partition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(C_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();


  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition_0;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_0;
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition_0;

  return computePartitions;
}

void task_6(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  task_6Args* args = (task_6Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  int64_t B1_dimension = args->B1_dimension;
  int64_t B2_dimension = args->B2_dimension;
  int64_t B3_dimension = args->B3_dimension;
  int64_t B4_dimension = args->B4_dimension;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  int64_t C2_dimension = args->C2_dimension;
  int64_t C4_dimension = args->C4_dimension;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int64_t an = args->an;
  int64_t bn = args->bn;
  int64_t cos = args->cos;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble4>(B_vals, B_vals_field_id);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble4>(C_vals, C_vals_field_id);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble4>(A_vals, A_vals_field_id);

  int64_t co = (bn + (an + cos)) % gridX;
  #pragma omp parallel for schedule(static)
  for (int64_t al = 0; al < ((B1_dimension + (gridX - 1)) / gridX); al++) {
    int64_t a = an * ((B1_dimension + (gridX - 1)) / gridX) + al;
    if (a >= B1_dimension)
      continue;

    if (a >= (an + 1) * ((B1_dimension + (gridX - 1)) / gridX))
      continue;

    for (int64_t bl = 0; bl < ((C2_dimension + (gridY - 1)) / gridY); bl++) {
      int64_t b = bn * ((C2_dimension + (gridY - 1)) / gridY) + bl;
      if (b >= C2_dimension)
        continue;

      if (b >= (bn + 1) * ((C2_dimension + (gridY - 1)) / gridY))
        continue;

      for (int64_t ci = 0; ci < ((B2_dimension + (gridX - 1)) / gridX); ci++) {
        int64_t c = co * ((B2_dimension + (gridX - 1)) / gridX) + ci;
        if (c >= B2_dimension)
          continue;

        if (c >= (co + 1) * ((B2_dimension + (gridX - 1)) / gridX))
          continue;

        for (int64_t i = 0; i < B3_dimension; i++) {
          for (int64_t j = 0; j < C4_dimension; j++) {
            for (int64_t k = 0; k < B4_dimension; k++) {
              A_vals_rw_accessor[Point<4>(a, b, i, j)] = A_vals_rw_accessor[Point<4>(a, b, i, j)] + B_vals_ro_accessor[Point<4>(a, c, i, k)] * C_vals_ro_accessor[Point<4>(c, b, k, j)];
            }
          }
        }
      }
    }
  }
}

void task_7(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_7Args* args = (task_7Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  int64_t B1_dimension = args->B1_dimension;
  int64_t B2_dimension = args->B2_dimension;
  int64_t B3_dimension = args->B3_dimension;
  int64_t B4_dimension = args->B4_dimension;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  int64_t C2_dimension = args->C2_dimension;
  int64_t C4_dimension = args->C4_dimension;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t an = getIndexPoint(task, 0);
  int64_t bn = getIndexPoint(task, 1);
  int64_t pointID1 = an + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * gridY + bn;
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto cosIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(cosIndexSpace));
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t cos = (*itr);
    task_6Args taskArgsRaw6;
    taskArgsRaw6.A_vals_field_id = A_vals_field_id;
    taskArgsRaw6.B1_dimension = B1_dimension;
    taskArgsRaw6.B2_dimension = B2_dimension;
    taskArgsRaw6.B3_dimension = B3_dimension;
    taskArgsRaw6.B4_dimension = B4_dimension;
    taskArgsRaw6.B_vals_field_id = B_vals_field_id;
    taskArgsRaw6.C2_dimension = C2_dimension;
    taskArgsRaw6.C4_dimension = C4_dimension;
    taskArgsRaw6.C_vals_field_id = C_vals_field_id;
    taskArgsRaw6.an = an;
    taskArgsRaw6.bn = bn;
    taskArgsRaw6.cos = cos;
    taskArgsRaw6.gridX = gridX;
    taskArgsRaw6.gridY = gridY;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw6, sizeof(task_6Args));
    TaskLauncher launcher = TaskLauncher(taskID(6), taskArgs);
    launcher.add_region_requirement(RegionRequirement(get_logical_region(A_vals), READ_WRITE, EXCLUSIVE, A_vals_parent, 0).add_field(A_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(B_vals), pointID2), cos)), READ_ONLY, EXCLUSIVE, B_vals_parent, 0).add_field(B_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(C_vals), pointID2), cos)), READ_ONLY, EXCLUSIVE, C_vals_parent, 0).add_field(C_vals_field_id));
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegionNestedOMP(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegionNestedOMP* partitionPack, int32_t gridX, int32_t gridY) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  size_t B3_dimension = B->dims[2];
  size_t B4_dimension = B->dims[3];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  size_t C2_dimension = C->dims[1];
  size_t C4_dimension = C->dims[3];
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_7Args taskArgsRaw7;
  taskArgsRaw7.A_vals_field_id = A_vals_field_id;
  taskArgsRaw7.B1_dimension = B1_dimension;
  taskArgsRaw7.B2_dimension = B2_dimension;
  taskArgsRaw7.B3_dimension = B3_dimension;
  taskArgsRaw7.B4_dimension = B4_dimension;
  taskArgsRaw7.B_vals_field_id = B_vals_field_id;
  taskArgsRaw7.C2_dimension = C2_dimension;
  taskArgsRaw7.C4_dimension = C4_dimension;
  taskArgsRaw7.C_vals_field_id = C_vals_field_id;
  taskArgsRaw7.gridX = gridX;
  taskArgsRaw7.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw7, sizeof(task_7Args));
  IndexLauncher launcher = IndexLauncher(taskID(7), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(C_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

void task_9(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_9Args* args = (task_9Args*)(task->args);
  IndexPartition A_dense_run_0_Partition_0 = args->A_dense_run_0_Partition_0;
  int64_t B1_dimension = args->B1_dimension;
  int64_t B2_dimension = args->B2_dimension;
  int64_t B3_dimension = args->B3_dimension;
  int64_t B4_dimension = args->B4_dimension;
  IndexSpace B_dense_run_0 = args->B_dense_run_0;
  IndexPartition B_dense_run_0_Partition_0 = args->B_dense_run_0_Partition_0;
  int64_t C2_dimension = args->C2_dimension;
  int64_t C4_dimension = args->C4_dimension;
  IndexSpace C_dense_run_0 = args->C_dense_run_0;
  IndexPartition C_dense_run_0_Partition_0 = args->C_dense_run_0_Partition_0;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t an = getIndexPoint(task, 0);
  int64_t bn = getIndexPoint(task, 1);
  int64_t pointID1 = an + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * gridY + bn;
  B_dense_run_0 = runtime->get_index_subspace(ctx, B_dense_run_0_Partition_0, task->index_point);
  C_dense_run_0 = runtime->get_index_subspace(ctx, C_dense_run_0_Partition_0, task->index_point);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto cosIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(cosIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t cos = (*itr)[0];
    Point<4> BStart = Point<4>((an * ((B1_dimension + (gridX - 1)) / gridX)), (((bn + (an + cos)) % gridX) * ((B2_dimension + (gridX - 1)) / gridX)), 0, 0);
    Point<4> BEnd = Point<4>(TACO_MIN((an * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((((bn + (an + cos)) % gridX) * ((B2_dimension + (gridX - 1)) / gridX) + ((B2_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]), TACO_MIN(B4_dimension, BDomain.hi()[3]));
    Rect<4> BRect = Rect<4>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<4> CStart = Point<4>((((bn + (an + cos)) % gridX) * ((B2_dimension + (gridX - 1)) / gridX)), (bn * ((C2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> CEnd = Point<4>(TACO_MIN((((bn + (an + cos)) % gridX) * ((B2_dimension + (gridX - 1)) / gridX) + ((B2_dimension + (gridX - 1)) / gridX - 1)), CDomain.hi()[0]), TACO_MIN((bn * ((C2_dimension + (gridY - 1)) / gridY) + ((C2_dimension + (gridY - 1)) / gridY - 1)), CDomain.hi()[1]), TACO_MIN(B4_dimension, CDomain.hi()[2]), TACO_MIN(C4_dimension, CDomain.hi()[3]));
    Rect<4> CRect = Rect<4>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition B_dense_run_0_Partition_2 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition_2, get_logical_region(B_vals), pointID2);
  IndexPartition C_dense_run_0_Partition_2 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_2, get_logical_region(C_vals), pointID2);
}

partitionPackForcomputeLegionTblis partitionForcomputeLegionTblis(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gridX, int32_t gridY) {
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  size_t B3_dimension = B->dims[2];
  size_t B4_dimension = B->dims[3];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  size_t C2_dimension = C->dims[1];
  size_t C4_dimension = C->dims[3];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegionTblis();

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t an = (*itr)[0];
    int64_t bn = (*itr)[1];
    Point<4> AStart = Point<4>((an * ((B1_dimension + (gridX - 1)) / gridX)), (bn * ((C2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> AEnd = Point<4>(TACO_MIN((an * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN((bn * ((C2_dimension + (gridY - 1)) / gridY) + ((C2_dimension + (gridY - 1)) / gridY - 1)), ADomain.hi()[1]), TACO_MIN(B3_dimension, ADomain.hi()[2]), TACO_MIN(C4_dimension, ADomain.hi()[3]));
    Rect<4> ARect = Rect<4>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<4> BStart = Point<4>((an * ((B1_dimension + (gridX - 1)) / gridX)), 0, 0, 0);
    Point<4> BEnd = Point<4>(TACO_MIN((an * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN(((gridX - 1) * ((B2_dimension + (gridX - 1)) / gridX) + ((B2_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]), TACO_MIN(B4_dimension, BDomain.hi()[3]));
    Rect<4> BRect = Rect<4>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<4> CStart = Point<4>(0, (bn * ((C2_dimension + (gridY - 1)) / gridY)), 0, 0);
    Point<4> CEnd = Point<4>(TACO_MIN(((gridX - 1) * ((B2_dimension + (gridX - 1)) / gridX) + ((B2_dimension + (gridX - 1)) / gridX - 1)), CDomain.hi()[0]), TACO_MIN((bn * ((C2_dimension + (gridY - 1)) / gridY) + ((C2_dimension + (gridY - 1)) / gridY - 1)), CDomain.hi()[1]), TACO_MIN(B4_dimension, CDomain.hi()[2]), TACO_MIN(C4_dimension, CDomain.hi()[3]));
    Rect<4> CRect = Rect<4>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition A_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition_0, get_logical_region(A_vals));
  IndexPartition B_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition_0, get_logical_region(B_vals));
  IndexPartition C_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_0, get_logical_region(C_vals));
  task_9Args taskArgsRaw9;
  taskArgsRaw9.A_dense_run_0_Partition_0 = A_dense_run_0_Partition_0;
  taskArgsRaw9.B1_dimension = B1_dimension;
  taskArgsRaw9.B2_dimension = B2_dimension;
  taskArgsRaw9.B3_dimension = B3_dimension;
  taskArgsRaw9.B4_dimension = B4_dimension;
  taskArgsRaw9.B_dense_run_0 = B_dense_run_0;
  taskArgsRaw9.B_dense_run_0_Partition_0 = B_dense_run_0_Partition_0;
  taskArgsRaw9.C2_dimension = C2_dimension;
  taskArgsRaw9.C4_dimension = C4_dimension;
  taskArgsRaw9.C_dense_run_0 = C_dense_run_0;
  taskArgsRaw9.C_dense_run_0_Partition_0 = C_dense_run_0_Partition_0;
  taskArgsRaw9.gridX = gridX;
  taskArgsRaw9.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw9, sizeof(task_9Args));
  IndexLauncher launcher = IndexLauncher(taskID(9), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(A_vals_partition, 0, READ_ONLY, EXCLUSIVE, A_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(B_vals_partition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(C_vals_partition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(C_vals_field_id));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();


  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition_0;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_0;
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(4);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(4);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition_0;

  return computePartitions;
}

void task_10(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  task_10Args* args = (task_10Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int64_t cos = args->cos;
  int32_t gridX = args->gridX;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble4>(B_vals, B_vals_field_id);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble4>(C_vals, C_vals_field_id);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble4>(A_vals, A_vals_field_id);

  auto aDomain = runtime->get_index_space_domain(ctx, get_logical_region(A_vals).get_index_space());
  auto bDomain = runtime->get_index_space_domain(ctx, get_logical_region(B_vals).get_index_space());
  auto cDomain = runtime->get_index_space_domain(ctx, get_logical_region(C_vals).get_index_space());
  if (bDomain.get_volume() == 0 || cDomain.get_volume() == 0)
    return ;

  TblisSetThreads();
  std::array<tblis::len_type, 4> aShape = {};
  aShape[0] = 1 + (aDomain.hi()[0] - aDomain.lo()[0]);
  aShape[1] = 1 + (aDomain.hi()[1] - aDomain.lo()[1]);
  aShape[2] = 1 + (aDomain.hi()[2] - aDomain.lo()[2]);
  aShape[3] = 1 + (aDomain.hi()[3] - aDomain.lo()[3]);
  std::array<tblis::len_type, 4> bShape = {};
  bShape[0] = 1 + (bDomain.hi()[0] - bDomain.lo()[0]);
  bShape[1] = 1 + (bDomain.hi()[1] - bDomain.lo()[1]);
  bShape[2] = 1 + (bDomain.hi()[2] - bDomain.lo()[2]);
  bShape[3] = 1 + (bDomain.hi()[3] - bDomain.lo()[3]);
  std::array<tblis::len_type, 4> cShape = {};
  cShape[0] = 1 + (cDomain.hi()[0] - cDomain.lo()[0]);
  cShape[1] = 1 + (cDomain.hi()[1] - cDomain.lo()[1]);
  cShape[2] = 1 + (cDomain.hi()[2] - cDomain.lo()[2]);
  cShape[3] = 1 + (cDomain.hi()[3] - cDomain.lo()[3]);
  std::array<tblis::stride_type, 4> aStrides = {};
  aStrides[0] = A_vals_rw_accessor.accessor.strides[0] / sizeof(double);
  aStrides[1] = A_vals_rw_accessor.accessor.strides[1] / sizeof(double);
  aStrides[2] = A_vals_rw_accessor.accessor.strides[2] / sizeof(double);
  aStrides[3] = A_vals_rw_accessor.accessor.strides[3] / sizeof(double);
  std::array<tblis::stride_type, 4> bStrides = {};
  bStrides[0] = B_vals_ro_accessor.accessor.strides[0] / sizeof(double);
  bStrides[1] = B_vals_ro_accessor.accessor.strides[1] / sizeof(double);
  bStrides[2] = B_vals_ro_accessor.accessor.strides[2] / sizeof(double);
  bStrides[3] = B_vals_ro_accessor.accessor.strides[3] / sizeof(double);
  std::array<tblis::stride_type, 4> cStrides = {};
  cStrides[0] = C_vals_ro_accessor.accessor.strides[0] / sizeof(double);
  cStrides[1] = C_vals_ro_accessor.accessor.strides[1] / sizeof(double);
  cStrides[2] = C_vals_ro_accessor.accessor.strides[2] / sizeof(double);
  cStrides[3] = C_vals_ro_accessor.accessor.strides[3] / sizeof(double);
  auto aTensor = tblis::varray_view<double>(aShape, A_vals_rw_accessor.ptr(aDomain.lo()), aStrides);
  auto bTensor = tblis::varray_view<const double>(bShape, B_vals_ro_accessor.ptr(bDomain.lo()), bStrides);
  auto cTensor = tblis::varray_view<const double>(cShape, C_vals_ro_accessor.ptr(cDomain.lo()), cStrides);
  tblis::mult(
    1.0000000000000000,
    bTensor,
    "acdf",
    cTensor,
    "cbfe",
    0.0,
    aTensor,
    "abde"
  );
}

void task_11(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_11Args* args = (task_11Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


  int64_t an = getIndexPoint(task, 0);
  int64_t bn = getIndexPoint(task, 1);
  int64_t pointID1 = an + TACO_PARTITION_COLOR_OFFSET;
  int64_t pointID2 = pointID1 * gridY + bn;
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto cosIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(cosIndexSpace));
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t cos = (*itr);
    task_10Args taskArgsRaw10;
    taskArgsRaw10.A_vals_field_id = A_vals_field_id;
    taskArgsRaw10.B_vals_field_id = B_vals_field_id;
    taskArgsRaw10.C_vals_field_id = C_vals_field_id;
    taskArgsRaw10.cos = cos;
    taskArgsRaw10.gridX = gridX;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw10, sizeof(task_10Args));
    TaskLauncher launcher = TaskLauncher(taskID(10), taskArgs);
    launcher.add_region_requirement(RegionRequirement(get_logical_region(A_vals), READ_WRITE, EXCLUSIVE, A_vals_parent, 0).add_field(A_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(B_vals), pointID2), cos)), READ_ONLY, EXCLUSIVE, B_vals_parent, 0).add_field(B_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(C_vals), pointID2), cos)), READ_ONLY, EXCLUSIVE, C_vals_parent, 0).add_field(C_vals_field_id));
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegionTblis(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegionTblis* partitionPack, int32_t gridX, int32_t gridY) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_11Args taskArgsRaw11;
  taskArgsRaw11.A_vals_field_id = A_vals_field_id;
  taskArgsRaw11.B_vals_field_id = B_vals_field_id;
  taskArgsRaw11.C_vals_field_id = C_vals_field_id;
  taskArgsRaw11.gridX = gridX;
  taskArgsRaw11.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw11, sizeof(task_11Args));
  IndexLauncher launcher = IndexLauncher(taskID(11), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(C_vals_field_id));
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
  {
    TaskVariantRegistrar registrar(taskID(9), "task_9");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_9>(registrar, "task_9");
  }
  {
    TaskVariantRegistrar registrar(taskID(10), "task_10");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_10>(registrar, "task_10");
  }
  {
    TaskVariantRegistrar registrar(taskID(11), "task_11");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_11>(registrar, "task_11");
  }
}
