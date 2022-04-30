#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
#include "leaf_kernels.h"
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorROdouble3;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;

struct task_1Args {
  int64_t sfID;
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
  int64_t sfID;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};

struct task_4Args {
  int64_t sfID;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};

struct task_5Args {
  Legion::FieldID A_vals_field_id;
  Legion::FieldID B_vals_field_id;
  Legion::FieldID C_vals_field_id;
  Legion::FieldID D_vals_field_id;
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};


Legion::LogicalPartition partitionLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX) {
  size_t A1_dimension = A->dims[0];
  size_t A2_dimension = A->dims[1];
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (gridX - 1)) / gridX)), 0);
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN(A2_dimension, ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto A_dense_run_0_Partition = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition, get_logical_region(A_vals));
  return A_vals_partition;
}

Legion::LogicalPartition partitionLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY, int32_t gridZ) {
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  size_t B3_dimension = B->dims[2];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    int64_t kn = (*itr)[2];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX)), (jn * ((B2_dimension + (gridY - 1)) / gridY)), (kn * ((B3_dimension + (gridZ - 1)) / gridZ)));
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)), BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition, get_logical_region(B_vals));
  return B_vals_partition;
}

Legion::LogicalPartition partitionLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridY) {
  size_t C1_dimension = C->dims[0];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridY - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    Point<2> CStart = Point<2>((in * ((C1_dimension + (gridY - 1)) / gridY)), 0);
    Point<2> CEnd = Point<2>(TACO_MIN((in * ((C1_dimension + (gridY - 1)) / gridY) + ((C1_dimension + (gridY - 1)) / gridY - 1)), CDomain.hi()[0]), TACO_MIN(C2_dimension, CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto C_dense_run_0_Partition = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition, get_logical_region(C_vals));
  return C_vals_partition;
}

Legion::LogicalPartition partitionLegionD(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* D, int32_t gridZ) {
  size_t D1_dimension = D->dims[0];
  size_t D2_dimension = D->dims[1];
  RegionWrapper D_vals = D->vals;
  IndexSpace D_dense_run_0 = D->denseLevelRuns[0];

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridZ - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto DDomain = runtime->get_index_space_domain(ctx, D_dense_run_0);
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    Point<2> DStart = Point<2>((in * ((D1_dimension + (gridZ - 1)) / gridZ)), 0);
    Point<2> DEnd = Point<2>(TACO_MIN((in * ((D1_dimension + (gridZ - 1)) / gridZ) + ((D1_dimension + (gridZ - 1)) / gridZ - 1)), DDomain.hi()[0]), TACO_MIN(D2_dimension, DDomain.hi()[1]));
    Rect<2> DRect = Rect<2>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  auto D_dense_run_0_Partition = runtime->create_index_partition(ctx, D_dense_run_0, domain, DColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto D_vals_partition = copyPartition(ctx, runtime, D_dense_run_0_Partition, get_logical_region(D_vals));
  return D_vals_partition;
}

partitionPackForplaceLegionA partitionForplaceLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX) {
  size_t A1_dimension = A->dims[0];
  size_t A2_dimension = A->dims[1];
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionA();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), 0, 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (gridX - 1)) / gridX)), 0);
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN(A2_dimension, ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto A_dense_run_0_Partition = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_COMPUTE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition, get_logical_region(A_vals));
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition;

  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


}

void placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, partitionPackForplaceLegionA* partitionPack, int32_t gridX, int32_t gridY, int32_t gridZ) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), 0, 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridX);
  dims.push_back(gridY);
  dims.push_back(gridZ);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(1), dims);
  task_1Args taskArgsRaw1;
  taskArgsRaw1.sfID = shardingID(1);
  taskArgsRaw1.gridX = gridX;
  taskArgsRaw1.gridY = gridY;
  taskArgsRaw1.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, A_vals_parent).add_field(A_vals_field_id));
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

partitionPackForplaceLegionB partitionForplaceLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY, int32_t gridZ) {
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  size_t B3_dimension = B->dims[2];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionB();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    int64_t kn = (*itr)[2];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX)), (jn * ((B2_dimension + (gridY - 1)) / gridY)), (kn * ((B3_dimension + (gridZ - 1)) / gridZ)));
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)), BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition, get_logical_region(B_vals));
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(3);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition;

  return computePartitions;
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B_vals = regions[0];
  LogicalRegion B_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


}

void placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, partitionPackForplaceLegionB* partitionPack, int32_t gridX, int32_t gridY, int32_t gridZ) {
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.gridX = gridX;
  taskArgsRaw2.gridY = gridY;
  taskArgsRaw2.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

partitionPackForplaceLegionC partitionForplaceLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridY) {
  size_t C1_dimension = C->dims[0];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionC();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, (gridY - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[1];
    Point<2> CStart = Point<2>((in * ((C1_dimension + (gridY - 1)) / gridY)), 0);
    Point<2> CEnd = Point<2>(TACO_MIN((in * ((C1_dimension + (gridY - 1)) / gridY) + ((C1_dimension + (gridY - 1)) / gridY - 1)), CDomain.hi()[0]), TACO_MIN(C2_dimension, CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto C_dense_run_0_Partition = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition, get_logical_region(C_vals));
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition;

  return computePartitions;
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion C_vals = regions[0];
  LogicalRegion C_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


}

void placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, partitionPackForplaceLegionC* partitionPack, int32_t gridY, int32_t gridX, int32_t gridZ) {
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, (gridY - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridX);
  dims.push_back(gridY);
  dims.push_back(gridZ);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(3), dims);
  task_3Args taskArgsRaw3;
  taskArgsRaw3.sfID = shardingID(3);
  taskArgsRaw3.gridX = gridX;
  taskArgsRaw3.gridY = gridY;
  taskArgsRaw3.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(C_vals_field_id));
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

partitionPackForplaceLegionD partitionForplaceLegionD(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* D, int32_t gridZ) {
  size_t D1_dimension = D->dims[0];
  size_t D2_dimension = D->dims[1];
  RegionWrapper D_vals = D->vals;
  IndexSpace D_dense_run_0 = D->denseLevelRuns[0];

  auto computePartitions = partitionPackForplaceLegionD();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, 0, (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto DDomain = runtime->get_index_space_domain(ctx, D_dense_run_0);
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[2];
    Point<2> DStart = Point<2>((in * ((D1_dimension + (gridZ - 1)) / gridZ)), 0);
    Point<2> DEnd = Point<2>(TACO_MIN((in * ((D1_dimension + (gridZ - 1)) / gridZ) + ((D1_dimension + (gridZ - 1)) / gridZ - 1)), DDomain.hi()[0]), TACO_MIN(D2_dimension, DDomain.hi()[1]));
    Rect<2> DRect = Rect<2>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  auto D_dense_run_0_Partition = runtime->create_index_partition(ctx, D_dense_run_0, domain, DColoring, LEGION_COMPUTE_KIND);
  auto D_vals_partition = copyPartition(ctx, runtime, D_dense_run_0_Partition, get_logical_region(D_vals));
  computePartitions.DPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.DPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.DPartition.valsPartition = D_vals_partition;
  computePartitions.DPartition.denseLevelRunPartitions[0] = D_dense_run_0_Partition;

  return computePartitions;
}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion D_vals = regions[0];
  LogicalRegion D_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


}

void placeLegionD(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* D, partitionPackForplaceLegionD* partitionPack, int32_t gridZ, int32_t gridX, int32_t gridY) {
  auto D_vals_parent = D->valsParent;
  auto D_vals_field_id = D->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, 0, (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gridX);
  dims.push_back(gridY);
  dims.push_back(gridZ);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(5), dims);
  task_4Args taskArgsRaw4;
  taskArgsRaw4.sfID = shardingID(5);
  taskArgsRaw4.gridX = gridX;
  taskArgsRaw4.gridY = gridY;
  taskArgsRaw4.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw4, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->DPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, D_vals_parent).add_field(D_vals_field_id));
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, int32_t gridX, int32_t gridY, int32_t gridZ) {
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  size_t B3_dimension = B->dims[2];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];
  RegionWrapper D_vals = D->vals;
  IndexSpace D_dense_run_0 = D->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegion();

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  auto DDomain = runtime->get_index_space_domain(ctx, D_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    int64_t kn = (*itr)[2];
    Point<2> AStart = Point<2>((in * ((B1_dimension + (gridX - 1)) / gridX)), 0);
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN(C2_dimension, ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX)), (jn * ((B2_dimension + (gridY - 1)) / gridY)), (kn * ((B3_dimension + (gridZ - 1)) / gridZ)));
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)), BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> CStart = Point<2>((jn * ((B2_dimension + (gridY - 1)) / gridY)), 0);
    Point<2> CEnd = Point<2>(TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), CDomain.hi()[0]), TACO_MIN(C2_dimension, CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
    Point<2> DStart = Point<2>((kn * ((B3_dimension + (gridZ - 1)) / gridZ)), 0);
    Point<2> DEnd = Point<2>(TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)), DDomain.hi()[0]), TACO_MIN(C2_dimension, DDomain.hi()[1]));
    Rect<2> DRect = Rect<2>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  auto A_dense_run_0_Partition = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_COMPUTE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition, get_logical_region(A_vals));
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition, get_logical_region(B_vals));
  auto C_dense_run_0_Partition = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition, get_logical_region(C_vals));
  auto D_dense_run_0_Partition = runtime->create_index_partition(ctx, D_dense_run_0, domain, DColoring, LEGION_COMPUTE_KIND);
  auto D_vals_partition = copyPartition(ctx, runtime, D_dense_run_0_Partition, get_logical_region(D_vals));
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(3);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition;
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition;
  computePartitions.DPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.DPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.DPartition.valsPartition = D_vals_partition;
  computePartitions.DPartition.denseLevelRunPartitions[0] = D_dense_run_0_Partition;

  return computePartitions;
}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();
  PhysicalRegion D_vals = regions[3];
  LogicalRegion D_vals_parent = regions[3].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_5Args* args = (task_5Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  Legion::FieldID D_vals_field_id = args->D_vals_field_id;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;

  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, C_vals_field_id);
  auto D_vals_ro_accessor = createAccessor<AccessorROdouble2>(D_vals, D_vals_field_id);
  auto B_vals_ro_accessor = createAccessor<AccessorROdouble3>(B_vals, B_vals_field_id);
  auto A_vals_red_accessor = createAccessor<AccessorReducedouble2>(A_vals, A_vals_field_id, LEGION_REDOP_SUM_FLOAT64);

  auto aDomain = runtime->get_index_space_domain(ctx, A_vals.get_logical_region().get_index_space());
  auto bDomain = runtime->get_index_space_domain(ctx, B_vals.get_logical_region().get_index_space());
  auto cDomain = runtime->get_index_space_domain(ctx, C_vals.get_logical_region().get_index_space());
  auto dDomain = runtime->get_index_space_domain(ctx, D_vals.get_logical_region().get_index_space());
  if ((bDomain.get_volume() == 0 || cDomain.get_volume() == 0) || dDomain.get_volume() == 0)
    return ;

  MTTKRPPack pack = MTTKRPPack();
  pack.iDim = 1 + (bDomain.hi()[0] - bDomain.lo()[0]);
  pack.jDim = 1 + (bDomain.hi()[1] - bDomain.lo()[1]);
  pack.kDim = 1 + (bDomain.hi()[2] - bDomain.lo()[2]);
  pack.lDim = 1 + (aDomain.hi()[1] - aDomain.lo()[1]);
  pack.ldA = A_vals_red_accessor.accessor.strides[0] / sizeof(double);
  pack.ldC = C_vals_ro_accessor.accessor.strides[0] / sizeof(double);
  pack.ldD = D_vals_ro_accessor.accessor.strides[0] / sizeof(double);
  pack.ldB1 = B_vals_ro_accessor.accessor.strides[0] / sizeof(double);
  pack.ldB2 = (B_vals_ro_accessor.accessor.strides[0] / sizeof(double)) / (B_vals_ro_accessor.accessor.strides[1] / sizeof(double));
  pack.ldB3 = B_vals_ro_accessor.accessor.strides[1] / sizeof(double);
  mttkrp<double>(pack, A_vals_red_accessor.ptr(aDomain.lo()), B_vals_ro_accessor.ptr(bDomain.lo()), C_vals_ro_accessor.ptr(cDomain.lo()), D_vals_ro_accessor.ptr(dDomain.lo()));
}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, partitionPackForcomputeLegion* partitionPack, int32_t gridX, int32_t gridY, int32_t gridZ) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;
  auto D_vals_parent = D->valsParent;
  auto D_vals_field_id = D->valsFieldID;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  task_5Args taskArgsRaw5;
  taskArgsRaw5.A_vals_field_id = A_vals_field_id;
  taskArgsRaw5.B_vals_field_id = B_vals_field_id;
  taskArgsRaw5.C_vals_field_id = C_vals_field_id;
  taskArgsRaw5.D_vals_field_id = D_vals_field_id;
  taskArgsRaw5.gridX = gridX;
  taskArgsRaw5.gridY = gridY;
  taskArgsRaw5.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw5, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, A_vals_parent).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(C_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->DPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, D_vals_parent).add_field(D_vals_field_id));
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
