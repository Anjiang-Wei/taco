#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
#include "leaf_kernels.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorROdouble3;
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
};


Legion::LogicalPartition partitionLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX, int32_t gridY) {
  int A1_dimension = A->dims[0];
  int A2_dimension = A->dims[1];
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (gridX - 1)) / gridX)), (jn * ((A2_dimension + (gridY - 1)) / gridY)));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN((jn * ((A2_dimension + (gridY - 1)) / gridY) + ((A2_dimension + (gridY - 1)) / gridY - 1)), ADomain.hi()[1]));
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

Legion::LogicalPartition partitionLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY) {
  int B1_dimension = B->dims[0];
  int B2_dimension = B->dims[1];
  int B3_dimension = B->dims[2];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX)), (jn * ((B2_dimension + (gridY - 1)) / gridY)), 0);
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]));
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

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

Legion::LogicalPartition placeLegionA(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, int32_t gridX, int32_t gridY) {
  int A1_dimension = A->dims[0];
  int A2_dimension = A->dims[1];
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (gridX - 1)) / gridX)), (jn * ((A2_dimension + (gridY - 1)) / gridY)));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (gridX - 1)) / gridX) + ((A1_dimension + (gridX - 1)) / gridX - 1)), ADomain.hi()[0]), TACO_MIN((jn * ((A2_dimension + (gridY - 1)) / gridY) + ((A2_dimension + (gridY - 1)) / gridY - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto A_dense_run_0_Partition = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_COMPUTE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition, get_logical_region(A_vals));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.gridX = gridX;
  taskArgsRaw1.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(A_vals_partition, 0, READ_ONLY, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return A_vals_partition;

}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B_vals = regions[0];
  LogicalRegion B_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

Legion::LogicalPartition placeLegionB(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, int32_t gridX, int32_t gridY) {
  int B1_dimension = B->dims[0];
  int B2_dimension = B->dims[1];
  int B3_dimension = B->dims[2];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    int64_t jn = (*itr)[1];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX)), (jn * ((B2_dimension + (gridY - 1)) / gridY)), 0);
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition, get_logical_region(B_vals));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.gridX = gridX;
  taskArgsRaw2.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(B_vals_partition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return B_vals_partition;

}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion C_vals = regions[0];
  LogicalRegion C_vals_parent = regions[0].get_logical_region();

  int64_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;


}

Legion::LogicalPartition placeLegionC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* C, int32_t gridX, int32_t gridY) {
  int C1_dimension = C->dims[0];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    Point<1> CStart = Point<1>(0);
    Point<1> CEnd = Point<1>(TACO_MIN(C1_dimension, CDomain.hi()[0]));
    Rect<1> CRect = Rect<1>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto C_dense_run_0_Partition = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition, get_logical_region(C_vals));
  task_3Args taskArgsRaw3;
  taskArgsRaw3.gridX = gridX;
  taskArgsRaw3.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(C_vals_partition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return C_vals_partition;

}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B_vals = regions[1];
  LogicalRegion B_vals_parent = regions[1].get_logical_region();
  PhysicalRegion C_vals = regions[2];
  LogicalRegion C_vals_parent = regions[2].get_logical_region();

  int64_t distFused = task->index_point[0];
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble1>(C_vals, FID_VAL);
  auto B_vals_ro_accessor = createAccessor<AccessorROdouble3>(B_vals, FID_VAL);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble2>(A_vals, FID_VAL);

  auto aDomain = runtime->get_index_space_domain(ctx, A_vals.get_logical_region().get_index_space());
  auto bDomain = runtime->get_index_space_domain(ctx, B_vals.get_logical_region().get_index_space());
  auto cDomain = runtime->get_index_space_domain(ctx, C_vals.get_logical_region().get_index_space());
  if (bDomain.get_volume() == 0 || cDomain.get_volume() == 0)
    return ;

  TTVPack pack = TTVPack();
  pack.iDim = 1 + (bDomain.hi()[0] - bDomain.lo()[0]);
  pack.jDim = 1 + (bDomain.hi()[1] - bDomain.lo()[1]);
  pack.kDim = 1 + (bDomain.hi()[2] - bDomain.lo()[2]);
  pack.ldA = A_vals_rw_accessor.accessor.strides[0] / sizeof(double);
  pack.ldB2 = (B_vals_ro_accessor.accessor.strides[0] / sizeof(double)) / (B_vals_ro_accessor.accessor.strides[1] / sizeof(double));
  pack.ldB3 = B_vals_ro_accessor.accessor.strides[1] / sizeof(double);
  ttv<double>(pack, A_vals_rw_accessor.ptr(aDomain.lo()), B_vals_ro_accessor.ptr(bDomain.lo()), C_vals_ro_accessor.ptr(cDomain.lo()));
}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, Legion::LogicalPartition BPartition, Legion::LogicalPartition APartition, Legion::LogicalPartition CPartition) {
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  int B3_dimension = B->dims[2];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;

  DomainT<2> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(BPartition));
  auto distFusedIndexSpace = runtime->get_index_partition_color_space_name(ctx, get_index_partition(BPartition));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    DomainPoint domPoint = (*itr);
    auto BPartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, BPartition, domPoint).get_index_space());
    int64_t BPartitionBounds0lo = BPartitionBounds.lo()[0];
    int64_t BPartitionBounds0hi = BPartitionBounds.hi()[0];
    int64_t BPartitionBounds1lo = BPartitionBounds.lo()[1];
    int64_t BPartitionBounds1hi = BPartitionBounds.hi()[1];
    Point<3> BStart = Point<3>(BPartitionBounds0lo, BPartitionBounds1lo, 0);
    Point<3> BEnd = Point<3>(TACO_MIN(((((BPartitionBounds0hi - BPartitionBounds0lo) + 1) - 1) + BPartitionBounds0lo), BDomain.hi()[0]), TACO_MIN(((((BPartitionBounds1hi - BPartitionBounds1lo) + 1) - 1) + BPartitionBounds1lo), BDomain.hi()[1]), TACO_MIN(B3_dimension, BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> AStart = Point<2>(BPartitionBounds0lo, BPartitionBounds1lo);
    Point<2> AEnd = Point<2>(TACO_MIN(((((BPartitionBounds0hi - BPartitionBounds0lo) + 1) - 1) + BPartitionBounds0lo), ADomain.hi()[0]), TACO_MIN(((((BPartitionBounds1hi - BPartitionBounds1lo) + 1) - 1) + BPartitionBounds1lo), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto B_vals_partition = copyPartition(ctx, runtime, B_dense_run_0_Partition, get_logical_region(B_vals));
  auto A_dense_run_0_Partition = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition, get_logical_region(A_vals));
  task_4Args taskArgsRaw4;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw4, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(A_vals_partition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(B_vals_partition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(C_vals), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
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
