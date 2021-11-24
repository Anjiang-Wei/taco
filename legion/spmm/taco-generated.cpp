#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;

struct task_2Args {
  int32_t B2_dimension;
  int32_t C2_dimension;
  IndexSpace C_dense_run_0;
  int32_t gx;
};
struct partitionPackForcomputeLegion {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
};

struct task_3Args {
  int32_t A1_dimension;
  int32_t A2_dimension;
  int32_t B1_dimension;
  int32_t C1_dimension;
  int32_t C2_dimension;
  int32_t gx;
  int32_t io;
  int32_t jo;
  int64_t pointID1;
};
struct task_4Args {
  int32_t A1_dimension;
  int32_t A2_dimension;
  int32_t B1_dimension;
  int32_t C1_dimension;
  int32_t C2_dimension;
  int32_t gx;
};

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

  int32_t io = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t B2_dimension = args->B2_dimension;
  int32_t C2_dimension = args->C2_dimension;
  IndexSpace C_dense_run_0 = args->C_dense_run_0;
  int32_t gx = args->gx;


  int64_t pointID1 = io;
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gx - 1));
  auto joIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(joIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t jo = (*itr)[0];
    Point<2> CStart = Point<2>(0, (jo * ((C2_dimension + (gx - 1)) / gx) + 0 / gx));
    Point<2> CEnd = Point<2>(TACO_MIN(B2_dimension, CDomain.hi()[0]), TACO_MIN((jo * ((C2_dimension + (gx - 1)) / gx) + ((C2_dimension + (gx - 1)) / gx - 1)), CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto CPartition = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, CPartition, get_logical_region(C_vals), pointID1);
}

partitionPackForcomputeLegion* partitionForcomputeLegion(Context ctx, Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gx) {
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  int B1_dimension = B->dims[0];
  int B2_dimension = B->dims[1];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  int C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

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
    int32_t io = (*itr)[0];
    Point<2> AStart = Point<2>((io * ((B1_dimension + (gx - 1)) / gx) + 0 / gx), (0 / gx));
    Point<2> AEnd = Point<2>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), ADomain.hi()[0]), TACO_MIN(((gx - 1) * ((C2_dimension + (gx - 1)) / gx) + ((C2_dimension + (gx - 1)) / gx - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<1> BStart = Point<1>((io * ((B1_dimension + (gx - 1)) / gx) + 0 / gx));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, APartition, get_logical_region(A_vals));
  auto BPartition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition posPartB2 = copyPartition(ctx, runtime, BPartition, B2_pos);
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd_parent, runtime->create_partition_by_image_range(
    ctx,
    B2_crd.get_index_space(),
    posPartB2,
    B2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, posPartB2.get_index_partition())
  ));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  task_2Args taskArgsRaw;
  taskArgsRaw.B2_dimension = B2_dimension;
  taskArgsRaw.C2_dimension = C2_dimension;
  taskArgsRaw.C_dense_run_0 = C_dense_run_0;
  taskArgsRaw.gx = gx;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(A_vals_partition, 0, READ_ONLY, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(posPartB2, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(crdPartB2, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(B_vals_partition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(C_vals), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher);


  auto computePartitions = new(partitionPackForcomputeLegion);
  computePartitions->APartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions->APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions->APartition.valsPartition = A_vals_partition;
  computePartitions->APartition.denseLevelRunPartitions[0] = APartition;
  computePartitions->BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions->BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions->BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions->BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions->BPartition.valsPartition = B_vals_partition;
  computePartitions->BPartition.denseLevelRunPartitions[0] = BPartition;
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

  task_3Args* args = (task_3Args*)(task->args);
  int32_t A1_dimension = args->A1_dimension;
  int32_t A2_dimension = args->A2_dimension;
  int32_t B1_dimension = args->B1_dimension;
  int32_t C1_dimension = args->C1_dimension;
  int32_t C2_dimension = args->C2_dimension;
  int32_t gx = args->gx;
  int32_t io = args->io;
  int32_t jo = args->jo;
  int64_t pointID1 = args->pointID1;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, FID_VAL);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble2>(A_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  int64_t pointID2 = pointID1 * gx + jo;
  for (int32_t ii = 0 / gx; ii < ((B1_dimension + (gx - 1)) / gx); ii++) {
    int32_t i = io * (((B1_dimension - 0) + (gx - 1)) / gx) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (gx - 1)) / gx - 0 / gx))
      continue;

    int64_t pointID3 = pointID2 * ((B1_dimension + (gx - 1)) / gx) + ii;
    int32_t iA = 0 * A1_dimension + i;
    int32_t iB = 0 * B1_dimension + i;
    for (int32_t ji = 0 / gx; ji < ((C2_dimension + (gx - 1)) / gx); ji++) {
      int32_t j = jo * (((C2_dimension - 0) + (gx - 1)) / gx) + ji;
      if (j >= C2_dimension)
        continue;

      if (j >= (jo + 1) * ((C2_dimension + (gx - 1)) / gx - 0 / gx))
        continue;

      int64_t pointID4 = pointID3 * ((C2_dimension + (gx - 1)) / gx) + ji;
      int32_t jA = iA * A2_dimension + j;
      for (int32_t kB = B2_pos_accessor[Point<1>(i)].lo; kB < (B2_pos_accessor[Point<1>(i)].hi + 1); kB++) {
        int32_t k = B2_crd_accessor[(kB * 1)];
        int32_t kC = 0 * C1_dimension + k;
        int32_t jC = kC * C2_dimension + j;
        A_vals_rw_accessor[Point<2>(i, j)] = A_vals_rw_accessor[Point<2>(i, j)] + B_vals_ro_accessor[Point<1>(kB)] * C_vals_ro_accessor[Point<2>(k, j)];
      }
    }
  }
}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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

  int32_t io = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t A1_dimension = args->A1_dimension;
  int32_t A2_dimension = args->A2_dimension;
  int32_t B1_dimension = args->B1_dimension;
  int32_t C1_dimension = args->C1_dimension;
  int32_t C2_dimension = args->C2_dimension;
  int32_t gx = args->gx;


  int64_t pointID1 = io;
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gx - 1));
  auto joIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(joIndexSpace));
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t jo = (*itr);
    task_3Args taskArgsRaw;
    taskArgsRaw.A1_dimension = A1_dimension;
    taskArgsRaw.A2_dimension = A2_dimension;
    taskArgsRaw.B1_dimension = B1_dimension;
    taskArgsRaw.C1_dimension = C1_dimension;
    taskArgsRaw.C2_dimension = C2_dimension;
    taskArgsRaw.gx = gx;
    taskArgsRaw.io = io;
    taskArgsRaw.jo = jo;
    taskArgsRaw.pointID1 = pointID1;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
    TaskLauncher launcher = TaskLauncher(taskID(3), taskArgs);
    launcher.add_region_requirement(RegionRequirement(get_logical_region(A_vals), READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(B2_pos), READ_ONLY, EXCLUSIVE, B2_pos_parent).add_field(FID_RECT_1));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(B2_crd), READ_ONLY, EXCLUSIVE, B2_crd_parent).add_field(FID_COORD));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(B_vals), READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
    launcher.add_region_requirement(RegionRequirement(get_logical_region(runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition_by_color(ctx, get_logical_region(C_vals), pointID1), jo)), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
    launcher.tag = launcher.tag | TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegion(Context ctx, Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t gx) {
  int A1_dimension = A->dims[0];
  int A2_dimension = A->dims[1];
  auto A_vals_parent = A->valsParent;
  int B1_dimension = B->dims[0];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  int C1_dimension = C->dims[0];
  int C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gx - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_4Args taskArgsRaw;
  taskArgsRaw.A1_dimension = A1_dimension;
  taskArgsRaw.A2_dimension = A2_dimension;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.C1_dimension = C1_dimension;
  taskArgsRaw.C2_dimension = C2_dimension;
  taskArgsRaw.gx = gx;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(C_vals), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher);

}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
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
    registrar.set_inner();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
}
