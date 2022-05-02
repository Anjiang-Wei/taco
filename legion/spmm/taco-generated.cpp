#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  Legion::FieldID A_vals_field_id;
  int64_t B1_dimension;
  Legion::FieldID B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id;
  int64_t C2_dimension;
  Legion::FieldID C_vals_field_id;
  int32_t gx;
};

struct task_2Args {
  Legion::FieldID A_vals_field_id;
  int64_t B1_dimension;
  Legion::FieldID B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id;
  int64_t C2_dimension;
  Legion::FieldID C_vals_field_id;
  int32_t gx;
  int32_t gy;
};

struct task_3Args {
  Legion::FieldID A_vals_field_id;
  int64_t B1_dimension;
  Legion::FieldID B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id;
  int64_t C2_dimension;
  Legion::FieldID C_vals_field_id;
  int32_t gx;
  int32_t gy;
  int64_t jo;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gx) {
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  size_t C2_dimension = C->dims[1];

  auto computePartitions = partitionPackForcomputeLegion();


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gx - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t io = (*itr)[0];
    Point<2> AStart = Point<2>((io * ((B1_dimension + (gx - 1)) / gx)), 0);
    Point<2> AEnd = Point<2>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), ADomain.hi()[0]), TACO_MIN(C2_dimension, ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<1> BStart = Point<1>((io * ((B1_dimension + (gx - 1)) / gx)));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  IndexPartition A_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition_0, get_logical_region(A_vals));
  IndexPartition B_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  Legion::LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition_0, B2_pos);
  Legion::LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B2_crd.get_index_space(), posPartB2, B2_pos_parent, B2_indices_field_id_1_0));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition_0;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_0;

  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
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

  int64_t io = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  int64_t B1_dimension = args->B1_dimension;
  Legion::FieldID B2_indices_field_id_1_0 = args->B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1 = args->B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  int64_t C2_dimension = args->C2_dimension;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int32_t gx = args->gx;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, B_vals_field_id);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, C_vals_field_id);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble2>(A_vals, A_vals_field_id);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, B2_indices_field_id_1_0);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, B2_indices_field_id_1_1);

  #pragma omp parallel for schedule(dynamic, 128)
  for (int64_t ii = 0; ii < ((B1_dimension + (gx - 1)) / gx); ii++) {
    int64_t i = io * ((B1_dimension + (gx - 1)) / gx) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (gx - 1)) / gx))
      continue;

    for (int64_t kB = B2_pos_accessor[Point<1>(i)].lo; kB < (B2_pos_accessor[Point<1>(i)].hi + 1); kB++) {
      int64_t k = B2_crd_accessor[kB];
      for (int64_t j = 0; j < C2_dimension; j++) {
        A_vals_rw_accessor[Point<2>(i, j)] = A_vals_rw_accessor[Point<2>(i, j)] + B_vals_ro_accessor[Point<1>(kB)] * C_vals_ro_accessor[Point<2>(k, j)];
      }
    }
  }
}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t gx) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  size_t B1_dimension = B->dims[0];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B2_indices_field_id_1_1 = B->indicesFieldIDs[1][1];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gx - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.A_vals_field_id = A_vals_field_id;
  taskArgsRaw1.B1_dimension = B1_dimension;
  taskArgsRaw1.B2_indices_field_id_1_0 = B2_indices_field_id_1_0;
  taskArgsRaw1.B2_indices_field_id_1_1 = B2_indices_field_id_1_1;
  taskArgsRaw1.B_vals_field_id = B_vals_field_id;
  taskArgsRaw1.C2_dimension = C2_dimension;
  taskArgsRaw1.C_vals_field_id = C_vals_field_id;
  taskArgsRaw1.gx = gx;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(B2_indices_field_id_1_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(B2_indices_field_id_1_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(C_vals), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(C_vals_field_id));
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionMemCons partitionForcomputeLegionMemCons(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gx, int32_t gy) {
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegionMemCons();


  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gx - 1), (gy - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int64_t io = (*itr)[0];
    int64_t jo = (*itr)[1];
    Point<2> AStart = Point<2>((io * ((B1_dimension + (gx - 1)) / gx)), (jo * ((C2_dimension + (gy - 1)) / gy)));
    Point<2> AEnd = Point<2>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), ADomain.hi()[0]), TACO_MIN((jo * ((C2_dimension + (gy - 1)) / gy) + ((C2_dimension + (gy - 1)) / gy - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<1> BStart = Point<1>((io * ((B1_dimension + (gx - 1)) / gx)));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> CStart = Point<2>(0, (jo * ((C2_dimension + (gy - 1)) / gy)));
    Point<2> CEnd = Point<2>(TACO_MIN(B2_dimension, CDomain.hi()[0]), TACO_MIN((jo * ((C2_dimension + (gy - 1)) / gy) + ((C2_dimension + (gy - 1)) / gy - 1)), CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  IndexPartition A_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition_0, get_logical_region(A_vals));
  IndexPartition B_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_COMPUTE_KIND);
  Legion::LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition_0, B2_pos);
  Legion::LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B2_crd.get_index_space(), posPartB2, B2_pos_parent, B2_indices_field_id_1_0));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  IndexPartition C_dense_run_0_Partition_0 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
  auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_0, get_logical_region(C_vals));
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition_0;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_0;
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition_0;

  return computePartitions;
}

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

  int64_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  int64_t B1_dimension = args->B1_dimension;
  Legion::FieldID B2_indices_field_id_1_0 = args->B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1 = args->B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  int64_t C2_dimension = args->C2_dimension;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int32_t gx = args->gx;
  int32_t gy = args->gy;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, B_vals_field_id);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, C_vals_field_id);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble2>(A_vals, A_vals_field_id);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, B2_indices_field_id_1_0);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, B2_indices_field_id_1_1);

  int64_t io = getIndexPoint(task, 0);
  int64_t jo = getIndexPoint(task, 1);
  for (int64_t ii = 0; ii < ((B1_dimension + (gx - 1)) / gx); ii++) {
    int64_t i = io * ((B1_dimension + (gx - 1)) / gx) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (gx - 1)) / gx))
      continue;

    for (int64_t kB = B2_pos_accessor[Point<1>(i)].lo; kB < (B2_pos_accessor[Point<1>(i)].hi + 1); kB++) {
      int64_t k = B2_crd_accessor[kB];
      for (int64_t ji = 0; ji < ((C2_dimension + (gy - 1)) / gy); ji++) {
        int64_t j = jo * ((C2_dimension + (gy - 1)) / gy) + ji;
        if (j >= C2_dimension)
          continue;

        if (j >= (jo + 1) * ((C2_dimension + (gy - 1)) / gy))
          continue;

        A_vals_rw_accessor[Point<2>(i, j)] = A_vals_rw_accessor[Point<2>(i, j)] + B_vals_ro_accessor[Point<1>(kB)] * C_vals_ro_accessor[Point<2>(k, j)];
      }
    }
  }
}

void computeLegionMemCons(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegionMemCons* partitionPack, int32_t gx, int32_t gy) {
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  size_t B1_dimension = B->dims[0];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B2_indices_field_id_1_1 = B->indicesFieldIDs[1][1];
  size_t C2_dimension = C->dims[1];
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;


  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gx - 1), (gy - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.A_vals_field_id = A_vals_field_id;
  taskArgsRaw2.B1_dimension = B1_dimension;
  taskArgsRaw2.B2_indices_field_id_1_0 = B2_indices_field_id_1_0;
  taskArgsRaw2.B2_indices_field_id_1_1 = B2_indices_field_id_1_1;
  taskArgsRaw2.B_vals_field_id = B_vals_field_id;
  taskArgsRaw2.C2_dimension = C2_dimension;
  taskArgsRaw2.C_vals_field_id = C_vals_field_id;
  taskArgsRaw2.gx = gx;
  taskArgsRaw2.gy = gy;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(A_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(B2_indices_field_id_1_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(B2_indices_field_id_1_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(C_vals_field_id));
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionBatched partitionForcomputeLegionBatched(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gy, int32_t gx) {
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  size_t B1_dimension = B->dims[0];
  size_t B2_dimension = B->dims[1];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegionBatched();


  for (int64_t jo = 0; jo < gy; jo++) {
    int64_t pointID1 = jo + TACO_PARTITION_COLOR_OFFSET;
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
      int64_t io = (*itr)[0];
      Point<2> AStart = Point<2>((io * ((B1_dimension + (gx - 1)) / gx)), (jo * ((C2_dimension + (gy - 1)) / gy)));
      Point<2> AEnd = Point<2>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), ADomain.hi()[0]), TACO_MIN((jo * ((C2_dimension + (gy - 1)) / gy) + ((C2_dimension + (gy - 1)) / gy - 1)), ADomain.hi()[1]));
      Rect<2> ARect = Rect<2>(AStart, AEnd);
      if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
        ARect = ARect.make_empty();
      }
      AColoring[(*itr)] = ARect;
      Point<1> BStart = Point<1>((io * ((B1_dimension + (gx - 1)) / gx)));
      Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (gx - 1)) / gx) + ((B1_dimension + (gx - 1)) / gx - 1)), BDomain.hi()[0]));
      Rect<1> BRect = Rect<1>(BStart, BEnd);
      if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
        BRect = BRect.make_empty();
      }
      BColoring[(*itr)] = BRect;
      Point<2> CStart = Point<2>(0, (jo * ((C2_dimension + (gy - 1)) / gy)));
      Point<2> CEnd = Point<2>(TACO_MIN(B2_dimension, CDomain.hi()[0]), TACO_MIN((jo * ((C2_dimension + (gy - 1)) / gy) + ((C2_dimension + (gy - 1)) / gy - 1)), CDomain.hi()[1]));
      Rect<2> CRect = Rect<2>(CStart, CEnd);
      if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
        CRect = CRect.make_empty();
      }
      CColoring[(*itr)] = CRect;
    }
    IndexPartition A_dense_run_0_Partition_1 = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_COMPUTE_KIND);
    auto A_vals_partition = copyPartition(ctx, runtime, A_dense_run_0_Partition_1, get_logical_region(A_vals), pointID1);
    IndexPartition B_dense_run_0_Partition_1 = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
    Legion::LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition_1, B2_pos, pointID1);
    Legion::LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(
      ctx,
      runtime,
      B2_crd.get_index_space(),
      posPartB2,
      B2_pos_parent,
      B2_indices_field_id_1_0,
      pointID1
    ));
    auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals), pointID1);
    IndexPartition C_dense_run_0_Partition_1 = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_COMPUTE_KIND);
    auto C_vals_partition = copyPartition(ctx, runtime, C_dense_run_0_Partition_1, get_logical_region(C_vals), pointID1);
    computePartitions.APartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
    computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
    computePartitions.APartition.valsPartition = A_vals_partition;
    computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition_1;
    computePartitions.BPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
    computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
    computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
    computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
    computePartitions.BPartition.valsPartition = B_vals_partition;
    computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition_1;
    computePartitions.CPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(2);
    computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
    computePartitions.CPartition.valsPartition = C_vals_partition;
    computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition_1;
  }

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

  int64_t io = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  Legion::FieldID A_vals_field_id = args->A_vals_field_id;
  int64_t B1_dimension = args->B1_dimension;
  Legion::FieldID B2_indices_field_id_1_0 = args->B2_indices_field_id_1_0;
  Legion::FieldID B2_indices_field_id_1_1 = args->B2_indices_field_id_1_1;
  Legion::FieldID B_vals_field_id = args->B_vals_field_id;
  int64_t C2_dimension = args->C2_dimension;
  Legion::FieldID C_vals_field_id = args->C_vals_field_id;
  int32_t gx = args->gx;
  int32_t gy = args->gy;
  int64_t jo = args->jo;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, B_vals_field_id);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, C_vals_field_id);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble2>(A_vals, A_vals_field_id);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, B2_indices_field_id_1_0);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, B2_indices_field_id_1_1);

  for (int64_t ii = 0; ii < ((B1_dimension + (gx - 1)) / gx); ii++) {
    int64_t i = io * ((B1_dimension + (gx - 1)) / gx) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (gx - 1)) / gx))
      continue;

    for (int64_t kB = B2_pos_accessor[Point<1>(i)].lo; kB < (B2_pos_accessor[Point<1>(i)].hi + 1); kB++) {
      int64_t k = B2_crd_accessor[kB];
      for (int64_t ji = 0; ji < ((C2_dimension + (gy - 1)) / gy); ji++) {
        int64_t j = jo * ((C2_dimension + (gy - 1)) / gy) + ji;
        if (j >= C2_dimension)
          continue;

        if (j >= (jo + 1) * ((C2_dimension + (gy - 1)) / gy))
          continue;

        A_vals_rw_accessor[Point<2>(i, j)] = A_vals_rw_accessor[Point<2>(i, j)] + B_vals_ro_accessor[Point<1>(kB)] * C_vals_ro_accessor[Point<2>(k, j)];
      }
    }
  }
}

void computeLegionBatched(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegionBatched* partitionPack, int32_t gy, int32_t gx) {
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  auto A_vals_field_id = A->valsFieldID;
  size_t B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  RegionWrapper B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  auto B_vals_field_id = B->valsFieldID;
  auto B2_indices_field_id_1_0 = B->indicesFieldIDs[1][0];
  auto B2_indices_field_id_1_1 = B->indicesFieldIDs[1][1];
  size_t C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  auto C_vals_field_id = C->valsFieldID;


  for (int64_t jo = 0; jo < gy; jo++) {
    int64_t pointID1 = jo + TACO_PARTITION_COLOR_OFFSET;
    Point<1> lowerBound = Point<1>(0);
    Point<1> upperBound = Point<1>((gx - 1));
    auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
    DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
    task_3Args taskArgsRaw3;
    taskArgsRaw3.A_vals_field_id = A_vals_field_id;
    taskArgsRaw3.B1_dimension = B1_dimension;
    taskArgsRaw3.B2_indices_field_id_1_0 = B2_indices_field_id_1_0;
    taskArgsRaw3.B2_indices_field_id_1_1 = B2_indices_field_id_1_1;
    taskArgsRaw3.B_vals_field_id = B_vals_field_id;
    taskArgsRaw3.C2_dimension = C2_dimension;
    taskArgsRaw3.C_vals_field_id = C_vals_field_id;
    taskArgsRaw3.gx = gx;
    taskArgsRaw3.gy = gy;
    taskArgsRaw3.jo = jo;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
    IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(A_vals), pointID1), 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(A_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(B2_pos), pointID1), 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(B2_indices_field_id_1_0));
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(B2_crd), pointID1), 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(B2_indices_field_id_1_1));
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(B_vals), pointID1), 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(B_vals_field_id));
    launcher.add_region_requirement(RegionRequirement(runtime->get_logical_partition_by_color(ctx, get_logical_region(C_vals), pointID1), 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(C_vals_field_id));
    runtime->execute_index_space(ctx, launcher);

  }
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
}
