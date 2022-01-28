#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef ReductionAccessor<SumReduction<double>,false,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorReduceNonExcldouble1;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int32_t B1_dimension;
  int32_t pieces;
};

struct task_2Args {
  int64_t B3Size;
  int32_t pieces;
};

struct task_3Args {
  int64_t B2Size;
  int32_t pieces;
};


partitionPackForcomputeLegionDSS partitionForcomputeLegionDSS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper A2_pos = A->indices[1][0];
  RegionWrapper A2_crd = A->indices[1][1];
  auto A2_pos_parent = A->indicesParents[1][0];
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  int B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_pos = B->indices[2][0];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B3_pos_parent = B->indicesParents[2][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  IndexSpace c_dense_run_0 = c->denseLevelRuns[0];

  int64_t A2Size = runtime->get_index_space_domain(ctx, get_index_space(A2_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t io = (*itr)[0];
    Point<1> BStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<1> AStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> AEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), ADomain.hi()[0]));
    Rect<1> ARect = Rect<1>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition, B2_pos);
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B2_crd.get_index_space(), posPartB2, B2_pos_parent, FID_RECT_1));
  LogicalPartition posPartB3 = copyPartition(ctx, runtime, crdPartB2, B3_pos);
  LogicalPartition crdPartB3 = runtime->get_logical_partition(ctx, B3_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B3_crd.get_index_space(), posPartB3, B3_pos_parent, FID_RECT_1));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB3, get_logical_region(B_vals));
  auto A_dense_run_0_Partition = runtime->create_index_partition(ctx, A_dense_run_0, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition posPartA2 = copyPartition(ctx, runtime, A_dense_run_0_Partition, A2_pos);
  LogicalPartition crdPartA2 = runtime->get_logical_partition(ctx, A2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, A2_crd.get_index_space(), posPartA2, A2_pos_parent, FID_RECT_1));
  auto A_vals_partition = copyPartition(ctx, runtime, crdPartA2, get_logical_region(A_vals));
  auto computePartitions = partitionPackForcomputeLegionDSS();
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.indicesPartitions[1].push_back(posPartA2);
  computePartitions.APartition.indicesPartitions[1].push_back(crdPartA2);
  computePartitions.APartition.valsPartition = A_vals_partition;
  computePartitions.APartition.denseLevelRunPartitions[0] = A_dense_run_0_Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(3);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.indicesPartitions[2].push_back(posPartB3);
  computePartitions.BPartition.indicesPartitions[2].push_back(crdPartB3);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition;

  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A2_pos = regions[0];
  LogicalRegion A2_pos_parent = regions[0].get_logical_region();
  PhysicalRegion A2_crd = regions[1];
  LogicalRegion A2_crd_parent = regions[1].get_logical_region();
  PhysicalRegion A_vals = regions[2];
  LogicalRegion A_vals_parent = regions[2].get_logical_region();
  PhysicalRegion B2_pos = regions[3];
  LogicalRegion B2_pos_parent = regions[3].get_logical_region();
  PhysicalRegion B2_crd = regions[4];
  LogicalRegion B2_crd_parent = regions[4].get_logical_region();
  PhysicalRegion B3_pos = regions[5];
  LogicalRegion B3_pos_parent = regions[5].get_logical_region();
  PhysicalRegion B3_crd = regions[6];
  LogicalRegion B3_crd_parent = regions[6].get_logical_region();
  PhysicalRegion B_vals = regions[7];
  LogicalRegion B_vals_parent = regions[7].get_logical_region();
  PhysicalRegion c_vals = regions[8];
  LogicalRegion c_vals_parent = regions[8].get_logical_region();

  int32_t io = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t B1_dimension = args->B1_dimension;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble1>(A_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);
  auto B3_pos_accessor = createAccessor<AccessorRORect_1_1>(B3_pos, FID_RECT_1);
  auto B3_crd_accessor = createAccessor<AccessorROint32_t1>(B3_crd, FID_COORD);

  int64_t pointID1 = io;
  #pragma omp parallel for schedule(dynamic, 1024)
  for (int32_t ii = 0; ii < ((B1_dimension + (pieces - 1)) / pieces); ii++) {
    int32_t i = io * ((B1_dimension + (pieces - 1)) / pieces) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (pieces - 1)) / pieces))
      continue;

    int64_t pointID2 = pointID1 * ((B1_dimension + (pieces - 1)) / pieces) + ii;
    int32_t iA = i;
    int32_t iB = i;
    for (int32_t jB = B2_pos_accessor[Point<1>(i)].lo; jB < (B2_pos_accessor[Point<1>(i)].hi + 1); jB++) {
      int32_t j = B2_crd_accessor[(jB * 1)];
      int32_t jA = jB;
      for (int32_t kB = B3_pos_accessor[Point<1>(jB)].lo; kB < (B3_pos_accessor[Point<1>(jB)].hi + 1); kB++) {
        int32_t k = B3_crd_accessor[(kB * 1)];
        int32_t kc = k;
        A_vals_rw_accessor[Point<1>(jA)] = A_vals_rw_accessor[Point<1>(jA)] + B_vals_ro_accessor[Point<1>(kB)] * c_vals_ro_accessor[Point<1>(k)];
      }
    }
  }
}

void computeLegionDSS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSS* partitionPack, int32_t pieces) {
  RegionWrapper A2_crd = A->indices[1][1];
  auto A2_pos_parent = A->indicesParents[1][0];
  auto A2_crd_parent = A->indicesParents[1][1];
  auto A_vals_parent = A->valsParent;
  int B1_dimension = B->dims[0];
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B3_pos_parent = B->indicesParents[2][0];
  auto B3_crd_parent = B->indicesParents[2][1];
  auto B_vals_parent = B->valsParent;
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t A2Size = runtime->get_index_space_domain(ctx, get_index_space(A2_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.B1_dimension = B1_dimension;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(A2_pos_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(A2_crd_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionDSSPosSplit partitionForcomputeLegionDSSPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper A2_crd = A->indices[1][1];
  auto A2_pos_parent = A->indicesParents[1][0];
  auto A2_crd_parent = A->indicesParents[1][1];
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_pos = B->indices[2][0];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B3_pos_parent = B->indicesParents[2][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  int64_t A2Size = runtime->get_index_space_domain(ctx, get_index_space(A2_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ffposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ffposoIndexSpace));
  DomainT<1> B3_crd_domain = runtime->get_index_space_domain(ctx, B3_crd.get_index_space());
  DomainPointColoring B3_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t ffposo = (*itr)[0];
    Point<1> B3CrdStart = Point<1>((ffposo * ((B3Size + (pieces - 1)) / pieces)));
    Point<1> B3CrdEnd = Point<1>(TACO_MIN((ffposo * ((B3Size + (pieces - 1)) / pieces) + ((B3Size + (pieces - 1)) / pieces - 1)), B3_crd_domain.bounds.hi[0]));
    Rect<1> B3CrdRect = Rect<1>(B3CrdStart, B3CrdEnd);
    if (!B3_crd_domain.contains(B3CrdRect.lo) || !B3_crd_domain.contains(B3CrdRect.hi)) {
      B3CrdRect = B3CrdRect.make_empty();
    }
    B3_crd_coloring[(*itr)] = B3CrdRect;
  }
  IndexPartition B3_crd_index_part = runtime->create_index_partition(ctx, B3_crd.get_index_space(), domain, B3_crd_coloring, LEGION_COMPUTE_KIND);
  LogicalPartition B3_crd_part = runtime->get_logical_partition(ctx, B3_crd, B3_crd_index_part);
  IndexPartition posSparsePartB3 = runtime->create_partition_by_preimage_range(ctx, B3_crd_index_part, B3_pos, B3_pos_parent, FID_RECT_1, runtime->get_index_partition_color_space_name(ctx, B3_crd_index_part));
  IndexPartition posIndexPartB3 = densifyPartition(ctx, runtime, get_index_space(B3_pos), posSparsePartB3);
  LogicalPartition posPartB3 = runtime->get_logical_partition(ctx, B3_pos, posIndexPartB3);
  LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, B3_crd_part, B_vals);
  LogicalPartition crdPartB2 = copyPartition(ctx, runtime, posPartB3, B2_crd);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(ctx, crdPartB2.get_index_partition(), B2_pos, B2_pos_parent, FID_RECT_1, runtime->get_index_partition_color_space_name(ctx, crdPartB2.get_index_partition()));
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  IndexPartition BDenseRun0Partition = copyPartition(ctx, runtime, posPartB2, B_dense_run_0);
  LogicalPartition AValsLogicalPart = copyPartition(ctx, runtime, posPartB3, A_vals);
  LogicalPartition posPartA2 = copyPartition(ctx, runtime, posPartB2, A2_pos_parent);
  LogicalPartition crdPartA2 = copyPartition(ctx, runtime, crdPartB2, A2_crd_parent);
  IndexPartition ADenseRun0Partition = copyPartition(ctx, runtime, posPartA2, A_dense_run_0);
  auto computePartitions = partitionPackForcomputeLegionDSSPosSplit();
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.indicesPartitions[1].push_back(posPartA2);
  computePartitions.APartition.indicesPartitions[1].push_back(crdPartA2);
  computePartitions.APartition.valsPartition = AValsLogicalPart;
  computePartitions.APartition.denseLevelRunPartitions[0] = ADenseRun0Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(3);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.indicesPartitions[2].push_back(posPartB3);
  computePartitions.BPartition.indicesPartitions[2].push_back(B3_crd_part);
  computePartitions.BPartition.valsPartition = BValsLogicalPart;
  computePartitions.BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;

  return computePartitions;
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A2_pos = regions[0];
  LogicalRegion A2_pos_parent = regions[0].get_logical_region();
  PhysicalRegion A2_crd = regions[1];
  LogicalRegion A2_crd_parent = regions[1].get_logical_region();
  PhysicalRegion A_vals = regions[2];
  LogicalRegion A_vals_parent = regions[2].get_logical_region();
  PhysicalRegion B2_pos = regions[3];
  LogicalRegion B2_pos_parent = regions[3].get_logical_region();
  PhysicalRegion B2_crd = regions[4];
  LogicalRegion B2_crd_parent = regions[4].get_logical_region();
  PhysicalRegion B3_pos = regions[5];
  LogicalRegion B3_pos_parent = regions[5].get_logical_region();
  PhysicalRegion B3_crd = regions[6];
  LogicalRegion B3_crd_parent = regions[6].get_logical_region();
  PhysicalRegion B_vals = regions[7];
  LogicalRegion B_vals_parent = regions[7].get_logical_region();
  PhysicalRegion c_vals = regions[8];
  LogicalRegion c_vals_parent = regions[8].get_logical_region();

  int32_t ffposo = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int64_t B3Size = args->B3Size;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto A_vals_red_accessor_non_excl = createAccessor<AccessorReduceNonExcldouble1>(A_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);
  auto B3_pos_accessor = createAccessor<AccessorRORect_1_1>(B3_pos, FID_RECT_1);
  auto B3_crd_accessor = createAccessor<AccessorROint32_t1>(B3_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).empty())
    return ;

  DomainT<1> B3PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B3_pos));
  DomainT<1> B3CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B3_crd));
  int32_t pB3_begin = B3PosDomain.bounds.lo;
  int32_t pB3_end = B3PosDomain.bounds.hi;
  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  int32_t pB2_begin = B2PosDomain.bounds.lo;
  int32_t pB2_end = B2PosDomain.bounds.hi;
  int64_t pointID1 = ffposo;
  #pragma omp parallel for schedule(static)
  for (int32_t ffposio = 0; ffposio < (((B3Size + (pieces - 1)) / pieces + 2047) / 2048); ffposio++) {
    int64_t pointID2 = pointID1 * (((B3Size + (pieces - 1)) / pieces + 2047) / 2048) + ffposio;
    int32_t ffposi = ffposio * 2048;
    int32_t ffposB = ffposo * ((B3Size + (pieces - 1)) / pieces) + ffposi;
    int32_t j_pos = taco_binarySearchBefore(B3_pos_accessor, pB3_begin, pB3_end, ffposB);
    int32_t j = B2_crd_accessor[j_pos];
    int32_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, j_pos);
    int32_t i = i_pos;
    for (int32_t ffposii = 0; ffposii < 2048; ffposii++) {
      int32_t ffposi = ffposio * 2048 + ffposii;
      int32_t ffposB = ffposo * ((B3Size + (pieces - 1)) / pieces) + ffposi;
      if (ffposB >= (ffposo + 1) * ((B3Size + (pieces - 1)) / pieces))
        continue;

      if (ffposB >= B3Size)
        continue;

      int32_t ff = B3_crd_accessor[ffposB];
      int32_t k = ff;
      if (!(B3_pos_accessor[j_pos].contains(ffposB))) {
        j_pos = j_pos + 1;
        j = B2_crd_accessor[j_pos];
        while (!(B2_pos_accessor[i_pos].contains(j_pos))) {
          i_pos = i_pos + 1;
          i = i_pos;
        }
      }
      int32_t jB = j_pos;
      int32_t iA = i;
      int32_t kc = k;
      int32_t jA = jB;
      A_vals_red_accessor_non_excl[Point<1>(jA)] <<= B_vals_ro_accessor[Point<1>(ffposB)] * c_vals_ro_accessor[Point<1>(k)];
    }
  }
}

void computeLegionDSSPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSSPosSplit* partitionPack, int32_t pieces) {
  RegionWrapper A2_crd = A->indices[1][1];
  auto A2_pos_parent = A->indicesParents[1][0];
  auto A2_crd_parent = A->indicesParents[1][1];
  auto A_vals_parent = A->valsParent;
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B3_pos_parent = B->indicesParents[2][0];
  auto B3_crd_parent = B->indicesParents[2][1];
  auto B_vals_parent = B->valsParent;
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t A2Size = runtime->get_index_space_domain(ctx, get_index_space(A2_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ffposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ffposoIndexSpace));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.B3Size = B3Size;
  taskArgsRaw2.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(A2_pos_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(A2_crd_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);

}

partitionPackForcomputeLegionDSSPartialPosSplit partitionForcomputeLegionDSSPartialPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, int32_t pieces) {
  RegionWrapper A2_crd = A->indices[1][1];
  auto A2_pos_parent = A->indicesParents[1][0];
  auto A2_crd_parent = A->indicesParents[1][1];
  RegionWrapper A_vals = A->vals;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_pos = B->indices[2][0];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B3_pos_parent = B->indicesParents[2][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];

  int64_t A2Size = runtime->get_index_space_domain(ctx, get_index_space(A2_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ffposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ffposoIndexSpace));
  DomainT<1> B2_crd_domain = runtime->get_index_space_domain(ctx, B2_crd.get_index_space());
  DomainPointColoring B2_crd_coloring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t ffposo = (*itr)[0];
    Point<1> B2CrdStart = Point<1>((ffposo * ((B2Size + (pieces - 1)) / pieces)));
    Point<1> B2CrdEnd = Point<1>(TACO_MIN((ffposo * ((B2Size + (pieces - 1)) / pieces) + ((B2Size + (pieces - 1)) / pieces - 1)), B2_crd_domain.bounds.hi[0]));
    Rect<1> B2CrdRect = Rect<1>(B2CrdStart, B2CrdEnd);
    if (!B2_crd_domain.contains(B2CrdRect.lo) || !B2_crd_domain.contains(B2CrdRect.hi)) {
      B2CrdRect = B2CrdRect.make_empty();
    }
    B2_crd_coloring[(*itr)] = B2CrdRect;
  }
  IndexPartition B2_crd_index_part = runtime->create_index_partition(ctx, B2_crd.get_index_space(), domain, B2_crd_coloring, LEGION_COMPUTE_KIND);
  LogicalPartition B2_crd_part = runtime->get_logical_partition(ctx, B2_crd, B2_crd_index_part);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(ctx, B2_crd_index_part, B2_pos, B2_pos_parent, FID_RECT_1, runtime->get_index_partition_color_space_name(ctx, B2_crd_index_part));
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  LogicalPartition posPartB3 = copyPartition(ctx, runtime, B2_crd_part, B3_pos);
  LogicalPartition crdPartB3 = runtime->get_logical_partition(ctx, B3_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B3_crd.get_index_space(), posPartB3, B3_pos_parent, FID_RECT_1));
  LogicalPartition BValsLogicalPart = copyPartition(ctx, runtime, crdPartB3, B_vals);
  IndexPartition BDenseRun0Partition = copyPartition(ctx, runtime, posPartB2, B_dense_run_0);
  LogicalPartition AValsLogicalPart = copyPartition(ctx, runtime, B2_crd_part, A_vals);
  LogicalPartition posPartA2 = copyPartition(ctx, runtime, posPartB2, A2_pos_parent);
  LogicalPartition crdPartA2 = copyPartition(ctx, runtime, B2_crd_part, A2_crd_parent);
  IndexPartition ADenseRun0Partition = copyPartition(ctx, runtime, posPartA2, A_dense_run_0);
  auto computePartitions = partitionPackForcomputeLegionDSSPartialPosSplit();
  computePartitions.APartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.APartition.indicesPartitions[1].push_back(posPartA2);
  computePartitions.APartition.indicesPartitions[1].push_back(crdPartA2);
  computePartitions.APartition.valsPartition = AValsLogicalPart;
  computePartitions.APartition.denseLevelRunPartitions[0] = ADenseRun0Partition;
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(3);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(B2_crd_part);
  computePartitions.BPartition.indicesPartitions[2].push_back(posPartB3);
  computePartitions.BPartition.indicesPartitions[2].push_back(crdPartB3);
  computePartitions.BPartition.valsPartition = BValsLogicalPart;
  computePartitions.BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;

  return computePartitions;
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A2_pos = regions[0];
  LogicalRegion A2_pos_parent = regions[0].get_logical_region();
  PhysicalRegion A2_crd = regions[1];
  LogicalRegion A2_crd_parent = regions[1].get_logical_region();
  PhysicalRegion A_vals = regions[2];
  LogicalRegion A_vals_parent = regions[2].get_logical_region();
  PhysicalRegion B2_pos = regions[3];
  LogicalRegion B2_pos_parent = regions[3].get_logical_region();
  PhysicalRegion B2_crd = regions[4];
  LogicalRegion B2_crd_parent = regions[4].get_logical_region();
  PhysicalRegion B3_pos = regions[5];
  LogicalRegion B3_pos_parent = regions[5].get_logical_region();
  PhysicalRegion B3_crd = regions[6];
  LogicalRegion B3_crd_parent = regions[6].get_logical_region();
  PhysicalRegion B_vals = regions[7];
  LogicalRegion B_vals_parent = regions[7].get_logical_region();
  PhysicalRegion c_vals = regions[8];
  LogicalRegion c_vals_parent = regions[8].get_logical_region();

  int32_t ffposo = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int64_t B2Size = args->B2Size;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble1>(A_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);
  auto B3_pos_accessor = createAccessor<AccessorRORect_1_1>(B3_pos, FID_RECT_1);
  auto B3_crd_accessor = createAccessor<AccessorROint32_t1>(B3_crd, FID_COORD);

  if (runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).empty())
    return ;

  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  DomainT<1> B2CrdDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_crd));
  int32_t pB2_begin = B2PosDomain.bounds.lo;
  int32_t pB2_end = B2PosDomain.bounds.hi;
  int64_t pointID1 = ffposo;
  #pragma omp parallel for schedule(static)
  for (int32_t ffposio = 0; ffposio < (((B2Size + (pieces - 1)) / pieces + 2047) / 2048); ffposio++) {
    int64_t pointID2 = pointID1 * (((B2Size + (pieces - 1)) / pieces + 2047) / 2048) + ffposio;
    int32_t ffposi = ffposio * 2048;
    int32_t ffposB = ffposo * ((B2Size + (pieces - 1)) / pieces) + ffposi;
    int32_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, ffposB);
    int32_t i = i_pos;
    for (int32_t ffposii = 0; ffposii < 2048; ffposii++) {
      int32_t ffposi = ffposio * 2048 + ffposii;
      int32_t ffposB = ffposo * ((B2Size + (pieces - 1)) / pieces) + ffposi;
      if (ffposB >= (ffposo + 1) * ((B2Size + (pieces - 1)) / pieces))
        continue;

      if (ffposB >= B2Size)
        continue;

      int32_t f = B2_crd_accessor[ffposB];
      int32_t j = f;
      while (!(B2_pos_accessor[i_pos].contains(ffposB))) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iA = i;
      int32_t jA = ffposB;
      for (int32_t kB = B3_pos_accessor[Point<1>(ffposB)].lo; kB < (B3_pos_accessor[Point<1>(ffposB)].hi + 1); kB++) {
        int32_t k = B3_crd_accessor[(kB * 1)];
        int32_t kc = k;
        A_vals_rw_accessor[Point<1>(jA)] = A_vals_rw_accessor[Point<1>(jA)] + B_vals_ro_accessor[Point<1>(kB)] * c_vals_ro_accessor[Point<1>(k)];
      }
    }
  }
}

void computeLegionDSSPartialPosSplit(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSSPartialPosSplit* partitionPack, int32_t pieces) {
  RegionWrapper A2_crd = A->indices[1][1];
  auto A2_pos_parent = A->indicesParents[1][0];
  auto A2_crd_parent = A->indicesParents[1][1];
  auto A_vals_parent = A->valsParent;
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B3_pos_parent = B->indicesParents[2][0];
  auto B3_crd_parent = B->indicesParents[2][1];
  auto B_vals_parent = B->valsParent;
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  int64_t A2Size = runtime->get_index_space_domain(ctx, get_index_space(A2_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ffposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ffposoIndexSpace));
  task_3Args taskArgsRaw3;
  taskArgsRaw3.B2Size = B2Size;
  taskArgsRaw3.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw3, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(A2_pos_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(A2_crd_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(c_vals), READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
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
}
