#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;

struct partitionPackForcomputeLegion {
  LegionTensorPartition APartition;
  LegionTensorPartition BPartition;
};

struct task_1Args {
  int32_t A1_dimension;
  int32_t A2_dimension;
  int64_t B3Size;
  int32_t C1_dimension;
  int32_t C2_dimension;
  int32_t D1_dimension;
  int32_t D2_dimension;
  int32_t pieces;
};

partitionPackForcomputeLegion* partitionForcomputeLegion(Context ctx, Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, int32_t pieces) {
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

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  auto BValsDomain = runtime->get_index_space_domain(ctx, get_index_space(B_vals));
  auto BValsColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t fposo = (*itr)[0];
    Point<1> BStart = Point<1>((fposo * ((B3Size + (pieces - 1)) / pieces) + 0 / pieces));
    Point<1> BEnd = Point<1>(TACO_MIN((fposo * ((B3Size + (pieces - 1)) / pieces) + ((B3Size + (pieces - 1)) / pieces - 1)), BValsDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BValsDomain.contains(BRect.lo) || !BValsDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BValsColoring[(*itr)] = BRect;
  }
  IndexPartition BValsIndexPart = runtime->create_index_partition(ctx, get_index_space(B_vals), domain, BValsColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition BValsLogicalPart = runtime->get_logical_partition(ctx, B_vals, BValsIndexPart);
  LogicalPartition crdPartB3 = copyPartition(ctx, runtime, BValsLogicalPart, B3_crd);
  IndexPartition posSparsePartB3 = runtime->create_partition_by_preimage_range(
    ctx,
    crdPartB3.get_index_partition(),
    B3_pos,
    B3_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, crdPartB3.get_index_partition())
  );
  IndexPartition posIndexPartB3 = densifyPartition(ctx, runtime, get_index_space(B3_pos), posSparsePartB3);
  LogicalPartition posPartB3 = runtime->get_logical_partition(ctx, B3_pos, posIndexPartB3);
  LogicalPartition crdPartB2 = copyPartition(ctx, runtime, posPartB3, B2_crd);
  IndexPartition posSparsePartB2 = runtime->create_partition_by_preimage_range(
    ctx,
    crdPartB2.get_index_partition(),
    B2_pos,
    B2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, crdPartB2.get_index_partition())
  );
  IndexPartition posIndexPartB2 = densifyPartition(ctx, runtime, get_index_space(B2_pos), posSparsePartB2);
  LogicalPartition posPartB2 = runtime->get_logical_partition(ctx, B2_pos, posIndexPartB2);
  IndexPartition BDenseRun0Partition = copyPartition(ctx, runtime, posPartB2, B_dense_run_0);
  IndexPartition ADenseRun0Partition = AffineProjection(0).apply(ctx, runtime, BDenseRun0Partition, A_dense_run_0);
  auto A_vals_partition = copyPartition(ctx, runtime, ADenseRun0Partition, get_logical_region(A_vals));
  auto computePartitions = new(partitionPackForcomputeLegion);
  computePartitions->APartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions->APartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions->APartition.valsPartition = A_vals_partition;
  computePartitions->APartition.denseLevelRunPartitions[0] = ADenseRun0Partition;
  computePartitions->BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(3);
  computePartitions->BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions->BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions->BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions->BPartition.indicesPartitions[2].push_back(posPartB3);
  computePartitions->BPartition.indicesPartitions[2].push_back(crdPartB3);
  computePartitions->BPartition.valsPartition = BValsLogicalPart;
  computePartitions->BPartition.denseLevelRunPartitions[0] = BDenseRun0Partition;
  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A_vals = regions[0];
  LogicalRegion A_vals_parent = regions[0].get_logical_region();
  PhysicalRegion B2_pos = regions[1];
  LogicalRegion B2_pos_parent = regions[1].get_logical_region();
  PhysicalRegion B2_crd = regions[2];
  LogicalRegion B2_crd_parent = regions[2].get_logical_region();
  PhysicalRegion B3_pos = regions[3];
  LogicalRegion B3_pos_parent = regions[3].get_logical_region();
  PhysicalRegion B3_crd = regions[4];
  LogicalRegion B3_crd_parent = regions[4].get_logical_region();
  PhysicalRegion B_vals = regions[5];
  LogicalRegion B_vals_parent = regions[5].get_logical_region();
  PhysicalRegion C_vals = regions[6];
  LogicalRegion C_vals_parent = regions[6].get_logical_region();
  PhysicalRegion D_vals = regions[7];
  LogicalRegion D_vals_parent = regions[7].get_logical_region();

  int32_t fposo = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t A1_dimension = args->A1_dimension;
  int32_t A2_dimension = args->A2_dimension;
  int64_t B3Size = args->B3Size;
  int32_t C1_dimension = args->C1_dimension;
  int32_t C2_dimension = args->C2_dimension;
  int32_t D1_dimension = args->D1_dimension;
  int32_t D2_dimension = args->D2_dimension;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble2>(C_vals, FID_VAL);
  auto D_vals_ro_accessor = createAccessor<AccessorROdouble2>(D_vals, FID_VAL);
  auto A_vals_red_accessor = createAccessor<AccessorReducedouble2>(A_vals, FID_VAL, LEGION_REDOP_SUM_FLOAT64);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);
  auto B3_pos_accessor = createAccessor<AccessorRORect_1_1>(B3_pos, FID_RECT_1);
  auto B3_crd_accessor = createAccessor<AccessorROint32_t1>(B3_crd, FID_COORD);

  DomainT<1> BValsDomain = runtime->get_index_space_domain(ctx, get_index_space(B_vals));
  if (BValsDomain.empty())
    return ;

  DomainT<1> B3PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B3_pos));
  int32_t pB3_begin = B3PosDomain.bounds.lo;
  int32_t pB3_end = B3PosDomain.bounds.hi;
  DomainT<1> B2PosDomain = runtime->get_index_space_domain(ctx, get_index_space(B2_pos));
  int32_t pB2_begin = B2PosDomain.bounds.lo;
  int32_t pB2_end = B2PosDomain.bounds.hi;
  int64_t pointID1 = fposo;
  #pragma omp parallel for schedule(static)
  for (int32_t fposio = (0 / pieces) / 2048; fposio < (((B3Size + (pieces - 1)) / pieces + 2047) / 2048); fposio++) {
    int64_t pointID2 = pointID1 * (((B3Size + (pieces - 1)) / pieces + 2047) / 2048) + fposio;
    int32_t fposi = fposio * 2048 + 0 / pieces;
    int32_t fposB = fposo * ((B3Size + (pieces - 1)) / pieces) + fposi;
    int32_t j_pos = taco_binarySearchBefore(B3_pos_accessor, pB3_begin, pB3_end, fposB);
    int32_t j = B2_crd_accessor[j_pos];
    int32_t i_pos = taco_binarySearchBefore(B2_pos_accessor, pB2_begin, pB2_end, j_pos);
    int32_t i = i_pos;
    for (int32_t fposii = 0; fposii < 2048; fposii++) {
      int32_t fposi = (fposio * 2048 + fposii) + 0 / pieces;
      int32_t fposB = fposo * ((B3Size + (pieces - 1)) / pieces) + fposi;
      if (fposB >= (fposo + 1) * ((B3Size + (pieces - 1)) / pieces - 0 / pieces))
        continue;

      if (fposB >= B3Size)
        continue;

      int32_t f2 = B3_crd_accessor[fposB];
      int32_t f1 = f2;
      int32_t k = f1;
      if (!(B3_pos_accessor[j_pos].contains(fposB))) {
        j_pos = j_pos + 1;
        j = B2_crd_accessor[j_pos];
        while (!(B2_pos_accessor[i_pos].contains(j_pos))) {
          i_pos = i_pos + 1;
          i = i_pos;
        }
      }
      int32_t jB = j_pos;
      int32_t iA = 0 * A1_dimension + i;
      int32_t jC = 0 * C1_dimension + j;
      int32_t kD = 0 * D1_dimension + k;
      for (int32_t l = 0; l < C2_dimension; l++) {
        int64_t pointID3 = pointID2 * C2_dimension + l;
        int32_t lA = iA * A2_dimension + l;
        int32_t lC = jC * C2_dimension + l;
        int32_t lD = kD * D2_dimension + l;
        A_vals_red_accessor[Point<2>(i, l)] <<= (B_vals_ro_accessor[Point<1>(fposB)] * C_vals_ro_accessor[Point<2>(j, l)]) * D_vals_ro_accessor[Point<2>(k, l)];
      }
    }
  }
}

void computeLegion(Context ctx, Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, partitionPackForcomputeLegion* partitionPack, int32_t pieces) {
  int A1_dimension = A->dims[0];
  int A2_dimension = A->dims[1];
  auto A_vals_parent = A->valsParent;
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B3_pos_parent = B->indicesParents[2][0];
  auto B3_crd_parent = B->indicesParents[2][1];
  auto B_vals_parent = B->valsParent;
  int C1_dimension = C->dims[0];
  int C2_dimension = C->dims[1];
  RegionWrapper C_vals = C->vals;
  auto C_vals_parent = C->valsParent;
  int D1_dimension = D->dims[0];
  int D2_dimension = D->dims[1];
  RegionWrapper D_vals = D->vals;
  auto D_vals_parent = D->valsParent;

  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t B3Size = runtime->get_index_space_domain(ctx, get_index_space(B3_crd)).hi()[0] + 1;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto fposoIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(fposoIndexSpace));
  task_1Args taskArgsRaw;
  taskArgsRaw.A1_dimension = A1_dimension;
  taskArgsRaw.A2_dimension = A2_dimension;
  taskArgsRaw.B3Size = B3Size;
  taskArgsRaw.C1_dimension = C1_dimension;
  taskArgsRaw.C2_dimension = C2_dimension;
  taskArgsRaw.D1_dimension = D1_dimension;
  taskArgsRaw.D2_dimension = D2_dimension;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->APartition.valsPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, A_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(C_vals), READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(get_logical_region(D_vals), READ_ONLY, EXCLUSIVE, D_vals_parent).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher);

}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
}
