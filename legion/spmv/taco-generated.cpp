#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_WRITE,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorRWint32_t1;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;
typedef FieldAccessor<READ_WRITE,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRWRect_1_1;

struct partitionPackForcomputeLegion {
  LegionTensorPartition aPartition;
  LegionTensorPartition BPartition;
};

struct task_1Args {
  int32_t B1_dimension;
  int32_t a1_dimension;
  int32_t c1_dimension;
};

void packLegion(Context ctx, Runtime* runtime, LegionTensor* BCSR, LegionTensor* BCOO) {
  int BCSR1_dimension = BCSR->dims[0];
  RegionWrapper BCSR2_pos = BCSR->indices[1][0];
  RegionWrapper BCSR2_crd = BCSR->indices[1][1];
  auto BCSR2_pos_parent = BCSR->indicesParents[1][0];
  auto BCSR2_crd_parent = BCSR->indicesParents[1][1];
  RegionWrapper BCSR_vals = BCSR->vals;
  auto BCSR_vals_parent = BCSR->valsParent;
  auto BCSR_vals_rw_accessor = createAccessor<AccessorRWdouble1>(BCSR_vals, FID_VAL);
  auto BCSR2_pos_accessor = createAccessor<AccessorRWRect_1_1>(BCSR2_pos, FID_RECT_1);
  auto BCSR2_crd_accessor = createAccessor<AccessorRWint32_t1>(BCSR2_crd, FID_COORD);
  RegionWrapper BCOO1_pos = BCOO->indices[0][0];
  RegionWrapper BCOO1_crd = BCOO->indices[0][1];
  RegionWrapper BCOO2_crd = BCOO->indices[1][0];
  auto BCOO1_pos_parent = BCOO->indicesParents[0][0];
  auto BCOO1_crd_parent = BCOO->indicesParents[0][1];
  auto BCOO2_crd_parent = BCOO->indicesParents[1][0];
  RegionWrapper BCOO_vals = BCOO->vals;
  auto BCOO_vals_parent = BCOO->valsParent;
  auto BCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(BCOO_vals, FID_VAL);
  auto BCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(BCOO1_pos, FID_RECT_1);
  auto BCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(BCOO1_crd, FID_COORD);
  auto BCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(BCOO2_crd, FID_COORD);

  BCOO1_pos = legionMalloc(ctx, runtime, BCOO1_pos, BCOO1_pos_parent, FID_RECT_1);
  BCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(BCOO1_pos, FID_RECT_1);
  BCOO1_crd = legionMalloc(ctx, runtime, BCOO1_crd, BCOO1_crd_parent, FID_COORD);
  BCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(BCOO1_crd, FID_COORD);
  BCOO2_crd = legionMalloc(ctx, runtime, BCOO2_crd, BCOO2_crd_parent, FID_COORD);
  BCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(BCOO2_crd, FID_COORD);
  BCOO_vals = legionMalloc(ctx, runtime, BCOO_vals, BCOO_vals_parent, FID_VAL);
  BCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(BCOO_vals, FID_VAL);

  BCSR2_pos = legionMalloc(ctx, runtime, BCSR2_pos_parent, BCSR1_dimension, FID_RECT_1);
  BCSR2_pos_accessor = createAccessor<AccessorRWRect_1_1>(BCSR2_pos, FID_RECT_1);
  BCSR2_pos_accessor[0] = Rect<1>(0, 0);
  for (int32_t pBCSR2 = 1; pBCSR2 < BCSR1_dimension; pBCSR2++) {
    BCSR2_pos_accessor[pBCSR2] = Rect<1>(0, 0);
  }
  int32_t BCSR2_crd_size = 1048576;
  BCSR2_crd = legionMalloc(ctx, runtime, BCSR2_crd_parent, BCSR2_crd_size, FID_COORD);
  BCSR2_crd_accessor = createAccessor<AccessorRWint32_t1>(BCSR2_crd, FID_COORD);
  int32_t jBCSR = 0;
  int32_t BCSR_capacity = 1048576;
  BCSR_vals = legionMalloc(ctx, runtime, BCSR_vals_parent, BCSR_capacity, FID_VAL);
  BCSR_vals_rw_accessor = createAccessor<AccessorRWdouble1>(BCSR_vals, FID_VAL);

  int32_t iBCOO = BCOO1_pos_accessor[0].lo;
  int32_t pBCOO1_end = BCOO1_pos_accessor[0].hi + 1;

  while (iBCOO < pBCOO1_end) {
    int32_t i = BCOO1_crd_accessor[iBCOO];
    int32_t BCOO1_segend = iBCOO + 1;
    while (BCOO1_segend < pBCOO1_end && BCOO1_crd_accessor[BCOO1_segend] == i) {
      BCOO1_segend++;
    }
    int32_t pBCSR2_begin = jBCSR;

    for (int32_t jBCOO = iBCOO; jBCOO < BCOO1_segend; jBCOO++) {
      int32_t j = BCOO2_crd_accessor[jBCOO];
      if (BCSR_capacity <= jBCSR) {
        BCSR_vals = legionRealloc(ctx, runtime, BCSR_vals_parent, BCSR_vals, BCSR_capacity * 2, FID_VAL);
        BCSR_vals_rw_accessor = createAccessor<AccessorRWdouble1>(BCSR_vals, FID_VAL);
        BCSR_capacity *= 2;
      }
      BCSR_vals_rw_accessor[Point<1>(jBCSR)] = BCOO_vals_ro_accessor[Point<1>(jBCOO)];
      if (BCSR2_crd_size <= jBCSR) {
        BCSR2_crd = legionRealloc(ctx, runtime, BCSR2_crd_parent, BCSR2_crd, BCSR2_crd_size * 2, FID_COORD);
        BCSR2_crd_accessor = createAccessor<AccessorRWint32_t1>(BCSR2_crd, FID_COORD);
        BCSR2_crd_size *= 2;
      }
      BCSR2_crd_accessor[jBCSR] = j;
      jBCSR++;
    }

    BCSR2_pos_accessor[i].hi = (jBCSR - pBCSR2_begin) - 1;
    iBCOO = BCOO1_segend;
  }

  int64_t csBCSR2 = 0;
  for (int64_t pBCSR20 = 0; pBCSR20 < BCSR1_dimension; pBCSR20++) {
    int64_t numElemsBCSR2 = BCSR2_pos_accessor[pBCSR20].hi;
    BCSR2_pos_accessor[pBCSR20].lo = csBCSR2 + BCSR2_pos_accessor[pBCSR20].lo;
    BCSR2_pos_accessor[pBCSR20].hi = csBCSR2 + BCSR2_pos_accessor[pBCSR20].hi;
    csBCSR2 += numElemsBCSR2 + 1;
  }
  BCSR->indices[1][1] = getSubRegion(ctx, runtime, BCSR2_crd_parent, Rect<1>(0, (jBCSR - 1)));

  BCSR->vals = getSubRegion(ctx, runtime, BCSR_vals_parent, Rect<1>(0, (jBCSR - 1)));

  runtime->unmap_region(ctx, BCSR2_pos);
  runtime->unmap_region(ctx, BCSR2_crd);
  runtime->unmap_region(ctx, BCSR_vals);
  runtime->unmap_region(ctx, BCOO1_pos);
  runtime->unmap_region(ctx, BCOO1_crd);
  runtime->unmap_region(ctx, BCOO2_crd);
  runtime->unmap_region(ctx, BCOO_vals);
}

partitionPackForcomputeLegion* partitionForcomputeLegion(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c) {
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];
  int B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  IndexSpace c_dense_run_0 = c->denseLevelRuns[0];

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(3);
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t io = (*itr)[0];
    Point<1> BStart = Point<1>((io * ((B1_dimension + 3) / 4)));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + 3) / 4) + ((B1_dimension + 3) / 4 - 1)), BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<1> aStart = Point<1>((io * ((B1_dimension + 3) / 4)));
    Point<1> aEnd = Point<1>(TACO_MIN((io * ((B1_dimension + 3) / 4) + ((B1_dimension + 3) / 4 - 1)), aDomain.hi()[0]));
    Rect<1> aRect = Rect<1>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
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
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, B_vals);
  auto aPartition = runtime->create_index_partition(ctx, a_dense_run_0, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto a_vals_partition = copyPartition(ctx, runtime, aPartition, a_vals);
  auto computePartitions = new(partitionPackForcomputeLegion);
  computePartitions->aPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(1);
  computePartitions->aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions->aPartition.valsPartition = a_vals_partition;
  computePartitions->aPartition.denseLevelRunPartitions[0] = aPartition;
  computePartitions->BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions->BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions->BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions->BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions->BPartition.valsPartition = B_vals_partition;
  computePartitions->BPartition.denseLevelRunPartitions[0] = BPartition;
  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  PhysicalRegion B2_pos = regions[1];
  PhysicalRegion B2_crd = regions[2];
  PhysicalRegion B_vals = regions[3];
  PhysicalRegion c_vals = regions[4];

  int32_t io = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t B1_dimension = args->B1_dimension;
  int32_t a1_dimension = args->a1_dimension;
  int32_t c1_dimension = args->c1_dimension;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, FID_VAL);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble1>(a_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);

  for (int32_t ii = 0; ii < ((B1_dimension + 3) / 4); ii++) {
    int32_t i = io * ((B1_dimension + 3) / 4) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + 3) / 4))
      continue;

    for (int32_t jB = B2_pos_accessor[i].lo; jB < (B2_pos_accessor[i].hi + 1); jB++) {
      int32_t j = B2_crd_accessor[jB];
      a_vals_rw_accessor[Point<1>(i)] = a_vals_rw_accessor[Point<1>(i)] + B_vals_ro_accessor[Point<1>(jB)] * c_vals_ro_accessor[Point<1>(j)];
    }
  }
}

void computeLegion(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegion* partitionPack) {
  int a1_dimension = a->dims[0];
  auto a_vals_parent = a->valsParent;
  int B1_dimension = B->dims[0];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  int c1_dimension = c->dims[0];
  RegionWrapper c_vals = c->vals;
  auto c_vals_parent = c->valsParent;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(3);
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.a1_dimension = a1_dimension;
  taskArgsRaw.c1_dimension = c1_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, a_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(c_vals, READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(FID_VAL));
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
}
