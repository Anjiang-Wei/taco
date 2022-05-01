#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;

struct task_1Args {
  Legion::FieldID a_vals_field_id;
  int64_t b1_dimension;
  Legion::FieldID b1_indices_field_id_0_1;
  Legion::FieldID b_vals_field_id;
  Legion::FieldID c1_indices_field_id_0_1;
  Legion::FieldID c_vals_field_id;
  int32_t pieces;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, int32_t pieces) {
  RegionWrapper a_vals = a->vals;
  IndexSpace a_dense_run_0 = a->denseLevelRuns[0];
  size_t b1_dimension = b->dims[0];
  RegionWrapper b1_pos = b->indices[0][0];
  RegionWrapper b1_crd = b->indices[0][1];
  auto b1_pos_parent = b->indicesParents[0][0];
  auto b1_crd_parent = b->indicesParents[0][1];
  RegionWrapper b_vals = b->vals;
  auto b1_indices_field_id_0_0 = b->indicesFieldIDs[0][0];
  auto b1_indices_field_id_0_1 = b->indicesFieldIDs[0][1];
  RegionWrapper c1_pos = c->indices[0][0];
  RegionWrapper c1_crd = c->indices[0][1];
  auto c1_pos_parent = c->indicesParents[0][0];
  auto c1_crd_parent = c->indicesParents[0][1];
  RegionWrapper c_vals = c->vals;
  auto c1_indices_field_id_0_0 = c->indicesFieldIDs[0][0];
  auto c1_indices_field_id_0_1 = c->indicesFieldIDs[0][1];

  auto computePartitions = partitionPackForcomputeLegion();


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_dense_run_0);
  IndexSpace coordDomain = runtime->create_index_space(ctx, Rect<1>(0, b1_dimension));
  auto bDomain = runtime->get_index_space_domain(ctx, coordDomain);
  auto cDomain = runtime->get_index_space_domain(ctx, coordDomain);
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t io = (*itr)[0];
    Point<1> aStart = Point<1>((io * ((b1_dimension + (pieces - 1)) / pieces)));
    Point<1> aEnd = Point<1>(TACO_MIN((io * ((b1_dimension + (pieces - 1)) / pieces) + ((b1_dimension + (pieces - 1)) / pieces - 1)), aDomain.hi()[0]));
    Rect<1> aRect = Rect<1>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<1> bStart = Point<1>((io * ((b1_dimension + (pieces - 1)) / pieces)));
    Point<1> bEnd = Point<1>(TACO_MIN((io * ((b1_dimension + (pieces - 1)) / pieces) + ((b1_dimension + (pieces - 1)) / pieces - 1)), bDomain.hi()[0]));
    Rect<1> bRect = Rect<1>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<1> cStart = Point<1>((io * ((b1_dimension + (pieces - 1)) / pieces)));
    Point<1> cEnd = Point<1>(TACO_MIN((io * ((b1_dimension + (pieces - 1)) / pieces) + ((b1_dimension + (pieces - 1)) / pieces - 1)), cDomain.hi()[0]));
    Rect<1> cRect = Rect<1>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto a_dense_run_0_Partition = runtime->create_index_partition(ctx, a_dense_run_0, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto a_vals_partition = copyPartition(ctx, runtime, a_dense_run_0_Partition, get_logical_region(a_vals));
  Legion::LogicalPartition b1_crd_part = RectCompressedCoordinatePartition::apply(
    ctx,
    runtime,
    b1_crd,
    b1_crd_parent,
    b1_indices_field_id_0_1,
    bColoring,
    ioIndexSpace
  );
  IndexPartition posSparsePartb1 = runtime->create_partition_by_preimage_range(
    ctx,
    b1_crd_part.get_index_partition(),
    b1_pos,
    b1_pos_parent,
    b1_indices_field_id_0_0,
    runtime->get_index_partition_color_space_name(ctx, b1_crd_part.get_index_partition()),
    LEGION_ALIASED_INCOMPLETE_KIND
  );
  IndexPartition posIndexPartb1 = densifyPartition(ctx, runtime, get_index_space(b1_pos), posSparsePartb1);
  Legion::LogicalPartition posPartb1 = runtime->get_logical_partition(ctx, b1_pos, posIndexPartb1);
  auto b_vals_partition = copyPartition(ctx, runtime, b1_crd_part, get_logical_region(b_vals));
  Legion::LogicalPartition c1_crd_part = RectCompressedCoordinatePartition::apply(
    ctx,
    runtime,
    c1_crd,
    c1_crd_parent,
    c1_indices_field_id_0_1,
    cColoring,
    ioIndexSpace
  );
  IndexPartition posSparsePartc1 = runtime->create_partition_by_preimage_range(
    ctx,
    c1_crd_part.get_index_partition(),
    c1_pos,
    c1_pos_parent,
    c1_indices_field_id_0_0,
    runtime->get_index_partition_color_space_name(ctx, c1_crd_part.get_index_partition()),
    LEGION_ALIASED_INCOMPLETE_KIND
  );
  IndexPartition posIndexPartc1 = densifyPartition(ctx, runtime, get_index_space(c1_pos), posSparsePartc1);
  Legion::LogicalPartition posPartc1 = runtime->get_logical_partition(ctx, c1_pos, posIndexPartc1);
  auto c_vals_partition = copyPartition(ctx, runtime, c1_crd_part, get_logical_region(c_vals));
  computePartitions.aPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(1);
  computePartitions.aPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.aPartition.valsPartition = a_vals_partition;
  computePartitions.aPartition.denseLevelRunPartitions[0] = a_dense_run_0_Partition;
  computePartitions.bPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(1);
  computePartitions.bPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.bPartition.indicesPartitions[0].push_back(posPartb1);
  computePartitions.bPartition.indicesPartitions[0].push_back(b1_crd_part);
  computePartitions.bPartition.valsPartition = b_vals_partition;
  computePartitions.cPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(1);
  computePartitions.cPartition.denseLevelRunPartitions = std::vector<IndexPartition>(1);
  computePartitions.cPartition.indicesPartitions[0].push_back(posPartc1);
  computePartitions.cPartition.indicesPartitions[0].push_back(c1_crd_part);
  computePartitions.cPartition.valsPartition = c_vals_partition;

  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a_vals = regions[0];
  LogicalRegion a_vals_parent = regions[0].get_logical_region();
  PhysicalRegion b1_pos = regions[1];
  LogicalRegion b1_pos_parent = regions[1].get_logical_region();
  PhysicalRegion b1_crd = regions[2];
  LogicalRegion b1_crd_parent = regions[2].get_logical_region();
  PhysicalRegion b_vals = regions[3];
  LogicalRegion b_vals_parent = regions[3].get_logical_region();
  PhysicalRegion c1_pos = regions[4];
  LogicalRegion c1_pos_parent = regions[4].get_logical_region();
  PhysicalRegion c1_crd = regions[5];
  LogicalRegion c1_crd_parent = regions[5].get_logical_region();
  PhysicalRegion c_vals = regions[6];
  LogicalRegion c_vals_parent = regions[6].get_logical_region();

  int64_t io = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  Legion::FieldID a_vals_field_id = args->a_vals_field_id;
  int64_t b1_dimension = args->b1_dimension;
  Legion::FieldID b1_indices_field_id_0_1 = args->b1_indices_field_id_0_1;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  Legion::FieldID c1_indices_field_id_0_1 = args->c1_indices_field_id_0_1;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t pieces = args->pieces;

  auto b_vals_ro_accessor = createAccessor<AccessorROdouble1>(b_vals, b_vals_field_id);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble1>(c_vals, c_vals_field_id);
  auto a_vals_rw_accessor = createAccessor<AccessorRWdouble1>(a_vals, a_vals_field_id);
  auto b1_crd_accessor = createAccessor<AccessorROint32_t1>(b1_crd, b1_indices_field_id_0_1);
  auto c1_crd_accessor = createAccessor<AccessorROint32_t1>(c1_crd, c1_indices_field_id_0_1);

  DomainT<1> b1_crd_domain = runtime->get_index_space_domain(ctx, get_index_space(b1_crd));
  DomainT<1> c1_crd_domain = runtime->get_index_space_domain(ctx, get_index_space(c1_crd));
  int64_t pb1_begin = io * ((b1_dimension + (pieces - 1)) / pieces);
  int64_t ib = 0;
  if (b1_crd_domain.empty()) {
    ib = 1 + b1_crd_domain.bounds.hi;
  }
  else {
    ib = taco_binarySearchAfter(b1_crd_accessor, b1_crd_domain.bounds.lo, (1 + b1_crd_domain.bounds.hi), pb1_begin);
  }
  int64_t pb1_end = 1 + b1_crd_domain.bounds.hi;
  int64_t pc1_begin = io * ((b1_dimension + (pieces - 1)) / pieces);
  int64_t ic = 0;
  if (c1_crd_domain.empty()) {
    ic = 1 + c1_crd_domain.bounds.hi;
  }
  else {
    ic = taco_binarySearchAfter(c1_crd_accessor, c1_crd_domain.bounds.lo, (1 + c1_crd_domain.bounds.hi), pc1_begin);
  }
  int64_t pc1_end = 1 + c1_crd_domain.bounds.hi;
  int64_t ib0 = 0;
  if (b1_crd_domain.contains(ib)) {
    ib0 = b1_crd_accessor[ib];
  }
  else {
    ib0 = b1_dimension;
  }
  int64_t ic0 = 0;
  if (c1_crd_domain.contains(ic)) {
    ic0 = c1_crd_accessor[ic];
  }
  else {
    ic0 = b1_dimension;
  }
  int64_t i = TACO_MIN(ib0, ic0);
  int64_t ii = i - ((b1_dimension + (pieces - 1)) / pieces) * io;
  int64_t ii_end = (b1_dimension + (pieces - 1)) / pieces;

  while ((ib < pb1_end && ii < ii_end) && ic < pc1_end) {
    ib0 = b1_crd_accessor[ib];
    ic0 = c1_crd_accessor[ic];
    i = TACO_MIN(ib0, ic0);
    if (ib0 == i && ic0 == i) {
      a_vals_rw_accessor[Point<1>(i)] = b_vals_ro_accessor[Point<1>(ib)] + c_vals_ro_accessor[Point<1>(ic)];
    }
    else if (ib0 == i) {
      a_vals_rw_accessor[Point<1>(i)] = b_vals_ro_accessor[Point<1>(ib)];
    }
    else if (ic0 == i) {
      a_vals_rw_accessor[Point<1>(i)] = c_vals_ro_accessor[Point<1>(ic)];
    }
    ib = ib + (int64_t)(ib0 == i);
    ic = ic + (int64_t)(ic0 == i);
    if (b1_crd_domain.contains(ib)) {
      ib0 = b1_crd_accessor[ib];
    }
    else {
      ib0 = b1_dimension;
    }
    if (c1_crd_domain.contains(ic)) {
      ic0 = c1_crd_accessor[ic];
    }
    else {
      ic0 = b1_dimension;
    }
    i = TACO_MIN(ib0, ic0);
    ii = i - ((b1_dimension + (pieces - 1)) / pieces) * io;
  }
  while (ib < pb1_end && ii < ii_end) {
    ib0 = b1_crd_accessor[ib];
    i = b1_crd_accessor[ib];
    if (ib0 == i) {
      a_vals_rw_accessor[Point<1>(i)] = b_vals_ro_accessor[Point<1>(ib)];
    }
    ib = ib + (int64_t)(ib0 == i);
    if (b1_crd_domain.contains(ib)) {
      ib0 = b1_crd_accessor[ib];
    }
    else {
      ib0 = b1_dimension;
    }
    i = b1_crd_accessor[ib];
    ii = i - ((b1_dimension + (pieces - 1)) / pieces) * io;
  }
  while (ic < pc1_end && ii < ii_end) {
    ic0 = c1_crd_accessor[ic];
    i = c1_crd_accessor[ic];
    if (ic0 == i) {
      a_vals_rw_accessor[Point<1>(i)] = c_vals_ro_accessor[Point<1>(ic)];
    }
    ic = ic + (int64_t)(ic0 == i);
    if (c1_crd_domain.contains(ic)) {
      ic0 = c1_crd_accessor[ic];
    }
    else {
      ic0 = b1_dimension;
    }
    i = c1_crd_accessor[ic];
    ii = i - ((b1_dimension + (pieces - 1)) / pieces) * io;
  }
}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* a, LegionTensor* b, LegionTensor* c, partitionPackForcomputeLegion* partitionPack, int32_t pieces) {
  auto a_vals_parent = a->valsParent;
  auto a_vals_field_id = a->valsFieldID;
  size_t b1_dimension = b->dims[0];
  auto b1_pos_parent = b->indicesParents[0][0];
  auto b1_crd_parent = b->indicesParents[0][1];
  auto b_vals_parent = b->valsParent;
  auto b_vals_field_id = b->valsFieldID;
  auto b1_indices_field_id_0_0 = b->indicesFieldIDs[0][0];
  auto b1_indices_field_id_0_1 = b->indicesFieldIDs[0][1];
  auto c1_pos_parent = c->indicesParents[0][0];
  auto c1_crd_parent = c->indicesParents[0][1];
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;
  auto c1_indices_field_id_0_0 = c->indicesFieldIDs[0][0];
  auto c1_indices_field_id_0_1 = c->indicesFieldIDs[0][1];


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.a_vals_field_id = a_vals_field_id;
  taskArgsRaw1.b1_dimension = b1_dimension;
  taskArgsRaw1.b1_indices_field_id_0_1 = b1_indices_field_id_0_1;
  taskArgsRaw1.b_vals_field_id = b_vals_field_id;
  taskArgsRaw1.c1_indices_field_id_0_1 = c1_indices_field_id_0_1;
  taskArgsRaw1.c_vals_field_id = c_vals_field_id;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->aPartition.valsPartition, 0, READ_WRITE, EXCLUSIVE, a_vals_parent).add_field(a_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->bPartition.indicesPartitions[0][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(b1_pos_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(b1_indices_field_id_0_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->bPartition.indicesPartitions[0][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(b1_crd_parent)).add_field(b1_indices_field_id_0_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->bPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, b_vals_parent).add_field(b_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.indicesPartitions[0][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(c1_pos_parent), Mapping::DefaultMapper::VIRTUAL_MAP).add_field(c1_indices_field_id_0_0));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.indicesPartitions[0][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(c1_crd_parent)).add_field(c1_indices_field_id_0_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(c_vals_field_id));
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
}
