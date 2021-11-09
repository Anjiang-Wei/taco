#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int32_t B1_dimension;
  int32_t a1_dimension;
  int32_t c1_dimension;
};

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

  AccessorROdouble1 c_vals_ro_accessor(c_vals, FID_VAL);
  AccessorROdouble1 B_vals_ro_accessor(B_vals, FID_VAL);
  AccessorRWdouble1 a_vals_rw_accessor(a_vals, FID_VAL);
  AccessorRORect_1_1 B2_pos_accessor(B2_pos, FID_RECT_1);
  AccessorROint32_t1 B2_crd_accessor(B2_crd, FID_COORD);

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

void computeLegion(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c) {
  int a1_dimension = a->dims[0];
  auto a_vals = a->vals;
  auto a_vals_parent = a->valsParent;
  auto a_dense_run_0 = a->denseLevelRuns[0];
  int B1_dimension = B->dims[0];
  auto B2_pos = B->indices[1][0];
  auto B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals = B->vals;
  auto B_vals_parent = B->valsParent;
  auto B_dense_run_0 = B->denseLevelRuns[0];
  int c1_dimension = c->dims[0];
  auto c_vals = c->vals;
  auto c_vals_parent = c->valsParent;
  auto c_dense_run_0 = c->denseLevelRuns[0];

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
  task_1Args taskArgsRaw;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.a1_dimension = a1_dimension;
  taskArgsRaw.c1_dimension = c1_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(a_vals_partition, 0, READ_WRITE, EXCLUSIVE, a_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(posPartB2, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(crdPartB2, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(B_vals_partition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
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
