#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorROdouble3;

struct task_1Args {
  double a_val;
  int64_t b1_dimension;
  int64_t b2_dimension;
  int64_t b3_dimension;
  Legion::FieldID b_vals_field_id;
  Legion::FieldID c_vals_field_id;
  int32_t pieces;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, LegionTensor* c, int32_t pieces) {
  size_t b1_dimension = b->dims[0];
  size_t b2_dimension = b->dims[1];
  size_t b3_dimension = b->dims[2];
  RegionWrapper b_vals = b->vals;
  IndexSpace b_dense_run_0 = b->denseLevelRuns[0];
  RegionWrapper c_vals = c->vals;
  IndexSpace c_dense_run_0 = c->denseLevelRuns[0];

  auto computePartitions = partitionPackForcomputeLegion();

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_dense_run_0);
  auto cDomain = runtime->get_index_space_domain(ctx, c_dense_run_0);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int64_t in = (*itr)[0];
    Point<3> bStart = Point<3>((in * ((b1_dimension + (pieces - 1)) / pieces)), 0, 0);
    Point<3> bEnd = Point<3>(TACO_MIN((in * ((b1_dimension + (pieces - 1)) / pieces) + ((b1_dimension + (pieces - 1)) / pieces - 1)), bDomain.hi()[0]), TACO_MIN(b2_dimension, bDomain.hi()[1]), TACO_MIN(b3_dimension, bDomain.hi()[2]));
    Rect<3> bRect = Rect<3>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<3> cStart = Point<3>((in * ((b1_dimension + (pieces - 1)) / pieces)), 0, 0);
    Point<3> cEnd = Point<3>(TACO_MIN((in * ((b1_dimension + (pieces - 1)) / pieces) + ((b1_dimension + (pieces - 1)) / pieces - 1)), cDomain.hi()[0]), TACO_MIN(b2_dimension, cDomain.hi()[1]), TACO_MIN(b3_dimension, cDomain.hi()[2]));
    Rect<3> cRect = Rect<3>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto b_dense_run_0_Partition = runtime->create_index_partition(ctx, b_dense_run_0, domain, bColoring, LEGION_COMPUTE_KIND);
  auto b_vals_partition = copyPartition(ctx, runtime, b_dense_run_0_Partition, get_logical_region(b_vals));
  auto c_dense_run_0_Partition = runtime->create_index_partition(ctx, c_dense_run_0, domain, cColoring, LEGION_COMPUTE_KIND);
  auto c_vals_partition = copyPartition(ctx, runtime, c_dense_run_0_Partition, get_logical_region(c_vals));
  computePartitions.bPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(3);
  computePartitions.bPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.bPartition.valsPartition = b_vals_partition;
  computePartitions.bPartition.denseLevelRunPartitions[0] = b_dense_run_0_Partition;
  computePartitions.cPartition.indicesPartitions = std::vector<std::vector<Legion::LogicalPartition>>(3);
  computePartitions.cPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.cPartition.valsPartition = c_vals_partition;
  computePartitions.cPartition.denseLevelRunPartitions[0] = c_dense_run_0_Partition;

  return computePartitions;
}

double task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b_vals = regions[0];
  LogicalRegion b_vals_parent = regions[0].get_logical_region();
  PhysicalRegion c_vals = regions[1];
  LogicalRegion c_vals_parent = regions[1].get_logical_region();

  int64_t in = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  double a_val = args->a_val;
  int64_t b1_dimension = args->b1_dimension;
  int64_t b2_dimension = args->b2_dimension;
  int64_t b3_dimension = args->b3_dimension;
  Legion::FieldID b_vals_field_id = args->b_vals_field_id;
  Legion::FieldID c_vals_field_id = args->c_vals_field_id;
  int32_t pieces = args->pieces;

  auto b_vals_ro_accessor = createAccessor<AccessorROdouble3>(b_vals, b_vals_field_id);
  auto c_vals_ro_accessor = createAccessor<AccessorROdouble3>(c_vals, c_vals_field_id);

  #pragma omp parallel for schedule(static)
  for (int64_t io = 0; io < ((((b1_dimension + (pieces - 1)) / pieces) * b2_dimension + 3) / 4); io++) {
    double tiia_val = 0.0;
    #pragma clang loop interleave(enable) vectorize(enable)
    for (int64_t ii = 0; ii < 4; ii++) {
      int64_t f = io * 4 + ii;
      int64_t il = f / b2_dimension;
      int64_t i = in * ((b1_dimension + (pieces - 1)) / pieces) + il;
      if (i >= b1_dimension)
        continue;

      if (i >= (in + 1) * ((b1_dimension + (pieces - 1)) / pieces))
        continue;

      int64_t j = f % b2_dimension;
      if (j >= b2_dimension)
        continue;

      for (int64_t k = 0; k < b3_dimension; k++) {
        tiia_val = tiia_val + b_vals_ro_accessor[Point<3>(i, j, k)] * c_vals_ro_accessor[Point<3>(i, j, k)];
      }
    }
    #pragma omp atomic
    a_val = a_val + tiia_val;
  }
  return a_val;
}

double computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* b, LegionTensor* c, partitionPackForcomputeLegion* partitionPack, int32_t pieces) {
  size_t b1_dimension = b->dims[0];
  size_t b2_dimension = b->dims[1];
  size_t b3_dimension = b->dims[2];
  auto b_vals_parent = b->valsParent;
  auto b_vals_field_id = b->valsFieldID;
  auto c_vals_parent = c->valsParent;
  auto c_vals_field_id = c->valsFieldID;

  double a_val = 0.0;

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.a_val = a_val;
  taskArgsRaw1.b1_dimension = b1_dimension;
  taskArgsRaw1.b2_dimension = b2_dimension;
  taskArgsRaw1.b3_dimension = b3_dimension;
  taskArgsRaw1.b_vals_field_id = b_vals_field_id;
  taskArgsRaw1.c_vals_field_id = c_vals_field_id;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->bPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, b_vals_parent).add_field(b_vals_field_id));
  launcher.add_region_requirement(RegionRequirement(partitionPack->cPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, c_vals_parent).add_field(c_vals_field_id));
  a_val = runtime->execute_index_space(ctx, launcher, LEGION_REDOP_SUM_FLOAT64).get<double>();


  return a_val;
}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<double,task_1>(registrar, "task_1");
  }
}
