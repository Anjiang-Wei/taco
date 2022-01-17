#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.cuh"
#include "cublas_v2.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;

struct task_1Args {
  int32_t B1_dimension;
  double a_val;
  int32_t pieces;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, LegionTensor* C, int32_t pieces) {
  int B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  RegionWrapper B3_pos = B->indices[2][0];
  RegionWrapper B3_crd = B->indices[2][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B3_pos_parent = B->indicesParents[2][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  RegionWrapper C2_pos = C->indices[1][0];
  RegionWrapper C2_crd = C->indices[1][1];
  RegionWrapper C3_pos = C->indices[2][0];
  RegionWrapper C3_crd = C->indices[2][1];
  auto C2_pos_parent = C->indicesParents[1][0];
  auto C3_pos_parent = C->indicesParents[2][0];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];



  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t io = (*itr)[0];
    Point<1> BStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> BEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)),BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<1> CStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> CEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)),CDomain.hi()[0]));
    Rect<1> CRect = Rect<1>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_ALIASED_COMPLETE_KIND);
  LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition, B2_pos);
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, runtime->create_partition_by_image_range(
    ctx,
    B2_crd.get_index_space(),
    posPartB2,
    B2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, posPartB2.get_index_partition())
  ));
  LogicalPartition posPartB3 = copyPartition(ctx, runtime, crdPartB2, B3_pos);
  LogicalPartition crdPartB3 = runtime->get_logical_partition(ctx, B3_crd, runtime->create_partition_by_image_range(
    ctx,
    B3_crd.get_index_space(),
    posPartB3,
    B3_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, posPartB3.get_index_partition())
  ));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB3, get_logical_region(B_vals));
  auto C_dense_run_0_Partition = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_ALIASED_COMPLETE_KIND);
  LogicalPartition posPartC2 = copyPartition(ctx, runtime, C_dense_run_0_Partition, C2_pos);
  LogicalPartition crdPartC2 = runtime->get_logical_partition(ctx, C2_crd, runtime->create_partition_by_image_range(
    ctx,
    C2_crd.get_index_space(),
    posPartC2,
    C2_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, posPartC2.get_index_partition())
  ));
  LogicalPartition posPartC3 = copyPartition(ctx, runtime, crdPartC2, C3_pos);
  LogicalPartition crdPartC3 = runtime->get_logical_partition(ctx, C3_crd, runtime->create_partition_by_image_range(
    ctx,
    C3_crd.get_index_space(),
    posPartC3,
    C3_pos_parent,
    FID_RECT_1,
    runtime->get_index_partition_color_space_name(ctx, posPartC3.get_index_partition())
  ));
  auto C_vals_partition = copyPartition(ctx, runtime, crdPartC3, get_logical_region(C_vals));
  auto computePartitions = partitionPackForcomputeLegion();
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(3);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.indicesPartitions[2].push_back(posPartB3);
  computePartitions.BPartition.indicesPartitions[2].push_back(crdPartB3);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition;
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(3);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(3);
  computePartitions.CPartition.indicesPartitions[1].push_back(posPartC2);
  computePartitions.CPartition.indicesPartitions[1].push_back(crdPartC2);
  computePartitions.CPartition.indicesPartitions[2].push_back(posPartC3);
  computePartitions.CPartition.indicesPartitions[2].push_back(crdPartC3);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition;

  return computePartitions;
}

__global__
void task_1DeviceKernel0(double* bufPtr, AccessorRORect_1_1 B2_pos_accessor, AccessorRORect_1_1 C2_pos_accessor, AccessorROint32_t1 B2_crd_accessor, AccessorROint32_t1 C2_crd_accessor, AccessorRORect_1_1 B3_pos_accessor, AccessorRORect_1_1 C3_pos_accessor, AccessorROint32_t1 B3_crd_accessor, AccessorROint32_t1 C3_crd_accessor, AccessorROdouble1 B_vals_ro_accessor, AccessorROdouble1 C_vals_ro_accessor, int32_t B1_dimension, double a_val, int32_t pieces, int32_t io) {

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (256));
  if (threadIdx.x >= 256) {
    return;
  }

  int64_t pointID2 = io * (((B1_dimension + (pieces - 1)) / pieces + 255) / 256) + block;
  double tthreada_val = 0.0;
  int32_t ii = block * 256 + thread;
  int32_t i = io * ((B1_dimension + (pieces - 1)) / pieces) + ii;
  if (i >= B1_dimension)
    return;

  if (i >= (io + 1) * ((B1_dimension + (pieces - 1)) / pieces))
    return;

  int32_t jB = B2_pos_accessor[Point<1>(i)].lo;
  int32_t pB2_end = B2_pos_accessor[Point<1>(i)].hi + 1;
  int32_t jC = C2_pos_accessor[Point<1>(i)].lo;
  int32_t pC2_end = C2_pos_accessor[Point<1>(i)].hi + 1;

  while (jB < pB2_end && jC < pC2_end) {
    int32_t jB0 = B2_crd_accessor[jB];
    int32_t jC0 = C2_crd_accessor[jC];
    int32_t j = TACO_MIN(jB0,jC0);
    if (jB0 == j && jC0 == j) {
      int32_t kB = B3_pos_accessor[Point<1>(jB)].lo;
      int32_t pB3_end = B3_pos_accessor[Point<1>(jB)].hi + 1;
      int32_t kC = C3_pos_accessor[Point<1>(jC)].lo;
      int32_t pC3_end = C3_pos_accessor[Point<1>(jC)].hi + 1;

      while (kB < pB3_end && kC < pC3_end) {
        int32_t kB0 = B3_crd_accessor[kB];
        int32_t kC0 = C3_crd_accessor[kC];
        int32_t k = TACO_MIN(kB0,kC0);
        if (kB0 == k && kC0 == k) {
          tthreada_val = tthreada_val + B_vals_ro_accessor[Point<1>(kB)] * C_vals_ro_accessor[Point<1>(kC)];
        }
        kB = kB + (int32_t)(kB0 == k);
        kC = kC + (int32_t)(kC0 == k);
      }
    }
    jB = jB + (int32_t)(jB0 == j);
    jC = jC + (int32_t)(jC0 == j);
  }
  atomicAddWarp(&bufPtr[0], 0, tthreada_val);
}

double task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B2_pos = regions[0];
  LogicalRegion B2_pos_parent = regions[0].get_logical_region();
  PhysicalRegion B2_crd = regions[1];
  LogicalRegion B2_crd_parent = regions[1].get_logical_region();
  PhysicalRegion B3_pos = regions[2];
  LogicalRegion B3_pos_parent = regions[2].get_logical_region();
  PhysicalRegion B3_crd = regions[3];
  LogicalRegion B3_crd_parent = regions[3].get_logical_region();
  PhysicalRegion B_vals = regions[4];
  LogicalRegion B_vals_parent = regions[4].get_logical_region();
  PhysicalRegion C2_pos = regions[5];
  LogicalRegion C2_pos_parent = regions[5].get_logical_region();
  PhysicalRegion C2_crd = regions[6];
  LogicalRegion C2_crd_parent = regions[6].get_logical_region();
  PhysicalRegion C3_pos = regions[7];
  LogicalRegion C3_pos_parent = regions[7].get_logical_region();
  PhysicalRegion C3_crd = regions[8];
  LogicalRegion C3_crd_parent = regions[8].get_logical_region();
  PhysicalRegion C_vals = regions[9];
  LogicalRegion C_vals_parent = regions[9].get_logical_region();

  int32_t io = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t B1_dimension = args->B1_dimension;
  double a_val = args->a_val;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble1>(C_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto C2_pos_accessor = createAccessor<AccessorRORect_1_1>(C2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);
  auto C2_crd_accessor = createAccessor<AccessorROint32_t1>(C2_crd, FID_COORD);
  auto B3_pos_accessor = createAccessor<AccessorRORect_1_1>(B3_pos, FID_RECT_1);
  auto C3_pos_accessor = createAccessor<AccessorRORect_1_1>(C3_pos, FID_RECT_1);
  auto B3_crd_accessor = createAccessor<AccessorROint32_t1>(B3_crd, FID_COORD);
  auto C3_crd_accessor = createAccessor<AccessorROint32_t1>(C3_crd, FID_COORD);

  double init = 0;
  Legion::DeferredBuffer<double, 1> buf = Legion::DeferredBuffer<double, 1>(Legion::Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, 0)), &(init));
  double* bufPtr = buf.ptr(0);

  if ((((B1_dimension + (pieces - 1)) / pieces + 255) / 256) > 0) {
    task_1DeviceKernel0<<<(((B1_dimension + (pieces - 1)) / pieces + 255) / 256), 256>>>(bufPtr, B2_pos_accessor, C2_pos_accessor, B2_crd_accessor, C2_crd_accessor, B3_pos_accessor, C3_pos_accessor, B3_crd_accessor, C3_crd_accessor, B_vals_ro_accessor, C_vals_ro_accessor, B1_dimension, a_val, pieces, io);
  }

  cudaMemcpy(&(a_val), bufPtr, sizeof(a_val), cudaMemcpyHostToDevice);
  return a_val;
}

double computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t pieces) {
  int B1_dimension = B->dims[0];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B3_pos_parent = B->indicesParents[2][0];
  auto B3_crd_parent = B->indicesParents[2][1];
  auto B_vals_parent = B->valsParent;
  auto C2_pos_parent = C->indicesParents[1][0];
  auto C2_crd_parent = C->indicesParents[1][1];
  auto C3_pos_parent = C->indicesParents[2][0];
  auto C3_crd_parent = C->indicesParents[2][1];
  auto C_vals_parent = C->valsParent;

  double a_val = 0.0;


  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  task_1Args taskArgsRaw1;
  taskArgsRaw1.B1_dimension = B1_dimension;
  taskArgsRaw1.a_val = a_val;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[2][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B3_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(C2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(C2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.indicesPartitions[2][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(C3_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.indicesPartitions[2][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(C3_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
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
