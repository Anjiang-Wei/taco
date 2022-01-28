#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorRWint32_t1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_WRITE,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorRWint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;
typedef FieldAccessor<READ_WRITE,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRWRect_1_1;

struct task_2Args {
  int32_t A1_dimension;
  int32_t B1_dimension;
  int32_t pieces;
};

struct task_1Args {
  int32_t A1_dimension;
  IndexSpace A_dense_run_0;
  int32_t B1_dimension;
  int32_t pieces;
};


partitionPackForcomputeLegion partitionForcomputeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, int32_t pieces) {
  RegionWrapper A2_crd = A->indices[1][1];
  int B1_dimension = B->dims[0];
  RegionWrapper B2_pos = B->indices[1][0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  RegionWrapper B_vals = B->vals;
  IndexSpace B_dense_run_0 = B->denseLevelRuns[0];
  RegionWrapper C2_pos = C->indices[1][0];
  RegionWrapper C2_crd = C->indices[1][1];
  auto C2_pos_parent = C->indicesParents[1][0];
  RegionWrapper C_vals = C->vals;
  IndexSpace C_dense_run_0 = C->denseLevelRuns[0];
  RegionWrapper D2_pos = D->indices[1][0];
  RegionWrapper D2_crd = D->indices[1][1];
  auto D2_pos_parent = D->indicesParents[1][0];
  RegionWrapper D_vals = D->vals;
  IndexSpace D_dense_run_0 = D->denseLevelRuns[0];
  RegionWrapper A2_nnz_vals;

  int64_t A2Size = runtime->get_index_space_domain(ctx, get_index_space(A2_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t C2Size = runtime->get_index_space_domain(ctx, get_index_space(C2_crd)).hi()[0] + 1;
  int64_t D2Size = runtime->get_index_space_domain(ctx, get_index_space(D2_crd)).hi()[0] + 1;

  IndexSpace A2_nnzispace = runtime->create_index_space(ctx, createSimpleDomain(Point<1>((B1_dimension - 1))));
  FieldSpace A2_nnzfspace = createFieldSpaceWithSize(ctx, runtime, FID_VAL, sizeof(int32_t));
  A2_nnz_vals = runtime->create_logical_region(ctx, A2_nnzispace, A2_nnzfspace);
  runtime->fill_field(ctx, A2_nnz_vals, A2_nnz_vals, FID_VAL, (int32_t)0);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto qioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(qioIndexSpace));
  auto A2_nnzDomain = runtime->get_index_space_domain(ctx, A2_nnz_vals.get_index_space());
  auto BDomain = runtime->get_index_space_domain(ctx, B_dense_run_0);
  auto CDomain = runtime->get_index_space_domain(ctx, C_dense_run_0);
  auto DDomain = runtime->get_index_space_domain(ctx, D_dense_run_0);
  DomainPointColoring A2_nnzColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t qio = (*itr)[0];
    Point<1> A2_nnzStart = Point<1>((qio * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> A2_nnzEnd = Point<1>(TACO_MIN((qio * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), A2_nnzDomain.hi()[0]));
    Rect<1> A2_nnzRect = Rect<1>(A2_nnzStart, A2_nnzEnd);
    if (!A2_nnzDomain.contains(A2_nnzRect.lo) || !A2_nnzDomain.contains(A2_nnzRect.hi)) {
      A2_nnzRect = A2_nnzRect.make_empty();
    }
    A2_nnzColoring[(*itr)] = A2_nnzRect;
    Point<1> BStart = Point<1>((qio * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> BEnd = Point<1>(TACO_MIN((qio * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), BDomain.hi()[0]));
    Rect<1> BRect = Rect<1>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<1> CStart = Point<1>((qio * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> CEnd = Point<1>(TACO_MIN((qio * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), CDomain.hi()[0]));
    Rect<1> CRect = Rect<1>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
    Point<1> DStart = Point<1>((qio * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> DEnd = Point<1>(TACO_MIN((qio * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), DDomain.hi()[0]));
    Rect<1> DRect = Rect<1>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  IndexPartition A2_nnz_index_partition = runtime->create_index_partition(ctx, A2_nnz_vals.get_index_space(), domain, A2_nnzColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition A2_nnz_logical_partition = runtime->get_logical_partition(ctx, A2_nnz_vals, A2_nnz_index_partition);
  auto B_dense_run_0_Partition = runtime->create_index_partition(ctx, B_dense_run_0, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition posPartB2 = copyPartition(ctx, runtime, B_dense_run_0_Partition, B2_pos);
  LogicalPartition crdPartB2 = runtime->get_logical_partition(ctx, B2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, B2_crd.get_index_space(), posPartB2, B2_pos_parent, FID_RECT_1));
  auto B_vals_partition = copyPartition(ctx, runtime, crdPartB2, get_logical_region(B_vals));
  auto C_dense_run_0_Partition = runtime->create_index_partition(ctx, C_dense_run_0, domain, CColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition posPartC2 = copyPartition(ctx, runtime, C_dense_run_0_Partition, C2_pos);
  LogicalPartition crdPartC2 = runtime->get_logical_partition(ctx, C2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, C2_crd.get_index_space(), posPartC2, C2_pos_parent, FID_RECT_1));
  auto C_vals_partition = copyPartition(ctx, runtime, crdPartC2, get_logical_region(C_vals));
  auto D_dense_run_0_Partition = runtime->create_index_partition(ctx, D_dense_run_0, domain, DColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition posPartD2 = copyPartition(ctx, runtime, D_dense_run_0_Partition, D2_pos);
  LogicalPartition crdPartD2 = runtime->get_logical_partition(ctx, D2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, D2_crd.get_index_space(), posPartD2, D2_pos_parent, FID_RECT_1));
  auto D_vals_partition = copyPartition(ctx, runtime, crdPartD2, get_logical_region(D_vals));
  auto computePartitions = partitionPackForcomputeLegion();
  computePartitions.BPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.BPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.BPartition.indicesPartitions[1].push_back(posPartB2);
  computePartitions.BPartition.indicesPartitions[1].push_back(crdPartB2);
  computePartitions.BPartition.valsPartition = B_vals_partition;
  computePartitions.BPartition.denseLevelRunPartitions[0] = B_dense_run_0_Partition;
  computePartitions.CPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.CPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.CPartition.indicesPartitions[1].push_back(posPartC2);
  computePartitions.CPartition.indicesPartitions[1].push_back(crdPartC2);
  computePartitions.CPartition.valsPartition = C_vals_partition;
  computePartitions.CPartition.denseLevelRunPartitions[0] = C_dense_run_0_Partition;
  computePartitions.DPartition.indicesPartitions = std::vector<std::vector<LogicalPartition>>(2);
  computePartitions.DPartition.denseLevelRunPartitions = std::vector<IndexPartition>(2);
  computePartitions.DPartition.indicesPartitions[1].push_back(posPartD2);
  computePartitions.DPartition.indicesPartitions[1].push_back(crdPartD2);
  computePartitions.DPartition.valsPartition = D_vals_partition;
  computePartitions.DPartition.denseLevelRunPartitions[0] = D_dense_run_0_Partition;

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
  PhysicalRegion B_vals = regions[5];
  LogicalRegion B_vals_parent = regions[5].get_logical_region();
  PhysicalRegion C2_pos = regions[6];
  LogicalRegion C2_pos_parent = regions[6].get_logical_region();
  PhysicalRegion C2_crd = regions[7];
  LogicalRegion C2_crd_parent = regions[7].get_logical_region();
  PhysicalRegion C_vals = regions[8];
  LogicalRegion C_vals_parent = regions[8].get_logical_region();
  PhysicalRegion D2_pos = regions[9];
  LogicalRegion D2_pos_parent = regions[9].get_logical_region();
  PhysicalRegion D2_crd = regions[10];
  LogicalRegion D2_crd_parent = regions[10].get_logical_region();
  PhysicalRegion D_vals = regions[11];
  LogicalRegion D_vals_parent = regions[11].get_logical_region();

  int32_t io = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t A1_dimension = args->A1_dimension;
  int32_t B1_dimension = args->B1_dimension;
  int32_t pieces = args->pieces;

  auto B_vals_ro_accessor = createAccessor<AccessorROdouble1>(B_vals, FID_VAL);
  auto C_vals_ro_accessor = createAccessor<AccessorROdouble1>(C_vals, FID_VAL);
  auto D_vals_ro_accessor = createAccessor<AccessorROdouble1>(D_vals, FID_VAL);
  auto A_vals_rw_accessor = createAccessor<AccessorRWdouble1>(A_vals, FID_VAL);
  auto A2_pos_accessor = createAccessor<AccessorRWRect_1_1>(A2_pos, FID_RECT_1);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto C2_pos_accessor = createAccessor<AccessorRORect_1_1>(C2_pos, FID_RECT_1);
  auto D2_pos_accessor = createAccessor<AccessorRORect_1_1>(D2_pos, FID_RECT_1);
  auto A2_crd_accessor = createAccessor<AccessorRWint32_t1>(A2_crd, FID_COORD);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);
  auto C2_crd_accessor = createAccessor<AccessorROint32_t1>(C2_crd, FID_COORD);
  auto D2_crd_accessor = createAccessor<AccessorROint32_t1>(D2_crd, FID_COORD);

  int64_t pointID1 = io;
  #pragma omp parallel for schedule(dynamic, 1024)
  for (int32_t ii = 0; ii < ((B1_dimension + (pieces - 1)) / pieces); ii++) {
    int32_t i = io * ((B1_dimension + (pieces - 1)) / pieces) + ii;
    if (i >= B1_dimension)
      continue;

    if (i >= (io + 1) * ((B1_dimension + (pieces - 1)) / pieces))
      continue;

    int64_t pointID2 = pointID1 * ((B1_dimension + (pieces - 1)) / pieces) + ii;
    int32_t iB = i;
    int32_t iC = i;
    int32_t iD = i;
    int32_t iA = i;
    int32_t jB = B2_pos_accessor[Point<1>(i)].lo;
    int32_t pB2_end = B2_pos_accessor[Point<1>(i)].hi + 1;
    int32_t jC = C2_pos_accessor[Point<1>(i)].lo;
    int32_t pC2_end = C2_pos_accessor[Point<1>(i)].hi + 1;
    int32_t jD = D2_pos_accessor[Point<1>(i)].lo;
    int32_t pD2_end = D2_pos_accessor[Point<1>(i)].hi + 1;

    while ((jB < pB2_end && jC < pC2_end) && jD < pD2_end) {
      int32_t jB0 = B2_crd_accessor[(jB * 1)];
      int32_t jC0 = C2_crd_accessor[(jC * 1)];
      int32_t jD0 = D2_crd_accessor[(jD * 1)];
      int32_t j = TACO_MIN(jB0, TACO_MIN(jC0, jD0));
      if ((jB0 == j && jC0 == j) && jD0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA2 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA2;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = (B_vals_ro_accessor[Point<1>(jB)] + C_vals_ro_accessor[Point<1>(jC)]) + D_vals_ro_accessor[Point<1>(jD)];
      }
      else if (jB0 == j && jD0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA20 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA20;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = B_vals_ro_accessor[Point<1>(jB)] + D_vals_ro_accessor[Point<1>(jD)];
      }
      else if (jC0 == j && jD0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA21 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA21;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = C_vals_ro_accessor[Point<1>(jC)] + D_vals_ro_accessor[Point<1>(jD)];
      }
      else if (jB0 == j && jC0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA22 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA22;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = B_vals_ro_accessor[Point<1>(jB)] + C_vals_ro_accessor[Point<1>(jC)];
      }
      else if (jB0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA23 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA23;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = B_vals_ro_accessor[Point<1>(jB)];
      }
      else if (jC0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA24 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA24;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = C_vals_ro_accessor[Point<1>(jC)];
      }
      else {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA25 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA25;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = D_vals_ro_accessor[Point<1>(jD)];
      }
      jB = jB + (int32_t)(jB0 == j);
      jC = jC + (int32_t)(jC0 == j);
      jD = jD + (int32_t)(jD0 == j);
    }
    while (jB < pB2_end && jD < pD2_end) {
      int32_t jB0 = B2_crd_accessor[(jB * 1)];
      int32_t jD0 = D2_crd_accessor[(jD * 1)];
      int32_t j = TACO_MIN(jB0, jD0);
      if (jB0 == j && jD0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA26 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA26;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = B_vals_ro_accessor[Point<1>(jB)] + D_vals_ro_accessor[Point<1>(jD)];
      }
      else if (jB0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA27 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA27;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = B_vals_ro_accessor[Point<1>(jB)];
      }
      else {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA28 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA28;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = D_vals_ro_accessor[Point<1>(jD)];
      }
      jB = jB + (int32_t)(jB0 == j);
      jD = jD + (int32_t)(jD0 == j);
    }
    while (jC < pC2_end && jD < pD2_end) {
      int32_t jC0 = C2_crd_accessor[(jC * 1)];
      int32_t jD0 = D2_crd_accessor[(jD * 1)];
      int32_t j = TACO_MIN(jC0, jD0);
      if (jC0 == j && jD0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA29 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA29;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = C_vals_ro_accessor[Point<1>(jC)] + D_vals_ro_accessor[Point<1>(jD)];
      }
      else if (jC0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA210 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA210;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = C_vals_ro_accessor[Point<1>(jC)];
      }
      else {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA211 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA211;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = D_vals_ro_accessor[Point<1>(jD)];
      }
      jC = jC + (int32_t)(jC0 == j);
      jD = jD + (int32_t)(jD0 == j);
    }
    while (jB < pB2_end && jC < pC2_end) {
      int32_t jB0 = B2_crd_accessor[(jB * 1)];
      int32_t jC0 = C2_crd_accessor[(jC * 1)];
      int32_t j = TACO_MIN(jB0, jC0);
      if (jB0 == j && jC0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA212 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA212;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = B_vals_ro_accessor[Point<1>(jB)] + C_vals_ro_accessor[Point<1>(jC)];
      }
      else if (jB0 == j) {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA213 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA213;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = B_vals_ro_accessor[Point<1>(jB)];
      }
      else {
        int32_t iA = 0 * A1_dimension + i;
        int32_t pA214 = A2_pos_accessor[Point<1>(i)].lo;
        A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
        int32_t jA = pA214;
        A2_crd_accessor[jA * 1] = j;
        A_vals_rw_accessor[Point<1>(jA)] = C_vals_ro_accessor[Point<1>(jC)];
      }
      jB = jB + (int32_t)(jB0 == j);
      jC = jC + (int32_t)(jC0 == j);
    }
    while (jB < pB2_end) {
      int32_t j = B2_crd_accessor[(jB * 1)];
      int32_t iA = 0 * A1_dimension + i;
      int32_t pA215 = A2_pos_accessor[Point<1>(i)].lo;
      A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
      int32_t jA = pA215;
      A2_crd_accessor[jA * 1] = j;
      A_vals_rw_accessor[Point<1>(jA)] = B_vals_ro_accessor[Point<1>(jB)];
      jB = jB + 1;
    }
    while (jC < pC2_end) {
      int32_t j = C2_crd_accessor[(jC * 1)];
      int32_t iA = 0 * A1_dimension + i;
      int32_t pA216 = A2_pos_accessor[Point<1>(i)].lo;
      A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
      int32_t jA = pA216;
      A2_crd_accessor[jA * 1] = j;
      A_vals_rw_accessor[Point<1>(jA)] = C_vals_ro_accessor[Point<1>(jC)];
      jC = jC + 1;
    }
    while (jD < pD2_end) {
      int32_t j = D2_crd_accessor[(jD * 1)];
      int32_t iA = 0 * A1_dimension + i;
      int32_t pA217 = A2_pos_accessor[Point<1>(i)].lo;
      A2_pos_accessor[Point<1>(i)].lo = A2_pos_accessor[Point<1>(i)].lo + 1;
      int32_t jA = pA217;
      A2_crd_accessor[jA * 1] = j;
      A_vals_rw_accessor[Point<1>(jA)] = D_vals_ro_accessor[Point<1>(jD)];
      jD = jD + 1;
    }
  }
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B2_pos = regions[0];
  LogicalRegion B2_pos_parent = regions[0].get_logical_region();
  PhysicalRegion B2_crd = regions[1];
  LogicalRegion B2_crd_parent = regions[1].get_logical_region();
  PhysicalRegion B_vals = regions[2];
  LogicalRegion B_vals_parent = regions[2].get_logical_region();
  PhysicalRegion C2_pos = regions[3];
  LogicalRegion C2_pos_parent = regions[3].get_logical_region();
  PhysicalRegion C2_crd = regions[4];
  LogicalRegion C2_crd_parent = regions[4].get_logical_region();
  PhysicalRegion C_vals = regions[5];
  LogicalRegion C_vals_parent = regions[5].get_logical_region();
  PhysicalRegion D2_pos = regions[6];
  LogicalRegion D2_pos_parent = regions[6].get_logical_region();
  PhysicalRegion D2_crd = regions[7];
  LogicalRegion D2_crd_parent = regions[7].get_logical_region();
  PhysicalRegion D_vals = regions[8];
  LogicalRegion D_vals_parent = regions[8].get_logical_region();
  PhysicalRegion A2_nnz_vals = regions[9];
  LogicalRegion A2_nnz_vals_parent = regions[9].get_logical_region();

  int32_t qio = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t A1_dimension = args->A1_dimension;
  IndexSpace A_dense_run_0 = args->A_dense_run_0;
  int32_t B1_dimension = args->B1_dimension;
  int32_t pieces = args->pieces;

  auto A2_nnz_vals_rw_accessor = createAccessor<AccessorRWint32_t1>(A2_nnz_vals, FID_VAL);
  auto B2_pos_accessor = createAccessor<AccessorRORect_1_1>(B2_pos, FID_RECT_1);
  auto C2_pos_accessor = createAccessor<AccessorRORect_1_1>(C2_pos, FID_RECT_1);
  auto D2_pos_accessor = createAccessor<AccessorRORect_1_1>(D2_pos, FID_RECT_1);
  auto B2_crd_accessor = createAccessor<AccessorROint32_t1>(B2_crd, FID_COORD);
  auto C2_crd_accessor = createAccessor<AccessorROint32_t1>(C2_crd, FID_COORD);
  auto D2_crd_accessor = createAccessor<AccessorROint32_t1>(D2_crd, FID_COORD);

  int64_t pointID1 = qio;
  #pragma omp parallel for schedule(dynamic, 1024)
  for (int32_t qii = 0; qii < ((B1_dimension + (pieces - 1)) / pieces); qii++) {
    int32_t qi = qio * ((B1_dimension + (pieces - 1)) / pieces) + qii;
    if (qi >= B1_dimension)
      continue;

    if (qi >= (qio + 1) * ((B1_dimension + (pieces - 1)) / pieces))
      continue;

    int64_t pointID2 = pointID1 * ((B1_dimension + (pieces - 1)) / pieces) + qii;
    int32_t qiB = qi;
    int32_t qiC = qi;
    int32_t qiD = qi;
    int32_t qjB = B2_pos_accessor[Point<1>(qi)].lo;
    int32_t pB2_end = B2_pos_accessor[Point<1>(qi)].hi + 1;
    int32_t qjC = C2_pos_accessor[Point<1>(qi)].lo;
    int32_t pC2_end = C2_pos_accessor[Point<1>(qi)].hi + 1;
    int32_t qjD = D2_pos_accessor[Point<1>(qi)].lo;
    int32_t pD2_end = D2_pos_accessor[Point<1>(qi)].hi + 1;

    while ((qjB < pB2_end && qjC < pC2_end) && qjD < pD2_end) {
      int32_t qjB0 = B2_crd_accessor[(qjB * 1)];
      int32_t qjC0 = C2_crd_accessor[(qjC * 1)];
      int32_t qjD0 = D2_crd_accessor[(qjD * 1)];
      int32_t qj = TACO_MIN(qjB0, TACO_MIN(qjC0, qjD0));
      if ((qjB0 == qj && qjC0 == qj) && qjD0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else if (qjB0 == qj && qjD0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else if (qjC0 == qj && qjD0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else if (qjB0 == qj && qjC0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else if (qjB0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else if (qjC0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      qjB = qjB + (int32_t)(qjB0 == qj);
      qjC = qjC + (int32_t)(qjC0 == qj);
      qjD = qjD + (int32_t)(qjD0 == qj);
    }
    while (qjB < pB2_end && qjD < pD2_end) {
      int32_t qjB0 = B2_crd_accessor[(qjB * 1)];
      int32_t qjD0 = D2_crd_accessor[(qjD * 1)];
      int32_t qj = TACO_MIN(qjB0, qjD0);
      if (qjB0 == qj && qjD0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else if (qjB0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      qjB = qjB + (int32_t)(qjB0 == qj);
      qjD = qjD + (int32_t)(qjD0 == qj);
    }
    while (qjC < pC2_end && qjD < pD2_end) {
      int32_t qjC0 = C2_crd_accessor[(qjC * 1)];
      int32_t qjD0 = D2_crd_accessor[(qjD * 1)];
      int32_t qj = TACO_MIN(qjC0, qjD0);
      if (qjC0 == qj && qjD0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else if (qjC0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      qjC = qjC + (int32_t)(qjC0 == qj);
      qjD = qjD + (int32_t)(qjD0 == qj);
    }
    while (qjB < pB2_end && qjC < pC2_end) {
      int32_t qjB0 = B2_crd_accessor[(qjB * 1)];
      int32_t qjC0 = C2_crd_accessor[(qjC * 1)];
      int32_t qj = TACO_MIN(qjB0, qjC0);
      if (qjB0 == qj && qjC0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else if (qjB0 == qj) {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      else {
        A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      }
      qjB = qjB + (int32_t)(qjB0 == qj);
      qjC = qjC + (int32_t)(qjC0 == qj);
    }
    while (qjB < pB2_end) {
      int32_t qj = B2_crd_accessor[(qjB * 1)];
      A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      qjB = qjB + 1;
    }
    while (qjC < pC2_end) {
      int32_t qj = C2_crd_accessor[(qjC * 1)];
      A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      qjC = qjC + 1;
    }
    while (qjD < pD2_end) {
      int32_t qj = D2_crd_accessor[(qjD * 1)];
      A2_nnz_vals_rw_accessor[Point<1>(qi)] = A2_nnz_vals_rw_accessor[Point<1>(qi)] + (int32_t)1;
      qjD = qjD + 1;
    }
  }
}

void computeLegion(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, LegionTensor* D, partitionPackForcomputeLegion* partitionPack, int32_t pieces) {
  int A1_dimension = A->dims[0];
  RegionWrapper A2_pos = A->indices[1][0];
  RegionWrapper A2_crd = A->indices[1][1];
  auto A2_pos_parent = A->indicesParents[1][0];
  auto A2_crd_parent = A->indicesParents[1][1];
  RegionWrapper A_vals = A->vals;
  auto A_vals_parent = A->valsParent;
  IndexSpace A_dense_run_0 = A->denseLevelRuns[0];
  int B1_dimension = B->dims[0];
  RegionWrapper B2_crd = B->indices[1][1];
  auto B2_pos_parent = B->indicesParents[1][0];
  auto B2_crd_parent = B->indicesParents[1][1];
  auto B_vals_parent = B->valsParent;
  RegionWrapper C2_crd = C->indices[1][1];
  auto C2_pos_parent = C->indicesParents[1][0];
  auto C2_crd_parent = C->indicesParents[1][1];
  auto C_vals_parent = C->valsParent;
  RegionWrapper D2_crd = D->indices[1][1];
  auto D2_pos_parent = D->indicesParents[1][0];
  auto D2_crd_parent = D->indicesParents[1][1];
  auto D_vals_parent = D->valsParent;
  RegionWrapper A2_nnz_vals;

  int64_t A2Size = runtime->get_index_space_domain(ctx, get_index_space(A2_crd)).hi()[0] + 1;
  int64_t B2Size = runtime->get_index_space_domain(ctx, get_index_space(B2_crd)).hi()[0] + 1;
  int64_t C2Size = runtime->get_index_space_domain(ctx, get_index_space(C2_crd)).hi()[0] + 1;
  int64_t D2Size = runtime->get_index_space_domain(ctx, get_index_space(D2_crd)).hi()[0] + 1;

  IndexSpace A2_nnzispace = runtime->create_index_space(ctx, createSimpleDomain(Point<1>((B1_dimension - 1))));
  FieldSpace A2_nnzfspace = createFieldSpaceWithSize(ctx, runtime, FID_VAL, sizeof(int32_t));
  A2_nnz_vals = runtime->create_logical_region(ctx, A2_nnzispace, A2_nnzfspace);
  runtime->fill_field(ctx, A2_nnz_vals, A2_nnz_vals, FID_VAL, (int32_t)0);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto qioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(qioIndexSpace));
  auto A2_nnzDomain = runtime->get_index_space_domain(ctx, A2_nnz_vals.get_index_space());
  DomainPointColoring A2_nnzColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t qio = (*itr)[0];
    Point<1> A2_nnzStart = Point<1>((qio * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> A2_nnzEnd = Point<1>(TACO_MIN((qio * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), A2_nnzDomain.hi()[0]));
    Rect<1> A2_nnzRect = Rect<1>(A2_nnzStart, A2_nnzEnd);
    if (!A2_nnzDomain.contains(A2_nnzRect.lo) || !A2_nnzDomain.contains(A2_nnzRect.hi)) {
      A2_nnzRect = A2_nnzRect.make_empty();
    }
    A2_nnzColoring[(*itr)] = A2_nnzRect;
  }
  IndexPartition A2_nnz_index_partition = runtime->create_index_partition(ctx, A2_nnz_vals.get_index_space(), domain, A2_nnzColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition A2_nnz_logical_partition = runtime->get_logical_partition(ctx, A2_nnz_vals, A2_nnz_index_partition);
  task_1Args taskArgsRaw1;
  taskArgsRaw1.A1_dimension = A1_dimension;
  taskArgsRaw1.A_dense_run_0 = A_dense_run_0;
  taskArgsRaw1.B1_dimension = B1_dimension;
  taskArgsRaw1.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw1, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(C2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(C2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(partitionPack->DPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(D2_pos_parent)).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(partitionPack->DPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(D2_crd_parent)).add_field(FID_COORD));
  launcher.add_region_requirement(RegionRequirement(partitionPack->DPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, D_vals_parent, Mapping::DefaultMapper::VIRTUAL_MAP).add_field(FID_VAL));
  launcher.add_region_requirement(RegionRequirement(A2_nnz_logical_partition, 0, READ_WRITE, EXCLUSIVE, A2_nnz_vals).add_field(FID_VAL));
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);


  auto A2_seq_insert_edges_result = RectCompressedGetSeqInsertEdges::compute(
    ctx,
    runtime,
    qioIndexSpace,
    A2_pos,
    FID_RECT_1,
    A2_nnz_vals,
    FID_VAL
  );
  A2_crd = getSubRegion(ctx, runtime, A2_crd_parent, Rect<1>(0, (A2_seq_insert_edges_result.scanResult - 1)));
  A->indices[1][1] = A2_crd;
  A_vals = getSubRegion(ctx, runtime, A_vals_parent, Rect<1>(0, (A2_seq_insert_edges_result.scanResult - 1)));
  A->vals = A_vals;

  Point<1> lowerBound0 = Point<1>(0);
  Point<1> upperBound0 = Point<1>((pieces - 1));
  auto ioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound0, upperBound0));
  DomainT<1> domain0 = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(ioIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_dense_run_0);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr0 = PointInDomainIterator<1>(domain0); itr0.valid(); itr0++) {
    int32_t io = (*itr0)[0];
    Point<1> AStart = Point<1>((io * ((B1_dimension + (pieces - 1)) / pieces)));
    Point<1> AEnd = Point<1>(TACO_MIN((io * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)), ADomain.hi()[0]));
    Rect<1> ARect = Rect<1>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr0)] = ARect;
  }
  auto A_dense_run_0_Partition = runtime->create_index_partition(ctx, A_dense_run_0, domain0, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition posPartA2 = copyPartition(ctx, runtime, A_dense_run_0_Partition, A2_pos);
  LogicalPartition crdPartA2 = runtime->get_logical_partition(ctx, A2_crd, RectCompressedPosPartitionDownwards::apply(ctx, runtime, A2_crd.get_index_space(), posPartA2, A2_pos_parent, FID_RECT_1));
  auto A_vals_partition = copyPartition(ctx, runtime, crdPartA2, get_logical_region(A_vals));
  task_2Args taskArgsRaw2;
  taskArgsRaw2.A1_dimension = A1_dimension;
  taskArgsRaw2.B1_dimension = B1_dimension;
  taskArgsRaw2.pieces = pieces;
  TaskArgument taskArgs0 = TaskArgument(&taskArgsRaw2, sizeof(task_2Args));
  IndexLauncher launcher0 = IndexLauncher(taskID(2), domain0, taskArgs0, ArgumentMap());
  launcher0.add_region_requirement(RegionRequirement(posPartA2, 0, READ_WRITE, EXCLUSIVE, get_logical_region(A2_pos_parent)).add_field(FID_RECT_1));
  launcher0.add_region_requirement(RegionRequirement(crdPartA2, 0, READ_WRITE, EXCLUSIVE, get_logical_region(A2_crd_parent)).add_field(FID_COORD));
  launcher0.add_region_requirement(RegionRequirement(A_vals_partition, 0, READ_WRITE, EXCLUSIVE, A_vals_parent).add_field(FID_VAL));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_pos_parent)).add_field(FID_RECT_1));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->BPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(B2_crd_parent)).add_field(FID_COORD));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->BPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, B_vals_parent).add_field(FID_VAL));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->CPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(C2_pos_parent)).add_field(FID_RECT_1));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->CPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(C2_crd_parent)).add_field(FID_COORD));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->CPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, C_vals_parent).add_field(FID_VAL));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->DPartition.indicesPartitions[1][0], 0, READ_ONLY, EXCLUSIVE, get_logical_region(D2_pos_parent)).add_field(FID_RECT_1));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->DPartition.indicesPartitions[1][1], 0, READ_ONLY, EXCLUSIVE, get_logical_region(D2_crd_parent)).add_field(FID_COORD));
  launcher0.add_region_requirement(RegionRequirement(partitionPack->DPartition.valsPartition, 0, READ_ONLY, EXCLUSIVE, D_vals_parent).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher0);


  RectCompressedFinalizeYieldPositions A2_finalize_yield_pos_launcher = RectCompressedFinalizeYieldPositions(ctx, runtime, A2_pos, A2_seq_insert_edges_result.partition, FID_RECT_1);
  runtime->execute_index_space(ctx, A2_finalize_yield_pos_launcher);

  runtime->destroy_field_space(ctx, A2_nnzfspace);
  runtime->destroy_index_space(ctx, A2_nnzispace);
  runtime->destroy_logical_region(ctx, A2_nnz_vals);
}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
}
