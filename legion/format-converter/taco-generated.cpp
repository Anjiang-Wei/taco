#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;

#include "taco-generated.h"
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,int64_t,1,coord_t,Realm::AffineAccessor<int64_t,1,coord_t>> AccessorRWint64_t1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef FieldAccessor<READ_WRITE,int64_t,2,coord_t,Realm::AffineAccessor<int64_t,2,coord_t>> AccessorRWint64_t2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_WRITE,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorRWint32_t1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;
typedef FieldAccessor<READ_WRITE,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRWRect_1_1;
typedef FieldAccessor<READ_WRITE,Rect<1>,2,coord_t,Realm::AffineAccessor<Rect<1>,2,coord_t>> AccessorRWRect_1_2;


void packLegionCOOToCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  RegionWrapper T2_pos = T->indices[1][0];
  RegionWrapper T2_crd = T->indices[1][1];
  auto T2_pos_parent = T->indicesParents[1][0];
  auto T2_crd_parent = T->indicesParents[1][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
  auto T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  auto T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int TCOO1_dimension = TCOO->dims[0];
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  RegionWrapper TCOO2_crd = TCOO->indices[1][0];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  auto TCOO2_crd_parent = TCOO->indicesParents[1][0];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  auto TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  RegionWrapper T2_nnz_vals;
  auto T2_nnz_vals_rw_accessor = createAccessor<AccessorRWint64_t1>(T2_nnz_vals, FID_VAL);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD, READ_ONLY);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);



  IndexSpace T2_nnzispace = runtime->create_index_space(ctx, createSimpleDomain(Point<1>((TCOO1_dimension - 1))));
  FieldSpace T2_nnzfspace = createFieldSpaceWithSize(ctx, runtime, FID_VAL, sizeof(int64_t));
  T2_nnz_vals = runtime->create_logical_region(ctx, T2_nnzispace, T2_nnzfspace);
  runtime->fill_field(ctx, T2_nnz_vals, T2_nnz_vals, FID_VAL, (int64_t)0);
  T2_nnz_vals = legionMalloc(ctx, runtime, T2_nnz_vals, T2_nnz_vals, FID_VAL, READ_WRITE);
  T2_nnz_vals_rw_accessor = createAccessor<AccessorRWint64_t1>(T2_nnz_vals, FID_VAL);

  int64_t qiTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (qiTCOO < pTCOO1_end) {
    int64_t qi = TCOO1_crd_accessor[qiTCOO];
    int64_t TCOO1_segend = qiTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == qi) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    for (int64_t qjTCOO = qiTCOO; qjTCOO < ((TCOO1_segend - 1) + 1); qjTCOO++) {
      T2_nnz_vals_rw_accessor[Point<1>(qi)] = T2_nnz_vals_rw_accessor[Point<1>(qi)] + (int64_t)1;
    }
    qiTCOO = TCOO1_segend;
  }

  auto T2_seq_insert_edges_result = RectCompressedGetSeqInsertEdges::compute(
    ctx,
    runtime,
    runtime->create_index_space(ctx, Rect<1>(0, 0)),
    T2_pos,
    FID_RECT_1,
    T2_nnz_vals,
    FID_VAL
  );
  T2_crd = getSubRegion(ctx, runtime, T2_crd_parent, Rect<1>(0, (T2_seq_insert_edges_result.scanResult - 1)));
  T->indices[1][1] = T2_crd;
  T_vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (T2_seq_insert_edges_result.scanResult - 1)));
  T->vals = T_vals;

  T2_pos = legionMalloc(ctx, runtime, T2_pos, T2_pos_parent, FID_RECT_1, READ_WRITE);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_crd = legionMalloc(ctx, runtime, T2_crd, T2_crd_parent, FID_COORD, READ_WRITE);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  T_vals = legionMalloc(ctx, runtime, T_vals, T_vals_parent, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end0 = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end0) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    int64_t TCOO1_segend0 = iTCOO + 1;
    while (TCOO1_segend0 < pTCOO1_end0 && TCOO1_crd_accessor[TCOO1_segend0] == i) {
      TCOO1_segend0 = TCOO1_segend0 + 1;
    }
    for (int64_t jTCOO = iTCOO; jTCOO < ((TCOO1_segend0 - 1) + 1); jTCOO++) {
      int64_t j = TCOO2_crd_accessor[jTCOO];
      int64_t pT2 = T2_pos_accessor[Point<1>(i)].lo;
      T2_pos_accessor[Point<1>(i)].lo = T2_pos_accessor[Point<1>(i)].lo + 1;
      T2_crd_accessor[pT2] = j;
      T_vals_rw_accessor[Point<1>(pT2)] = TCOO_vals_ro_accessor[Point<1>(jTCOO)];
    }
    iTCOO = TCOO1_segend0;
  }

  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T_vals);

  RectCompressedFinalizeYieldPositions::compute(ctx, runtime, T2_pos, T2_seq_insert_edges_result.partition, FID_RECT_1);

  runtime->unmap_region(ctx, T2_nnz_vals);
  runtime->destroy_field_space(ctx, T2_nnzfspace);
  runtime->destroy_index_space(ctx, T2_nnzispace);
  runtime->destroy_logical_region(ctx, T2_nnz_vals);

  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T2_nnz_vals);
  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToSSS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  RegionWrapper T1_pos = T->indices[0][0];
  RegionWrapper T1_crd = T->indices[0][1];
  RegionWrapper T2_pos = T->indices[1][0];
  RegionWrapper T2_crd = T->indices[1][1];
  RegionWrapper T3_pos = T->indices[2][0];
  RegionWrapper T3_crd = T->indices[2][1];
  auto T1_pos_parent = T->indicesParents[0][0];
  auto T1_crd_parent = T->indicesParents[0][1];
  auto T2_pos_parent = T->indicesParents[1][0];
  auto T2_crd_parent = T->indicesParents[1][1];
  auto T3_pos_parent = T->indicesParents[2][0];
  auto T3_crd_parent = T->indicesParents[2][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
  auto T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  auto T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  auto T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  auto T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  auto T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
  auto T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  RegionWrapper TCOO2_crd = TCOO->indices[1][0];
  RegionWrapper TCOO3_crd = TCOO->indices[2][0];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  auto TCOO2_crd_parent = TCOO->indicesParents[1][0];
  auto TCOO3_crd_parent = TCOO->indicesParents[2][0];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  auto TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  auto TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD, READ_ONLY);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO3_crd = legionMalloc(ctx, runtime, TCOO3_crd, TCOO3_crd_parent, FID_COORD, READ_ONLY);
  TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);



  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1, READ_WRITE);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD, READ_WRITE);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int64_t iT = 0;
  int64_t T2_pos_size = 1;
  T2_pos = legionMalloc(ctx, runtime, T2_pos_parent, T2_pos_size, FID_RECT_1, READ_WRITE);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T2_crd_size = 1;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD, READ_WRITE);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int64_t jT = 0;
  int64_t T3_pos_size = 1;
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, T3_pos_size, FID_RECT_1, READ_WRITE);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
  T3_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T3_crd_size = 1;
  T3_crd = legionMalloc(ctx, runtime, T3_crd_parent, T3_crd_size, FID_COORD, READ_WRITE);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int64_t kT = 0;
  int64_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int64_t pT1_begin = iT;

  int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    int64_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int64_t pT2_begin = jT;
    if (T2_pos_size <= iT) {
      T2_pos = legionRealloc(ctx, runtime, T2_pos_parent, T2_pos, T2_pos_size * 2, FID_RECT_1, READ_WRITE);
      T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
      T2_pos_size = T2_pos_size * 2;
    }

    int64_t jTCOO = iTCOO;
    int64_t pTCOO2_end = (TCOO1_segend - 1) + 1;

    while (jTCOO < pTCOO2_end) {
      int64_t j = TCOO2_crd_accessor[jTCOO];
      int64_t TCOO2_segend = jTCOO + 1;
      while (TCOO2_segend < pTCOO2_end && TCOO2_crd_accessor[TCOO2_segend] == j) {
        TCOO2_segend = TCOO2_segend + 1;
      }
      int64_t pT3_begin = kT;
      if (T3_pos_size <= jT) {
        T3_pos = legionRealloc(ctx, runtime, T3_pos_parent, T3_pos, T3_pos_size * 2, FID_RECT_1, READ_WRITE);
        T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
        T3_pos_size = T3_pos_size * 2;
      }

      for (int64_t kTCOO = jTCOO; kTCOO < ((TCOO2_segend - 1) + 1); kTCOO++) {
        int64_t k = TCOO3_crd_accessor[kTCOO];
        if (T_capacity <= kT) {
          T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL, READ_WRITE);
          T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
          T_capacity = T_capacity * 2;
        }
        T_vals_rw_accessor[Point<1>(kT)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
        if (T3_crd_size <= kT) {
          T3_crd = legionRealloc(ctx, runtime, T3_crd_parent, T3_crd, T3_crd_size * 2, FID_COORD, READ_WRITE);
          T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
          T3_crd_size = T3_crd_size * 2;
        }
        T3_crd_accessor[kT] = k;
        kT = kT + 1;
      }

      T3_pos_accessor[Point<1>(jT)].lo = pT3_begin;
      T3_pos_accessor[Point<1>(jT)].hi = kT - 1;
      if (pT3_begin < kT) {
        if (T2_crd_size <= jT) {
          T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD, READ_WRITE);
          T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
          T2_crd_size = T2_crd_size * 2;
        }
        T2_crd_accessor[jT] = j;
        jT = jT + 1;
      }
      jTCOO = TCOO2_segend;
    }

    T2_pos_accessor[Point<1>(iT)].lo = pT2_begin;
    T2_pos_accessor[Point<1>(iT)].hi = jT - 1;
    if (pT2_begin < jT) {
      if (T1_crd_size <= iT) {
        T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD, READ_WRITE);
        T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
        T1_crd_size = T1_crd_size * 2;
      }
      T1_crd_accessor[iT] = i;
      iT = iT + 1;
    }
    iTCOO = TCOO1_segend;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, 0));
  T->indices[0][1] = getSubRegion(ctx, runtime, T1_crd_parent, Rect<1>(0, (iT - 1)));

  T->indices[1][0] = getSubRegion(ctx, runtime, T2_pos_parent, Rect<1>(0, (iT - 1)));
  T->indices[1][1] = getSubRegion(ctx, runtime, T2_crd_parent, Rect<1>(0, (jT - 1)));

  T->indices[2][0] = getSubRegion(ctx, runtime, T3_pos_parent, Rect<1>(0, (jT - 1)));
  T->indices[2][1] = getSubRegion(ctx, runtime, T3_crd_parent, Rect<1>(0, (kT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (kT - 1)));

  runtime->unmap_region(ctx, T1_crd);
  runtime->unmap_region(ctx, T1_pos);
  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, T3_crd);
  runtime->unmap_region(ctx, T3_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO3_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToDSS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  RegionWrapper T2_pos = T->indices[1][0];
  RegionWrapper T2_crd = T->indices[1][1];
  RegionWrapper T3_pos = T->indices[2][0];
  RegionWrapper T3_crd = T->indices[2][1];
  auto T2_pos_parent = T->indicesParents[1][0];
  auto T2_crd_parent = T->indicesParents[1][1];
  auto T3_pos_parent = T->indicesParents[2][0];
  auto T3_crd_parent = T->indicesParents[2][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
  auto T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  auto T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  auto T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
  auto T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  RegionWrapper TCOO2_crd = TCOO->indices[1][0];
  RegionWrapper TCOO3_crd = TCOO->indices[2][0];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  auto TCOO2_crd_parent = TCOO->indicesParents[1][0];
  auto TCOO3_crd_parent = TCOO->indicesParents[2][0];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  auto TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  auto TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD, READ_ONLY);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO3_crd = legionMalloc(ctx, runtime, TCOO3_crd, TCOO3_crd_parent, FID_COORD, READ_ONLY);
  TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);



  DomainT<1> T2_pos_domain = runtime->get_index_space_domain(ctx, T2_pos.get_index_space());
  T2_pos = legionMalloc(ctx, runtime, T2_pos, T2_pos_parent, FID_RECT_1, READ_WRITE);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  for (int64_t pT20 = 0; pT20 < (T2_pos_domain.bounds.hi[0] + 1); pT20++) {
    T2_pos_accessor[Point<1>(pT20)] = Rect<1>(0, -1);
  }
  int64_t T2_crd_size = 1;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD, READ_WRITE);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int64_t jT = 0;
  int64_t T3_pos_size = 1;
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, T3_pos_size, FID_RECT_1, READ_WRITE);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
  T3_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T3_crd_size = 1;
  T3_crd = legionMalloc(ctx, runtime, T3_crd_parent, T3_crd_size, FID_COORD, READ_WRITE);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int64_t kT = 0;
  int64_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    int64_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int64_t pT2_begin = jT;

    int64_t jTCOO = iTCOO;
    int64_t pTCOO2_end = (TCOO1_segend - 1) + 1;

    while (jTCOO < pTCOO2_end) {
      int64_t j = TCOO2_crd_accessor[jTCOO];
      int64_t TCOO2_segend = jTCOO + 1;
      while (TCOO2_segend < pTCOO2_end && TCOO2_crd_accessor[TCOO2_segend] == j) {
        TCOO2_segend = TCOO2_segend + 1;
      }
      int64_t pT3_begin = kT;
      if (T3_pos_size <= jT) {
        T3_pos = legionRealloc(ctx, runtime, T3_pos_parent, T3_pos, T3_pos_size * 2, FID_RECT_1, READ_WRITE);
        T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
        T3_pos_size = T3_pos_size * 2;
      }

      for (int64_t kTCOO = jTCOO; kTCOO < ((TCOO2_segend - 1) + 1); kTCOO++) {
        int64_t k = TCOO3_crd_accessor[kTCOO];
        if (T_capacity <= kT) {
          T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL, READ_WRITE);
          T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
          T_capacity = T_capacity * 2;
        }
        T_vals_rw_accessor[Point<1>(kT)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
        if (T3_crd_size <= kT) {
          T3_crd = legionRealloc(ctx, runtime, T3_crd_parent, T3_crd, T3_crd_size * 2, FID_COORD, READ_WRITE);
          T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
          T3_crd_size = T3_crd_size * 2;
        }
        T3_crd_accessor[kT] = k;
        kT = kT + 1;
      }

      T3_pos_accessor[Point<1>(jT)].lo = pT3_begin;
      T3_pos_accessor[Point<1>(jT)].hi = kT - 1;
      if (pT3_begin < kT) {
        if (T2_crd_size <= jT) {
          T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD, READ_WRITE);
          T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
          T2_crd_size = T2_crd_size * 2;
        }
        T2_crd_accessor[jT] = j;
        jT = jT + 1;
      }
      jTCOO = TCOO2_segend;
    }

    T2_pos_accessor[Point<1>(i)].hi = (jT - pT2_begin) - 1;
    iTCOO = TCOO1_segend;
  }

  int64_t csT2 = 0;
  for (int64_t pT200 = 0; pT200 < (T2_pos_domain.bounds.hi[0] + 1); pT200++) {
    int64_t numElemsT2 = T2_pos_accessor[Point<1>(pT200)].hi;
    T2_pos_accessor[Point<1>(pT200)].lo = csT2 + T2_pos_accessor[Point<1>(pT200)].lo;
    T2_pos_accessor[Point<1>(pT200)].hi = csT2 + T2_pos_accessor[Point<1>(pT200)].hi;
    csT2 = csT2 + (numElemsT2 + 1);
  }
  T->indices[1][1] = getSubRegion(ctx, runtime, T2_crd_parent, Rect<1>(0, (jT - 1)));

  T->indices[2][0] = getSubRegion(ctx, runtime, T3_pos_parent, Rect<1>(0, (jT - 1)));
  T->indices[2][1] = getSubRegion(ctx, runtime, T3_crd_parent, Rect<1>(0, (kT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (kT - 1)));

  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, T3_crd);
  runtime->unmap_region(ctx, T3_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO3_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToDDS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  RegionWrapper T3_pos = T->indices[2][0];
  RegionWrapper T3_crd = T->indices[2][1];
  auto T3_pos_parent = T->indicesParents[2][0];
  auto T3_crd_parent = T->indicesParents[2][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
  auto T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
  auto T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int TCOO1_dimension = TCOO->dims[0];
  int TCOO2_dimension = TCOO->dims[1];
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  RegionWrapper TCOO2_crd = TCOO->indices[1][0];
  RegionWrapper TCOO3_crd = TCOO->indices[2][0];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  auto TCOO2_crd_parent = TCOO->indicesParents[1][0];
  auto TCOO3_crd_parent = TCOO->indicesParents[2][0];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  auto TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  auto TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);
  RegionWrapper T3_nnz_vals;
  auto T3_nnz_vals_rw_accessor = createAccessor<AccessorRWint64_t2>(T3_nnz_vals, FID_VAL);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD, READ_ONLY);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO3_crd = legionMalloc(ctx, runtime, TCOO3_crd, TCOO3_crd_parent, FID_COORD, READ_ONLY);
  TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);



  IndexSpace T3_nnzispace = runtime->create_index_space(ctx, createSimpleDomain(Point<2>((TCOO1_dimension - 1), (TCOO2_dimension - 1))));
  FieldSpace T3_nnzfspace = createFieldSpaceWithSize(ctx, runtime, FID_VAL, sizeof(int64_t));
  T3_nnz_vals = runtime->create_logical_region(ctx, T3_nnzispace, T3_nnzfspace);
  runtime->fill_field(ctx, T3_nnz_vals, T3_nnz_vals, FID_VAL, (int64_t)0);
  T3_nnz_vals = legionMalloc(ctx, runtime, T3_nnz_vals, T3_nnz_vals, FID_VAL, READ_WRITE);
  T3_nnz_vals_rw_accessor = createAccessor<AccessorRWint64_t2>(T3_nnz_vals, FID_VAL);

  int64_t qiTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (qiTCOO < pTCOO1_end) {
    int64_t qi = TCOO1_crd_accessor[qiTCOO];
    int64_t TCOO1_segend = qiTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == qi) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int64_t qjTCOO = qiTCOO;
    int64_t pTCOO2_end = (TCOO1_segend - 1) + 1;

    while (qjTCOO < pTCOO2_end) {
      int64_t qj = TCOO2_crd_accessor[qjTCOO];
      int64_t TCOO2_segend = qjTCOO + 1;
      while (TCOO2_segend < pTCOO2_end && TCOO2_crd_accessor[TCOO2_segend] == qj) {
        TCOO2_segend = TCOO2_segend + 1;
      }
      for (int64_t qkTCOO = qjTCOO; qkTCOO < ((TCOO2_segend - 1) + 1); qkTCOO++) {
        T3_nnz_vals_rw_accessor[Point<2>(qi, qj)] = T3_nnz_vals_rw_accessor[Point<2>(qi, qj)] + (int64_t)1;
      }
      qjTCOO = TCOO2_segend;
    }
    qiTCOO = TCOO1_segend;
  }

  auto T3_seq_insert_edges_result = RectCompressedGetSeqInsertEdges::compute(
    ctx,
    runtime,
    runtime->create_index_space(ctx, Rect<1>(0, 0)),
    T3_pos,
    FID_RECT_1,
    T3_nnz_vals,
    FID_VAL
  );
  T3_crd = getSubRegion(ctx, runtime, T3_crd_parent, Rect<1>(0, (T3_seq_insert_edges_result.scanResult - 1)));
  T->indices[2][1] = T3_crd;
  T_vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (T3_seq_insert_edges_result.scanResult - 1)));
  T->vals = T_vals;

  T3_pos = legionMalloc(ctx, runtime, T3_pos, T3_pos_parent, FID_RECT_1, READ_WRITE);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
  T3_crd = legionMalloc(ctx, runtime, T3_crd, T3_crd_parent, FID_COORD, READ_WRITE);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  T_vals = legionMalloc(ctx, runtime, T_vals, T_vals_parent, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end0 = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end0) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    int64_t TCOO1_segend0 = iTCOO + 1;
    while (TCOO1_segend0 < pTCOO1_end0 && TCOO1_crd_accessor[TCOO1_segend0] == i) {
      TCOO1_segend0 = TCOO1_segend0 + 1;
    }
    int64_t jTCOO = iTCOO;
    int64_t pTCOO2_end0 = (TCOO1_segend0 - 1) + 1;

    while (jTCOO < pTCOO2_end0) {
      int64_t j = TCOO2_crd_accessor[jTCOO];
      int64_t TCOO2_segend0 = jTCOO + 1;
      while (TCOO2_segend0 < pTCOO2_end0 && TCOO2_crd_accessor[TCOO2_segend0] == j) {
        TCOO2_segend0 = TCOO2_segend0 + 1;
      }
      for (int64_t kTCOO = jTCOO; kTCOO < ((TCOO2_segend0 - 1) + 1); kTCOO++) {
        int64_t k = TCOO3_crd_accessor[kTCOO];
        int64_t pT3 = T3_pos_accessor[Point<2>(i, j)].lo;
        T3_pos_accessor[Point<2>(i, j)].lo = T3_pos_accessor[Point<2>(i, j)].lo + 1;
        T3_crd_accessor[pT3] = k;
        T_vals_rw_accessor[Point<1>(pT3)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
      }
      jTCOO = TCOO2_segend0;
    }
    iTCOO = TCOO1_segend0;
  }

  runtime->unmap_region(ctx, T3_pos);
  runtime->unmap_region(ctx, T3_crd);
  runtime->unmap_region(ctx, T_vals);

  RectCompressedFinalizeYieldPositions::compute(ctx, runtime, T3_pos, T3_seq_insert_edges_result.partition, FID_RECT_1);

  runtime->unmap_region(ctx, T3_nnz_vals);
  runtime->destroy_field_space(ctx, T3_nnzfspace);
  runtime->destroy_index_space(ctx, T3_nnzispace);
  runtime->destroy_logical_region(ctx, T3_nnz_vals);

  runtime->unmap_region(ctx, T3_crd);
  runtime->unmap_region(ctx, T3_nnz_vals);
  runtime->unmap_region(ctx, T3_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO3_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToSDS(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  RegionWrapper T1_pos = T->indices[0][0];
  RegionWrapper T1_crd = T->indices[0][1];
  RegionWrapper T3_pos = T->indices[2][0];
  RegionWrapper T3_crd = T->indices[2][1];
  auto T1_pos_parent = T->indicesParents[0][0];
  auto T1_crd_parent = T->indicesParents[0][1];
  auto T3_pos_parent = T->indicesParents[2][0];
  auto T3_crd_parent = T->indicesParents[2][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
  auto T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  auto T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  auto T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
  auto T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  RegionWrapper TCOO2_crd = TCOO->indices[1][0];
  RegionWrapper TCOO3_crd = TCOO->indices[2][0];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  auto TCOO2_crd_parent = TCOO->indicesParents[1][0];
  auto TCOO3_crd_parent = TCOO->indicesParents[2][0];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  auto TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  auto TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD, READ_ONLY);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO3_crd = legionMalloc(ctx, runtime, TCOO3_crd, TCOO3_crd_parent, FID_COORD, READ_ONLY);
  TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);



  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1, READ_WRITE);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD, READ_WRITE);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int64_t iT = 0;
  DomainT<2> T3_pos_domain = runtime->get_index_space_domain(ctx, T3_pos.get_index_space());
  int64_t T3_pos_size = 1;
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, T3_pos_size, FID_RECT_1, READ_WRITE);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
  T3_pos_accessor[Point<2>(0, 0)] = Rect<1>(0, -1);
  int64_t T3_crd_size = 1;
  T3_crd = legionMalloc(ctx, runtime, T3_crd_parent, T3_crd_size, FID_COORD, READ_WRITE);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int64_t kT = 0;
  int64_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int64_t pT1_begin = iT;

  int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    int64_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    if (T3_pos_size <= iT) {
      T3_pos = legionRealloc(ctx, runtime, T3_pos_parent, T3_pos, T3_pos_size * 2, FID_RECT_1, READ_WRITE);
      T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
      T3_pos_size = T3_pos_size * 2;
    }
    for (int64_t pT31 = 0; pT31 < (T3_pos_domain.bounds.hi[1] + 1); pT31++) {
      T3_pos_accessor[Point<2>(iT, pT31)] = Rect<1>(0, -1);
    }

    int64_t jTCOO = iTCOO;
    int64_t pTCOO2_end = (TCOO1_segend - 1) + 1;

    while (jTCOO < pTCOO2_end) {
      int64_t j = TCOO2_crd_accessor[jTCOO];
      int64_t TCOO2_segend = jTCOO + 1;
      while (TCOO2_segend < pTCOO2_end && TCOO2_crd_accessor[TCOO2_segend] == j) {
        TCOO2_segend = TCOO2_segend + 1;
      }
      int64_t pT3_begin = kT;

      for (int64_t kTCOO = jTCOO; kTCOO < ((TCOO2_segend - 1) + 1); kTCOO++) {
        int64_t k = TCOO3_crd_accessor[kTCOO];
        if (T_capacity <= kT) {
          T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL, READ_WRITE);
          T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
          T_capacity = T_capacity * 2;
        }
        T_vals_rw_accessor[Point<1>(kT)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
        if (T3_crd_size <= kT) {
          T3_crd = legionRealloc(ctx, runtime, T3_crd_parent, T3_crd, T3_crd_size * 2, FID_COORD, READ_WRITE);
          T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
          T3_crd_size = T3_crd_size * 2;
        }
        T3_crd_accessor[kT] = k;
        kT = kT + 1;
      }

      T3_pos_accessor[Point<2>(iT, j)].hi = (kT - pT3_begin) - 1;
      jTCOO = TCOO2_segend;
    }
    if (T1_crd_size <= iT) {
      T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD, READ_WRITE);
      T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
      T1_crd_size = T1_crd_size * 2;
    }
    T1_crd_accessor[iT] = i;
    iT = iT + 1;
    iTCOO = TCOO1_segend;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, 0));
  T->indices[0][1] = getSubRegion(ctx, runtime, T1_crd_parent, Rect<1>(0, (iT - 1)));

  runtime->unmap_region(ctx, T3_pos);
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, iT, FID_RECT_1, READ_WRITE);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
  int64_t csT3 = 0;
  for (int64_t pT30 = 0; pT30 < iT; pT30++) {
    for (int64_t pT310 = 0; pT310 < (T3_pos_domain.bounds.hi[1] + 1); pT310++) {
      int64_t numElemsT3 = T3_pos_accessor[Point<2>(pT30, pT310)].hi;
      T3_pos_accessor[Point<2>(pT30, pT310)].lo = csT3 + T3_pos_accessor[Point<2>(pT30, pT310)].lo;
      T3_pos_accessor[Point<2>(pT30, pT310)].hi = csT3 + T3_pos_accessor[Point<2>(pT30, pT310)].hi;
      csT3 = csT3 + (numElemsT3 + 1);
    }
  }
  T->indices[2][0] = getSubRegion(ctx, runtime, T3_pos_parent, Rect<1>(0, (iT - 1)));
  T->indices[2][1] = getSubRegion(ctx, runtime, T3_crd_parent, Rect<1>(0, (kT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (kT - 1)));

  runtime->unmap_region(ctx, T1_crd);
  runtime->unmap_region(ctx, T1_pos);
  runtime->unmap_region(ctx, T3_crd);
  runtime->unmap_region(ctx, T3_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO3_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToDCSR(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  RegionWrapper T1_pos = T->indices[0][0];
  RegionWrapper T1_crd = T->indices[0][1];
  RegionWrapper T2_pos = T->indices[1][0];
  RegionWrapper T2_crd = T->indices[1][1];
  auto T1_pos_parent = T->indicesParents[0][0];
  auto T1_crd_parent = T->indicesParents[0][1];
  auto T2_pos_parent = T->indicesParents[1][0];
  auto T2_crd_parent = T->indicesParents[1][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
  auto T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  auto T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  auto T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  auto T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  RegionWrapper TCOO2_crd = TCOO->indices[1][0];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  auto TCOO2_crd_parent = TCOO->indicesParents[1][0];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  auto TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD, READ_ONLY);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);



  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1, READ_WRITE);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD, READ_WRITE);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int64_t iT = 0;
  int64_t T2_pos_size = 1;
  T2_pos = legionMalloc(ctx, runtime, T2_pos_parent, T2_pos_size, FID_RECT_1, READ_WRITE);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T2_crd_size = 1;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD, READ_WRITE);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int64_t jT = 0;
  int64_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int64_t pT1_begin = iT;

  int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    int64_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int64_t pT2_begin = jT;
    if (T2_pos_size <= iT) {
      T2_pos = legionRealloc(ctx, runtime, T2_pos_parent, T2_pos, T2_pos_size * 2, FID_RECT_1, READ_WRITE);
      T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
      T2_pos_size = T2_pos_size * 2;
    }

    for (int64_t jTCOO = iTCOO; jTCOO < ((TCOO1_segend - 1) + 1); jTCOO++) {
      int64_t j = TCOO2_crd_accessor[jTCOO];
      if (T_capacity <= jT) {
        T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL, READ_WRITE);
        T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
        T_capacity = T_capacity * 2;
      }
      T_vals_rw_accessor[Point<1>(jT)] = TCOO_vals_ro_accessor[Point<1>(jTCOO)];
      if (T2_crd_size <= jT) {
        T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD, READ_WRITE);
        T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
        T2_crd_size = T2_crd_size * 2;
      }
      T2_crd_accessor[jT] = j;
      jT = jT + 1;
    }

    T2_pos_accessor[Point<1>(iT)].lo = pT2_begin;
    T2_pos_accessor[Point<1>(iT)].hi = jT - 1;
    if (pT2_begin < jT) {
      if (T1_crd_size <= iT) {
        T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD, READ_WRITE);
        T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
        T1_crd_size = T1_crd_size * 2;
      }
      T1_crd_accessor[iT] = i;
      iT = iT + 1;
    }
    iTCOO = TCOO1_segend;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, 0));
  T->indices[0][1] = getSubRegion(ctx, runtime, T1_crd_parent, Rect<1>(0, (iT - 1)));

  T->indices[1][0] = getSubRegion(ctx, runtime, T2_pos_parent, Rect<1>(0, (iT - 1)));
  T->indices[1][1] = getSubRegion(ctx, runtime, T2_crd_parent, Rect<1>(0, (jT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (jT - 1)));

  runtime->unmap_region(ctx, T1_crd);
  runtime->unmap_region(ctx, T1_pos);
  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToSD(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  int T2_dimension = T->dims[1];
  RegionWrapper T1_pos = T->indices[0][0];
  RegionWrapper T1_crd = T->indices[0][1];
  auto T1_pos_parent = T->indicesParents[0][0];
  auto T1_crd_parent = T->indicesParents[0][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble2>(T_vals, FID_VAL);
  auto T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  auto T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  RegionWrapper TCOO2_crd = TCOO->indices[1][0];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  auto TCOO2_crd_parent = TCOO->indicesParents[1][0];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  auto TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD, READ_ONLY);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);



  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1, READ_WRITE);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD, READ_WRITE);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int64_t iT = 0;
  int64_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble2>(T_vals, FID_VAL);

  int64_t pT1_begin = iT;

  int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    int64_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    if (T_capacity <= iT) {
      int64_t T_vals_new_size = TACO_MAX(T_capacity * 2,(iT + 1));
      T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_vals_new_size, FID_VAL, READ_WRITE);
      T_vals_rw_accessor = createAccessor<AccessorRWdouble2>(T_vals, FID_VAL);
      T_capacity = T_vals_new_size;
    }
    for (int64_t j = 0; j < T2_dimension; j++) {
      T_vals_rw_accessor[Point<2>(iT, j)] = 0.0;
    }

    for (int64_t jTCOO = iTCOO; jTCOO < ((TCOO1_segend - 1) + 1); jTCOO++) {
      int64_t j = TCOO2_crd_accessor[jTCOO];
      T_vals_rw_accessor[Point<2>(iT, j)] = TCOO_vals_ro_accessor[Point<1>(jTCOO)];
    }
    if (T1_crd_size <= iT) {
      T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD, READ_WRITE);
      T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
      T1_crd_size = T1_crd_size * 2;
    }
    T1_crd_accessor[iT] = i;
    iT = iT + 1;
    iTCOO = TCOO1_segend;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, 0));
  T->indices[0][1] = getSubRegion(ctx, runtime, T1_crd_parent, Rect<1>(0, (iT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (iT - 1)));

  runtime->unmap_region(ctx, T1_crd);
  runtime->unmap_region(ctx, T1_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToCSC(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  RegionWrapper T2_pos = T->indices[1][0];
  RegionWrapper T2_crd = T->indices[1][1];
  auto T2_pos_parent = T->indicesParents[1][0];
  auto T2_crd_parent = T->indicesParents[1][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
  auto T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  auto T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int TCOO2_dimension = TCOO->dims[1];
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  RegionWrapper TCOO2_crd = TCOO->indices[1][0];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  auto TCOO2_crd_parent = TCOO->indicesParents[1][0];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  auto TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  RegionWrapper T2_nnz_vals;
  auto T2_nnz_vals_rw_accessor = createAccessor<AccessorRWint64_t1>(T2_nnz_vals, FID_VAL);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD, READ_ONLY);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);



  IndexSpace T2_nnzispace = runtime->create_index_space(ctx, createSimpleDomain(Point<1>((TCOO2_dimension - 1))));
  FieldSpace T2_nnzfspace = createFieldSpaceWithSize(ctx, runtime, FID_VAL, sizeof(int64_t));
  T2_nnz_vals = runtime->create_logical_region(ctx, T2_nnzispace, T2_nnzfspace);
  runtime->fill_field(ctx, T2_nnz_vals, T2_nnz_vals, FID_VAL, (int64_t)0);
  T2_nnz_vals = legionMalloc(ctx, runtime, T2_nnz_vals, T2_nnz_vals, FID_VAL, READ_WRITE);
  T2_nnz_vals_rw_accessor = createAccessor<AccessorRWint64_t1>(T2_nnz_vals, FID_VAL);

  int64_t qiTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (qiTCOO < pTCOO1_end) {
    int64_t qi = TCOO1_crd_accessor[qiTCOO];
    int64_t TCOO1_segend = qiTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == qi) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    for (int64_t qjTCOO = qiTCOO; qjTCOO < ((TCOO1_segend - 1) + 1); qjTCOO++) {
      int64_t qj = TCOO2_crd_accessor[qjTCOO];
      T2_nnz_vals_rw_accessor[Point<1>(qj)] = T2_nnz_vals_rw_accessor[Point<1>(qj)] + (int64_t)1;
    }
    qiTCOO = TCOO1_segend;
  }

  auto T2_seq_insert_edges_result = RectCompressedGetSeqInsertEdges::compute(
    ctx,
    runtime,
    runtime->create_index_space(ctx, Rect<1>(0, 0)),
    T2_pos,
    FID_RECT_1,
    T2_nnz_vals,
    FID_VAL
  );
  T2_crd = getSubRegion(ctx, runtime, T2_crd_parent, Rect<1>(0, (T2_seq_insert_edges_result.scanResult - 1)));
  T->indices[1][1] = T2_crd;
  T_vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (T2_seq_insert_edges_result.scanResult - 1)));
  T->vals = T_vals;

  T2_pos = legionMalloc(ctx, runtime, T2_pos, T2_pos_parent, FID_RECT_1, READ_WRITE);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_crd = legionMalloc(ctx, runtime, T2_crd, T2_crd_parent, FID_COORD, READ_WRITE);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  T_vals = legionMalloc(ctx, runtime, T_vals, T_vals_parent, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int64_t pTCOO1_end0 = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end0) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    int64_t TCOO1_segend0 = iTCOO + 1;
    while (TCOO1_segend0 < pTCOO1_end0 && TCOO1_crd_accessor[TCOO1_segend0] == i) {
      TCOO1_segend0 = TCOO1_segend0 + 1;
    }
    for (int64_t jTCOO = iTCOO; jTCOO < ((TCOO1_segend0 - 1) + 1); jTCOO++) {
      int64_t j = TCOO2_crd_accessor[jTCOO];
      int64_t pT2 = T2_pos_accessor[Point<1>(j)].lo;
      T2_pos_accessor[Point<1>(j)].lo = T2_pos_accessor[Point<1>(j)].lo + 1;
      T2_crd_accessor[pT2] = i;
      T_vals_rw_accessor[Point<1>(pT2)] = TCOO_vals_ro_accessor[Point<1>(jTCOO)];
    }
    iTCOO = TCOO1_segend0;
  }

  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T_vals);

  RectCompressedFinalizeYieldPositions::compute(ctx, runtime, T2_pos, T2_seq_insert_edges_result.partition, FID_RECT_1);

  runtime->unmap_region(ctx, T2_nnz_vals);
  runtime->destroy_field_space(ctx, T2_nnzfspace);
  runtime->destroy_index_space(ctx, T2_nnzispace);
  runtime->destroy_logical_region(ctx, T2_nnz_vals);

  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T2_nnz_vals);
  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToVec(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  RegionWrapper T1_pos = T->indices[0][0];
  RegionWrapper T1_crd = T->indices[0][1];
  auto T1_pos_parent = T->indicesParents[0][0];
  auto T1_crd_parent = T->indicesParents[0][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
  auto T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  auto T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  RegionWrapper TCOO1_pos = TCOO->indices[0][0];
  RegionWrapper TCOO1_crd = TCOO->indices[0][1];
  auto TCOO1_pos_parent = TCOO->indicesParents[0][0];
  auto TCOO1_crd_parent = TCOO->indicesParents[0][1];
  RegionWrapper TCOO_vals = TCOO->vals;
  auto TCOO_vals_parent = TCOO->valsParent;
  auto TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);
  auto TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  auto TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1, READ_ONLY);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD, READ_ONLY);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL, READ_ONLY);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);


  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1, READ_WRITE);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int64_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD, READ_WRITE);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int64_t iT = 0;
  int64_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL, READ_WRITE);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int64_t pT1_begin = iT;

  for (int64_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo; iTCOO < (TCOO1_pos_accessor[Point<1>(0)].hi + 1); iTCOO++) {
    int64_t i = TCOO1_crd_accessor[iTCOO];
    if (T_capacity <= iT) {
      T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL, READ_WRITE);
      T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
      T_capacity = T_capacity * 2;
    }
    T_vals_rw_accessor[Point<1>(iT)] = TCOO_vals_ro_accessor[Point<1>(iTCOO)];
    if (T1_crd_size <= iT) {
      T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD, READ_WRITE);
      T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
      T1_crd_size = T1_crd_size * 2;
    }
    T1_crd_accessor[iT] = i;
    iT = iT + 1;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, 0));
  T->indices[0][1] = getSubRegion(ctx, runtime, T1_crd_parent, Rect<1>(0, (iT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (iT - 1)));

  runtime->unmap_region(ctx, T1_crd);
  runtime->unmap_region(ctx, T1_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}
void registerTacoTasks() {
}
