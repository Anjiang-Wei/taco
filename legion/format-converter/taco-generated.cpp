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


void packLegionCOOToCSR(Context ctx, Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  int T1_dimension = T->dims[0];
  RegionWrapper T2_pos = T->indices[1][0];
  RegionWrapper T2_crd = T->indices[1][1];
  auto T2_pos_parent = T->indicesParents[1][0];
  auto T2_crd_parent = T->indicesParents[1][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
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

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);

  T2_pos = legionMalloc(ctx, runtime, T2_pos_parent, T1_dimension, FID_RECT_1);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[0] = Rect<1>(0, 0);
  for (int32_t pT2 = 1; pT2 < T1_dimension; pT2++) {
    T2_pos_accessor[pT2] = Rect<1>(0, 0);
  }
  int32_t T2_crd_size = 1048576;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int32_t jT = 0;
  int32_t T_capacity = 1048576;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int32_t iTCOO = TCOO1_pos_accessor[0].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[0].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[iTCOO];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == i) {
      TCOO1_segend++;
    }
    int32_t pT2_begin = jT;

    for (int32_t jTCOO = iTCOO; jTCOO < TCOO1_segend; jTCOO++) {
      int32_t j = TCOO2_crd_accessor[jTCOO];
      if (T_capacity <= jT) {
        T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL);
        T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
        T_capacity *= 2;
      }
      T_vals_rw_accessor[Point<1>(jT)] = TCOO_vals_ro_accessor[Point<1>(jTCOO)];
      if (T2_crd_size <= jT) {
        T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD);
        T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
        T2_crd_size *= 2;
      }
      T2_crd_accessor[jT] = j;
      jT++;
    }

    T2_pos_accessor[i].hi = (jT - pT2_begin) - 1;
    iTCOO = TCOO1_segend;
  }

  int64_t csT2 = 0;
  for (int64_t pT20 = 0; pT20 < T1_dimension; pT20++) {
    int64_t numElemsT2 = T2_pos_accessor[pT20].hi;
    T2_pos_accessor[pT20].lo = csT2 + T2_pos_accessor[pT20].lo;
    T2_pos_accessor[pT20].hi = csT2 + T2_pos_accessor[pT20].hi;
    csT2 += numElemsT2 + 1;
  }
  T->indices[1][1] = getSubRegion(ctx, runtime, T2_crd_parent, Rect<1>(0, (jT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (jT - 1)));

  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T_vals);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO_vals);
}

void packLegionCOOToDSS(Context ctx, Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  int T1_dimension = T->dims[0];
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

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO3_crd = legionMalloc(ctx, runtime, TCOO3_crd, TCOO3_crd_parent, FID_COORD);
  TCOO3_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO3_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);

  T2_pos = legionMalloc(ctx, runtime, T2_pos_parent, T1_dimension, FID_RECT_1);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[0] = Rect<1>(0, 0);
  for (int32_t pT2 = 1; pT2 < T1_dimension; pT2++) {
    T2_pos_accessor[pT2] = Rect<1>(0, 0);
  }
  int32_t T2_crd_size = 1048576;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int32_t jT = 0;
  int32_t T3_pos_size = 1048576;
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, T3_pos_size, FID_RECT_1);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
  T3_pos_accessor[0] = Rect<1>(0, 0);
  int32_t T3_crd_size = 1048576;
  T3_crd = legionMalloc(ctx, runtime, T3_crd_parent, T3_crd_size, FID_COORD);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int32_t kT = 0;
  int32_t T_capacity = 1048576;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int32_t iTCOO = TCOO1_pos_accessor[0].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[0].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[iTCOO];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[TCOO1_segend] == i) {
      TCOO1_segend++;
    }
    int32_t pT2_begin = jT;

    int32_t jTCOO = iTCOO;

    while (jTCOO < TCOO1_segend) {
      int32_t j = TCOO2_crd_accessor[jTCOO];
      int32_t TCOO2_segend = jTCOO + 1;
      while (TCOO2_segend < TCOO1_segend && TCOO2_crd_accessor[TCOO2_segend] == j) {
        TCOO2_segend++;
      }
      int32_t pT3_begin = kT;
      if (T3_pos_size <= jT + 1) {
        T3_pos = legionRealloc(ctx, runtime, T3_pos_parent, T3_pos, T3_pos_size * 2, FID_RECT_1);
        T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
        T3_pos_size *= 2;
      }

      for (int32_t kTCOO = jTCOO; kTCOO < TCOO2_segend; kTCOO++) {
        int32_t k = TCOO3_crd_accessor[kTCOO];
        if (T_capacity <= kT) {
          T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL);
          T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
          T_capacity *= 2;
        }
        T_vals_rw_accessor[Point<1>(kT)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
        if (T3_crd_size <= kT) {
          T3_crd = legionRealloc(ctx, runtime, T3_crd_parent, T3_crd, T3_crd_size * 2, FID_COORD);
          T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
          T3_crd_size *= 2;
        }
        T3_crd_accessor[kT] = k;
        kT++;
      }

      T3_pos_accessor[jT].lo = pT3_begin;
      T3_pos_accessor[jT].hi = kT - 1;
      if (pT3_begin < kT) {
        if (T2_crd_size <= jT) {
          T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD);
          T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
          T2_crd_size *= 2;
        }
        T2_crd_accessor[jT] = j;
        jT++;
      }
      jTCOO = TCOO2_segend;
    }

    T2_pos_accessor[i].hi = (jT - pT2_begin) - 1;
    iTCOO = TCOO1_segend;
  }

  int64_t csT2 = 0;
  for (int64_t pT20 = 0; pT20 < T1_dimension; pT20++) {
    int64_t numElemsT2 = T2_pos_accessor[pT20].hi;
    T2_pos_accessor[pT20].lo = csT2 + T2_pos_accessor[pT20].lo;
    T2_pos_accessor[pT20].hi = csT2 + T2_pos_accessor[pT20].hi;
    csT2 += numElemsT2 + 1;
  }
  T->indices[1][1] = getSubRegion(ctx, runtime, T2_crd_parent, Rect<1>(0, (jT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (kT - 1)));

  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T3_pos);
  runtime->unmap_region(ctx, T3_crd);
  runtime->unmap_region(ctx, T_vals);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO3_crd);
  runtime->unmap_region(ctx, TCOO_vals);
}
void registerTacoTasks() {
}
