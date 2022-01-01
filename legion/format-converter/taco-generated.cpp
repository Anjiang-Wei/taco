#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorROint32_t1;
typedef FieldAccessor<READ_WRITE,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorRWint32_t1;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRWdouble1;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRORect_1_1;
typedef FieldAccessor<READ_WRITE,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorRWRect_1_1;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;
typedef FieldAccessor<READ_WRITE,Rect<1>,2,coord_t,Realm::AffineAccessor<Rect<1>,2,coord_t>> AccessorRWRect_1_2;


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

  int64_t T2Size = runtime->get_index_space_domain(ctx, get_index_space(T2_crd)).hi()[0] + 1;
  int64_t TCOO1Size = runtime->get_index_space_domain(ctx, get_index_space(TCOO1_crd)).hi()[0] + 1;

  DomainT<1> T2_pos_domain = runtime->get_index_space_domain(ctx, T2_pos.get_index_space());
  T2_pos = legionMalloc(ctx, runtime, T2_pos, T2_pos_parent, FID_RECT_1);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  for (int32_t pT20 = 0; pT20 < (T2_pos_domain.bounds.hi[0] + 1); pT20++) {
    T2_pos_accessor[Point<1>(pT20)] = Rect<1>(0, -1);
  }
  int32_t T2_crd_size = 1;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int32_t jT = 0;
  int32_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int32_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[(iTCOO * 1)];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[(TCOO1_segend * 1)] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int32_t iT = 0 * T1_dimension + i;
    int32_t pT2_begin = jT;

    for (int32_t jTCOO = iTCOO; jTCOO < ((TCOO1_segend - 1) + 1); jTCOO++) {
      int32_t j = TCOO2_crd_accessor[(jTCOO * 1 + 0)];
      if (T_capacity <= jT) {
        T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL);
        T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
        T_capacity = T_capacity * 2;
      }
      T_vals_rw_accessor[Point<1>(jT)] = TCOO_vals_ro_accessor[Point<1>(jTCOO)];
      if (T2_crd_size <= jT) {
        T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD);
        T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
        T2_crd_size = T2_crd_size * 2;
      }
      T2_crd_accessor[jT * 1] = j;
      jT = jT + 1;
    }

    T2_pos_accessor[Point<1>(i)].hi = (jT - pT2_begin) - 1;
    iTCOO = TCOO1_segend;
  }

  int64_t csT2 = 0;
  for (int32_t pT200 = 0; pT200 < (T2_pos_domain.bounds.hi[0] + 1); pT200++) {
    int64_t numElemsT2 = T2_pos_accessor[Point<1>(pT200)].hi;
    T2_pos_accessor[Point<1>(pT200)].lo = csT2 + T2_pos_accessor[Point<1>(pT200)].lo;
    T2_pos_accessor[Point<1>(pT200)].hi = csT2 + T2_pos_accessor[Point<1>(pT200)].hi;
    csT2 = csT2 + (numElemsT2 + 1);
  }
  T->indices[1][1] = getSubRegion(ctx, runtime, T2_crd_parent, Rect<1>(0, (jT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (jT - 1)));

  runtime->unmap_region(ctx, T2_crd);
  runtime->unmap_region(ctx, T2_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToSSS(Context ctx, Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
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

  int64_t T1Size = runtime->get_index_space_domain(ctx, get_index_space(T1_crd)).hi()[0] + 1;
  int64_t T2Size = runtime->get_index_space_domain(ctx, get_index_space(T2_crd)).hi()[0] + 1;
  int64_t T3Size = runtime->get_index_space_domain(ctx, get_index_space(T3_crd)).hi()[0] + 1;
  int64_t TCOO1Size = runtime->get_index_space_domain(ctx, get_index_space(TCOO1_crd)).hi()[0] + 1;

  DomainT<1> T1_pos_domain = runtime->get_index_space_domain(ctx, T1_pos.get_index_space());
  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int32_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int32_t iT = 0;
  DomainT<1> T2_pos_domain = runtime->get_index_space_domain(ctx, T2_pos.get_index_space());
  int32_t T2_pos_size = 1;
  T2_pos = legionMalloc(ctx, runtime, T2_pos_parent, T2_pos_size, FID_RECT_1);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int32_t T2_crd_size = 1;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int32_t jT = 0;
  DomainT<1> T3_pos_domain = runtime->get_index_space_domain(ctx, T3_pos.get_index_space());
  int32_t T3_pos_size = 1;
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, T3_pos_size, FID_RECT_1);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
  T3_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int32_t T3_crd_size = 1;
  T3_crd = legionMalloc(ctx, runtime, T3_crd_parent, T3_crd_size, FID_COORD);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int32_t kT = 0;
  int32_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int32_t pT1_begin = iT;

  int32_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[(iTCOO * 1)];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[(TCOO1_segend * 1)] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int32_t pT2_begin = jT;
    if (T2_pos_size <= iT) {
      T2_pos = legionRealloc(ctx, runtime, T2_pos_parent, T2_pos, T2_pos_size * 2, FID_RECT_1);
      T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
      T2_pos_size = T2_pos_size * 2;
    }

    int32_t jTCOO = iTCOO;
    int32_t pTCOO2_end = (TCOO1_segend - 1) + 1;

    while (jTCOO < pTCOO2_end) {
      int32_t j = TCOO2_crd_accessor[(jTCOO * 1 + 0)];
      int32_t TCOO2_segend = jTCOO + 1;
      while (TCOO2_segend < pTCOO2_end && TCOO2_crd_accessor[(TCOO2_segend * 1 + 0)] == j) {
        TCOO2_segend = TCOO2_segend + 1;
      }
      int32_t pT3_begin = kT;
      if (T3_pos_size <= jT) {
        T3_pos = legionRealloc(ctx, runtime, T3_pos_parent, T3_pos, T3_pos_size * 2, FID_RECT_1);
        T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
        T3_pos_size = T3_pos_size * 2;
      }

      for (int32_t kTCOO = jTCOO; kTCOO < ((TCOO2_segend - 1) + 1); kTCOO++) {
        int32_t k = TCOO3_crd_accessor[(kTCOO * 1 + 0)];
        if (T_capacity <= kT) {
          T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL);
          T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
          T_capacity = T_capacity * 2;
        }
        T_vals_rw_accessor[Point<1>(kT)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
        if (T3_crd_size <= kT) {
          T3_crd = legionRealloc(ctx, runtime, T3_crd_parent, T3_crd, T3_crd_size * 2, FID_COORD);
          T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
          T3_crd_size = T3_crd_size * 2;
        }
        T3_crd_accessor[kT * 1] = k;
        kT = kT + 1;
      }

      T3_pos_accessor[Point<1>(jT)].lo = pT3_begin;
      T3_pos_accessor[Point<1>(jT)].hi = kT - 1;
      if (pT3_begin < kT) {
        if (T2_crd_size <= jT) {
          T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD);
          T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
          T2_crd_size = T2_crd_size * 2;
        }
        T2_crd_accessor[jT * 1] = j;
        jT = jT + 1;
      }
      jTCOO = TCOO2_segend;
    }

    T2_pos_accessor[Point<1>(iT)].lo = pT2_begin;
    T2_pos_accessor[Point<1>(iT)].hi = jT - 1;
    if (pT2_begin < jT) {
      if (T1_crd_size <= iT) {
        T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD);
        T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
        T1_crd_size = T1_crd_size * 2;
      }
      T1_crd_accessor[iT * 1] = i;
      iT = iT + 1;
    }
    iTCOO = TCOO1_segend;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, (1 - 1)));
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

  int64_t T2Size = runtime->get_index_space_domain(ctx, get_index_space(T2_crd)).hi()[0] + 1;
  int64_t T3Size = runtime->get_index_space_domain(ctx, get_index_space(T3_crd)).hi()[0] + 1;
  int64_t TCOO1Size = runtime->get_index_space_domain(ctx, get_index_space(TCOO1_crd)).hi()[0] + 1;

  DomainT<1> T2_pos_domain = runtime->get_index_space_domain(ctx, T2_pos.get_index_space());
  T2_pos = legionMalloc(ctx, runtime, T2_pos, T2_pos_parent, FID_RECT_1);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  for (int32_t pT20 = 0; pT20 < (T2_pos_domain.bounds.hi[0] + 1); pT20++) {
    T2_pos_accessor[Point<1>(pT20)] = Rect<1>(0, -1);
  }
  int32_t T2_crd_size = 1;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int32_t jT = 0;
  DomainT<1> T3_pos_domain = runtime->get_index_space_domain(ctx, T3_pos.get_index_space());
  int32_t T3_pos_size = 1;
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, T3_pos_size, FID_RECT_1);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
  T3_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int32_t T3_crd_size = 1;
  T3_crd = legionMalloc(ctx, runtime, T3_crd_parent, T3_crd_size, FID_COORD);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int32_t kT = 0;
  int32_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int32_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[(iTCOO * 1)];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[(TCOO1_segend * 1)] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int32_t iT = 0 * T1_dimension + i;
    int32_t pT2_begin = jT;

    int32_t jTCOO = iTCOO;
    int32_t pTCOO2_end = (TCOO1_segend - 1) + 1;

    while (jTCOO < pTCOO2_end) {
      int32_t j = TCOO2_crd_accessor[(jTCOO * 1 + 0)];
      int32_t TCOO2_segend = jTCOO + 1;
      while (TCOO2_segend < pTCOO2_end && TCOO2_crd_accessor[(TCOO2_segend * 1 + 0)] == j) {
        TCOO2_segend = TCOO2_segend + 1;
      }
      int32_t pT3_begin = kT;
      if (T3_pos_size <= jT) {
        T3_pos = legionRealloc(ctx, runtime, T3_pos_parent, T3_pos, T3_pos_size * 2, FID_RECT_1);
        T3_pos_accessor = createAccessor<AccessorRWRect_1_1>(T3_pos, FID_RECT_1);
        T3_pos_size = T3_pos_size * 2;
      }

      for (int32_t kTCOO = jTCOO; kTCOO < ((TCOO2_segend - 1) + 1); kTCOO++) {
        int32_t k = TCOO3_crd_accessor[(kTCOO * 1 + 0)];
        if (T_capacity <= kT) {
          T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL);
          T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
          T_capacity = T_capacity * 2;
        }
        T_vals_rw_accessor[Point<1>(kT)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
        if (T3_crd_size <= kT) {
          T3_crd = legionRealloc(ctx, runtime, T3_crd_parent, T3_crd, T3_crd_size * 2, FID_COORD);
          T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
          T3_crd_size = T3_crd_size * 2;
        }
        T3_crd_accessor[kT * 1] = k;
        kT = kT + 1;
      }

      T3_pos_accessor[Point<1>(jT)].lo = pT3_begin;
      T3_pos_accessor[Point<1>(jT)].hi = kT - 1;
      if (pT3_begin < kT) {
        if (T2_crd_size <= jT) {
          T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD);
          T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
          T2_crd_size = T2_crd_size * 2;
        }
        T2_crd_accessor[jT * 1] = j;
        jT = jT + 1;
      }
      jTCOO = TCOO2_segend;
    }

    T2_pos_accessor[Point<1>(i)].hi = (jT - pT2_begin) - 1;
    iTCOO = TCOO1_segend;
  }

  int64_t csT2 = 0;
  for (int32_t pT200 = 0; pT200 < (T2_pos_domain.bounds.hi[0] + 1); pT200++) {
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

void packLegionCOOToDDS(Context ctx, Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  int T1_dimension = T->dims[0];
  int T2_dimension = T->dims[1];
  RegionWrapper T3_pos = T->indices[2][0];
  RegionWrapper T3_crd = T->indices[2][1];
  auto T3_pos_parent = T->indicesParents[2][0];
  auto T3_crd_parent = T->indicesParents[2][1];
  RegionWrapper T_vals = T->vals;
  auto T_vals_parent = T->valsParent;
  auto T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
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

  int64_t T3Size = runtime->get_index_space_domain(ctx, get_index_space(T3_crd)).hi()[0] + 1;
  int64_t TCOO1Size = runtime->get_index_space_domain(ctx, get_index_space(TCOO1_crd)).hi()[0] + 1;

  DomainT<2> T3_pos_domain = runtime->get_index_space_domain(ctx, T3_pos.get_index_space());
  T3_pos = legionMalloc(ctx, runtime, T3_pos, T3_pos_parent, FID_RECT_1);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
  T3_pos_accessor[Point<2>(0, 0)] = Rect<1>(0, -1);
  for (int32_t pT30 = 0; pT30 < (T3_pos_domain.bounds.hi[0] + 1); pT30++) {
    for (int32_t pT31 = 0; pT31 < (T3_pos_domain.bounds.hi[1] + 1); pT31++) {
      T3_pos_accessor[Point<2>(pT30, pT31)] = Rect<1>(0, -1);
    }
  }
  int32_t T3_crd_size = 1;
  T3_crd = legionMalloc(ctx, runtime, T3_crd_parent, T3_crd_size, FID_COORD);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int32_t kT = 0;
  int32_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int32_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[(iTCOO * 1)];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[(TCOO1_segend * 1)] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int32_t iT = 0 * T1_dimension + i;
    int32_t jTCOO = iTCOO;
    int32_t pTCOO2_end = (TCOO1_segend - 1) + 1;

    while (jTCOO < pTCOO2_end) {
      int32_t j = TCOO2_crd_accessor[(jTCOO * 1 + 0)];
      int32_t TCOO2_segend = jTCOO + 1;
      while (TCOO2_segend < pTCOO2_end && TCOO2_crd_accessor[(TCOO2_segend * 1 + 0)] == j) {
        TCOO2_segend = TCOO2_segend + 1;
      }
      int32_t jT = iT * T2_dimension + j;
      int32_t pT3_begin = kT;

      for (int32_t kTCOO = jTCOO; kTCOO < ((TCOO2_segend - 1) + 1); kTCOO++) {
        int32_t k = TCOO3_crd_accessor[(kTCOO * 1 + 0)];
        if (T_capacity <= kT) {
          T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL);
          T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
          T_capacity = T_capacity * 2;
        }
        T_vals_rw_accessor[Point<1>(kT)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
        if (T3_crd_size <= kT) {
          T3_crd = legionRealloc(ctx, runtime, T3_crd_parent, T3_crd, T3_crd_size * 2, FID_COORD);
          T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
          T3_crd_size = T3_crd_size * 2;
        }
        T3_crd_accessor[kT * 1] = k;
        kT = kT + 1;
      }

      T3_pos_accessor[Point<2>(i, j)].hi = (kT - pT3_begin) - 1;
      jTCOO = TCOO2_segend;
    }
    iTCOO = TCOO1_segend;
  }

  int64_t csT3 = 0;
  for (int32_t pT300 = 0; pT300 < (T3_pos_domain.bounds.hi[0] + 1); pT300++) {
    for (int32_t pT310 = 0; pT310 < (T3_pos_domain.bounds.hi[1] + 1); pT310++) {
      int64_t numElemsT3 = T3_pos_accessor[Point<2>(pT300, pT310)].hi;
      T3_pos_accessor[Point<2>(pT300, pT310)].lo = csT3 + T3_pos_accessor[Point<2>(pT300, pT310)].lo;
      T3_pos_accessor[Point<2>(pT300, pT310)].hi = csT3 + T3_pos_accessor[Point<2>(pT300, pT310)].hi;
      csT3 = csT3 + (numElemsT3 + 1);
    }
  }
  T->indices[2][1] = getSubRegion(ctx, runtime, T3_crd_parent, Rect<1>(0, (kT - 1)));

  T->vals = getSubRegion(ctx, runtime, T_vals_parent, Rect<1>(0, (kT - 1)));

  runtime->unmap_region(ctx, T3_crd);
  runtime->unmap_region(ctx, T3_pos);
  runtime->unmap_region(ctx, TCOO1_crd);
  runtime->unmap_region(ctx, TCOO1_pos);
  runtime->unmap_region(ctx, TCOO2_crd);
  runtime->unmap_region(ctx, TCOO3_crd);
  runtime->unmap_region(ctx, TCOO_vals);
  runtime->unmap_region(ctx, T_vals);
}

void packLegionCOOToSDS(Context ctx, Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
  int T2_dimension = T->dims[1];
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

  int64_t T1Size = runtime->get_index_space_domain(ctx, get_index_space(T1_crd)).hi()[0] + 1;
  int64_t T3Size = runtime->get_index_space_domain(ctx, get_index_space(T3_crd)).hi()[0] + 1;
  int64_t TCOO1Size = runtime->get_index_space_domain(ctx, get_index_space(TCOO1_crd)).hi()[0] + 1;

  DomainT<1> T1_pos_domain = runtime->get_index_space_domain(ctx, T1_pos.get_index_space());
  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int32_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int32_t iT = 0;
  DomainT<2> T3_pos_domain = runtime->get_index_space_domain(ctx, T3_pos.get_index_space());
  int32_t T3_pos_size = 1;
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, T3_pos_size, FID_RECT_1);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
  T3_pos_accessor[Point<2>(0, 0)] = Rect<1>(0, -1);
  int32_t T3_crd_size = 1;
  T3_crd = legionMalloc(ctx, runtime, T3_crd_parent, T3_crd_size, FID_COORD);
  T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
  int32_t kT = 0;
  int32_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int32_t pT1_begin = iT;

  int32_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[(iTCOO * 1)];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[(TCOO1_segend * 1)] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    if (T3_pos_size <= iT) {
      T3_pos = legionRealloc(ctx, runtime, T3_pos_parent, T3_pos, T3_pos_size * 2, FID_RECT_1);
      T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
      T3_pos_size = T3_pos_size * 2;
    }
    for (int32_t pT31 = 0; pT31 < (T3_pos_domain.bounds.hi[1] + 1); pT31++) {
      T3_pos_accessor[Point<2>(iT, pT31)] = Rect<1>(0, -1);
    }

    int32_t jTCOO = iTCOO;
    int32_t pTCOO2_end = (TCOO1_segend - 1) + 1;

    while (jTCOO < pTCOO2_end) {
      int32_t j = TCOO2_crd_accessor[(jTCOO * 1 + 0)];
      int32_t TCOO2_segend = jTCOO + 1;
      while (TCOO2_segend < pTCOO2_end && TCOO2_crd_accessor[(TCOO2_segend * 1 + 0)] == j) {
        TCOO2_segend = TCOO2_segend + 1;
      }
      int32_t jT = iT * T2_dimension + j;
      int32_t pT3_begin = kT;

      for (int32_t kTCOO = jTCOO; kTCOO < ((TCOO2_segend - 1) + 1); kTCOO++) {
        int32_t k = TCOO3_crd_accessor[(kTCOO * 1 + 0)];
        if (T_capacity <= kT) {
          T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL);
          T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
          T_capacity = T_capacity * 2;
        }
        T_vals_rw_accessor[Point<1>(kT)] = TCOO_vals_ro_accessor[Point<1>(kTCOO)];
        if (T3_crd_size <= kT) {
          T3_crd = legionRealloc(ctx, runtime, T3_crd_parent, T3_crd, T3_crd_size * 2, FID_COORD);
          T3_crd_accessor = createAccessor<AccessorRWint32_t1>(T3_crd, FID_COORD);
          T3_crd_size = T3_crd_size * 2;
        }
        T3_crd_accessor[kT * 1] = k;
        kT = kT + 1;
      }

      T3_pos_accessor[Point<2>(iT, j)].hi = (kT - pT3_begin) - 1;
      jTCOO = TCOO2_segend;
    }
    if (T1_crd_size <= iT) {
      T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD);
      T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
      T1_crd_size = T1_crd_size * 2;
    }
    T1_crd_accessor[iT * 1] = i;
    iT = iT + 1;
    iTCOO = TCOO1_segend;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, (1 - 1)));
  T->indices[0][1] = getSubRegion(ctx, runtime, T1_crd_parent, Rect<1>(0, (iT - 1)));

  runtime->unmap_region(ctx, T3_pos);
  T3_pos = legionMalloc(ctx, runtime, T3_pos_parent, iT, FID_RECT_1);
  T3_pos_accessor = createAccessor<AccessorRWRect_1_2>(T3_pos, FID_RECT_1);
  int64_t csT3 = 0;
  for (int32_t pT30 = 0; pT30 < iT; pT30++) {
    for (int32_t pT310 = 0; pT310 < (T3_pos_domain.bounds.hi[1] + 1); pT310++) {
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

void packLegionCOOToDCSR(Context ctx, Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
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

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);

  int64_t T1Size = runtime->get_index_space_domain(ctx, get_index_space(T1_crd)).hi()[0] + 1;
  int64_t T2Size = runtime->get_index_space_domain(ctx, get_index_space(T2_crd)).hi()[0] + 1;
  int64_t TCOO1Size = runtime->get_index_space_domain(ctx, get_index_space(TCOO1_crd)).hi()[0] + 1;

  DomainT<1> T1_pos_domain = runtime->get_index_space_domain(ctx, T1_pos.get_index_space());
  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int32_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int32_t iT = 0;
  DomainT<1> T2_pos_domain = runtime->get_index_space_domain(ctx, T2_pos.get_index_space());
  int32_t T2_pos_size = 1;
  T2_pos = legionMalloc(ctx, runtime, T2_pos_parent, T2_pos_size, FID_RECT_1);
  T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
  T2_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int32_t T2_crd_size = 1;
  T2_crd = legionMalloc(ctx, runtime, T2_crd_parent, T2_crd_size, FID_COORD);
  T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
  int32_t jT = 0;
  int32_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);

  int32_t pT1_begin = iT;

  int32_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[(iTCOO * 1)];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[(TCOO1_segend * 1)] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    int32_t pT2_begin = jT;
    if (T2_pos_size <= iT) {
      T2_pos = legionRealloc(ctx, runtime, T2_pos_parent, T2_pos, T2_pos_size * 2, FID_RECT_1);
      T2_pos_accessor = createAccessor<AccessorRWRect_1_1>(T2_pos, FID_RECT_1);
      T2_pos_size = T2_pos_size * 2;
    }

    for (int32_t jTCOO = iTCOO; jTCOO < ((TCOO1_segend - 1) + 1); jTCOO++) {
      int32_t j = TCOO2_crd_accessor[(jTCOO * 1 + 0)];
      if (T_capacity <= jT) {
        T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_capacity * 2, FID_VAL);
        T_vals_rw_accessor = createAccessor<AccessorRWdouble1>(T_vals, FID_VAL);
        T_capacity = T_capacity * 2;
      }
      T_vals_rw_accessor[Point<1>(jT)] = TCOO_vals_ro_accessor[Point<1>(jTCOO)];
      if (T2_crd_size <= jT) {
        T2_crd = legionRealloc(ctx, runtime, T2_crd_parent, T2_crd, T2_crd_size * 2, FID_COORD);
        T2_crd_accessor = createAccessor<AccessorRWint32_t1>(T2_crd, FID_COORD);
        T2_crd_size = T2_crd_size * 2;
      }
      T2_crd_accessor[jT * 1] = j;
      jT = jT + 1;
    }

    T2_pos_accessor[Point<1>(iT)].lo = pT2_begin;
    T2_pos_accessor[Point<1>(iT)].hi = jT - 1;
    if (pT2_begin < jT) {
      if (T1_crd_size <= iT) {
        T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD);
        T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
        T1_crd_size = T1_crd_size * 2;
      }
      T1_crd_accessor[iT * 1] = i;
      iT = iT + 1;
    }
    iTCOO = TCOO1_segend;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, (1 - 1)));
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

void packLegionCOOToSD(Context ctx, Runtime* runtime, LegionTensor* T, LegionTensor* TCOO) {
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

  TCOO1_pos = legionMalloc(ctx, runtime, TCOO1_pos, TCOO1_pos_parent, FID_RECT_1);
  TCOO1_pos_accessor = createAccessor<AccessorRORect_1_1>(TCOO1_pos, FID_RECT_1);
  TCOO1_crd = legionMalloc(ctx, runtime, TCOO1_crd, TCOO1_crd_parent, FID_COORD);
  TCOO1_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO1_crd, FID_COORD);
  TCOO2_crd = legionMalloc(ctx, runtime, TCOO2_crd, TCOO2_crd_parent, FID_COORD);
  TCOO2_crd_accessor = createAccessor<AccessorROint32_t1>(TCOO2_crd, FID_COORD);
  TCOO_vals = legionMalloc(ctx, runtime, TCOO_vals, TCOO_vals_parent, FID_VAL);
  TCOO_vals_ro_accessor = createAccessor<AccessorROdouble1>(TCOO_vals, FID_VAL);

  int64_t T1Size = runtime->get_index_space_domain(ctx, get_index_space(T1_crd)).hi()[0] + 1;
  int64_t TCOO1Size = runtime->get_index_space_domain(ctx, get_index_space(TCOO1_crd)).hi()[0] + 1;

  DomainT<1> T1_pos_domain = runtime->get_index_space_domain(ctx, T1_pos.get_index_space());
  T1_pos = legionMalloc(ctx, runtime, T1_pos_parent, 1, FID_RECT_1);
  T1_pos_accessor = createAccessor<AccessorRWRect_1_1>(T1_pos, FID_RECT_1);
  T1_pos_accessor[Point<1>(0)] = Rect<1>(0, -1);
  int32_t T1_crd_size = 1;
  T1_crd = legionMalloc(ctx, runtime, T1_crd_parent, T1_crd_size, FID_COORD);
  T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
  int32_t iT = 0;
  int32_t T_capacity = 1;
  T_vals = legionMalloc(ctx, runtime, T_vals_parent, T_capacity, FID_VAL);
  T_vals_rw_accessor = createAccessor<AccessorRWdouble2>(T_vals, FID_VAL);

  int32_t pT1_begin = iT;

  int32_t iTCOO = TCOO1_pos_accessor[Point<1>(0)].lo;
  int32_t pTCOO1_end = TCOO1_pos_accessor[Point<1>(0)].hi + 1;

  while (iTCOO < pTCOO1_end) {
    int32_t i = TCOO1_crd_accessor[(iTCOO * 1)];
    int32_t TCOO1_segend = iTCOO + 1;
    while (TCOO1_segend < pTCOO1_end && TCOO1_crd_accessor[(TCOO1_segend * 1)] == i) {
      TCOO1_segend = TCOO1_segend + 1;
    }
    if (T_capacity <= iT) {
      int32_t T_vals_new_size = TACO_MAX(T_capacity * 2,(iT + 1));
      T_vals = legionRealloc(ctx, runtime, T_vals_parent, T_vals, T_vals_new_size, FID_VAL);
      T_vals_rw_accessor = createAccessor<AccessorRWdouble2>(T_vals, FID_VAL);
      T_capacity = T_vals_new_size;
    }
    for (int32_t j = 0; j < T2_dimension; j++) {
      T_vals_rw_accessor[Point<2>(iT, j)] = 0.0;
    }

    for (int32_t jTCOO = iTCOO; jTCOO < ((TCOO1_segend - 1) + 1); jTCOO++) {
      int32_t j = TCOO2_crd_accessor[(jTCOO * 1 + 0)];
      int32_t jT = iT * T2_dimension + j;
      T_vals_rw_accessor[Point<2>(iT, j)] = TCOO_vals_ro_accessor[Point<1>(jTCOO)];
    }
    if (T1_crd_size <= iT) {
      T1_crd = legionRealloc(ctx, runtime, T1_crd_parent, T1_crd, T1_crd_size * 2, FID_COORD);
      T1_crd_accessor = createAccessor<AccessorRWint32_t1>(T1_crd, FID_COORD);
      T1_crd_size = T1_crd_size * 2;
    }
    T1_crd_accessor[iT * 1] = i;
    iT = iT + 1;
    iTCOO = TCOO1_segend;
  }

  T1_pos_accessor[Point<1>(0)].lo = pT1_begin;
  T1_pos_accessor[Point<1>(0)].hi = iT - 1;

  T->indices[0][0] = getSubRegion(ctx, runtime, T1_pos_parent, Rect<1>(0, (1 - 1)));
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
void registerTacoTasks() {
}
