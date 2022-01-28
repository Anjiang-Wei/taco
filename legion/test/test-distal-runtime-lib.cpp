#include "test.h"
#include "taco_legion_header.h"
#include <random>

using namespace Legion;

template<typename T, int DIM, PrivilegeMode MODE>
using Accessor = FieldAccessor<MODE,T,DIM,coord_t,Realm::AffineAccessor<T,DIM,coord_t>>;

template<int DIM, typename T>
void ASSERT_REGION_EQ(Context ctx, Runtime* runtime, LogicalRegion r1, LogicalRegion r2, FieldID field) {
  SCOPED_TRACE("RegionEQ:");
  Domain d1 = runtime->get_index_space_domain(ctx, r1.get_index_space());
  Domain d2 = runtime->get_index_space_domain(ctx, r2.get_index_space());
  ASSERT_TRUE(d1.dense());
  ASSERT_TRUE(d2.dense());
  EXPECT_EQ(d1, d2);
  auto pr1 = legionMalloc(ctx, runtime, r1, r1, field, READ_ONLY);
  auto pr2 = legionMalloc(ctx, runtime, r2, r2, field, READ_ONLY);
  Accessor<T, DIM, READ_ONLY> acc1(pr1, field), acc2(pr2, field);
  for (PointInDomainIterator<DIM> itr(d1); itr(); itr++) {
    EXPECT_EQ(acc1[*itr], acc2[*itr]);
  }
  runtime->unmap_region(ctx, pr1);
  runtime->unmap_region(ctx, pr2);
}

TEST_F(DISTALRuntime, getPrevPoint) {
  Rect<3> bounds({0, 0, 0}, {9, 14, 19});
  EXPECT_EQ(getPreviousPoint(Point<3>({0, 0, 1}), bounds), Point<3>({0, 0, 0}));
  EXPECT_EQ(getPreviousPoint(Point<3>({0, 14, 0}), bounds), Point<3>({0, 13, 19}));
  EXPECT_EQ(getPreviousPoint(Point<3>({9, 0, 1}), bounds), Point<3>({9, 0, 0}));
  EXPECT_EQ(getPreviousPoint(Point<3>({9, 14, 1}), bounds), Point<3>({9, 14, 0}));
  EXPECT_EQ(getPreviousPoint(Point<3>({9, 14, 0}), bounds), Point<3>({9, 13, 19}));
  EXPECT_EQ(getPreviousPoint(Point<3>({9, 0, 0}), bounds), Point<3>({8, 14, 19}));
}

TEST_F(DISTALRuntime, SparseGatherProjection) {
  auto size = 5;
  auto ispace = runtime->create_index_space(ctx, Rect<1>(0, size - 1));
  auto fspace = runtime->create_field_space(ctx);
  {
    auto falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(sizeof(int32_t), FID_COORD);
  }
  auto reg = runtime->create_logical_region(ctx, ispace, fspace);
  auto ipart = runtime->create_equal_partition(ctx, ispace, ispace);
  auto lpart = runtime->get_logical_partition(ctx, reg, ipart);
  std::vector<int32_t> mapping({1, 4, 6, 7, 9});
  {
    auto preg = legionMalloc(ctx, runtime, reg, reg, FID_COORD, WRITE_ONLY);
    Accessor<int32_t, 1, WRITE_ONLY> acc(preg, FID_COORD);
    for (size_t i = 0; i < mapping.size(); i++) {
      acc[i] = mapping[i];
    }
    runtime->unmap_region(ctx, preg);
  }

  // These tests ensure that the correct dimension is projected into
  // when using the SparseGatherProjection, and that the remaining
  // dimensions are set to the right boundary values.
  auto xDim = 10, yDim = 20;
  auto target = runtime->create_index_space(ctx, Rect<2>({0, 0}, {xDim - 1, yDim - 1}));
  {
    auto part = SparseGatherProjection(0).apply(ctx, runtime, reg, lpart, FID_COORD, target);
    for (int i = 0; i < size; i++) {
      auto subspace = runtime->get_index_subspace(ctx, part, Color(i));
      auto bounds = runtime->get_index_space_domain(ctx, subspace);
      EXPECT_EQ(mapping[i], bounds.lo()[0]);
      EXPECT_EQ(mapping[i], bounds.hi()[0]);
      EXPECT_EQ(0, bounds.lo()[1]);
      EXPECT_EQ(yDim - 1, bounds.hi()[1]);
    }
  }
  {
    auto part = SparseGatherProjection(1).apply(ctx, runtime, reg, lpart, FID_COORD, target);
    for (int i = 0; i < size; i++) {
      auto subspace = runtime->get_index_subspace(ctx, part, Color(i));
      auto bounds = runtime->get_index_space_domain(ctx, subspace);
      EXPECT_EQ(mapping[i], bounds.lo()[1]);
      EXPECT_EQ(mapping[i], bounds.hi()[1]);
      EXPECT_EQ(0, bounds.lo()[0]);
      EXPECT_EQ(xDim - 1, bounds.hi()[0]);
    }
  }

  // Cleanup after ourselves.
  runtime->destroy_logical_region(ctx, reg);
  runtime->destroy_index_space(ctx, ispace);
  runtime->destroy_field_space(ctx, fspace);
}

TEST_F(DISTALRuntime, RectCompressedFinalizeYieldPos) {
  auto colorSpace = runtime->create_index_space(ctx, Rect<1>(0, 3));
  auto fspace = createFieldSpaceWithSize(ctx, runtime, FID_RECT_1, sizeof(Rect<1>));

  // 1-D test.
  {
    auto n = 10;
    auto ispace = runtime->create_index_space(ctx, Rect<1>{0, n - 1});
    auto reg = runtime->create_logical_region(ctx, ispace, fspace);
    std::vector<Rect<1>> data = {
      {2, 1},
      {2, 1},
      {4, 3},
      {4, 3},
      {6, 5},
      {8, 7},
      {8, 7},
      {10, 9},
      {10, 9},
      {13, 12},
    };
    std::vector<Rect<1>> expected = {
        {0, 1},
        {2, 1},
        {2, 3},
        {4, 3},
        {4, 5},
        {6, 7},
        {8, 7},
        {8, 9},
        {10, 9},
        {10, 12},
    };

    {
      auto preg = legionMalloc(ctx, runtime, reg, reg, FID_RECT_1, WRITE_ONLY);
      Accessor<Rect<1>, 1, WRITE_ONLY> acc(preg, FID_RECT_1);
      for (int i = 0; i < n; i++) {
        acc[i] = data[i];
      }
      runtime->unmap_region(ctx, preg);
    }
    auto ipart = runtime->create_equal_partition(ctx, ispace, colorSpace);
    auto lpart = runtime->get_logical_partition(ctx, reg, ipart);
    RectCompressedFinalizeYieldPositions launcher(ctx, runtime, reg, lpart, FID_RECT_1);
    runtime->execute_index_space(ctx, launcher).wait_all_results();

    {
      auto preg = legionMalloc(ctx, runtime, reg, reg, FID_RECT_1, READ_ONLY);
      Accessor<Rect<1>, 1, READ_ONLY> acc(preg, FID_RECT_1);
      for (int i = 0; i < n; i++) {
        EXPECT_EQ(acc[i], expected[i]);
      }
      runtime->unmap_region(ctx, preg);
    }

    runtime->destroy_logical_region(ctx, reg);
    runtime->destroy_index_space(ctx, ispace);
  }

  // TODO (rohany): Add some data into the 2-D and 3-D tests.

  // 2-D test.
  {
    auto n = 4;
    auto ispace = runtime->create_index_space(ctx, Rect<2>{{0, 0}, {n - 1, n - 1}});
    auto reg = runtime->create_logical_region(ctx, ispace, fspace);
    runtime->fill_field(ctx, reg, reg, FID_RECT_1, Rect<1>{0, 1});
    auto ipart = runtime->create_equal_partition(ctx, ispace, colorSpace);
    auto lpart = runtime->get_logical_partition(ctx, reg, ipart);
    RectCompressedFinalizeYieldPositions launcher(ctx, runtime, reg, lpart, FID_RECT_1);
    runtime->execute_index_space(ctx, launcher).wait_all_results();
    runtime->destroy_logical_region(ctx, reg);
    runtime->destroy_index_space(ctx, ispace);
  }

  // 3-D test.
  {
    auto n = 4;
    auto ispace = runtime->create_index_space(ctx, Rect<3>{{0, 0, 0}, {n - 1, n - 1, n - 1}});
    auto reg = runtime->create_logical_region(ctx, ispace, fspace);
    runtime->fill_field(ctx, reg, reg, FID_RECT_1, Rect<1>{0, 1});
    auto ipart = runtime->create_equal_partition(ctx, ispace, colorSpace);
    auto lpart = runtime->get_logical_partition(ctx, reg, ipart);
    RectCompressedFinalizeYieldPositions launcher(ctx, runtime, reg, lpart, FID_RECT_1);
    runtime->execute_index_space(ctx, launcher).wait_all_results();
    runtime->destroy_logical_region(ctx, reg);
    runtime->destroy_index_space(ctx, ispace);
  }

  runtime->destroy_field_space(ctx, fspace);
  runtime->destroy_index_space(ctx, colorSpace);
}

template<int DIM>
int32_t scanExpectedResult(Context ctx, Runtime* runtime, LogicalRegion posL, LogicalRegion nnzL) {
  Rect<DIM> bounds = runtime->get_index_space_domain(ctx, posL.get_index_space());
  auto posP = legionMalloc(ctx, runtime, posL, posL, FID_RECT_1, WRITE_ONLY);
  auto nnzP = legionMalloc(ctx, runtime, nnzL, nnzL, FID_VAL, READ_ONLY);
  Accessor<Rect<1>, DIM, WRITE_ONLY> posAcc(posP, FID_RECT_1);
  Accessor<int32_t, DIM, READ_ONLY> nnzAcc(nnzP, FID_VAL);
  for (PointInRectIterator<DIM> itr(bounds, false /* column_major_order */); itr(); itr++) {
    auto accPoint = *itr;
    if (accPoint == accPoint.ZEROES()) {
      posAcc[accPoint] = Rect<1>(0, (nnzAcc[accPoint] - 1));
    }
    else {
      int64_t prev = posAcc[getPreviousPoint(accPoint, bounds)].hi + 1;
      posAcc[accPoint] = Rect<1>(prev, ((prev + nnzAcc[accPoint]) - 1));
    }
  }
  auto result = posAcc[bounds.hi].hi + 1;
  runtime->unmap_region(ctx, posP);
  runtime->unmap_region(ctx, nnzP);
  return result;
}

template<int DIM>
void testRectCompressedSeqInsertEdges(
    Context ctx,
    Runtime* runtime,
    std::mt19937 mt,
    std::uniform_real_distribution<double> dist,
    Rect<DIM> regionBounds,
    IndexSpace colorSpace,
    FieldSpace rectfspace,
    FieldSpace i32fspace
) {
  auto ispace = runtime->create_index_space(ctx, regionBounds);
  auto pos = runtime->create_logical_region(ctx, ispace, rectfspace);
  auto posExpected = runtime->create_logical_region(ctx, ispace, rectfspace);
  auto nnz = runtime->create_logical_region(ctx, ispace, i32fspace);
  {
    auto preg = legionMalloc(ctx, runtime, nnz, nnz, FID_VAL, WRITE_ONLY);
    Accessor<int32_t, DIM, WRITE_ONLY> acc(preg, FID_VAL);
    for (PointInRectIterator<DIM> itr(regionBounds); itr(); itr++) {
      auto isZero = dist(mt) < 4;
      if (isZero) {
        acc[*itr] = 0;
      } else {
        acc[*itr] = dist(mt);
      }
    }
    runtime->unmap_region(ctx, preg);
  }

  auto result = RectCompressedGetSeqInsertEdges::compute(ctx, runtime, colorSpace, pos, FID_RECT_1, nnz, FID_VAL);
  auto expected = scanExpectedResult<DIM>(ctx, runtime, posExpected, nnz);
  ASSERT_REGION_EQ<DIM, Rect<1>>(ctx, runtime, posExpected, pos, FID_RECT_1);
  EXPECT_EQ(expected, result.scanResult);

  runtime->destroy_logical_region(ctx, pos);
  runtime->destroy_logical_region(ctx, nnz);
  runtime->destroy_index_space(ctx, ispace);
}

TEST_F(DISTALRuntime, RectCompressedGetSeqInsertEdges) {
  auto pieces = 5;
  auto colorSpace = runtime->create_index_space(ctx, Rect<1>(0, pieces - 1));
  auto rectfspace = createFieldSpaceWithSize(ctx, runtime, FID_RECT_1, sizeof(Rect<1>));
  auto i32fspace = createFieldSpaceWithSize(ctx, runtime, FID_VAL, sizeof(int32_t));

  // We'll initialize the nnz regions with random data.
  auto seed = initRandomDevice();
  std::mt19937 mt(seed);
  std::uniform_real_distribution<double> dist(1, 10);

  // Test each of 1-D, 2-D and 3-D.
  int n = 100;
  testRectCompressedSeqInsertEdges(ctx, runtime, mt, dist, Rect<1>(0, n - 1), colorSpace, rectfspace, i32fspace);
  testRectCompressedSeqInsertEdges(ctx, runtime, mt, dist, Rect<2>({0, 0}, {n - 1, n - 1}), colorSpace, rectfspace, i32fspace);
  testRectCompressedSeqInsertEdges(ctx, runtime, mt, dist, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}), colorSpace, rectfspace, i32fspace);

  runtime->destroy_index_space(ctx, colorSpace);
  runtime->destroy_field_space(ctx, rectfspace);
  runtime->destroy_field_space(ctx, i32fspace);
}

TEST_F(DISTALRuntime, RectCompressedCoordinatePartition) {
  auto pieces = 4;
  auto colorSpace = runtime->create_index_space(ctx, Rect<1>(0, pieces - 1));
  auto i32fspace = createFieldSpaceWithSize(ctx, runtime, FID_COORD, sizeof(int32_t));
  auto n = 10;
  auto ispace = runtime->create_index_space(ctx, Rect<1>{0, n - 1});
  auto reg = runtime->create_logical_region(ctx, ispace, i32fspace);
  {
    auto preg = legionMalloc(ctx, runtime, reg, reg, FID_COORD, WRITE_ONLY);
    Accessor<int32_t, 1, WRITE_ONLY> acc(preg, FID_COORD);
    acc[0] = 0;
    acc[1] = 1;
    acc[2] = 2;
    acc[3] = 5;
    acc[4] = 8;
    acc[5] = 10;
    acc[6] = 15;
    acc[7] = 16;
    acc[8] = 20;
    acc[9] = 21;
    runtime->unmap_region(ctx, preg);
  }
  DomainPointColoring coloring;
  coloring[0] = Rect<1>(0, 5);
  coloring[1] = Rect<1>(6, 11);
  coloring[2] = Rect<1>(12, 17);
  coloring[3] = Rect<1>(18, 23);

  auto result = RectCompressedCoordinatePartition::apply(ctx, runtime, reg, reg, FID_COORD, coloring, colorSpace).get_index_partition();
  std::vector<Rect<1>> expected = {
      {0, 3},
      {4, 5},
      {6, 7},
      {8, 9},
  };
  for (int i = 0; i < pieces; i++) {
    auto subspace = runtime->get_index_subspace(ctx, result, i);
    auto bounds = runtime->get_index_space_domain(ctx, subspace).bounds<1, Legion::coord_t>();
    EXPECT_EQ(expected[i], bounds);
  }
}