#include "test.h"
#include "taco_legion_header.h"

using namespace Legion;

template<typename T, int DIM, PrivilegeMode MODE>
using Accessor = FieldAccessor<MODE,T,DIM,coord_t,Realm::AffineAccessor<T,DIM,coord_t>>;

TEST_F(DISTALRuntime, getPrevPoint) {
  Rect<3> bounds({0, 0, 0}, {9, 14, 19});
  EXPECT_EQ(getPreviousPoint(Point<3>({0, 0, 1}), bounds), Point<3>({0, 0, 0}));
  EXPECT_EQ(getPreviousPoint(Point<3>({0, 14, 0}), bounds), Point<3>({0, 13, 19}));
  EXPECT_EQ(getPreviousPoint(Point<3>({9, 0, 1}), bounds), Point<3>({9, 0, 0}));
  EXPECT_EQ(getPreviousPoint(Point<3>({9, 14, 1}), bounds), Point<3>({9, 14, 0}));
  EXPECT_EQ(getPreviousPoint(Point<3>({9, 14, 0}), bounds), Point<3>({9, 13, 19}));
  EXPECT_EQ(getPreviousPoint(Point<3>({9, 0, 0}), bounds), Point<3>({8, 14, 19}));
}

TEST_F(DISTALRuntime, sparseGatherProjection) {
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

TEST_F(DISTALRuntime, rectCompressedFinalizeYieldPos) {
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

