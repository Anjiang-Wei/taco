#include "test.h"
#include "taco_legion_header.h"

using namespace Legion;

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
  Memory sysMem = Machine::MemoryQuery(Machine::get_machine())
                          .only_kind(Memory::SYSTEM_MEM)
                          .first();

  // Just attach the mapping itself as the base region.
  AttachLauncher launcher(EXTERNAL_INSTANCE, reg, reg);
  launcher.attach_array_soa(mapping.data(), false /* columnMajor */, {FID_COORD}, sysMem);
  auto preg = runtime->attach_external_resource(ctx, launcher);

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
  runtime->detach_external_resource(ctx, preg);
  runtime->destroy_logical_region(ctx, reg);
  runtime->destroy_index_space(ctx, ispace);
  runtime->destroy_field_space(ctx, fspace);
}

