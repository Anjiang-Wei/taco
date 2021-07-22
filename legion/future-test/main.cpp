#include "legion.h"
#include "legion_utils.h"

using namespace Legion;

const int TID_WORK = 101;

void work(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::cout << "Executing work" << std::endl;
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  int n = 10;
  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<double>(ctx, runtime, fspace);
  auto aISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto bISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  auto cISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto dISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, aISpace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, bISpace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, cISpace, fspace); runtime->attach_name(C, "C");
  auto D = runtime->create_logical_region(ctx, dISpace, fspace); runtime->attach_name(D, "D");
  // Fill all the fields.
  runtime->fill_field(ctx, A, A, FID_VAL, double(0.0));
  runtime->fill_field(ctx, B, B, FID_VAL, double(0.0));
  runtime->fill_field(ctx, C, C, FID_VAL, double(0.0));
  runtime->fill_field(ctx, D, D, FID_VAL, double(0.0));

  // Make a few dummy regions.
  Future f;
  for (int i = 0; i < 10; i++) {
    TaskLauncher launcher(TID_WORK, TaskArgument());
    RegionRequirement AReq = RegionRequirement(A, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(A));
    AReq.add_field(FID_VAL);
    RegionRequirement BReq = RegionRequirement(get_logical_region(B), READ_ONLY, EXCLUSIVE, get_logical_region(B));
    BReq.add_field(FID_VAL);
    RegionRequirement CReq = RegionRequirement(C, READ_ONLY, EXCLUSIVE, get_logical_region(C));
    CReq.add_field(FID_VAL);
    RegionRequirement DReq = RegionRequirement(D, READ_ONLY, EXCLUSIVE, get_logical_region(D));
    DReq.add_field(FID_VAL);

    launcher.add_region_requirement(AReq);
    launcher.add_region_requirement(BReq);
    launcher.add_region_requirement(CReq);
    launcher.add_region_requirement(DReq);

    if (f.valid()) {
      launcher.add_future(f);
    }

    f = runtime->execute_task(ctx, launcher);
  }
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_WORK, "work");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<work>(registrar, "work");
  }
  return Runtime::start(argc, argv);
}