#include "legion.h"
#include <unistd.h>

using namespace Legion;


enum FieldIDs {
  FID_VAL,
};

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_WORKER,
};

void workerGPU(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto workerIdx = *(int32_t*)task->args;
  std::cout << "starting worker " << workerIdx << std::endl;

  // Now let's allocate a buffer to yield control back to the runtime.
  double initVal = 0.f;
  // Allocate and fill 4 GB of data.
  DeferredBuffer<double, 1> buf(Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, 50000000)), &initVal);
  // DeferredBuffer<double, 1> buf(Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, 50000000)));
  double* inter = buf.ptr(0);

  usleep(500 * 1000);
  
  std::cout << "finishing worker " << workerIdx << std::endl;
  // return 0;
}

void workerCPU(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto workerIdx = *(int32_t*)task->args;
  std::cout << "starting worker " << workerIdx << std::endl;

  // Now let's allocate a buffer to yield control back to the runtime.
  double initVal = 0.f;
  // Allocate and fill 4 GB of data.
  DeferredBuffer<double, 1> buf(Memory::Kind::SYSTEM_MEM, DomainT<1>(Rect<1>(0, 500000)), &initVal);
  // DeferredBuffer<double, 1> buf(Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, 50000000)), &initVal);
  // DeferredBuffer<double, 1> buf(Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, 50000000)));
  double* inter = buf.ptr(0);

  usleep(500 * 1000);
  
  std::cout << "finishing worker " << workerIdx << std::endl;
  // return 0;
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create a region.
  auto fspace = runtime->create_field_space(ctx);
  Legion::FieldAllocator allocator = runtime->create_field_allocator(ctx, fspace);
  allocator.allocate_field(sizeof(double), FID_VAL);
  auto ispace = runtime->create_index_space(ctx, Rect<1>(0, 9));
  auto reg = runtime->create_logical_region(ctx, ispace, fspace);

  Future f;
  for (int i = 0; i < 10; i++) {
    int32_t idx = i;
    TaskLauncher launcher(TID_WORKER, TaskArgument(&idx, sizeof(int32_t)));
    launcher.add_region_requirement(RegionRequirement(reg, LEGION_REDOP_SUM_FLOAT64, SIMULTANEOUS, reg).add_field(FID_VAL));
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
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_WORKER, "worker");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    // Runtime::preregister_task_variant<int,worker>(registrar, "worker");
    Runtime::preregister_task_variant<worker>(registrar, "worker");
  }
  {
    TaskVariantRegistrar registrar(TID_WORKER, "worker");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    // Runtime::preregister_task_variant<int,worker>(registrar, "worker");
    Runtime::preregister_task_variant<worker>(registrar, "worker");
  }
  return Runtime::start(argc, argv);
}
