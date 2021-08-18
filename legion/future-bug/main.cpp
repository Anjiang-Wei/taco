#include "legion.h"
#include "mappers/default_mapper.h"
#include <unistd.h>

using namespace Legion;
using namespace Legion::Mapping;

enum FieldIDs {
  FID_VAL,
};

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_WORKER,
};

// Let's write a custom mapper to try and buffer the tasks.
class MyMapper : public Mapping::DefaultMapper {
public:
  MyMapper(MapperRuntime* rt, Machine machine, Processor local) : DefaultMapper(rt, machine, local) {}
  void select_tasks_to_map(const MapperContext ctx,
                           const SelectMappingInput& input,
                                 SelectMappingOutput& output) override {
    MapperEvent returnEvent;
    // Find the depth of the deepest task
    int max_depth = 0;
    for (std::list<const Task*>::const_iterator it =
        input.ready_tasks.begin(); it != input.ready_tasks.end(); it++)
    {
      int depth = (*it)->get_depth();
      if (depth > max_depth)
        max_depth = depth;
    }
    unsigned count = 0;
    // Only schedule tasks from the max depth in any pass
    for (std::list<const Task*>::const_iterator it =
        input.ready_tasks.begin(); (count < max_schedule_count) &&
                                   (it != input.ready_tasks.end()); it++)
    {
      auto task = *it;
      bool schedule = true;
      if (task->task_id == TID_WORKER) {
        // TODO (rohany): Is target proc valid to do here? What is the ordering
        //  of mapper calls between select_task_options and this?
        auto waitEvent = this->queue[task->target_proc];
        if (waitEvent.exists()) {
          // If there is an event, then wait behind it.
          schedule = false;
          returnEvent = waitEvent;
        } else {
          // Otherwise, create an event to listen on.
          this->queue[task->target_proc] = this->runtime->create_mapper_event(ctx);
        }
      }
      if (schedule && (*it)->get_depth() == max_depth)
      {
        output.map_tasks.insert(*it);
        count++;
      }
    }
    if (output.map_tasks.size() == 0) {
      assert(returnEvent.exists());
      output.deferral_event = returnEvent;
    }
  }

  void map_task(const MapperContext ctx,
                const Task& task,
                const MapTaskInput& input,
                MapTaskOutput& output) override {
    DefaultMapper::map_task(ctx, task, input, output);
    if (task.task_id == TID_WORKER) {
      output.task_prof_requests.add_measurement<ProfilingMeasurements::OperationStatus>();
    }
  }

  void report_profiling(const MapperContext ctx,
                        const Task& task,
                        const TaskProfilingInfo& input) override {
    assert(task.task_id == TID_WORKER);
    auto prof = input.profiling_responses.get_measurement<ProfilingMeasurements::OperationStatus>();
    assert(prof->result == Realm::ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY);
    auto event = this->queue[task.target_proc];
    assert(event.exists());
    this->queue[task.target_proc] = MapperEvent();
    this->runtime->trigger_mapper_event(ctx, event);
  }

  MapperSyncModel get_mapper_sync_model() const override {
    // TODO (rohany): If I wanted to loosen the mapper serialization model here, where would I
    // need locks? Probably around the queue itself? Might also be easier to just
    // take a lock on entry to these overridden functions.
    return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
  }

  // Map from processor to MapperEvents.
  std::map<Processor, MapperEvent> queue;
};

void worker(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto workerIdx = *(int32_t*)task->args;
  std::cout << "starting worker " << workerIdx << std::endl;
  double initVal = 0.f;
  DeferredBuffer<double, 1> buf(Memory::Kind::SYSTEM_MEM, DomainT<1>(Rect<1>(0, 500000)), &initVal);
  usleep(500 * 1000);
  std::cout << "finishing worker " << workerIdx << std::endl;
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

void register_mapper(Machine machine, Runtime* runtime, const std::set<Processor> &local_procs) {
  MyMapper* m = new MyMapper(runtime->get_mapper_runtime(), machine, *local_procs.begin());
  runtime->replace_default_mapper(m);
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
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<worker>(registrar, "worker");
  }
  Runtime::add_registration_callback(register_mapper);
  return Runtime::start(argc, argv);
}
