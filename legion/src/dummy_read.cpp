#include "dummy_read.h"
#include "legion.h"
#include "task_ids.h"
#include "mappers/default_mapper.h"
#include "taco_mapper.h"

using namespace Legion;

LogicalPartition launchDummyRead(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion reg, Legion::FieldID fid, bool wait, bool untrack, bool cpuOnly) {
  auto numNodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  auto numCPUMemsPerNode = runtime->select_tunable_value(ctx, TACOMapper::TUNABLE_LOCAL_CPU_MEMS).get<size_t>();
  auto numPieces = numNodes * numCPUMemsPerNode;
  auto partSpace = runtime->create_index_space(ctx, Rect<1>(0, numPieces - 1));
  auto ipart = runtime->create_equal_partition(ctx, reg.get_index_space(), partSpace);
  auto lpart = runtime->get_logical_partition(ctx, reg, ipart);
  launchDummyReadOverPartition(ctx, runtime, reg, lpart, fid, runtime->get_index_space_domain(ctx, partSpace), wait, untrack, cpuOnly);
  return lpart;
}

void launchDummyReadOverPartition(Context ctx, Runtime* runtime, LogicalRegion reg, LogicalPartition part, FieldID fid, Domain launchDim, bool wait, bool untrack, bool cpuOnly) {
  IndexTaskLauncher launcher(TID_DUMMY_READ_REGION, launchDim, TaskArgument(), ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(part, 0, READ_ONLY, EXCLUSIVE, reg).add_field(fid));
  if (untrack) {
    launcher.tag |= TACOMapper::UNTRACK_VALID_REGIONS;
  }
  if (cpuOnly) {
    launcher.tag |= TACOMapper::MAP_TO_OMP_OR_LOC;
  }
  auto fut = runtime->execute_index_space(ctx, launcher);
  if (wait) {
    fut.wait_all_results();
  }
}

void dummyReadTask(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {}

// Guard against repeated initializations of the dummy read tasks.
static bool registeredDummyTasks = false;
void registerDummyReadTasks() {
  if (!registeredDummyTasks) {
    {
      TaskVariantRegistrar registrar(TID_DUMMY_READ_REGION, "dummyReadTask");
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
      Runtime::preregister_task_variant<dummyReadTask>(registrar, "dummyReadTask");
    }
    {
      TaskVariantRegistrar registrar(TID_DUMMY_READ_REGION, "dummyReadTask");
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
      Runtime::preregister_task_variant<dummyReadTask>(registrar, "dummyReadTask");
    }
    {
      TaskVariantRegistrar registrar(TID_DUMMY_READ_REGION, "dummyReadTask");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      Runtime::preregister_task_variant<dummyReadTask>(registrar, "dummyReadTask");
    }
    registeredDummyTasks = true;
  }
}
