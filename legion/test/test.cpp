#include "test.h"
#include "taco_legion_header.h"
#include <random>

using namespace Legion;

// Initialize the DISTALRuntime static variables.
Context DISTALRuntime::ctx = Context();
Runtime* DISTALRuntime::runtime = nullptr;

void register_mapper(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs) {
  auto proc = *local_procs.begin();
  runtime->replace_default_mapper(
      new DISTALRuntimeTestMapper(runtime->get_mapper_runtime(), machine, proc, "DISTALRuntimeTestMapper"),
      Processor::NO_PROC);
}

void DISTALRuntime::SetUpTestCase() {
  // Register all of the tasks that we may need for tests.
  registerTacoRuntimeLibTasks();

  // Register the test mapper.
  Runtime::add_registration_callback(register_mapper);

  // Start the runtime in the background.
  Legion::Runtime::start(my_argc, my_argv, true /* background */);
  DISTALRuntime::runtime = Legion::Runtime::get_runtime();
  DISTALRuntime::ctx = runtime->begin_implicit_task(TID_TOP_LEVEL, 0 /* mapper_id */, Legion::Processor::LOC_PROC, "top_level", true /* control replicable */);
}

void DISTALRuntime::TearDownTestCase() {
  // Close up the runtime.
  runtime->finish_implicit_task(ctx);
  Legion::Runtime::wait_for_shutdown();
}

DISTALRuntimeTestMapper::DISTALRuntimeTestMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine &machine,
                                                 const Legion::Processor &local, const char *name)
    : Mapping::DefaultMapper(rt, machine, local, name) {}

Legion::Memory DISTALRuntimeTestMapper::default_policy_select_target_memory(Legion::Mapping::MapperContext ctx,
                                                                            Legion::Processor target_proc,
                                                                            const Legion::RegionRequirement &req,
                                                                            Legion::MemoryConstraint mc) {
  // Return the first CPU memory. If we have OpenMP though, then use NUMA memories.
#ifdef REALM_USE_OPENMP
  return Machine::MemoryQuery(this->machine).only_kind(Realm::Memory::SOCKET_MEM).first();
#else
  return Machine::MemoryQuery(this->machine).only_kind(Realm::Memory::SYSTEM_MEM).first();
#endif
}

// Redeclarations of the extern variables so that everything links.
int my_argc;
char** my_argv;

int main(int argc, char **argv) {
  // If we have OpenMP-enabled Legion, we'll include some extra arguments to the runtime to
  // enable use of the OpenMP processors and memories. We have to do it this way because ctest
  // does not allow for passing custom arguments to tests.
#ifdef REALM_USE_OPENMP
  std::vector<std::string> extraArguments = {
    "-ll:onuma", "1",
    "-ll:ocpu", "1",
    "-ll:othr", "10",
    "-ll:nsize", "15G",
    "-ll:ncsize", "0",
    "-lg:eager_alloc_percentage", "50",
  };
  std::vector<char*> newArgv;
  for (int i = 0; i < argc; i++) {
    newArgv.push_back(argv[i]);
  }
  for (auto it : extraArguments) {
    newArgv.push_back(const_cast<char*>(it.c_str()));
  }
  argc = newArgv.size();
  argv = newArgv.data();
#endif

  // If there is just one argument and it is not a gtest option, then filter
  // the tests using that argument surrounded by wildcards.
  std::string filter;
  if (argc == 2 && std::string(argv[argc-1]).substr(0,2) != "--") {
    filter = std::string(argv[1]);
    filter = "*" + filter + "*";
    filter = std::string("--gtest_filter=") + filter;
    argv[1] = (char*)filter.c_str();
  }
  ::testing::InitGoogleTest(&argc, argv);
  my_argc = argc;
  my_argv = argv;
  return RUN_ALL_TESTS();
}

unsigned initRandomDevice() {
  std::random_device rd;
  auto seed = rd();
  std::cout << "USING RANDOM SEED: " << seed << std::endl;
  return seed;
}
