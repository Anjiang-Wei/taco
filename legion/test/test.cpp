#include "test.h"
#include "taco_legion_header.h"

using namespace Legion;

// Initialize the DISTALRuntime static variables.
Context DISTALRuntime::ctx = Context();
Runtime* DISTALRuntime::runtime = nullptr;

void DISTALRuntime::SetUpTestCase() {
  // Register all of the tasks that we may need for tests.
  registerTacoRuntimeLibTasks();

  // Start the runtime in the background.
  Legion::Runtime::start(my_argc, my_argv, true /* backgroun */);
  DISTALRuntime::runtime = Legion::Runtime::get_runtime();
  DISTALRuntime::ctx = runtime->begin_implicit_task(TID_TOP_LEVEL, 0 /* mapper_id */, Legion::Processor::LOC_PROC, "top_level", true /* control replicable */);
}

void DISTALRuntime::TearDownTestCase() {
  // Close up the runtime.
  runtime->finish_implicit_task(ctx);
  Legion::Runtime::wait_for_shutdown();
}

// Redeclarations of the extern variables so that everything links.
int my_argc;
char** my_argv;

int main(int argc, char **argv) {
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
