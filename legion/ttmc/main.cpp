#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

LogicalPartition partition3Tensor(Context ctx, Runtime* runtime, LogicalRegion A, int32_t pieces);
LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t pieces);
LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t pieces);
LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t pieces);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition APartition, LogicalPartition BPartition, LogicalPartition CPartition);


void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto args = runtime->get_input_args();
  int n = -1;
  int pieces = -1;

  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-pieces") == 0) {
      pieces = atoi(args.argv[++i]);
      continue;
    }
  }
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (pieces == -1) {
    std::cout << "Please provide the number of pieces with -pieces." << std::endl;
    return;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);

  auto aISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  auto bISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  // Make C "replicated".
  auto cISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {(pieces * n) - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, aISpace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, bISpace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, cISpace, fspace); runtime->attach_name(C, "C");

  // Partition the tensors.
  auto aPart = partition3Tensor(ctx, runtime, A, pieces);
  auto bPart = partition3Tensor(ctx, runtime, B, pieces);

  // Create a "replicated" partition of C.
  DomainPointColoring cColoring;
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(pieces - 1);
  DomainT<1> dom(Rect<1>{lowerBound, upperBound});
  for (PointInDomainIterator<1> itr(dom); itr(); itr++) {
    int idx = *itr;
    auto start = Point<2>(idx * n, 0);
    auto end = Point<2>((idx + 1) * n - 1, n - 1);
    cColoring[*itr] = Rect<2>(start, end);
  }
  auto cIndexPart = runtime->create_index_partition(ctx, cISpace, dom, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto cPart = runtime->get_logical_partition(ctx, C, cIndexPart);

  tacoFill<valType>(ctx, runtime, A, aPart, 0);
  tacoFill<valType>(ctx, runtime, B, bPart, 1);
  tacoFill<valType>(ctx, runtime, C, cPart, 1);

  // Run the program once for verification.
  computeLegion(ctx, runtime, A, B, C, aPart, bPart, cPart);
  tacoValidate<valType>(ctx, runtime, A, aPart, valType(n));

  std::vector<size_t> times;
  benchmarkAsyncCall(ctx, runtime, times, [&]() {
    for (int i = 0; i < 10; i++) {
      computeLegion(ctx, runtime, A, B, C, aPart, bPart, cPart);
    }
  });

  // Get the GFLOPS per node.
  auto avgTime = double(times[0]) / 10.f;
  auto flopCount = getTTMCFLOPCount(n, n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));
}

#include "../dsl_mapper.cc"

#define TACO_MAIN2(FillType) \
  int main(int argc, char **argv) { \
    Runtime::set_top_level_task_id(TID_TOP_LEVEL); \
    {               \
      TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level"); \
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
      registrar.set_replicable();   \
      Runtime::preregister_task_variant<top_level_task>(registrar, "top_level"); \
    }                       \
    if (TACO_FEATURE_OPENMP) {               \
      TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level"); \
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC)); \
      registrar.set_replicable();   \
      Runtime::preregister_task_variant<top_level_task>(registrar, "top_level"); \
    }                       \
    registerTACOFillTasks<FillType>();             \
    registerTACOValidateTasks<FillType>();             \
    bool dslmapper = false; \
    for (int i = 1; i < argc; i++) { \
      if (strcmp(argv[i], "-dslmapper") == 0) { \
        register_mappers(); \
        dslmapper = true; \
        break; \
      } \
    } \
    if (dslmapper) { \
      register_mappers(); \
    } \
    else \
    { \
      Runtime::add_registration_callback(register_taco_mapper); \
    } \
    initCUDA(); \
    registerTacoTasks();    \
    Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor()); \
    return Runtime::start(argc, argv);             \
  }

TACO_MAIN2(valType)

// TACO_MAIN(valType)
