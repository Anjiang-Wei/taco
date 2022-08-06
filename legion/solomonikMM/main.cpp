#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"
#include "realm/cmdline.h"

#ifdef TACO_USE_CUDA
#include "taco-generated.cuh"
#else
#include "taco-generated.h"
#endif

using namespace Legion;

typedef double valType;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1;
  int rpoc = -1;
  int rpoc3 = -1;
  int c = -1;
  Realm::CommandLineParser parser;
  parser.add_option_int("-n", n);
  parser.add_option_int("-rpoc", rpoc);
  parser.add_option_int("-rpoc3", rpoc3);
  parser.add_option_int("-c", c);
  parser.parse_command_line(args.argc, args.argv);
  // TODO (rohany): Improve these messages.
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (rpoc == -1) {
    std::cout << "Please provide a rpoc." << std::endl;
    return;
  }
  if (rpoc3 == -1) {
    std::cout << "Please provide a rpoc3." << std::endl;
    return;
  }
  if (c == -1) {
    std::cout << "Please provide a c." << std::endl;
    return;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto A = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);
  auto B = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);
  auto C = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);


  // Partition all tensors.
  auto aPart = partitionForplaceLegionA(ctx, runtime, &A, rpoc);
  auto bPart = partitionForplaceLegionB(ctx, runtime, &B, rpoc);
  auto cPart = partitionForplaceLegionC(ctx, runtime, &C, rpoc);

  auto parts = partitionForcomputeLegion(ctx, runtime, &A, &B, &C, rpoc, c, rpoc3);

  std::vector<size_t> times;
  // Run the benchmark several times.
  for (int i = 0; i < 11; i++) {
    // TODO (rohany): We could potentially eliminate these fills to place the data right where
    //  we need it just like Johnson's algorithm. This would allow us to use larger values of c
    //  as well which might improve performance.
    tacoFill<valType>(ctx, runtime, A.vals, aPart.APartition.valsPartition, 0);
    tacoFill<valType>(ctx, runtime, B.vals, bPart.BPartition.valsPartition, 1);
    tacoFill<valType>(ctx, runtime, C.vals, cPart.CPartition.valsPartition, 1);

    // Place the tensors.
    placeLegionA(ctx, runtime, &A, &aPart, rpoc, c);
    placeLegionB(ctx, runtime, &B, &bPart, rpoc, c);
    placeLegionC(ctx, runtime, &C, &cPart, rpoc, c);

    auto bench = [&]() {
      computeLegion(ctx, runtime, &A, &B, &C, &parts, rpoc, c, rpoc3);
      placeLegionA(ctx, runtime, &A, &aPart, rpoc, c);
    };

    if (i == 0) {
      bench();
    } else {
      benchmark(ctx, runtime, times, bench);
    }
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = getGEMMFLOPCount(n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));

  tacoValidate<valType>(ctx, runtime, A.vals, aPart.APartition.valsPartition, valType(n));
}

#include "my_mapper.cc"

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
