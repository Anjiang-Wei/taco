#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"
#include "realm/cmdline.h"
#include "taco-generated.h"

using namespace Legion;

typedef double valType;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int n = -1, gx = -1, gy = -1, px = -1, py = -1;
  Realm::CommandLineParser parser;
  parser.add_option_int("-n", n);
  parser.add_option_int("-gx", gx);
  parser.add_option_int("-gy", gy);
  parser.add_option_int("-px", px);
  parser.add_option_int("-py", py);
  parser.parse_command_line(args.argc, args.argv);
  if (n == -1) {
    std::cout << "Please provide an input matrix size with -n." << std::endl;
    return;
  }
  if (gx == -1) {
    std::cout << "Please provide a grid x size with -gx." << std::endl;
    return;
  }
  if (gy == -1) {
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }
  if (px == -1) {
    px = gx;
  }
  if (py == -1) {
    py = gy;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto A = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);
  auto B = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);
  auto C = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);

  // These partitions are disjoint, so we can fill over them.
  auto aPart = partitionForplaceLegionA(ctx, runtime, &A, px, py);
  auto bPart = partitionForplaceLegionB(ctx, runtime, &B, px, py);
  auto cPart = partitionForplaceLegionC(ctx, runtime, &C, px, py);

  // Get partitions for the computation.
  auto parts = partitionForcomputeLegion(ctx, runtime, &A, &B, &C, gx, gy);

  std::vector<size_t> times;
  // Run the benchmark several times.
  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A.vals, aPart.aPartition.valsPartition, 0);
    tacoFill<valType>(ctx, runtime, B.vals, bPart.bPartition.valsPartition, 1);
    tacoFill<valType>(ctx, runtime, C.vals, cPart.cPartition.valsPartition, 1);

    // Place the tensors.
    placeLegionA(ctx, runtime, &A, &aPart, gx, gy);
    placeLegionB(ctx, runtime, &B, &bPart, gx, gy);
    placeLegionC(ctx, runtime, &C, &cPart, gx, gy);

    // Compute on the tensors.
    benchmark(ctx, runtime, times, [&]() { computeLegion(ctx, runtime, &A, &B, &C, &parts, gx, gy); });
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = getGEMMFLOPCount(n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));

  // The result should be equal to 1.
  tacoValidate<valType>(ctx, runtime, A.vals, aPart.aPartition.valsPartition, valType(n));
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
    register_mappers(); \
    initCUDA(); \
    registerTacoTasks();    \
    Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor()); \
    return Runtime::start(argc, argv);             \
  }

TACO_MAIN2(valType)
/*
    registerTACOFillTasks<FillType>();             \
    registerTACOValidateTasks<FillType>();             \
    Runtime::add_registration_callback(register_taco_mapper);    \

    register_mappers();
  */
