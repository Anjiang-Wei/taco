#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"

using namespace Legion;

typedef double valType;

// Defined by the generated TACO code.
void registerTacoTasks();

// Partitioning statements.
LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX);
LogicalPartition partitionLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY, int32_t gridZ);
LogicalPartition partitionLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t gridY);
LogicalPartition partitionLegionD(Context ctx, Runtime* runtime, LogicalRegion D, int32_t gridZ);

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t gridX);
void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, LogicalPartition aPart, int32_t gridX, int32_t gridY, int32_t gridZ);

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY, int32_t gridZ);
void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, LogicalPartition bPart, int32_t gridX, int32_t gridY, int32_t gridZ);

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t gridY);
void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, LogicalPartition cPart, int32_t gridY, int32_t gridX, int32_t gridZ);

std::vector<LogicalPartition> partitionForplaceLegionD(Context ctx, Runtime* runtime, LogicalRegion D, int32_t gridZ);
void placeLegionD(Context ctx, Runtime* runtime, LogicalRegion D, LogicalPartition dPart, int32_t gridZ, int32_t gridX, int32_t gridY);

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalRegion D, int32_t gridX, int32_t gridY, int32_t gridZ);
void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalRegion D, LogicalPartition APartition, LogicalPartition BPartition, LogicalPartition CPartition, LogicalPartition DPartition, int32_t gridX, int32_t gridY, int32_t gridZ);

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto args = runtime->get_input_args();
  int n = -1;
  int gx = -1;
  int gy = -1;
  int gz = -1;

  // Parse input args.
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-n") == 0) {
      n = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gx") == 0) {
      gx = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gy") == 0) {
      gy = atoi(args.argv[++i]);
      continue;
    }
    if (strcmp(args.argv[i], "-gz") == 0) {
      gz = atoi(args.argv[++i]);
      continue;
    }
  }
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
  if (gz == -1) {
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto aISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto bISpace = runtime->create_index_space(ctx, Rect<3>({0, 0, 0}, {n - 1, n - 1, n - 1}));
  auto cISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto dISpace = runtime->create_index_space(ctx, Rect<2>({0, 0}, {n - 1, n - 1}));
  auto A = runtime->create_logical_region(ctx, aISpace, fspace); runtime->attach_name(A, "A");
  auto B = runtime->create_logical_region(ctx, bISpace, fspace); runtime->attach_name(B, "B");
  auto C = runtime->create_logical_region(ctx, cISpace, fspace); runtime->attach_name(C, "C");
  auto D = runtime->create_logical_region(ctx, dISpace, fspace); runtime->attach_name(D, "D");

  // Partition all of the tensors.
  auto aPart = partitionLegionA(ctx, runtime, A, gx);
  auto bPart = partitionLegionB(ctx, runtime, B, gx, gy, gz);
  auto cPart = partitionLegionC(ctx, runtime, C, gy);
  auto dPart = partitionLegionD(ctx, runtime, D, gz);

  // Partitions for placement operators.
  auto pAPart = partitionForplaceLegionA(ctx, runtime, A, gx)[0];
  auto pBPart = partitionForplaceLegionB(ctx, runtime, B, gx, gy, gz)[0];
  auto pCPart = partitionForplaceLegionC(ctx, runtime, C, gy)[0];
  auto pDPart = partitionForplaceLegionD(ctx, runtime, D, gz)[0];

  // Partitions for computation.
  auto compParts = partitionForcomputeLegion(ctx, runtime, A, B, C, D, gx, gy, gz);

  std::vector<size_t> times;
  for (int i = 0; i < 11; i++) {
    tacoFill<valType>(ctx, runtime, A, aPart, 0);
    tacoFill<valType>(ctx, runtime, B, bPart, 1);
    tacoFill<valType>(ctx, runtime, C, cPart, 1);
    tacoFill<valType>(ctx, runtime, D, dPart, 1);

    placeLegionA(ctx, runtime, A, pAPart, gx, gy, gz);
    placeLegionB(ctx, runtime, B, pBPart, gx, gy, gz);
    placeLegionC(ctx, runtime, C, pCPart, gy, gx, gz);
    placeLegionD(ctx, runtime, D, pDPart, gz, gx, gy);

    auto bench = [&]() {
      computeLegion(ctx, runtime, A, B, C, D, compParts[0], compParts[1], compParts[2], compParts[3], gx, gy, gz);
      // Run the A placement routine again to force a reduction into the right place.
      placeLegionA(ctx, runtime, A, pAPart, gx, gy, gz);
    };

    if (i == 0) {
      bench();
    } else {
      benchmark(ctx, runtime, times, bench);
    }
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = getMTTKRPFLOPCount(n, n, n, n);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));
  tacoValidate<valType>(ctx, runtime, A, aPart, valType(n * n));
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
