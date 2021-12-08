#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "legion_utils.h"
#include "legion_string_utils.h"
#include "error.h"

using namespace Legion;
typedef double valType;


// Forward declarations.
struct partitionPackForcomputeLegionDSS;
struct partitionPackForcomputeLegionDDS;
partitionPackForcomputeLegionDSS* partitionForcomputeLegionDSS(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t gx);
void computeLegionDSS(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSS* partitionPack, int32_t gx);
partitionPackForcomputeLegionDDS* partitionForcomputeLegionDDS(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t gx, int32_t gy);
void computeLegionDDS(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDDS* partitionPack, int32_t gx, int32_t gy);
struct partitionPackForcomputeLegionDSSPosSplit;
partitionPackForcomputeLegionDSSPosSplit* partitionForcomputeLegionDSSPosSplit(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t gx);
void computeLegionDSSPosSplit(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSSPosSplit* partitionPack, int32_t gx);

void registerTacoTasks();

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  int bDenseDims = 1, gx = 1, gy = 1, n = 10, warmup = 5;
  bool dump = false, pos = false;
  std::string input;
  Realm::CommandLineParser parser;
  parser.add_option_int("-bdd", bDenseDims);
  parser.add_option_string("-tensor", input);
  parser.add_option_int("-gx", gx);
  parser.add_option_int("-gy", gy);
  parser.add_option_int("-n", n);
  parser.add_option_int("-warmup", warmup);
  parser.add_option_bool("-dump", dump);
  parser.add_option_bool("-pos", pos);
  auto args = Runtime::get_input_args();
  taco_iassert(parser.parse_command_line(args.argc, args.argv)) << "Parse failed.";
  taco_uassert(!input.empty()) << "Provide input with -tensor.";
  if (pos) {
    taco_uassert(bDenseDims == 1) << "If pos, bdd must equal 1";
  }

  LegionTensor B; ExternalHDF5LegionTensor Bex;
  if (bDenseDims == 1) {
    std::tie(B, Bex) = loadLegionTensorFromHDF5File(ctx, runtime, input, {Dense, Sparse, Sparse});
  } else {
    taco_uassert(bDenseDims == 2);
    std::tie(B, Bex) = loadLegionTensorFromHDF5File(ctx, runtime, input, {Dense, Dense, Sparse});
  }
  auto A = createDenseTensor<2, valType>(ctx, runtime, {B.dims[0], B.dims[1]}, FID_VAL);
  auto c = createDenseTensor<1, valType>(ctx, runtime, {B.dims[2]}, FID_VAL);
  runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0));
  runtime->fill_field(ctx, c.vals, c.valsParent, FID_VAL, valType(1));

  // Create an row-wise partition of A to force reduction operations to get run.
  auto eqIspace = runtime->create_index_space(ctx, Rect<1>(0, gx - 1));
  auto eqDomain = runtime->get_index_space_domain(ctx, eqIspace);
  auto AEqIndexPart = runtime->create_equal_partition(ctx, A.vals.get_index_space(), eqIspace);
  auto AEqLogPart = runtime->get_logical_partition(ctx, A.vals, AEqIndexPart);

  // Partition the computation.
  partitionPackForcomputeLegionDDS* ddsPart = nullptr;
  partitionPackForcomputeLegionDSS* dssPart = nullptr;
  partitionPackForcomputeLegionDSSPosSplit* dssPosPart = nullptr;
  if (bDenseDims == 1 && pos) {
    dssPosPart = partitionForcomputeLegionDSSPosSplit(ctx, runtime, &A, &B, &c, gx);
  } else if (bDenseDims == 1) {
    dssPart = partitionForcomputeLegionDSS(ctx, runtime, &A, &B, &c, gx);
  } else {
    ddsPart = partitionForcomputeLegionDDS(ctx, runtime, &A, &B, &c, gx, gy);
  }

  // Run the benchmark.
  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) { runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0)); }
    if (bDenseDims == 1 && pos) {
      computeLegionDSSPosSplit(ctx, runtime, &A, &B, &c, dssPosPart, gx);
      launchDummyReadOverPartition(ctx, runtime, A.vals, AEqLogPart, FID_VAL, eqDomain);
    } else if (bDenseDims == 1) {
      computeLegionDSS(ctx, runtime, &A, &B, &c, dssPart, gx);
    } else {
      computeLegionDDS(ctx, runtime, &A, &B, &c, ddsPart, gx, gy);
    }
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    printLegionTensor<valType>(ctx, runtime, A, {Dense, Dense});
  }

  if (dssPosPart != nullptr) delete dssPosPart;
  if (dssPart != nullptr) delete dssPart;
  if (ddsPart != nullptr) delete ddsPart;
  Bex.destroy(ctx, runtime);
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  if (TACO_FEATURE_OPENMP) {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  registerHDF5UtilTasks();
  registerTacoTasks();
  registerDummyReadTasks();
  Runtime::add_registration_callback(register_taco_mapper);
  Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor());
  return Runtime::start(argc, argv);
}
