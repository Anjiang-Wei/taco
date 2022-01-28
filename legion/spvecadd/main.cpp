#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "legion_utils.h"
#include "legion_string_utils.h"
#include "error.h"

#include "taco-generated.h"

using namespace Legion;
using namespace Legion::Mapping;

typedef double valType;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::string vecBFile, vecCFile;
  bool dump = false;
  int n = 10, pieces = 0, warmup = 5;
  Realm::CommandLineParser parser;
  parser.add_option_string("-vecb", vecBFile);
  parser.add_option_string("-vecc", vecCFile);
  parser.add_option_bool("-dump", dump);
  parser.add_option_int("-n", n);
  parser.add_option_int("-pieces", pieces);
  parser.add_option_int("-warmup", warmup);
  auto args = Runtime::get_input_args();
  taco_uassert(parser.parse_command_line(args.argc, args.argv)) << "Parse failure.";
  taco_uassert(!vecBFile.empty() && !vecCFile.empty()) << "Provide input vectors with -vecb and -vecc.";

  // Figure out how many pieces to chop up the data into.
  if (pieces == 0) {
    pieces = getNumPieces(ctx, runtime);
    taco_uassert(pieces != 0) << "Please provide a number of pieces to split into with -pieces. Unable to automatically find.";
  }

  LegionTensor b, c; ExternalHDF5LegionTensor bex, cex;
  std::tie(b, bex) = loadLegionTensorFromHDF5File(ctx, runtime, vecBFile, {Sparse});
  std::tie(c, cex) = loadLegionTensorFromHDF5File(ctx, runtime, vecCFile, {Sparse});

  auto a = createDenseTensor<1, valType>(ctx, runtime, {b.dims[0]}, FID_VAL);
  runtime->fill_field(ctx, a.vals, a.valsParent, FID_VAL, valType(0));

  partitionPackForcomputeLegion pack = partitionForcomputeLegion(ctx, runtime, &a, &b, &c, pieces);

  // Benchmark the computation.
  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) { runtime->fill_field(ctx, a.vals, a.valsParent, FID_VAL, valType(0)); }
    computeLegion(ctx, runtime, &a, &b, &c, &pack, pieces);
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    printLegionTensor<valType>(ctx, runtime, a);
  }
  bex.destroy(ctx, runtime);
  cex.destroy(ctx, runtime);
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
  registerTacoRuntimeLibTasks();
  Runtime::add_registration_callback(register_taco_mapper);
  Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor());
  return Runtime::start(argc, argv);
}
