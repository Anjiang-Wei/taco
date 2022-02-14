#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "mappers/default_mapper.h"
#include "legion_utils.h"
#include "legion_string_utils.h"
#include "error.h"
#include "taco-generated.h"

using namespace Legion;
typedef double valType;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::string fileNameB, fileNameC;
  bool dump = false, dds = false;
  int n = 10, pieces = 0, warmup = 5;
  Realm::CommandLineParser parser;
  parser.add_option_string("-tensorB", fileNameB);
  parser.add_option_string("-tensorC", fileNameC);
  parser.add_option_bool("-dump", dump);
  parser.add_option_int("-n", n);
  parser.add_option_int("-pieces", pieces);
  parser.add_option_int("-warmup", warmup);
  parser.add_option_bool("-dds", dds);
  auto args = Runtime::get_input_args();
  taco_uassert(parser.parse_command_line(args.argc, args.argv)) << "Parse failure.";
  taco_uassert(!fileNameB.empty()) << "Provide the B matrix with -tensorB";
  taco_uassert(!fileNameC.empty()) << "Provide the C matrix with -tensorC";

  // Figure out how many pieces to chop up the data into.
  if (pieces == 0) {
    pieces = getNumPieces(ctx, runtime);
    taco_uassert(pieces != 0) << "Please provide a number of pieces to split into with -pieces. Unable to automatically find.";
  }

  LegionTensor B, C; ExternalHDF5LegionTensor Bex, Cex;
  LegionTensorFormat format = {Dense, Sparse, Sparse};
  if (dds) {
    format = {Dense, Dense, Sparse};
  }
  std::tie(B, Bex) = loadLegionTensorFromHDF5File(ctx, runtime, fileNameB, format);
  std::tie(C, Cex) = loadLegionTensorFromHDF5File(ctx, runtime, fileNameC, format);

  partitionPackForcomputeLegion pack;
  partitionPackForcomputeLegionDDS packDDS;
  if (dds) {
    // TODO (rohany): Experiment with the decomposition here.
    packDDS = partitionForcomputeLegionDDS(ctx, runtime, &B, &C, pieces, 1);
  } else {
    pack = partitionForcomputeLegion(ctx, runtime, &B, &C, pieces);
  }

  valType a = 0;
  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dds) {
      // TODO (rohany): Experiment with the decomposition here.
      a = computeLegionDDS(ctx, runtime, &B, &C, &packDDS, pieces, 1);
    } else {
      a = computeLegion(ctx, runtime, &B, &C, &pack, pieces);
    }
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Result: %lf\n", a);
  }
  Bex.destroy(ctx, runtime);
  Cex.destroy(ctx, runtime);
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
  registerTacoRuntimeLibTasks();
  Runtime::add_registration_callback(register_taco_mapper);
  Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor());
  return Runtime::start(argc, argv);
}
