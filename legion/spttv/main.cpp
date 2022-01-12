#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "legion_utils.h"
#include "legion_string_utils.h"
#include "error.h"
#include "taco-generated.h"

using namespace Legion;
typedef double valType;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  int pieces = 0, n = 10, warmup = 5;
  bool dump = false, pos = false, partialPos = false;
  std::string input;
  Realm::CommandLineParser parser;
  parser.add_option_string("-tensor", input);
  parser.add_option_int("-pieces", pieces);
  parser.add_option_int("-n", n);
  parser.add_option_int("-warmup", warmup);
  parser.add_option_bool("-dump", dump);
  parser.add_option_bool("-pos", pos);
  parser.add_option_bool("-partial_pos", partialPos);
  auto args = Runtime::get_input_args();
  taco_iassert(parser.parse_command_line(args.argc, args.argv)) << "Parse failed.";
  taco_uassert(!input.empty()) << "Provide input with -tensor.";
  taco_uassert(!(pos && partialPos)) << "Cannot do pos and partial_pos.";

  // Figure out how many pieces to chop up the data into.
  if (pieces == 0) {
    pieces = getNumPieces(ctx, runtime);
    taco_uassert(pieces != 0) << "Please provide a number of pieces to split into with -pieces. Unable to automatically find.";
  }

  LegionTensor B; ExternalHDF5LegionTensor Bex;
  std::tie(B, Bex) = loadLegionTensorFromHDF5File(ctx, runtime, input, {Dense, Sparse, Sparse});
  auto A = copyNonZeroStructure<valType>(ctx, runtime, {Dense, Sparse}, B);
  auto c = createDenseTensor<1, valType>(ctx, runtime, {B.dims[2]}, FID_VAL);
  runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0));
  runtime->fill_field(ctx, c.vals, c.valsParent, FID_VAL, valType(1));

  // Create an row-wise partition of A to force reduction operations to get run.
  auto eqIspace = runtime->create_index_space(ctx, Rect<1>(0, pieces - 1));
  auto eqDomain = runtime->get_index_space_domain(ctx, eqIspace);
  auto AEqIndexPart = runtime->create_equal_partition(ctx, A.vals.get_index_space(), eqIspace);
  auto AEqLogPart = runtime->get_logical_partition(ctx, A.vals, AEqIndexPart);

  // Partition the computation.
  partitionPackForcomputeLegionDSS dssPart;
  partitionPackForcomputeLegionDSSPosSplit dssPosPart;
  partitionPackForcomputeLegionDSSPartialPosSplit dssPartialPosPart;
  if (pos) {
    dssPosPart = partitionForcomputeLegionDSSPosSplit(ctx, runtime, &A, &B, &c, pieces);
  } else if (partialPos) {
    dssPartialPosPart = partitionForcomputeLegionDSSPartialPosSplit(ctx, runtime, &A, &B, &c, pieces);
  } else {
    dssPart = partitionForcomputeLegionDSS(ctx, runtime, &A, &B, &c, pieces);
  }

  // Run the benchmark.
  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) { runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0)); }
    if (pos) {
      computeLegionDSSPosSplit(ctx, runtime, &A, &B, &c, &dssPosPart, pieces);
      launchDummyReadOverPartition(ctx, runtime, A.vals, AEqLogPart, FID_VAL, eqDomain);
    } else if (partialPos) {
      computeLegionDSSPartialPosSplit(ctx, runtime, &A, &B, &c, &dssPartialPosPart, pieces);
      launchDummyReadOverPartition(ctx, runtime, A.vals, AEqLogPart, FID_VAL, eqDomain);
    } else {
      computeLegionDSS(ctx, runtime, &A, &B, &c, &dssPart, pieces);
    }
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    printLegionTensor<valType>(ctx, runtime, A);
  }
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
