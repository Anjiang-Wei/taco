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
  std::string fileName;
  bool dump = false;
  // The j-dimension if the computation will commonly have a small value
  // that is divisible by 32, as per Stephen and Chang-wan.
  int n = 10, pieces = 0, warmup = 5, lDim = 32;
  Realm::CommandLineParser parser;
  parser.add_option_string("-tensor", fileName);
  parser.add_option_bool("-dump", dump);
  parser.add_option_int("-n", n);
  parser.add_option_int("-pieces", pieces);
  parser.add_option_int("-warmup", warmup);
  parser.add_option_int("-ldim", lDim);
  auto args = Runtime::get_input_args();
  taco_uassert(parser.parse_command_line(args.argc, args.argv)) << "Parse failure.";
  taco_uassert(!fileName.empty()) << "Provide a tensor with -tensor";

  // Figure out how many pieces to chop up the data into.
  if (pieces == 0) {
    pieces = getNumPieces(ctx, runtime);
    taco_uassert(pieces != 0) << "Please provide a number of pieces to split into with -pieces. Unable to automatically find.";
  }

  LegionTensor B; ExternalHDF5LegionTensor Bex;
  std::tie(B, Bex) = loadLegionTensorFromHDF5File(ctx, runtime, fileName, {Dense, Sparse, Sparse});
  auto A = createDenseTensor<2, valType>(ctx, runtime, {B.dims[0], lDim}, FID_VAL);
  auto C = createDenseTensor<2, valType>(ctx, runtime, {B.dims[1], lDim}, FID_VAL);
  auto D = createDenseTensor<2, valType>(ctx, runtime, {B.dims[2], lDim}, FID_VAL);
  runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0));
  runtime->fill_field(ctx, C.vals, C.valsParent, FID_VAL, valType(1));
  runtime->fill_field(ctx, D.vals, D.valsParent, FID_VAL, valType(1));

  auto pack = partitionForcomputeLegion(ctx, runtime, &A, &B, &C, &D, pieces);

  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) { runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0)); }
    computeLegion(ctx, runtime, &A, &B, &C, &D, &pack, pieces);
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
  Runtime::add_registration_callback(register_taco_mapper);
  Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor());
  return Runtime::start(argc, argv);
}
