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
  std::string fileNameB, fileNameC, fileNameD;
  bool dump = false;
  int n = 10, pieces = 0, warmup = 5;
  Realm::CommandLineParser parser;
  parser.add_option_string("-tensorB", fileNameB);
  parser.add_option_string("-tensorC", fileNameC);
  parser.add_option_string("-tensorD", fileNameD);
  parser.add_option_bool("-dump", dump);
  parser.add_option_int("-n", n);
  parser.add_option_int("-pieces", pieces);
  parser.add_option_int("-warmup", warmup);
  auto args = Runtime::get_input_args();
  taco_uassert(parser.parse_command_line(args.argc, args.argv)) << "Parse failure.";
  taco_uassert(!fileNameB.empty()) << "Provide the B matrix with -tensorB";
  taco_uassert(!fileNameC.empty()) << "Provide the C matrix with -tensorC";
  taco_uassert(!fileNameC.empty()) << "Provide the D matrix with -tensorD";

  // Figure out how many pieces to chop up the data into.
  if (pieces == 0) {
    pieces = getNumPieces(ctx, runtime);
    taco_uassert(pieces != 0) << "Please provide a number of pieces to split into with -pieces. Unable to automatically find.";
  }

  LegionTensor B, C, D; ExternalHDF5LegionTensor Bex, Cex, Dex;
  std::tie(B, Bex) = loadLegionTensorFromHDF5File(ctx, runtime, fileNameB, {Dense, Sparse});
  std::tie(C, Cex) = loadLegionTensorFromHDF5File(ctx, runtime, fileNameC, {Dense, Sparse});
  std::tie(D, Dex) = loadLegionTensorFromHDF5File(ctx, runtime, fileNameD, {Dense, Sparse});
  auto A = createSparseTensorForPack<double>(ctx, runtime, {Dense, Sparse}, B.dims, FID_RECT_1, FID_COORD, FID_VAL);

  auto pack = partitionForcomputeLegion(ctx, runtime, &A, &B, &C, &D, pieces);

  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) {
      runtime->fill_field(ctx, A.indices[1][0], A.indicesParents[1][0], FID_RECT_1, Rect<1>({0, 0}));
      runtime->fill_field(ctx, A.indices[1][1], A.indicesParents[1][1], FID_COORD, int32_t(0));
      runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0));
    }
    computeLegion(ctx, runtime, &A, &B, &C, &D, &pack, pieces);
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    printLegionTensor<valType>(ctx, runtime, A);
  }
  Bex.destroy(ctx, runtime);
  Cex.destroy(ctx, runtime);
  Dex.destroy(ctx, runtime);
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
