#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "legion_utils.h"
#include "legion_string_utils.h"

using namespace Legion;
typedef double valType;


// Forward declarations.
struct partitionPackForcomputeLegionDSS;
struct partitionPackForcomputeLegionDDS;
partitionPackForcomputeLegionDSS* partitionForcomputeLegionDSS(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t gx);
void computeLegionDSS(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDSS* partitionPack, int32_t gx);
partitionPackForcomputeLegionDDS* partitionForcomputeLegionDDS(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t gx, int32_t gy);
void computeLegionDDS(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionDDS* partitionPack, int32_t gx, int32_t gy);
void registerTacoTasks();

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  int bDenseDims = 1, gx = 1, gy = 1, n = 10, warmup = 5;
  bool dump = false;
  std::string input;
  Realm::CommandLineParser parser;
  parser.add_option_int("-bdd", bDenseDims);
  parser.add_option_string("-tensor", input);
  parser.add_option_int("-gx", gx);
  parser.add_option_int("-gy", gy);
  parser.add_option_int("-n", n);
  parser.add_option_int("-warmup", warmup);
  parser.add_option_bool("-dump", dump);
  auto args = Runtime::get_input_args();
  assert(parser.parse_command_line(args.argc, args.argv));
  assert(!input.empty());

  LegionTensor B;
  if (bDenseDims == 1) {
    B = loadLegionTensorFromHDF5File(ctx, runtime, input, {Dense, Sparse, Sparse});
  } else {
    assert(bDenseDims == 2);
    B = loadLegionTensorFromHDF5File(ctx, runtime, input, {Dense, Dense, Sparse});
  }
  auto A = createDenseTensor<2, valType>(ctx, runtime, {B.dims[0], B.dims[1]}, FID_VAL);
  auto c = createDenseTensor<1, valType>(ctx, runtime, {B.dims[2]}, FID_VAL);
  runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0));
  runtime->fill_field(ctx, c.vals, c.valsParent, FID_VAL, valType(1));

  // Partition the computation.
  partitionPackForcomputeLegionDDS* ddsPart;
  partitionPackForcomputeLegionDSS* dssPart;
  if (bDenseDims == 1) {
    dssPart = partitionForcomputeLegionDSS(ctx, runtime, &A, &B, &c, gx);
  } else {
    ddsPart = partitionForcomputeLegionDDS(ctx, runtime, &A, &B, &c, gx, gy);
  }

  // Run the benchmark.
  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) { runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0)); }
    if (bDenseDims == 1) {
      computeLegionDSS(ctx, runtime, &A, &B, &c, dssPart, gx);
    } else {
      computeLegionDDS(ctx, runtime, &A, &B, &c, ddsPart, gx, gy);
    }
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    printLegionTensor<valType>(ctx, runtime, A, {Dense, Dense});
  }

  if (bDenseDims == 1) {
    delete dssPart;
  } else {
    delete ddsPart;
  }
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  registerHDF5UtilTasks();
  registerTacoTasks();
  return Runtime::start(argc, argv);
}
