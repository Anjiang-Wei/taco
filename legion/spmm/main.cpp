#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "mappers/default_mapper.h"
#include "legion_utils.h"
#include "legion_string_utils.h"

using namespace Legion;

typedef double valType;

struct partitionPackForcomputeLegion;
partitionPackForcomputeLegion* partitionForcomputeLegion(Context ctx, Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, int32_t gx);
void computeLegion(Context ctx, Runtime* runtime, LegionTensor* A, LegionTensor* B, LegionTensor* C, partitionPackForcomputeLegion* partitionPack, int32_t gx);
void registerTacoTasks();

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::string csrFileName;
  bool dump = false;
  int n = 10, pieces = 4, warmup = 5;
  Realm::CommandLineParser parser;
  parser.add_option_string("-tensor", csrFileName);
  parser.add_option_bool("-dump", dump);
  parser.add_option_int("-n", n);
  parser.add_option_int("-pieces", pieces);
  parser.add_option_int("-warmup", warmup);
  auto args = Runtime::get_input_args();
  assert(parser.parse_command_line(args.argc, args.argv));
  assert(!csrFileName.empty());

  auto B = loadLegionTensorFromHDF5File(ctx, runtime, csrFileName, {Dense, Sparse});
  // TODO (rohany): What should the value of the j dimension be? A(i, j) = B(i, k) * C(k, j).
  auto jDim = B.dims[1];
  auto A = createDenseTensor<2, valType>(ctx, runtime, {B.dims[0], jDim}, FID_VAL);
  auto C = createDenseTensor<2, valType>(ctx, runtime, {B.dims[1], jDim}, FID_VAL);
  runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0));
  runtime->fill_field(ctx, C.vals, C.valsParent, FID_VAL, valType(1));

  auto pack = partitionForcomputeLegion(ctx, runtime, &A, &B, &C, pieces);

  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) { runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0)); }
    computeLegion(ctx, runtime, &A, &B, &C, pack, pieces);
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    printLegionTensor<valType>(ctx, runtime, A, {Dense, Dense});
  }

  // Delete the partition pack.
  delete pack;
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
