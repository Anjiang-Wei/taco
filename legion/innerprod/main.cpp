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
  int n = 10, pieces = 1;
  Realm::CommandLineParser parser;
  parser.add_option_int("-n", n);
  parser.add_option_int("-pieces", pieces);
  parser.parse_command_line(args.argc, args.argv);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto B = createDenseTensor<3, valType>(ctx, runtime, {n, n, n}, FID_VAL);
  auto C = createDenseTensor<3, valType>(ctx, runtime, {n, n, n}, FID_VAL);
  auto parts = partitionForcomputeLegion(ctx, runtime, &B, &C, pieces);

  tacoFill<valType>(ctx, runtime, B.vals, parts.bPartition.valsPartition, valType(1));
  tacoFill<valType>(ctx, runtime, C.vals, parts.cPartition.valsPartition, valType(1));

  // Run one iteration to warm up the system.
  computeLegion(ctx, runtime, &B, &C, &parts, pieces);

  std::vector<size_t> times;
  for (int i = 0; i < 10; i++) {
    benchmark(ctx, runtime, times, [&]() {
      auto res = computeLegion(ctx, runtime, &B, &C, &parts, pieces);
      taco_iassert(res == valType(valType(n) * valType(n) * valType(n)));
    });
  }

  size_t elems = [](size_t n) { return 2 * n * n * n; }(n);
  size_t bytes = elems * sizeof(valType);
  double gbytes = double(bytes) / 1e9;
  auto avgTimeS = (double(average(times))) / 1e3;
  double bw = gbytes / (avgTimeS);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GB/s BW per node: %lf.\n", nodes, bw / double(nodes));
}

TACO_MAIN(valType)
