#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"
#include "taco-generated.h"
#include "realm/cmdline.h"

using namespace Legion;
typedef double valType;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  int n = 10, gx = 1, gy = 1, gz = 1;
  Realm::CommandLineParser parser;
  parser.add_option_int("-n", n);
  parser.add_option_int("-gx", gx);
  parser.add_option_int("-gy", gy);
  parser.add_option_int("-gz", gz);
  auto args = runtime->get_input_args();
  parser.parse_command_line(args.argc, args.argv);

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);

  auto A = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);
  auto B = createDenseTensor<3, valType>(ctx, runtime, {n, n, n}, FID_VAL);
  auto C = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);
  auto D = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);

  // Partition all of the tensors.
  // TODO (rohany): This needs to return LegionTensorPartitions.
  //  It seems like there should be a different way to do this though.
  auto aPart = partitionLegionA(ctx, runtime, &A, gx);
  auto bPart = partitionLegionB(ctx, runtime, &B, gx, gy, gz);
  auto cPart = partitionLegionC(ctx, runtime, &C, gy);
  auto dPart = partitionLegionD(ctx, runtime, &D, gz);

  // Partitions for placement operators.
  auto pAPart = partitionForplaceLegionA(ctx, runtime, &A, gx);
  auto pBPart = partitionForplaceLegionB(ctx, runtime, &B, gx, gy, gz);
  auto pCPart = partitionForplaceLegionC(ctx, runtime, &C, gy);
  auto pDPart = partitionForplaceLegionD(ctx, runtime, &D, gz);

  // Partitions for computation.
  auto compPart = partitionForcomputeLegion(ctx, runtime, &A, &B, &C, &D, gx, gy, gz);

  std::vector<size_t> times;
  for (int i = 0; i < 11; i++) {
    tacoFill<valType>(ctx, runtime, A.vals, aPart, 0);
    tacoFill<valType>(ctx, runtime, B.vals, bPart, 1);
    tacoFill<valType>(ctx, runtime, C.vals, cPart, 1);
    tacoFill<valType>(ctx, runtime, D.vals, dPart, 1);

    placeLegionA(ctx, runtime, &A, &pAPart, gx, gy, gz);
    placeLegionB(ctx, runtime, &B, &pBPart, gx, gy, gz);
    placeLegionC(ctx, runtime, &C, &pCPart, gy, gx, gz);
    placeLegionD(ctx, runtime, &D, &pDPart, gz, gx, gy);

    auto bench = [&]() {
      computeLegion(ctx, runtime, &A, &B, &C, &D, &compPart, gx, gy, gz);
      // Run the A placement routine again to force a reduction into the right place.
      placeLegionA(ctx, runtime, &A, &pAPart, gx, gy, gz);
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
  tacoValidate<valType>(ctx, runtime, A.vals, aPart, valType(n * n));
}

TACO_MAIN(valType)
