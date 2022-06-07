#include "legion.h"
#include "legion_utils.h"
#include "realm/cmdline.h"
#include "taco-generated.h"
#include "taco_mapper.h"

using namespace Legion;

using valType = double;

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions, Context ctx,
                    Runtime* runtime) {
  // Create the regions.
  auto args = runtime->get_input_args();
  int no = -1, nv = -1, gx = -1, gy = -1;
  bool use_tblis = false;
  Realm::CommandLineParser parser;
  parser.add_option_int("-nv", nv);
  parser.add_option_int("-no", no);
  parser.add_option_int("-gx", gx);
  parser.add_option_int("-gy", gy);
  parser.add_option_bool("-tblis", use_tblis);
  parser.parse_command_line(args.argc, args.argv);
  if (no == -1 || nv == -1) {
    std::cout << "Please provide input sizes with -no, -nv." << std::endl;
    return;
  }
  if (gx == -1) {
    std::cout << "Please provide a grid x size with -gx." << std::endl;
    return;
  }
  if (gy == -1) {
    std::cout << "Please provide a gris y size with -gy." << std::endl;
    return;
  }

  initCuBLAS(ctx, runtime);

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);
  auto A = createDenseTensor<4, valType>(ctx, runtime, {nv, nv, no, no}, FID_VAL);
  auto B = createDenseTensor<4, valType>(ctx, runtime, {nv, nv, no, no}, FID_VAL);
  auto C = createDenseTensor<4, valType>(ctx, runtime, {nv, nv, no, no}, FID_VAL);

  // These partitions are disjoint, so we can fill over them.
  auto aPart = partitionForplaceLegionA(ctx, runtime, &A, gx, gy);
  auto bPart = partitionForplaceLegionB(ctx, runtime, &B, gx, gy);
  auto cPart = partitionForplaceLegionC(ctx, runtime, &C, gx, gy);

  // Get partitions for the computation.
  auto partsTblis =
      partitionForcomputeLegionTblis(ctx, runtime, &A, &B, &C, gx, gy);
  auto partsNestedOMP =
      partitionForcomputeLegionNestedOMP(ctx, runtime, &A, &B, &C, gx, gy);

  std::vector<size_t> times;
  // Run the benchmark several times.
  for (int i = 0; i < 10; i++) {
    tacoFill<valType>(ctx, runtime, A.vals, aPart.APartition.valsPartition, 0);
    tacoFill<valType>(ctx, runtime, B.vals, bPart.BPartition.valsPartition, 1);
    tacoFill<valType>(ctx, runtime, C.vals, cPart.CPartition.valsPartition, 1);

    // Place the tensors.
    placeLegionA(ctx, runtime, &A, &aPart, gx, gy);
    placeLegionB(ctx, runtime, &B, &bPart, gx, gy);
    placeLegionC(ctx, runtime, &C, &cPart, gx, gy);

    // Compute on the tensors.
    if (use_tblis) {
      benchmark(ctx, runtime, times, [&, gx, gy]() {
        computeLegionTblis(ctx, runtime, &A, &B, &C, &partsTblis, gx, gy);
      });
    } else {
      benchmark(ctx, runtime, times, [&, gx, gy]() {
        computeLegionNestedOMP(ctx, runtime, &A, &B, &C, &partsNestedOMP, gx,
                               gy);
      });
    }
  }

  // Get the GFLOPS per node.
  auto avgTime = average(times);
  auto flopCount = size_t(no) * size_t(nv) * size_t(no) * size_t(nv) * (2 * size_t(no) * size_t(nv) - 1);
  auto gflops = getGFLOPS(flopCount, avgTime);
  auto nodes = runtime
                   ->select_tunable_value(
                       ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT)
                   .get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout,
                    "On %zu nodes achieved GFLOPS per node: %f.\n", nodes,
                    gflops / double(nodes));

  // TODO (rohany): I'm not sure what the resulting entries should validate to.
  // tacoValidate<valType>(ctx, runtime, A.vals, aPart.APartition.valsPartition,
  //                       valType(n * n));
}

TACO_MAIN(valType)
