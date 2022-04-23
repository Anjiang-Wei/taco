#include "legion.h"
#include "taco_mapper.h"
#include "legion_utils.h"
#include "realm/cmdline.h"

#include "taco-generated.h"

using namespace Legion;

typedef double valType;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  // Parse input args.
  Realm::CommandLineParser parser;
  int n = 10, gx = 2, gy = 2;
  bool validate = false;
  parser.add_option_int("-n", n);
  parser.add_option_int("-gx", gx);
  parser.add_option_int("-gy", gy);
  parser.add_option_bool("-validate", validate);

  // Create the tensors.
  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<valType>(ctx, runtime, fspace);

  auto A = createDenseTensor<2, valType>(ctx, runtime, {n, n}, FID_VAL);
  auto B = createDenseTensor<3, valType>(ctx, runtime, {n, n, n}, FID_VAL);
  // To get around a Legion issue (before we have collective instances),
  // we need to manually replicate the c vector.
  // TODO (rohany): Do the replication.

  // Create initial partitions.
  auto aValsPart = partitionLegionA(ctx, runtime, &A, gx, gy);
  auto bValsPart = partitionLegionB(ctx, runtime, &B, gx, gy);
  // Perform an initial fill on the values.
  // TODO (rohany): Add a function that fills over a LegionTensorPartition.
  tacoFill<valType>(ctx, runtime, A.vals, aValsPart, 0);
  tacoFill<valType>(ctx, runtime, B.vals, bValsPart, 1);


  // Create a "replicated" version of C.
  auto cISpace = runtime->create_index_space(ctx, Rect<1>({0, (n * gx * gy) - 1}));
  auto C = runtime->create_logical_region(ctx, cISpace, fspace); runtime->attach_name(C, "C");

  // Create a "replicated" partition of C.
  DomainPointColoring cColoring;
  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gx - 1), (gy - 1));
  DomainT<2> dom(Rect<2>{lowerBound, upperBound});
  for (PointInDomainIterator<2> itr(dom); itr(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int idx = in * gy + jn;
    auto start = idx * n;
    auto end = (idx + 1) * n - 1;
    cColoring[*itr] = Rect<1>(start, end);
  }
  auto cIndexPart = runtime->create_index_partition(ctx, cISpace, dom, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto cPart = runtime->get_logical_partition(ctx, C, cIndexPart);

  tacoFill<valType>(ctx, runtime, C, cPart, 1);

  placeLegionA(ctx, runtime, A, gx, gy);
  placeLegionB(ctx, runtime, B, gx, gy);

  // Run the function once to warm up the runtime system and validate the result.
  computeLegion(ctx, runtime, A, B, C, bPart, aPart, cPart);
  tacoValidate<valType>(ctx, runtime, A, aPart, valType(n));

  std::vector<size_t> times;
  benchmarkAsyncCall(ctx, runtime, times, [&]() {
    for (int i = 0; i < 10; i++) {
      computeLegion(ctx, runtime, A, B, C, bPart, aPart, cPart);
    }
  });

  // Calculate the total bandwidth.
  size_t elems = [](size_t n) { return n * n + n * n * n + n; }(n);
  size_t bytes = elems * sizeof(valType);
  double gbytes = double(bytes) / 1e9;
  auto avgTimeS = (double(times[0]) / 10.f) / 1e3;
  double bw = gbytes / (avgTimeS);
  auto nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %ld nodes achieved GB/s BW per node: %lf.\n", nodes, bw / double(nodes));
}

TACO_MAIN(valType)
