#include "legion.h"
#include "realm/cmdline.h"
#include "mappers/default_mapper.h"
#include "taco.h"
#include "taco/lower/lower.h"

#include "distal-compiler-jit.h"
#include "distal-runtime.h"

using namespace Legion;
enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime) {

  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<double>(ctx, runtime, fspace);

  int n = 10;
  Realm::CommandLineParser parser;
  parser.add_option_int("-n", n);
  auto args = runtime->get_input_args();
  parser.parse_command_line(args.argc, args.argv);

  // TODO (rohany): I don't know the best way of passing runtime arguments
  //  like processor grid sizes to the generated code in the JIT case.

  // Declare the computation and schedule it.
  taco::IndexStmt stmt;
  std::shared_ptr<taco::LeafCallInterface> gemm = std::make_shared<taco::GEMM>();
  {
    int omps = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_LOCAL_OMPS).get<size_t>();
    int cpus = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_LOCAL_CPUS).get<size_t>();
    taco::Grid m(omps > 0 ? omps : cpus);
    taco::DistVar x, y;
    taco::TensorDistributionNotation dist({x, y}, m, {x});
    taco::TensorVar a("a", {taco::Float64, {n, n}}, {dist});
    taco::TensorVar b("b", {taco::Float64, {n, n}}, {dist});
    taco::TensorVar c("c", {taco::Float64, {n, n}}, {dist});
    taco::IndexVar i("i"), j("j"), k("k"), io("io"), ii("ii");
    stmt = (a(i, j) = b(i, k) * c(k, j));
    stmt = stmt.concretize()
               .distribute({i}, {io}, {ii}, m)
               .communicate({a(i, j), b(i, k), c(k, j)}, io)
               .swapLeafKernel(ii, gemm)
               ;
  }
  // JIT the kernel.
  auto jit = DISTAL::Compiler::JIT::compile(ctx, runtime, stmt);

  // Create LegionTensor objects to call the compiled kernel upon.
  auto a = createDenseTensor<2, double>(ctx, runtime, {n, n}, FID_VAL);
  auto b = createDenseTensor<2, double>(ctx, runtime, {n, n}, FID_VAL);
  auto c = createDenseTensor<2, double>(ctx, runtime, {n, n}, FID_VAL);
  // TODO (rohany): This is kind of ugly. I might want a helper method that
  //  performs this zip.
  jit.distributions[0].partition(ctx, runtime, &a);
  jit.distributions[1].partition(ctx, runtime, &b);
  jit.distributions[2].partition(ctx, runtime, &c);
  jit.kernel.partition(ctx, runtime, {&a, &b, &c});

  // TODO (rohany): Same with here.
  jit.distributions[0].apply(ctx, runtime, &a);
  jit.distributions[1].apply(ctx, runtime, &b);
  jit.distributions[2].apply(ctx, runtime, &c);
  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, 5, 10, [&]() {
    jit.kernel.compute(ctx, runtime, {&a, &b, &c});
  });

  std::cout << "Average time: " << avgTime << std::endl;
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID,
                                   "top_level_variant");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar,"top_level_task");
  }
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID,
                                   "top_level_variant");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar,"top_level_task");
  }
  Runtime::add_registration_callback(register_taco_mapper);
  Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor());
  return Runtime::start(argc, argv);
}
