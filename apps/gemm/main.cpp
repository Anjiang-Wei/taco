#include "legion.h"
#include "realm/cmdline.h"
#include "mappers/default_mapper.h"

#include "taco.h"

#include "distal-compiler-jit.h"
#include "distal-runtime.h"

using namespace Legion;

std::pair<int, int> factorSquare(int n) {
  for (int i = sqrt(double(n)) + 1; i > 0; i--) {
    if (n % i == 0) {
      return {i, n / i};
    }
  }
  return {n, 1};
}

// GEMMAlgorithm is an interface for matrix-matrix multiplication algorithms.
// A GEMMAlgorithm implementation is responsible for constructing a matrix-multiply
// within TACO/DISTAL.
class GEMMAlgorithm {
public:
  virtual ~GEMMAlgorithm() {};
  virtual taco::IndexStmt schedule(taco::ParallelUnit pu, int procs) { return {}; }
  virtual bool hierarchy() { return false; }
  virtual taco::IndexStmt schedule(std::vector<taco::ParallelUnit> levels, std::vector<int> procs) { return {}; }
protected:
  std::shared_ptr<taco::LeafCallInterface> getGEMM(taco::ParallelUnit pu) {
    if (pu == taco::ParallelUnit::DistributedGPU) {
      return std::make_shared<taco::CuGEMM>();
    }
    return std::make_shared<taco::GEMM>();
  }
};

// An implementation of Cannon's algorithm.
class CannonsAlgorithm : public GEMMAlgorithm {
public:
  taco::IndexStmt schedule(taco::ParallelUnit pu, int procs) override {
    auto factors = factorSquare(procs);
    taco::Grid m(factors.first, factors.second);

    // Distribute all tensors in tiles.
    taco::DistVar x, y;
    taco::TensorDistributionNotation distribution({x, y}, m, {x, y}, pu);
    taco::TensorVar A("A", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar B("B", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar C("C", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});

    taco::IndexVar i("i"), j("j"), k("k");
    taco::IndexStmt stmt = A(i, j) = B(i, k) * C(k, j);

    taco::IndexVar io("io"), ii("ii"), jo("jo"), ji("ji"), ko("ko"), ki("ki"), kos("kos");
    return stmt.concretize()
               .distribute({i, j}, {io, jo}, {ii, ji}, m, pu)
               .divide(k, ko, ki, m.getDimSize(0))
               .reorder({ko, ii, ji})
               .stagger(ko, {io, jo}, kos)
               .communicate(A(i, j), jo)
               .communicate({B(i, k), C(k, j)}, kos)
               .swapLeafKernel(ii, this->getGEMM(pu))
               ;
  }
  bool hierarchy() override { return true; }
  taco::IndexStmt schedule(std::vector<taco::ParallelUnit> levels, std::vector<int> procs) override {
    taco_iassert(levels.size() == 2 && levels[0] == taco::ParallelUnit::DistributedNode);

    auto factorAllNodes = factorSquare(procs[0]);
    auto factorNode = factorSquare(procs[1]);
    std::vector<taco::Grid> machine = {
        {factorAllNodes.first, factorAllNodes.second},
        // TODO (rohany): We can do something smarter for the inter-node if that matters.
        {factorNode.first, factorNode.second},
    };

    taco::DistVar x, y;
    std::vector<taco::TensorDistributionNotation> distribution = {
        {{x, y}, machine[0], {x, y}, levels[0]},
        {{x, y}, machine[1], {x, y}, levels[1]},
    };
    taco::TensorVar A("A", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar B("B", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar C("C", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});

    taco::IndexVar i("i"), j("j"), k("k");
    taco::IndexStmt stmt = A(i, j) = B(i, k) * C(k, j);

    // Index variables for the first layer of distribution.
    taco::IndexVar io("io"), ii("ii"), jo("jo"), ji("ji"), ko("ko"), ki("ki"), kos("kos");
    // Index variables for the second level of distribution.
    taco::IndexVar iio("iio"), iii("iii"), jio("jio"), jii("jii"), kio("kio"), kii("kii"), kios("kios");
    return stmt.concretize()
               // Level 1 of distribution -- all nodes.
               .distribute({i, j}, {io, jo}, {ii, ji}, machine[0], levels[0])
               .divide(k, ko, ki, machine[0].getDimSize(0))
               .reorder({ko, ii, ji})
               .stagger(ko, {io, jo}, kos)
               .communicate(A(i, j), jo)
               .communicate({B(i, k), C(k, j)}, kos)
               // Level 2 of distribution -- all sockets or GPUs on a node.
               .distribute({ii, ji}, {iio, jio}, {iii, jii}, machine[1], levels[1])
               .divide(ki, kio, kii, machine[1].getDimSize(0))
               .reorder({kio, iii, jii})
               .stagger(kio, {iio, jio}, kios)
               .communicate(A(i, j), jio)
               .communicate({B(i, k), C(k, j)}, kios)
               .swapLeafKernel(iii, this->getGEMM(levels[1]))
               ;
  }
};

// An implementation of the SUMMA algorithm.
class SUMMAAlgorithm : public GEMMAlgorithm {
public:
  taco::IndexStmt schedule(taco::ParallelUnit pu, int procs) override {
    auto factors = factorSquare(procs);
    taco::Grid m(factors.first, factors.second);

    // Distribute all tensors in tiles.
    taco::DistVar x, y;
    taco::TensorDistributionNotation distribution({x, y}, m, {x, y}, pu);
    taco::TensorVar A("A", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar B("B", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar C("C", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});

    taco::IndexVar i("i"), j("j"), k("k");
    taco::IndexStmt stmt = A(i, j) = B(i, k) * C(k, j);

    taco::IndexVar io("io"), ii("ii"), jo("jo"), ji("ji"), ko("ko"), ki("ki");
    return stmt.concretize()
               .distribute({i, j}, {io, jo}, {ii, ji}, m, pu)
               .divide(k, ko, ki, m.getDimSize(0))
               .reorder({ko, ii, ji})
               .communicate(A(i, j), jo)
               .communicate({B(i, k), C(k, j)}, ko)
               .swapLeafKernel(ii, this->getGEMM(pu))
               ;
  }
  bool hierarchy() override { return true; }
  taco::IndexStmt schedule(std::vector<taco::ParallelUnit> levels, std::vector<int> procs) override {
    taco_iassert(levels.size() == 2 && levels[0] == taco::ParallelUnit::DistributedNode);

    auto factorAllNodes = factorSquare(procs[0]);
    auto factorNode = factorSquare(procs[1]);
    std::vector<taco::Grid> machine = {
        {factorAllNodes.first, factorAllNodes.second},
        // TODO (rohany): We can do something smarter for the inter-node if that matters.
        {factorNode.first, factorNode.second},
    };

    taco::DistVar x, y;
    std::vector<taco::TensorDistributionNotation> distribution = {
        {{x, y}, machine[0], {x, y}, levels[0]},
        {{x, y}, machine[1], {x, y}, levels[1]},
    };
    taco::TensorVar A("A", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar B("B", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar C("C", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});

    taco::IndexVar i("i"), j("j"), k("k");
    taco::IndexStmt stmt = A(i, j) = B(i, k) * C(k, j);

    // Index variables for the first layer of distribution.
    taco::IndexVar io("io"), ii("ii"), jo("jo"), ji("ji"), ko("ko"), ki("ki");
    // Index variables for the second level of distribution.
    taco::IndexVar iio("iio"), iii("iii"), jio("jio"), jii("jii"), kio("kio"), kii("kii");
    return stmt.concretize()
               // Level 1 of distribution -- all nodes.
               .distribute({i, j}, {io, jo}, {ii, ji}, machine[0], levels[0])
               .divide(k, ko, ki, machine[0].getDimSize(0))
               .reorder({ko, ii, ji})
               .communicate(A(i, j), jo)
               .communicate({B(i, k), C(k, j)}, ko)
               // Level 2 of distribution -- all sockets or GPUs on a node.
               .distribute({ii, ji}, {iio, jio}, {iii, jii}, machine[1], levels[1])
               .divide(ki, kio, kii, machine[1].getDimSize(0))
               .reorder({kio, iii, jii})
               .communicate(A(i, j), jio)
               .communicate({B(i, k), C(k, j)}, kio)
               .swapLeafKernel(iii, this->getGEMM(levels[1]))
               ;
  }
};

// An implementation of the PUMMA algorithm.
class PUMMAAlgorithm : public GEMMAlgorithm {
public:
  taco::IndexStmt schedule(taco::ParallelUnit pu, int procs) override {
    auto factors = factorSquare(procs);
    taco::Grid m(factors.first, factors.second);

    // Distribute all tensors in tiles.
    taco::DistVar x, y;
    taco::TensorDistributionNotation distribution({x, y}, m, {x, y}, pu);
    taco::TensorVar A("A", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar B("B", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar C("C", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});

    taco::IndexVar i("i"), j("j"), k("k");
    taco::IndexStmt stmt = A(i, j) = B(i, k) * C(k, j);

    taco::IndexVar io("io"), ii("ii"), jo("jo"), ji("ji"), ko("ko"), ki("ki"), kos("kos");
    return stmt.concretize()
        .distribute({i, j}, {io, jo}, {ii, ji}, m, pu)
        .divide(k, ko, ki, m.getDimSize(0))
        .reorder({ko, ii, ji})
        .stagger(ko, {io}, kos)
        .communicate(A(i, j), jo)
        .communicate({B(i, k), C(k, j)}, kos)
        .swapLeafKernel(ii, this->getGEMM(pu))
        ;
  }
  bool hierarchy() override { return true; }
  taco::IndexStmt schedule(std::vector<taco::ParallelUnit> levels, std::vector<int> procs) override {
    taco_iassert(levels.size() == 2 && levels[0] == taco::ParallelUnit::DistributedNode);

    auto factorAllNodes = factorSquare(procs[0]);
    auto factorNode = factorSquare(procs[1]);
    std::vector<taco::Grid> machine = {
        {factorAllNodes.first, factorAllNodes.second},
        // TODO (rohany): We can do something smarter for the inter-node if that matters.
        {factorNode.first, factorNode.second},
    };

    taco::DistVar x, y;
    std::vector<taco::TensorDistributionNotation> distribution = {
        {{x, y}, machine[0], {x, y}, levels[0]},
        {{x, y}, machine[1], {x, y}, levels[1]},
    };
    taco::TensorVar A("A", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar B("B", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar C("C", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});

    taco::IndexVar i("i"), j("j"), k("k");
    taco::IndexStmt stmt = A(i, j) = B(i, k) * C(k, j);

    // Index variables for the first layer of distribution.
    taco::IndexVar io("io"), ii("ii"), jo("jo"), ji("ji"), ko("ko"), ki("ki"), kos("kos");
    // Index variables for the second level of distribution.
    taco::IndexVar iio("iio"), iii("iii"), jio("jio"), jii("jii"), kio("kio"), kii("kii"), kios("kios");
    return stmt.concretize()
               // Level 1 of distribution -- all nodes.
               .distribute({i, j}, {io, jo}, {ii, ji}, machine[0], levels[0])
               .divide(k, ko, ki, machine[0].getDimSize(0))
               .reorder({ko, ii, ji})
               .stagger(ko, {io}, kos)
               .communicate(A(i, j), jo)
               .communicate({B(i, k), C(k, j)}, kos)
               // Level 2 of distribution -- all sockets or GPUs on a node.
               .distribute({ii, ji}, {iio, jio}, {iii, jii}, machine[1], levels[1])
               .divide(ki, kio, kii, machine[1].getDimSize(0))
               .reorder({kio, iii, jii})
               .stagger(kio, {iio}, kios)
               .communicate(A(i, j), jio)
               .communicate({B(i, k), C(k, j)}, kios)
               .swapLeafKernel(iii, this->getGEMM(levels[1]))
               ;
  }
};

// An implementation of Solomonik's algorithm. Since Legion does not support virtual
// reduction instances right now, this algorithm does not support heirarchy.
class SolomoniksAlgorithm : public GEMMAlgorithm {
public:
  SolomoniksAlgorithm(int c, int rpoc, int rpoc3) : c(c), rpoc(rpoc), rpoc3(rpoc3) {}
  taco::IndexStmt schedule(taco::ParallelUnit pu, int procs) override {
    // The parameters need to satisfy certain constraints.
    taco_iassert(rpoc * rpoc / c == procs);
    taco::Grid m(rpoc, rpoc, c);

    // Distribute all tensors in tiles.
    taco::DistVar x, y;
    taco::TensorDistributionNotation distribution({x, y}, m, {x, y, 0}, pu);
    taco::TensorVar A("A", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar B("B", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});
    taco::TensorVar C("C", {taco::Float64, {taco::Dimension(), taco::Dimension()}}, {distribution});

    taco::IndexVar i("i"), j("j"), k("k");
    taco::IndexStmt stmt = A(i, j) = B(i, k) * C(k, j);

    taco::IndexVar io("io"), ii("ii"), jo("jo"), ji("ji"), ko("ko"), ki("ki"),
                   kio("kio"), kii("kii"), kios("kios");
    // To schedule for Solomonik's algorithm, we'll distribute over i, j, k according to the
    // processor grid. Then, we divide the ki loop into kio and kii so that each partition of C
    // is operated on in chunks. Finally, we then stagger the kio loop so that along each parallel
    // slice of k, a Cannon style shifting occurs.
    return stmt.concretize()
               .distribute({i, j, k}, {io, jo, ko}, {ii, ji, ki}, m, pu)
               .divide(ki, kio, kii, rpoc3)
               .reorder({kio, ii, ji})
               .stagger(kio, {io, jo}, kios)
               .communicate(A(i, j), ko)
               .communicate({B(i, k), C(k, j)}, kios)
               .swapLeafKernel(ii, this->getGEMM(pu))
               ;
  }
private:
  int c;
  int rpoc;
  int rpoc3;
};

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime) {
  int n = -1, m = -1, k = -1;
  std::string algorithmStr;
  Realm::CommandLineParser parser;
  // Size parameters.
  parser.add_option_int("-n", n);
  parser.add_option_int("-m", m);
  parser.add_option_int("-k", k);
  parser.add_option_string("-alg", algorithmStr);
  // Algorithms for Solomonik's algorithm.
  int c = -1, rpoc = -1, rpoc3 = -1;
  parser.add_option_int("-c", c);
  parser.add_option_int("-rpoc", rpoc);
  parser.add_option_int("-rpoc3", rpoc3);
  auto args = runtime->get_input_args();
  auto ok = parser.parse_command_line(args.argc, args.argv);
  if (!ok) {
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Unable to parse command line.\n");
    return;
  }
  if (n == -1) {
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Provide matrix sizes with -n (-m, -k).\n");
    return;
  }
  if (n != -1 && ((m == -1) ^ (k == -1))) {
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Must provide either just -n or all of -n, -m and -k.\n");
    return;
  }
  // If only n was set, expand m and k to equal n.
  if (n != -1 && (m == -1 || k == -1)) {
    m = n;
    k = n;
  }

  // Parse the desired algorithm from the command line.
  std::unique_ptr<GEMMAlgorithm> algorithm;
  if (algorithmStr == "cannon") {
    algorithm = std::make_unique<CannonsAlgorithm>();
  } else if (algorithmStr == "summa") {
    algorithm = std::make_unique<SUMMAAlgorithm>();
  } else if (algorithmStr == "pumma") {
    algorithm = std::make_unique<PUMMAAlgorithm>();
  } else if (algorithmStr == "solomonik") {
    if (c == -1 || rpoc == -1 || rpoc3 == -1) {
      LEGION_PRINT_ONCE(runtime, ctx, stdout, "Must provide -c, -rpoc and -rpoc3 for Solomonik's algorithm.\n");
      return;
    }
    algorithm = std::make_unique<SolomoniksAlgorithm>(c, rpoc, rpoc3);
  }
  if (!algorithm) {
    LEGION_PRINT_ONCE(runtime, ctx, stdout, "Provide a supported algorithm with -alg: %s\n",
                      taco::util::join(std::vector<std::string>{"cannon", "summa", "pumma", "solomonik"}).c_str());
    return;
  }

  // Figure out some metadata about our machine, which will guide the
  // scheduling of each algorithm.
  int nodes = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_NODE_COUNT).get<size_t>();
  int gpus = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_GPUS).get<size_t>();
  int omps = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_OMPS).get<size_t>();
  int cpus = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_CPUS).get<size_t>();
  int localGPUs = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_LOCAL_GPUS).get<size_t>();
  int localOMPs = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_LOCAL_OMPS).get<size_t>();
  int localCPUs = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_LOCAL_CPUS).get<size_t>();
  bool wantHierarchy = ((gpus > 0 && localGPUs > 1) || (omps > 0 && localOMPs > 1));

  taco::IndexStmt result;
  if (wantHierarchy && algorithm->hierarchy()) {
    std::vector<taco::ParallelUnit> pu = {
        taco::ParallelUnit::DistributedNode,
        localGPUs > 0 ? taco::ParallelUnit::DistributedGPU : taco::ParallelUnit::DistributedNode
    };
    std::vector<int> procs = {
        nodes,
        (localGPUs > 0 ? localGPUs : (localOMPs > 0 ? localOMPs : localCPUs))
    };
    result = algorithm->schedule(pu, procs);
  } else {
    auto pu = gpus > 0 ? taco::ParallelUnit::DistributedGPU : taco::ParallelUnit::DistributedNode;
    auto procs = (gpus > 0 ? gpus : (omps > 0 ? omps : cpus));
    result = algorithm->schedule(pu, procs);
  }
  auto jit = DISTAL::Compiler::JIT::compile(ctx, runtime, result);

  // Finally, create the tensors that we'll execute on.
  auto fspace = runtime->create_field_space(ctx);
  allocate_tensor_fields<double>(ctx, runtime, fspace);
  auto A = createDenseTensor<2, double>(ctx, runtime, {n, m}, FID_VAL);
  auto B = createDenseTensor<2, double>(ctx, runtime, {n, k}, FID_VAL);
  auto C = createDenseTensor<2, double>(ctx, runtime, {k, m}, FID_VAL);
  auto kernel = jit.bind({&A, &B, &C});
  initCuBLAS(ctx, runtime);

  // Regardless of how the algorithm partitions its data, we want to have
  // a tiled, disjoint partition on hand to fill over and reset the data
  // to an initial layout instead of having left over instances everywhere.
  // TODO (rohany): If the partitions here don't line up well with the
  //  partitions created by DISTAL, then we can migrate this to doing the
  //  fill within DISTAL instead of calling out to the pre-written code.
  auto makeTiledPart = [&](LogicalRegion rg) {
    auto factors = factorSquare((gpus > 0 ? gpus : (omps > 0 ? omps : cpus)));
    auto cspace = runtime->create_index_space(ctx, Domain(Point<2>{0, 0}, Point<2>{factors.first - 1, factors.second - 1}));
    auto dom = runtime->get_index_space_domain(rg.get_index_space());
    auto extents = dom.hi();
    // Make each block for the transform the block size that we want.
    auto bounds = Point<2>((extents[0] + factors.first) / factors.first, (extents[1] + factors.second) / factors.second);
    Transform<2, 2> transform;
    transform[0][0] = bounds.x;
    transform[1][0] = 0;
    transform[0][1] = 0;
    transform[1][1] = bounds.y;
    // Then, apply partition by restriction, using an extent the same size as the
    // transform to get equally sized blocks.
    auto ipart = runtime->create_partition_by_restriction(
      ctx,
      rg.get_index_space(),
      cspace,
      transform,
      Rect<2>{{0, 0}, bounds - Point<2>::ONES()}
    );
    taco_iassert(runtime->is_index_partition_disjoint(ctx, ipart));
    taco_iassert(runtime->is_index_partition_complete(ctx, ipart));
    return runtime->get_logical_partition(ctx, rg, ipart);
  };
  auto APart = makeTiledPart(A.vals);
  auto BPart = makeTiledPart(B.vals);
  auto CPart = makeTiledPart(C.vals);

  std::vector<size_t> times;
  for (int i = 0; i < 11; i++) {
    // TODO (rohany): For some algorithms (like Solomonik's), we may create some instances that we
    //  don't need/want by filling over the tiled partition. If this becomes a problem, then we'll
    //  need to have a switch on the algorithm to tell us to choose which partition to fill over.
    tacoFill<double>(ctx, runtime, A.vals, APart, 0);
    tacoFill<double>(ctx, runtime, B.vals, BPart, 1);
    tacoFill<double>(ctx, runtime, C.vals, CPart, 1);

    kernel.distribute(ctx, runtime);
    if (i == 0) kernel.compute(ctx, runtime);
    else benchmark(ctx, runtime, times, [&]() { kernel.compute(ctx, runtime); });
  }

  auto avgTime = average(times);
  auto flopCount = getGEMMFLOPCount(n, m, k);
  auto gflops = getGFLOPS(flopCount, avgTime);
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "On %d nodes achieved GFLOPS per node: %lf.\n", nodes, gflops / double(nodes));

  // The result should be equal to 1 if m n and k are all the same. So, we'll only
  // do validation in that case.
  if (n == m && m == k) tacoValidate<double>(ctx, runtime, A.vals, APart, double(n));
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level_variant");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar,"top_level_task");
  }
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level_variant");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar,"top_level_task");
  }
  // TODO (rohany): Add this all to a DISTAL::Runtime::init() method.
  Runtime::add_registration_callback(register_taco_mapper);
  Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor());
  registerTACOFillTasks<double>();
  registerTACOValidateTasks<double>();
  initCUDA();
  return Runtime::start(argc, argv);
}
