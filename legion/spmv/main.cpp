#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "mappers/default_mapper.h"
#include "legion_utils.h"
#include "legion_string_utils.h"

using namespace Legion;
using namespace Legion::Mapping;

typedef double valType;

// This mapper is currently unused but I'll keep it around for debugging purposes.
class SpMVMapper : public DefaultMapper {
public:
  SpMVMapper(MapperRuntime* rt, Machine& machine, const Legion::Processor& local) : DefaultMapper(rt, machine, local) {}

  void map_task(const Legion::Mapping::MapperContext ctx,
                const Legion::Task &task,
                const MapTaskInput &input,
                MapTaskOutput &output) {
    std::cout << "Mapping task!" << std::endl;
    if (strcmp(task.get_task_name(), "task_1") == 0) {
      for (size_t i = 0; i < task.regions.size(); i++) {
        auto reg = task.regions[i];
        auto ispace = reg.region.get_index_space();
        auto dom = runtime->get_index_space_domain(ctx, ispace);
        std::cout << "arg: " << i << " : " << dom << std::endl;
      }
    }
    DefaultMapper::map_task(ctx, task, input, output);
  }
};

void register_mapper(Machine m, Runtime* runtime, const std::set<Processor>& local_procs) {
}

// Forward declarations for partitioning and computation.
struct partitionPackForcomputeLegionRowSplit;
partitionPackForcomputeLegionRowSplit* partitionForcomputeLegionRowSplit(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces);
void computeLegionRowSplit(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionRowSplit* partitionPack, int32_t pieces);

struct partitionPackForcomputeLegionPosSplit;
partitionPackForcomputeLegionPosSplit* partitionForcomputeLegionPosSplit(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, int32_t pieces);
void computeLegionPosSplit(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegionPosSplit* partitionPack, int32_t pieces);

void registerTacoTasks();

const int TID_INIT_X = 420;
void initX(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  typedef FieldAccessor<WRITE_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorD;
  auto x = regions[0];
  auto dom = runtime->get_index_space_domain(x.get_logical_region().get_index_space());
  AccessorD xAcc(x, FID_VAL);
  #pragma omp parallel for schedule(static)
  for (size_t i = dom.lo()[0]; i < size_t(dom.hi()[0]); i++) {
    xAcc[i] = i;
  }
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::string csrFileName;
  bool dump = false, pos = false;
  int n = 10, pieces = 4, warmup = 5;
  Realm::CommandLineParser parser;
  parser.add_option_string("-csr", csrFileName);
  parser.add_option_bool("-dump", dump);
  parser.add_option_int("-n", n);
  parser.add_option_int("-pieces", pieces);
  parser.add_option_int("-warmup", warmup);
  parser.add_option_bool("-pos", pos);
  auto args = Runtime::get_input_args();
  assert(parser.parse_command_line(args.argc, args.argv));
  assert(!csrFileName.empty());

  auto A = loadLegionTensorFromHDF5File(ctx, runtime, csrFileName, {Dense, Sparse});

  auto y = createDenseTensor<1, valType>(ctx, runtime, {A.dims[0]}, FID_VAL);
  auto x = createDenseTensor<1, valType>(ctx, runtime, {A.dims[1]}, FID_VAL);
  runtime->fill_field(ctx, y.vals, y.valsParent, FID_VAL, valType(0));

  // Initialize x.
  auto eqPartIspace = runtime->create_index_space(ctx, Rect<1>(0, pieces - 1));
  auto eqPartDomain = runtime->get_index_space_domain(eqPartIspace);
  auto xEqPart = runtime->create_equal_partition(ctx, x.vals.get_index_space(), eqPartIspace);
  auto xEqLPart = runtime->get_logical_partition(ctx, x.vals, xEqPart);
  {
    IndexTaskLauncher launcher(TID_INIT_X, eqPartDomain, TaskArgument(), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(xEqLPart, 0, WRITE_ONLY, EXCLUSIVE, x.valsParent).add_field(FID_VAL));
    runtime->execute_index_space(ctx, launcher);
  }

  // Create a partition of y for forcing reductions.
  auto yEqIndexPart = runtime->create_equal_partition(ctx, y.vals.get_index_space(), eqPartIspace);
  auto yEqLPart = runtime->get_logical_partition(ctx, y.vals, yEqIndexPart);

  // Partition the tensors.
  partitionPackForcomputeLegionPosSplit* posPack = nullptr;
  partitionPackForcomputeLegionRowSplit* rowPack = nullptr;
  if (pos) {
    posPack = partitionForcomputeLegionPosSplit(ctx, runtime, &y, &A, &x, pieces);
  } else {
    rowPack = partitionForcomputeLegionRowSplit(ctx, runtime, &y, &A, &x, pieces);
  }

  // Benchmark the computation.
  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) { runtime->fill_field(ctx, y.vals, y.valsParent, FID_VAL, valType(0)); }
    if (pos) {
      computeLegionPosSplit(ctx, runtime, &y, &A, &x, posPack, pieces);
      launchDummyReadOverPartition(ctx, runtime, y.vals, yEqLPart, FID_VAL, eqPartDomain);
    } else {
      computeLegionRowSplit(ctx, runtime, &y, &A, &x, rowPack, pieces);
    }
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    printLegionTensor<valType>(ctx, runtime, y, {Dense});
  }

  // Delete the partition packs.
  if (posPack != nullptr) delete posPack;
  if (rowPack != nullptr) delete rowPack;
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_INIT_X, "initXCPU");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<initX>(registrar, "initX");
  }
  {
    TaskVariantRegistrar registrar(TID_INIT_X, "initXOMP");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    Runtime::preregister_task_variant<initX>(registrar, "initX");
  }
  registerHDF5UtilTasks();
  registerTacoTasks();
  registerDummyReadTasks();
  Runtime::add_registration_callback(register_mapper);
  return Runtime::start(argc, argv);
}
