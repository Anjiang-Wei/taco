#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "mappers/default_mapper.h"

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
void packLegion(Context ctx, Runtime* runtime, LegionTensor* BCSR, LegionTensor* BCOO);
struct partitionPackForcomputeLegion;
partitionPackForcomputeLegion* partitionForcomputeLegion(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c);
void computeLegion(Context ctx, Runtime* runtime, LegionTensor* a, LegionTensor* B, LegionTensor* c, partitionPackForcomputeLegion* partitionPack);
void registerTacoTasks();

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::string cooFileName;
  bool dump = false;
  Realm::CommandLineParser parser;
  parser.add_option_string("-coofile", cooFileName);
  parser.add_option_bool("-dump", dump);
  auto args = Runtime::get_input_args();
  assert(parser.parse_command_line(args.argc, args.argv));
  assert(!cooFileName.empty());

  // Read in our COO matrix.
  auto coo = loadCOOFromHDF5(ctx, runtime, cooFileName, FID_RECT_1, FID_COORD, sizeof(int32_t), FID_VAL, sizeof(valType));

  auto y = createDenseTensor<1, valType>(ctx, runtime, {10}, FID_VAL);
  auto x = createDenseTensor<1, valType>(ctx, runtime, {10}, FID_VAL);
  runtime->fill_field(ctx, y.vals, y.valsParent, FID_VAL, valType(0));
  runtime->fill_field(ctx, x.vals, x.valsParent, FID_VAL, valType(1));

  // Initialize x.
  // TODO (rohany): Extract this into a task so that it can be reused.
  {
    typedef FieldAccessor<WRITE_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorD;
    auto x_phys = legionMalloc(ctx, runtime, x.vals, x.valsParent, FID_VAL);
    AccessorD xAcc(x_phys, FID_VAL);
    for (size_t i = 0; i < size_t(x.dims[0]); i++) {
      xAcc[i] = i;
    }
    runtime->unmap_region(ctx, x_phys);
  }

  // Pack the COO matrix into A.
  // TODO (rohany): In the future, I think I want this to a be two step process where we
  //  preprocess the COO file into an HDF5 file that describes exactly the CSR tensor. Then,
  //  to actually run the program we read in the CSR version of the tensor.
  auto A = createSparseTensorForPack<valType>(ctx, runtime, {Dense, Sparse}, {10, 10}, FID_RECT_1, FID_COORD, FID_VAL);

  packLegion(ctx, runtime, &A, &coo);

  // Partition the tensors.
  auto pack = partitionForcomputeLegion(ctx, runtime, &y, &A, &x);
  // TODO (rohany): Benchmark this computation.
  computeLegion(ctx, runtime, &y, &A, &x, pack);

  if (dump) {
    auto yreg = legionMalloc(ctx, runtime, y.vals, y.valsParent, FID_VAL);
    FieldAccessor<READ_WRITE,valType,1,coord_t, Realm::AffineAccessor<valType, 1, coord_t>> yrw(yreg, FID_VAL);
    for (int i = 0; i < y.dims[0]; i++) {
      std::cout << yrw[i] << " ";
    }
    std::cout << std::endl;
    runtime->unmap_region(ctx, yreg);
  }

  // Delete the partition pack.
  delete pack;
}

int TID_TOP_LEVEL = 420;
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
  Runtime::add_registration_callback(register_mapper);
  return Runtime::start(argc, argv);
}
