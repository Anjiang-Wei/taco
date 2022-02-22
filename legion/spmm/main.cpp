#include "legion.h"
#include "taco_legion_header.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "mappers/default_mapper.h"
#include "legion_utils.h"
#include "legion_string_utils.h"
#include "error.h"

#ifdef TACO_USE_CUDA
#include "taco-generated.cuh"
#else
#include "taco-generated.h"
#endif

using namespace Legion;
typedef double valType;

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  std::string csrFileName;
  bool dump = false, batched = false, CCpu = false;
  // The j-dimension if the computation will commonly have a small value
  // that is divisible by 32, as per Stephen and Chang-wan.
  int n = 10, pieces = 0, warmup = 5, jDim = 32, batchSize = 2;
  Realm::CommandLineParser parser;
  parser.add_option_string("-tensor", csrFileName);
  parser.add_option_bool("-dump", dump);
  parser.add_option_int("-n", n);
  parser.add_option_int("-pieces", pieces);
  parser.add_option_int("-warmup", warmup);
  parser.add_option_int("-jdim", jDim);
  parser.add_option_bool("-batched", batched);
  parser.add_option_int("-batchSize", batchSize);
  parser.add_option_bool("-Ccpu", CCpu);
  auto args = Runtime::get_input_args();
  taco_uassert(parser.parse_command_line(args.argc, args.argv)) << "Parse failure.";
  taco_uassert(!csrFileName.empty()) << "Provide a matrix with -tensor";

  // Figure out how many pieces to chop up the data into.
  if (pieces == 0) {
    pieces = getNumPieces(ctx, runtime);
    taco_uassert(pieces != 0) << "Please provide a number of pieces to split into with -pieces. Unable to automatically find.";
  }

  LegionTensor B; ExternalHDF5LegionTensor Bex;
  std::tie(B, Bex) = loadLegionTensorFromHDF5File(ctx, runtime, csrFileName, {Dense, Sparse});
  auto A = createDenseTensor<2, valType>(ctx, runtime, {B.dims[0], jDim}, FID_VAL);
  auto C = createDenseTensor<2, valType>(ctx, runtime, {B.dims[1], jDim}, FID_VAL);
  runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0));
  runtime->fill_field(ctx, C.vals, C.valsParent, FID_VAL, valType(1));

  auto piecesIspace = runtime->create_index_space(ctx, Rect<1>(0, pieces - 1));
  auto piecesDom = runtime->get_index_space_domain(ctx, piecesIspace);

  partitionPackForcomputeLegion pack;
  partitionPackForcomputeLegionBatched packBatched;
  IndexPartition commPart;
  if (batched) {
    packBatched = partitionForcomputeLegionBatched(ctx, runtime, &A, &B, &C, batchSize, pieces);
    commPart = createSparseAliasingPartitions(ctx, runtime, A.vals.get_index_space(), packBatched.APartition.valsPartition.get_index_partition());
  } else {
    pack = partitionForcomputeLegion(ctx, runtime, &A, &B, &C, pieces);
    commPart = createSparseAliasingPartitions(ctx, runtime, A.vals.get_index_space(), pack.APartition.valsPartition.get_index_partition());
  }
  auto commLPart = runtime->get_logical_partition(ctx, A.vals, commPart);

  // Create a long lasting instance for the C region to force communcation costs.
  if (batched) {
    auto colorSpace = runtime->create_index_space(ctx, Rect<1>(0, pieces - 1));
    taco_iassert(pieces <= 32);
    auto dom = runtime->get_index_space_domain(ctx, C.vals.get_index_space());
    auto size = jDim / pieces;
    DomainPointColoring coloring;
    for (int i = 0; i < pieces; i++) {
      coloring[i] = Rect<2>({dom.lo()[0], i * size}, {dom.hi()[0], (i+1) * size - 1});
    }
    auto ipart = runtime->create_index_partition(ctx, C.valsParent.get_index_space(), Rect<1>(0, pieces - 1), coloring);
    auto lpart = runtime->get_logical_partition(ctx, C.vals, ipart);
    tacoFill(ctx, runtime, C.vals, lpart, valType(1));
    if (CCpu) {
      auto numOMPs = runtime->select_tunable_value(ctx, Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_OMPS).get<size_t>();
      // TODO (rohany): I'm hacking here, but this distribution could be easily done using
      //  DISTAL's data distribution language.
      IndexTaskLauncher launcher(TID_DUMMY_READ_REGION, Rect<1>(0, numOMPs - 1), TaskArgument(), ArgumentMap());
      launcher.add_region_requirement(RegionRequirement(C.vals, READ_ONLY, EXCLUSIVE, C.valsParent).add_field(FID_VAL));
      launcher.tag |= TACOMapper::MAP_TO_OMP_OR_LOC;
      runtime->execute_index_space(ctx, launcher).wait_all_results();
    }
  }

  auto avgTime = benchmarkAsyncCallWithWarmup(ctx, runtime, warmup, n, [&]() {
    if (dump) { runtime->fill_field(ctx, A.vals, A.valsParent, FID_VAL, valType(0)); }
    if (batched) {
      computeLegionBatched(ctx, runtime, &A, &B, &C, &packBatched, batchSize, pieces);
      launchDummyReadOverPartition(ctx, runtime, A.vals, commLPart, FID_VAL, Rect<1>(0, pieces - 1), false/* wait */, true /* untrack */, false /* cpuOnly */, true /* sparse */);
    } else {
      computeLegion(ctx, runtime, &A, &B, &C, &pack, pieces);
    }
#ifdef TACO_USE_CUDA
    // Collapse our reduction buffers. We use sparse instances to force just the communication
    // that we want. We only do this for the GPU schedule, as the CPU schedule does not
    // use Legion reductions.
    launchDummyReadOverPartition(ctx, runtime, A.vals, commLPart, FID_VAL, Rect<1>(0, pieces - 1), false /* wait */, true /* untrack */, false /* cpuOnly */, true /* sparse */);
#endif
  });
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Average execution time: %lf ms\n", avgTime);

  if (dump) {
    printLegionTensor<valType>(ctx, runtime, A);
  }
  Bex.destroy(ctx, runtime);
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  if (TACO_FEATURE_OPENMP) {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC));
    registrar.set_replicable();
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  registerHDF5UtilTasks();
  registerTacoTasks();
  registerTacoRuntimeLibTasks();
  registerTACOFillTasks<valType>();
  Runtime::add_registration_callback(register_taco_mapper);
  Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor());
  return Runtime::start(argc, argv);
}
