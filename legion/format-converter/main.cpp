#include "legion.h"
#include "taco_legion_header.h"
#include "legion_string_utils.h"
#include "hdf5_utils.h"
#include "realm/cmdline.h"
#include "error.h"
#include "taco-generated.h"
#include <stdlib.h>
#include "task_ids.h"
#include "taco_mapper.h"

#include "mapping_utilities.h"

using namespace Legion;
using namespace Legion::Mapping;

Realm::Logger logApp("app");

// Until Legion can do AOS HDF5 writes (#1136), we need to manually transpose
// the data into SOA layout for HDF5 copies.
class HDF5CopyMapper : public TACOMapper {
public:
  HDF5CopyMapper(MapperRuntime* rt, Machine& machine, const Legion::Processor& local) : TACOMapper(rt, machine, local, "TACOMapper") {
    this->exact_region = true;
  }

  void map_copy(const MapperContext ctx,
                const Copy& copy,
                const MapCopyInput& input,
                MapCopyOutput& output) {
    if (copy.parent_task->task_id == TID_TOP_LEVEL) {
      default_create_copy_instance<false /* is src*/>(ctx, copy, copy.dst_requirements[0], 0, output.dst_instances[0]);
      // Map the source instance in SOA format for HDF5 to be happy.
      default_create_copy_instance<true /* is src*/>(ctx, copy, copy.src_requirements[0], 0, output.src_instances[0]);
    } else {
      DefaultMapper::map_copy(ctx, copy, input, output);
    }
  }
};

void register_mapper(Machine m, Runtime* runtime, const std::set<Processor>& local_procs) {
  runtime->replace_default_mapper(new HDF5CopyMapper(runtime->get_mapper_runtime(), m, *local_procs.begin()), Processor::NO_PROC);
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  bool roundTrip = false, dump = false, h5dump = false;
  std::string outputFormat, cooFile, outputFile, outputModeOrdering;
  Realm::CommandLineParser parser;
  parser.add_option_string("-coofile", cooFile);
  parser.add_option_string("-format", outputFormat);
  parser.add_option_string("-o", outputFile);
  parser.add_option_bool("-roundTrip", roundTrip);
  parser.add_option_bool("-dump", dump);
  parser.add_option_bool("-h5dump", h5dump);
  parser.add_option_string("-mode_ordering", outputModeOrdering);
  auto args = Runtime::get_input_args();
  taco_uassert(parser.parse_command_line(args.argc, args.argv)) << "Parse failure.";
  taco_uassert(!cooFile.empty()) << "provide a file with -coofile";
  taco_uassert(!outputFormat.empty()) << "provide an output format with -format";
  taco_uassert(!outputFile.empty()) << "provide an output file with -o";

  logApp.info() << "Loading input COO tensor.";
  // Load the input COO tensor.
  auto coo = loadCOOFromHDF5(ctx, runtime, cooFile, FID_RECT_1, FID_COORD, sizeof(int32_t), FID_VAL, sizeof(double));
  logApp.info() << "Done loading input COO tensor.";

  // Construct the desired output format.
  std::vector<LegionTensorLevelFormat> format;
  for (auto c : outputFormat) {
    switch (c) {
      case 'd':
        format.push_back(Dense); break;
      case 's':
        format.push_back(Sparse); break;
      default:
        taco_uassert(false) << "invalid sparse input";
    }
  }
  taco_uassert(int(format.size()) == coo.order);

  // Create the output tensor.
  auto output = createSparseTensorForPack<double>(ctx, runtime, format, coo.dims, FID_RECT_1, FID_COORD, FID_VAL);

  logApp.info() << "Packing COO to desired format.";

  // TODO (rohany): Is there any way that we could generate this map programmatically? We could
  //  probably emit it at the top of the taco generated file?
  typedef void (*ConvFunc)(Context, Runtime*, LegionTensor*, LegionTensor*);
  std::map<std::string, ConvFunc> converters = {
      {"s", packLegionCOOToVec},
      {"ds", packLegionCOOToCSR},
      {"ds10", packLegionCOOToCSC},
      {"ss", packLegionCOOToDCSR},
      {"sd", packLegionCOOToSD},
      {"sss", packLegionCOOToSSS},
      {"dss", packLegionCOOToDSS},
      {"dds", packLegionCOOToDDS},
      {"sds", packLegionCOOToSDS},
  };
  auto it = converters.find(outputFormat + outputModeOrdering);
  if (it == converters.end()) {
    taco_uassert(false) << "unsupported output format kind";
  }
  it->second(ctx, runtime, &output, &coo);

  logApp.info() << "Done packing COO to desired format.";

  logApp.info() << "Dumping output tensor to HDF5 file.";
  // Now, dump the output tensor to an HDF5 file.
  dumpLegionTensorToHDF5File(ctx, runtime, output, outputFile);
  logApp.info() << "Done!";

  if (roundTrip) {
    // Let's try and load it back in to see if it somewhat round-trips.
    auto test = loadLegionTensorFromHDF5File(ctx, runtime, outputFile, format);
    printLegionTensor<double>(ctx, runtime, test.first);
    test.second.destroy(ctx, runtime);
  }
  if (dump) {
    // Dump out the packed tensor to stdout.
    printLegionTensor<double>(ctx, runtime, output);
  }
  // If requested, dump data to a textual version of the HDF5 file. We do
  // this to avoid regressing.
  if (h5dump) {
    std::stringstream cmdSS;
    auto h5dumpFile = outputFile + ".h5dump";
    cmdSS << "h5dump -O -o " << h5dumpFile << " " << outputFile;
    taco_iassert(system(cmdSS.str().c_str()) == 0);
  }
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
  registerTacoRuntimeLibTasks();
  Runtime::add_registration_callback(register_mapper);
  return Runtime::start(argc, argv);
}
