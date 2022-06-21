#include "distal-compiler-core.h"
#include "distal-compiler-jit.h"

// A generated file by the CMake build.
#include "deps.h"

// Public headers exported by TACO.
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/lower/lower.h"
#include "taco/ir/simplify.h"
#include "taco/util/env.h"

// "Hidden" headers from within the TACO source tree.
#include "codegen/codegen_legion_c.h"
#include "codegen/codegen_legion_cuda.h"

#include <dlfcn.h>
#include <fstream>
#include <random>

namespace DISTAL {
namespace Compiler {
namespace JIT {

TensorDistribution::TensorDistribution(std::shared_ptr<Module> module, std::string funcName) : module(module), funcName(funcName) {}

void* TensorDistribution::partition(Legion::Context ctx, Legion::Runtime *runtime, LegionTensor *t) {
  if (!this->module) { return nullptr; }
  return this->module->callLinkedFunction<void*>("partitionFor" + this->funcName, {ctx, runtime, t});
}

void TensorDistribution::apply(Legion::Context ctx, Legion::Runtime *runtime, LegionTensor *t, void* partitions) {
  if (!this->module) { return; }
  this->module->callLinkedFunction<void>(this->funcName, {ctx, runtime, t, partitions});
}

Kernel::Kernel(std::shared_ptr<Module> module, bool hasPartitionMethod) : module(module), hasPartitionMethod(hasPartitionMethod) {}

void* Kernel::partition(Legion::Context ctx, Legion::Runtime *runtime, std::vector<LegionTensor *> tensors) {
  if (!hasPartitionMethod) {
    return nullptr;
  }
  std::vector<void*> args;
  args.reserve(tensors.size() + 2);
  args.push_back(ctx);
  args.push_back(runtime);
  for (auto it : tensors) {
    args.push_back(it);
  }
  return this->module->callLinkedFunction<void*>("partitionForcompute", args);
}

void Kernel::compute(Legion::Context ctx, Legion::Runtime *runtime, std::vector<LegionTensor *> tensors, void* partitions) {
  std::vector<void*> args;
  args.reserve(tensors.size() + 2);
  args.push_back(ctx);
  args.push_back(runtime);
  for (auto it : tensors) {
    args.push_back(it);
  }
  if (hasPartitionMethod) {
    args.push_back(partitions);
  }
  return this->module->callLinkedFunction<void>("compute", args);
}

JITResult::JITResult(std::vector<TensorDistribution> distributions, Kernel kernel) : distributions(distributions), kernel(kernel) {}

Computation JITResult::bind(std::vector<LegionTensor*> tensors) {
  return Computation(tensors, this->distributions, this->kernel);
}

JITResult compile(Legion::Context ctx, Legion::Runtime* runtime, taco::IndexStmt stmt) {
  // TODO (rohany): This doesn't handle a case where the data is distributed into
  //  the GPUs but the code is distributed onto CPUs.
  // Compile all data distribution statements and the kernel into the same module.
  auto module = std::make_shared<Module>(DISTAL::Compiler::Core::stmtUsesGPU(stmt) ? Module::GPU : Module::CPU);

  auto lowerDistribution = [&](std::string name, taco::IndexStmt stmt) {
    taco_iassert(DISTAL::Compiler::Core::stmtIsDistributed(stmt));
    taco::LowerOptions options;
    // First, lower the partitioning code.
    // TODO (rohany): We should be able to do separate assemble and compute.
    options.assemble = false;
    options.compute = false;
    options.pack = false;
    options.unpack = false;
    options.legion = true;
    options.waitOnFuture = true;
    options.partition = true;
    options.setPlacementPrivilege = false;
    // JIT code needs to use the partition pack as a pointer.
    options.partitionPackAsPointer = true;
    auto partition = taco::lower(stmt, "partitionFor" + name, options);
    // Now lower the compute code.
    options.compute = true;
    options.partition = false;
    auto compute = taco::lower(stmt, name, options);
    return taco::ir::Block::make(partition, compute);
  };

  // Compile each of the tensor distributions in the input statement.
  std::vector<std::string> distributionFuncNames;
  auto tensors = DISTAL::Compiler::Core::getTensors(stmt);
  for (auto t : tensors) {
    // We'll attempt to transparently handle when a distribution is not
    // set for input TensorVars.
    if (!t.getDistribution().empty()) {
      auto trans = t.translateDistribution();
      std::string funcName = "distribute" + t.getName();
      auto lowered = lowerDistribution(funcName, trans);
      distributionFuncNames.push_back(funcName);
      module->addFunction(lowered);
    }
  }

  auto lowerCompute = [&](std::string name, taco::IndexStmt stmt) {
    bool sparseLHS = DISTAL::Compiler::Core::stmtHasSparseLHS(stmt);
    if (DISTAL::Compiler::Core::stmtIsDistributed(stmt)) {
      taco::LowerOptions options;
      // First, lower the partitioning code.
      // TODO (rohany): We should be able to do separate assemble and compute.
      options.assemble = sparseLHS;
      options.compute = false;
      options.pack = false;
      options.unpack = false;
      options.legion = true;
      options.waitOnFuture = true;
      options.partition = true;
      options.setPlacementPrivilege = false;
      // JIT code needs to use the partition pack as a pointer.
      options.partitionPackAsPointer = true;
      auto partition = taco::lower(stmt, "partitionFor" + name, options);
      // Now lower the compute code.
      options.compute = true;
      options.partition = false;
      auto compute = taco::lower(stmt, name, options);
      return taco::ir::Block::make(partition, compute);
    } else {
      taco::LowerOptions options;
      // First, lower the partitioning code.
      // TODO (rohany): We should be able to do separate assemble and compute.
      options.assemble = sparseLHS;
      options.compute = true;
      options.pack = false;
      options.unpack = false;
      options.legion = true;
      options.partition = true;
      options.waitOnFuture = true;
      options.setPlacementPrivilege = false;
      auto lowered = taco::lower(stmt, name, options);
      if (sparseLHS) {
        // For some codes that use assemble, we need to explicitly
        // call simplify otherwise TACO can generate some dead code
        // that causes errors during host compilation.
        lowered = taco::ir::simplify(lowered);
      }
      return lowered;
    }
  };
  // TODO (rohany): Make these names a shared variable, and stop naming things computeLegion.
  // Next, lower the actual compute kernel into IR.
  auto lowered = lowerCompute("compute", stmt);
  module->addFunction(lowered);

  // Finally compile everything.
  module->compile();

  // Before we return, we need to register all of the tasks created by this module.
  module->callLinkedFunctionRaw<void>("dynamicallyRegisterDISTALTasks", {ctx, runtime});

  // Construct TensorDistribution kernels for each of the distribution statements.
  std::vector<TensorDistribution> distributions;
  int idx = 0;
  for (auto t : tensors) {
    if (t.getDistribution().empty()) {
      distributions.push_back({nullptr, ""});
    } else {
      distributions.push_back({module, distributionFuncNames[idx]});
      idx++;
    }
  }

  return JITResult(distributions, {module, DISTAL::Compiler::Core::stmtIsDistributed(stmt)});
}

Computation::Computation(std::vector<LegionTensor*> tensors, std::vector<TensorDistribution> distributions,
                         Kernel kernel) : tensors(tensors), distributions(distributions), kernel(kernel),
                                          distributionPartitions(tensors.size()), computePartitions(nullptr) {
  taco_iassert(this->tensors.size() == this->distributions.size());
}

void Computation::distribute(Legion::Context ctx, Legion::Runtime *runtime) {
  for (size_t i = 0; i < this->distributionPartitions.size(); i++) {
    if (this->distributionPartitions[i] == nullptr) {
      this->distributionPartitions[i] = this->distributions[i].partition(ctx, runtime, this->tensors[i]);
    }
  }
  for (size_t i = 0; i < this->distributions.size(); i++) {
    this->distributions[i].apply(ctx, runtime, this->tensors[i], this->distributionPartitions[i]);
  }
}

LegionTensorPartition Computation::getDistributionPartition(Legion::Context ctx, Legion::Runtime* runtime, int idx) {
  if (this->distributionPartitions[idx] == nullptr) {
    this->distributionPartitions[idx] = this->distributions[idx].partition(ctx, runtime, this->tensors[idx]);
  }
  return *(LegionTensorPartition*)this->distributionPartitions[idx];
}

void Computation::compute(Legion::Context ctx, Legion::Runtime *runtime) {
  if (this->computePartitions == nullptr) {
    this->computePartitions = this->kernel.partition(ctx, runtime, this->tensors);
  }
  this->kernel.compute(ctx, runtime, this->tensors, this->computePartitions);
}

// Definitions for DISTAL::Compiler::JIT::Module.

std::string Module::chars = "abcdefghijkmnpqrstuvwxyz0123456789";

Module::Module(Module::Target target) : libHandle(nullptr), target(target) {
  // TODO (rohany): Stop using TACO's temporary directory creation logic as it doesn't
  //  take things like shards into account.
  this->tmpdir = taco::util::getTmpdir();

  // Construct a random name for the generated library.
  std::default_random_engine gen = std::default_random_engine();
  std::uniform_int_distribution<int> randint = std::uniform_int_distribution<int>(0, chars.length() - 1);
  this->libname.resize(12);
  for (int i = 0; i < 12; i++) {
    this->libname[i] = Module::chars[randint(gen)];
  }
}

void Module::addFunction(taco::ir::Stmt stmt) {
  this->funcs.push_back(stmt);
}

void Module::compile() {
  // Compile the attached functions into a piece of source code.
  if (this->target == Module::CPU) {
    taco::ir::CodegenLegionC codegen(this->source, taco::ir::CodeGen::ImplementationNoHeaderGen);
    codegen.compile(taco::ir::Block::make(this->funcs));
  } else {
    taco::ir::CodegenLegionCuda codegen(this->source, taco::ir::CodeGen::ImplementationNoHeaderGen);
    codegen.compile(taco::ir::Block::make(this->funcs));
  }

  // Dump the source into a file in our temporary directory.
  std::ofstream sourceFile;
  std::string fileEnding = this->target == Module::CPU ? ".cpp" : ".cu";
  std::string sourceFileName = this->tmpdir + this->libname + fileEnding;
  sourceFile.open(sourceFileName);
  sourceFile << this->source.str();
  sourceFile.close();

  // Now, invoke the host compiler to actually compile the generated code
  // into a dynamic library.
  std::string libPath = this->tmpdir + this->libname + ".so";
  std::string cc;
  std::string cflags;
  if (this->target == Module::CPU) {
    cc = "c++";
    cflags = taco::util::getFromEnv("DISTAL_CPP_FLAGS", "-O3 -ffast-math") + " -shared -fPIC --std=c++11";
  } else {
    cc = "nvcc";
    cflags = taco::util::getFromEnv("DISTAL_CPP_FLAGS", "-O3 --use_fast_math") + " --std=c++11 -shared -Xcompiler -fPIC";
  }
#ifdef REALM_USE_OPENMP
  if (this->target == Module::CPU) {
    cflags += " -fopenmp";
  }
#endif
  std::string compileCmd = cc + " " + cflags + " " + sourceFileName + " -o " + libPath +
                           " " + DISTAL_JIT_INCLUDE_PATHS +
			   " " + DISTAL_JIT_LINK_FLAGS
			   ;
  bool printCompileOutput = taco::util::getFromEnv("DISTAL_PRINT_HOST_COMPILER_OUTPUT", "OFF") != "OFF";
  if (!printCompileOutput) {
    compileCmd += " > /dev/null 2> /dev/null";
  }

  bool printCompileCmd = taco::util::getFromEnv("DISTAL_PRINT_JIT_COMMAND", "OFF") != "OFF";
  if (printCompileCmd) {
    std::cout << compileCmd << std::endl;
  }

  int err = system(compileCmd.data());
  taco_uassert(err == 0) << "Compilation command failed:\n" << compileCmd
                         << "\nreturned " << err;

  // Finally, link in the compiled library.
  taco_uassert(this->libHandle == nullptr);
  this->libHandle = dlopen(libPath.data(), RTLD_NOW | RTLD_LOCAL);
  if (this->libHandle == nullptr) {
    taco_uassert(this->libHandle) << "Failed to load generated shared library with error: " << std::string(dlerror());
  }
}

std::string Module::getSource() {
  return this->source.str();
}

void* Module::getFuncPtr(std::string name) {
  return dlsym(this->libHandle, name.data());
}

// End definitions for DISTAL::Compiler::JIT::Module.

} // namespace JIT
} // namespace Compiler
} // namespace DISTAL
