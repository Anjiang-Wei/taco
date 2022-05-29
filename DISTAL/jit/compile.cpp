#include "distal-compiler-core.h"
#include "distal-compiler-jit.h"

// A generated file by the CMake build.
#include "deps.h"

// Public headers exported by TACO.
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/lower/lower.h"
#include "taco/util/env.h"

// "Hidden" headers from within the TACO source tree.
#include "codegen/codegen_legion_c.h"

#include <dlfcn.h>
#include <fstream>
#include <random>

namespace DISTAL {
namespace Compiler {
namespace JIT {

TensorDistribution::TensorDistribution(std::shared_ptr<Module> module, std::string funcName) : module(module), funcName(funcName) {}

void TensorDistribution::partition(Legion::Context ctx, Legion::Runtime *runtime, LegionTensor *t) {
  this->partitionPack = this->module->callLinkedFunction<void*>("partitionFor" + this->funcName, {ctx, runtime, t});
}

void TensorDistribution::apply(Legion::Context ctx, Legion::Runtime *runtime, LegionTensor *t) {
  this->module->callLinkedFunction<void>(this->funcName, {ctx, runtime, t, this->partitionPack});
}

Kernel::Kernel(std::shared_ptr<Module> module) : module(module) {}

void Kernel::partition(Legion::Context ctx, Legion::Runtime *runtime, std::vector<LegionTensor *> tensors) {
  std::vector<void*> args;
  args.reserve(tensors.size() + 2);
  args.push_back(ctx);
  args.push_back(runtime);
  for (auto it : tensors) {
    args.push_back(it);
  }
  this->partitions = this->module->callLinkedFunction<void*>("partitionForcompute", args);
}

void Kernel::compute(Legion::Context ctx, Legion::Runtime *runtime, std::vector<LegionTensor *> tensors) {
  std::vector<void*> args;
  args.reserve(tensors.size() + 2);
  args.push_back(ctx);
  args.push_back(runtime);
  for (auto it : tensors) {
    args.push_back(it);
  }
  taco_iassert(this->partitions != nullptr);
  args.push_back(this->partitions);
  return this->module->callLinkedFunction<void>("compute", args);
}

JITResult::JITResult(std::vector<TensorDistribution> distributions, Kernel kernel) : distributions(distributions), kernel(kernel) {}

JITResult compile(Legion::Context ctx, Legion::Runtime* runtime, taco::IndexStmt stmt) {
  // TODO (rohany): This doesn't handle a case where the data is distributed into
  //  the GPUs but the code is distributed onto CPUs.
  // Compile all data distribution statements and the kernel into the same module.
  auto module = std::make_shared<Module>(DISTAL::Compiler::Core::stmtUsesGPU(stmt) ? Module::GPU : Module::CPU);

  auto lower = [&](std::string name, taco::IndexStmt stmt) {
    taco::LowerOptions options;
    // First, lower the partitioning code.
    // TODO (rohany): Should assemble be true?
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
    taco_iassert(!t.getDistribution().empty());
    auto trans = t.translateDistribution();
    std::string funcName = "distribute" + t.getName();
    auto lowered = lower(funcName, trans);
    distributionFuncNames.push_back(funcName);
    module->addFunction(lowered);
  }

  // TODO (rohany): Make these names a shared variable, and stop naming things computeLegion.
  // Next, lower the actual compute kernel into IR.
  auto lowered = lower("compute", stmt);
  module->addFunction(lowered);

  // Finally compile everything.
  module->compile();

  // Before we return, we need to register all of the tasks created by this module.
  module->callLinkedFunctionRaw<void>("dynamicallyRegisterDISTALTasks", {ctx, runtime});

  // Construct TensorDistribution kernels for each of the distribution statements.
  std::vector<TensorDistribution> distributions;
  for (auto it : distributionFuncNames) {
    distributions.push_back({module, it});
  }

  return JITResult(distributions, {module});
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
  taco_iassert(this->target == Module::CPU);
  // Compile the attached functions into a piece of source code.
  taco::ir::CodegenLegionC codegen(this->source, taco::ir::CodeGen::ImplementationNoHeaderGen);
  codegen.compile(taco::ir::Block::make(this->funcs));

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
  std::string cc = "c++";
  std::string cflags = taco::util::getFromEnv("DISTAL_CPPFLAGS", "-O3 -ffast-math --std=c++11") + " -shared -fPIC";
#ifdef REALM_USE_OPENMP
  cflags += " -fopenmp";
#endif
  std::string compileCmd = cc + " " + cflags + " " + sourceFileName + " -o " + libPath +
                           " " + DISTAL_JIT_INCLUDE_PATHS +
                           " " + DISTAL_JIT_LINK_FLAGS
                           ;

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
