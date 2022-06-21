#ifndef DISTAL_COMPILER_JIT_H
#define DISTAL_COMPILER_JIT_H

#include "taco.h"
#include "legion.h"

#include "distal-runtime.h"

// TODO (rohany): This forward declaration will eventually need to be scoped within a namespace.
struct LegionTensor;

namespace DISTAL {
namespace Compiler {
namespace JIT {

// Necessary forward declarations.
struct JITResult;
class Computation;

// Module is a wrapper around a dynamically linked library used to invoke methods
// from the dynamically linked library. A module contains pointers and data structures
// that are specific to the CPU memory of a particular address space. It is _not_
// portable across machines and is not safe to serialize and transfer over the network.
class Module {
public:
  // Target backend to compile to.
  enum Target {
    CPU,
    GPU,
  };
  // Create a module.
  Module(Target target=CPU);
  // Add a compile TACO ir statement to the set of functions being compiled by this module.
  void addFunction(taco::ir::Stmt stmt);
  // Compile all functions attached to this module into a dynamic library.
  void compile();
  // Get the C++ source code of the compiled module.
  std::string getSource();

  // TODO (rohany): Potentially move these to a distal-compiler-jit.inl file.
  // Call a compiled function with the provided name from the compiled library.
  // The return type T must be specified by the caller. This function assumes that
  // the function with the provided name exists within the library.
  template<typename T>
  T callLinkedFunctionRaw(std::string name, std::vector<void*> args) {
    typedef T (*fnptr_t)(void**);
    static_assert(sizeof(void*) == sizeof(fnptr_t),
                  "Unable to cast dlsym() returned void pointer to function pointer");
    void* rawPtr = this->getFuncPtr(name);
    // TODO (rohany): Turn this into a DISTAL assertion rather than using TACO's
    //  error handling infrastructure.
    taco_iassert(rawPtr != nullptr) << "could not find " << name << " within dylib.";
    fnptr_t funcPtr;
    *reinterpret_cast<void**>(&funcPtr) = rawPtr;
    return funcPtr(args.data());
  }

  // callLinkedFunction is similar to callLinkedFunctionRaw but interacts with the
  // shim functions generated by DISTAL to simplify the calling convention.
  template<typename T>
  T callLinkedFunction(std::string name, std::vector<void*> args) {
    return this->callLinkedFunctionRaw<T>("_shim_" + name, args);
  }
private:
  // getFuncPtr returns a function pointer to a function with the provided
  // name within the currently loaded library. It returns nullptr if the
  // function does not exist within the library.
  void* getFuncPtr(std::string name);

  std::stringstream source;
  std::string libname;
  std::string tmpdir;
  void* libHandle;
  std::vector<taco::ir::Stmt> funcs;
  Target target;
  static std::string chars;
};

// TensorDistribution is a compiled kernel that moves an input tensor into a
// distribution described by the compiled Tensor Distribution Notation statement.
class TensorDistribution {
public:
  // partition generates all of the partitions needed to perform the distribution.
  void* partition(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* t);
  // apply invokes the compiled kernel to distribute the data.
  // TODO (rohany): can this return a FutureMap to wait on or ignore?
  void apply(Legion::Context ctx, Legion::Runtime* runtime, LegionTensor* t, void* partition);
  // TODO (rohany): Add methods to access pieces of the partition created by
  //  the generated code.
private:
  TensorDistribution(std::shared_ptr<Module> module, std::string funcName);

  // The compiler methods are able to construct TensorDistribution objects.
  friend JITResult compile(Legion::Context, Legion::Runtime*, taco::IndexStmt);

  std::shared_ptr<Module> module;
  std::string funcName;
};

// Kernel is a compiled kernel that performs the computation described by
// a scheduled Tensor Index Notation statement.
class Kernel {
public:
  // partition generates all of the partitions needed to perform the distribution.
  void* partition(Legion::Context ctx, Legion::Runtime* runtime, std::vector<LegionTensor*> tensors);
  // compute invokes the compiled kernel.
  void compute(Legion::Context ctx, Legion::Runtime* runtime, std::vector<LegionTensor*> tensors, void* partition);
private:
  Kernel(std::shared_ptr<Module> module, bool hasPartitionMethod=true);

  // The compiler methods are able to construct Kernel objects.
  friend JITResult compile(Legion::Context, Legion::Runtime*, taco::IndexStmt);

  std::shared_ptr<Module> module;
  bool hasPartitionMethod;
};

// TODO (rohany): This should eventually take some sort of configurations struct that enables
//  controlling all the knobs that are currently configured manually in tests.
// `compile` JIT compiles a taco::IndexStmt and dynamically links it into the current process,
// registering all tasks created by DISTAL. `compile` compiles both the provided statement as
// well as all data distribution code for each tensor in the statement.
struct JITResult {
  JITResult(std::vector<TensorDistribution> distributions, Kernel kernel);
  std::vector<TensorDistribution> distributions;
  Kernel kernel;
  // Construct a computation bound to a particular set of tensors from the JIT-ed kernels.
  Computation bind(std::vector<LegionTensor*> tensors);
};
JITResult compile(Legion::Context ctx, Legion::Runtime* runtime, taco::IndexStmt stmt);

// Computation represents a generated set of data distribution and computation operations
// bound to particular LegionTensor objects. It maintains the internal state to invoke
// the generated kernels upon the bound operands.
class Computation {
public:
  void distribute(Legion::Context ctx, Legion::Runtime* runtime);
  LegionTensorPartition getDistributionPartition(Legion::Context ctx, Legion::Runtime* runtime, int idx);
  void compute(Legion::Context ctx, Legion::Runtime* runtime);
private:
  friend JITResult;
  Computation(std::vector<LegionTensor*> tensors, std::vector<TensorDistribution> distributions, Kernel kernel);
  std::vector<LegionTensor*> tensors;
  std::vector<TensorDistribution> distributions;
  Kernel kernel;
  std::vector<void*> distributionPartitions;
  void* computePartitions;
};

} // namespace JIT
} // namespace Compiler
} // namespace DISTAL

#endif // DISTAL_COMPILER_JIT_H
