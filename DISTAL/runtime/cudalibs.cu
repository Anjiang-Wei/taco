#include "task_ids.h"
#include "legion.h"
#include "cudalibs.h"
#include "mappers/default_mapper.h"

using namespace Legion;

void checkCuBLAS(cublasStatus_t status, const char* file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal TACO CUBLAS failure with error code %d in file %s at line %d\n",
            status,
            file,
            line);
    exit(status);
  }
}

void checkCuSparse(cusparseStatus_t status, const char* file, int line) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr,
            "Internal TACO CUSPARSE failure with error code %d in file %s at line %d\n",
            status,
            file,
            line);
    exit(status);
  }
}

// This management of cublasHandle objects was taken from
// https://github.com/nv-legate/legate.core/blob/6b0d6f5a40c39154e0b08a6662e0753903d90867/src/cudalibs.cc.
struct cublasContext;
struct CUDALib {
  CUDALib() : cublas(NULL) {}
  cublasContext* get() {
    if (this->cublas == NULL) {
      CHECK_CUBLAS(cublasCreate(&this->cublas));
    }
    return cublas;
  }
protected:
  cublasContext* cublas;
};

CUDALib& getLib(Processor curProc) {
  static std::map<Processor, CUDALib> handles;
  auto it = handles.find(curProc);
  if (it == handles.end()) {
    return handles[curProc];
  } else {
    return it->second;
  }
}

cublasHandle_t getCuBLAS() {
  auto curProc = Processor::get_executing_processor();
  assert(curProc.kind() == Processor::TOC_PROC);
  return getLib(curProc).get();
}

// Duplicate set of structs and accessors for use with CuSparse. We don't
// bundle these up together as the handles for each of these libraries are
// relatively heavyweight, so applications should have to opt in if they want
// to take on that footprint.
struct cusparseContext;
struct CuSparseLib {
  CuSparseLib() : cusparse(NULL) {}
  cusparseContext* get() {
    if (this->cusparse == NULL) {
      CHECK_CUSPARSE(cusparseCreate(&this->cusparse));
    }
    return cusparse;
  }
  cusparseContext* cusparse;
};

CuSparseLib& getCuSparseLib(Processor curProc) {
  static std::map<Processor, CuSparseLib> handles;
  auto it = handles.find(curProc);
  if (it == handles.end()) {
    return handles[curProc];
  } else {
    return it->second;
  }
}

cusparseHandle_t getCuSparse() {
  auto curProc = Processor::get_executing_processor();
  assert(curProc.kind() == Processor::TOC_PROC);
  return getCuSparseLib(curProc).get();
}

// Definition of initAllCudaLibraries when CUDA is used.
void initAllCudaLibraries(Machine machine, Runtime*, const std::set<Processor>&) {
  // Call the initialization routine for each GPU.
  Machine::ProcessorQuery localGPUs(machine);
  localGPUs.local_address_space();
  localGPUs.only_kind(Processor::TOC_PROC);
  for (auto it = localGPUs.begin(); it != localGPUs.end(); it++) {
    // Create an entry for each GPU in the map of cublasHandle_t's, but
    // don't initialize the handle. This is important, as CuBLAS keeps
    // some sort of thread-level cache, so repeated initializations on
    // the same thread don't do anything, and result in a re-initialization
    // before use on a different thread.
    getLib(*it);
  }
}

void initCuBLASTask(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) { getCuBLAS(); }
void initCuBLAS(Context ctx, Runtime* runtime) {
  // Launch a point task for each GPU in the system to initialize.
  auto gpus = runtime->select_tunable_value(ctx, Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_GPUS).get<size_t>();
  if (gpus == 0) return;
  auto space = runtime->create_index_space(ctx, Rect<1>(0, gpus - 1));
  auto launcher = IndexLauncher(TID_INIT_CUBLAS, space, TaskArgument(), ArgumentMap());
  runtime->execute_index_space(ctx, launcher).wait_all_results();
}

// Intialize CuSparse handles on all processors in a similar manner to CuBLAS.
void initAllCuSparse(Machine machine, Runtime*, const std::set<Processor>&) {
  // Call the initialization routine for each GPU.
  Machine::ProcessorQuery localGPUs(machine);
  localGPUs.local_address_space();
  localGPUs.only_kind(Processor::TOC_PROC);
  for (auto it = localGPUs.begin(); it != localGPUs.end(); it++) {
    // Create an entry for each GPU in the map of cusparseHandle_t's, but
    // don't initialize the handle. This is important, as Cusparse keeps
    // some sort of thread-level cache, so repeated initializations on
    // the same thread don't do anything, and result in a re-initialization
    // before use on a different thread.
    getCuSparseLib(*it);
  }
}

void initCuSparseTask(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) { getCuSparse(); }
void initCuSparse(Context ctx, Runtime* runtime) {
  // Launch a point task for each GPU in the system to initialize.
  auto gpus = runtime->select_tunable_value(ctx, Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_GPUS).get<size_t>();
  if (gpus == 0) return;
  auto space = runtime->create_index_space(ctx, Rect<1>(0, gpus - 1));
  auto launcher = IndexLauncher(TID_INIT_CUSPARSE, space, TaskArgument(), ArgumentMap());
  runtime->execute_index_space(ctx, launcher).wait_all_results();
}

void initCUDA() {
  Runtime::add_registration_callback(initAllCudaLibraries);
  {
    TaskVariantRegistrar registrar(TID_INIT_CUBLAS, "init_cublas");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<initCuBLASTask>(registrar, "init_cublas");
  }
}

void initCuSparseAtStartup() {
  Runtime::add_registration_callback(initAllCuSparse);
  {
    TaskVariantRegistrar registrar(TID_INIT_CUSPARSE, "init_cusparse");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<initCuSparseTask>(registrar, "init_cusparse");
  }
}
