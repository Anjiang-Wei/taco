cmake_policy(SET CMP0026 OLD)

find_package(BLAS REQUIRED)
find_package(HDF5 REQUIRED)
find_package(Legion REQUIRED)
if(OPENMP)
  find_package(OpenMP REQUIRED)
endif(OPENMP)
# Since some architectures don't support TBLIS, we can't expect it to be included.
find_package(TBLIS)
if(Legion_USE_CUDA)
  find_package(CUDA REQUIRED)
endif()

# We're targeting GPUs with at least -sm_60 capability. We need this so that
# we can use the atomicAdd() function in leaf kernels.
# On Lassen, we need to target -sm_70.
if (DEFINED ENV{LLNL_COMPUTE_NODES})
    set(CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
else()
    set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
endif()

# Find the BLAS library to use. We have to manually add the include path
# to the BLAS library because CMake doesn't set this for us.
find_path(BLAS_INCLUDE_DIR cblas.h ${CMAKE_PREFIX_PATH})
# On some setups the find_path to get BLAS_INCLUDE_DIR sometimes includes
# the actual 'include' suffix, so just set both as valid paths.
set(BLAS_INCLUDE_DIRS ${BLAS_INCLUDE_DIR} "${BLAS_INCLUDE_DIR}/include")

option(DISTAL_USE_LOGGING_MAPPER OFF)
if (DISTAL_USE_LOGGING_MAPPER)
    add_definitions(-DTACO_USE_LOGGING_MAPPER)
endif()
# DISTAL-Runtime is the runtime library to interact with DISTAL generated code,
# as well as support that DISTAL generated kernels need.
if (Legion_USE_CUDA)
    file(GLOB DISTAL_RUNTIME_SOURCE runtime/distal-runtime.h runtime/*.cpp runtime/distal-runtime.h runtime/*.cu)
    cuda_add_library(distal-runtime SHARED ${DISTAL_RUNTIME_SOURCE})
else()
    file(GLOB DISTAL_RUNTIME_SOURCE runtime/distal-runtime.h runtime/*.cpp)
    add_library(distal-runtime SHARED ${DISTAL_RUNTIME_SOURCE})
endif()

target_include_directories(distal-runtime PRIVATE include/ include/runtime/ "${CMAKE_BINARY_DIR}/include")
target_link_libraries(distal-runtime Legion::Legion ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})
if (TBLIS_FOUND)
    target_link_libraries(distal-runtime TBLIS::TBLIS)
    target_compile_definitions(distal-runtime PUBLIC DISTAL_USE_TBLIS)
endif()
if (Legion_USE_OpenMP)
    target_compile_options(distal-runtime PRIVATE ${OpenMP_CXX_FLAGS})
endif()
if (Legion_USE_CUDA)
    target_compile_definitions(distal-runtime PUBLIC TACO_USE_CUDA)
    target_include_directories(distal-runtime PRIVATE ../deps/cub/)
    target_link_libraries(distal-runtime ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
endif()
# TODO (rohany): Do I want this to be global instead?
# TODO (rohany): Eventually stop directly including include/runtime.
set(DISTAL_RUNTIME_INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/runtime"
        "${CMAKE_BINARY_DIR}/include"
	${BLAS_INCLUDE_DIRS}
        # This may not be found, but it will be undefined if not found, so no harm done.
        "${TBLIS_INCLUDE_DIRS}"
)
set(DISTAL_RUNTIME_INCLUDE_DIRS ${DISTAL_RUNTIME_INCLUDE_DIRS} PARENT_SCOPE)

# DISTAL-Compiler-Core contains common utilities for DISTAL compiler utilities.
file(GLOB DISTAL_CORE_SOURCE core/*.cpp)
add_library(distal-compiler-core SHARED include/distal-compiler-core.h ${DISTAL_CORE_SOURCE})
target_include_directories(distal-compiler-core PRIVATE include)
target_link_libraries(distal-compiler-core PUBLIC taco)
install(TARGETS distal-compiler-core DESTINATION lib)

# DISTAL-Compiler-AOT is all packages needed for ahead-of-time compilation with DISTAL.
file(GLOB DISTAL_COMPILER_AOT_SOURCE distal-compiler-aot.h aot/*.cpp)
add_library(distal-compiler-aot SHARED include/distal-compiler-aot.h ${DISTAL_COMPILER_AOT_SOURCE})
target_include_directories(distal-compiler-aot PRIVATE include)
# DISTAL's CMakeLists.txt should get included after the TACO library has been defined.
target_link_libraries(distal-compiler-aot PUBLIC taco distal-compiler-core)
# TODO (rohany): We need to include some "hidden" header files for now to compile
#  everything the way that we want to for DISTAL. These files should be moved to
#  external facing headers for a final version.
target_include_directories(distal-compiler-aot PRIVATE ../src/)
install(TARGETS distal-compiler-aot DESTINATION lib)

# TODO (rohany): Enable just including distal-runtime.h directly, rather than
#  the individual files.
# Collect include paths and library locations for the DISTAL runtime library.
get_filename_component(DISTAL_RUNTIME_LIB_INCLUDE "include/runtime" ABSOLUTE)
get_property(DISTAL_RUNTIME_LIB_LOCATION TARGET distal-runtime PROPERTY LOCATION)
# The Legion::Legion target is composed of Legion::LegionRuntime and Legion::RealmRuntime.
# We can find the locations of each of these explicitly as the logic to iterate over lists
# etc within CMake is too annoying.
get_target_property(LEGION_INCLUDE_PATHS Legion::LegionRuntime INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(REALM_INCLUDE_PATHS Legion::RealmRuntime INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(LEGION_LIB_LOCATION Legion::LegionRuntime INTERFACE_LOCATION)
get_target_property(REALM_LIB_LOCATION Legion::RealmRuntime INTERFACE_LOCATION)
# We have to clean up the physical paths constructed by CMake as if the full
# shared library path is passed, NVCC gets unhappy and can't figure out what
# kind of file a shared library is. So, we have to make sure all of the shared
# libraries that we construct here actually end with .so, and nothing else. We
# also have to make sure not to do anything if we end with .dylib for local Mac
# development.
if (NOT ${LEGION_LIB_LOCATION} MATCHES ".*.so$" AND NOT ${LEGION_LIB_LOCATION} MATCHES ".*.dylib$")
  get_filename_component(LEGION_LIB_DIR ${LEGION_LIB_LOCATION} DIRECTORY)
  get_filename_component(LEGION_LIB_NAME ${LEGION_LIB_LOCATION} NAME_WLE)
  set(LEGION_LIB_LOCATION "${LEGION_LIB_DIR}/${LEGION_LIB_NAME}")
endif()
if (NOT ${REALM_LIB_LOCATION} MATCHES ".*.so$" AND NOT ${REALM_LIB_LOCATION} MATCHES ".*.dylib$")
  get_filename_component(REALM_LIB_DIR ${REALM_LIB_LOCATION} DIRECTORY)
  get_filename_component(REALM_LIB_NAME ${REALM_LIB_LOCATION} NAME_WLE)
  set(REALM_LIB_LOCATION "${REALM_LIB_DIR}/${REALM_LIB_NAME}")
endif()

# Package up all of the include paths into properly formatted include paths for the
# call to the host compiler.
foreach(PATH IN LISTS LEGION_INCLUDE_PATHS REALM_INCLUDE_PATHS)
    list(APPEND DISTAL_JIT_INCLUDE_PATHS ${PATH})
endforeach()
list(APPEND DISTAL_JIT_INCLUDE_PATHS ${DISTAL_RUNTIME_LIB_INCLUDE} ${BLAS_INCLUDE_DIRS})
list(TRANSFORM DISTAL_JIT_INCLUDE_PATHS PREPEND "-I")
list(JOIN DISTAL_JIT_INCLUDE_PATHS " " DISTAL_JIT_INCLUDE_PATHS)
# Package up all of the individual libraries needed by JIT-ed code.
set(DISTAL_JIT_LINK_FLAGS "${LEGION_LIB_LOCATION} ${REALM_LIB_LOCATION} -lstdc++ ${BLAS_LIBRARIES} ${DISTAL_RUNTIME_LIB_LOCATION}")
# Using these defined variables, generate a header file preprocessed with the variables.
configure_file(jit/deps.h.in jit/deps.h @ONLY)

# DISTAL-Compiler-JIT is all packages needed for JIT compilation with DISTAL.
file(GLOB DISTAL_COMPILER_JIT_SOURCE jit/*.cpp)
add_library(distal-compiler-jit SHARED include/distal-compiler-jit.h ${DISTAL_COMPILER_JIT_SOURCE})
target_include_directories(distal-compiler-jit PRIVATE include ${DISTAL_RUNTIME_INCLUDE_DIRS})
target_link_libraries(distal-compiler-jit PUBLIC taco Legion::Legion distal-compiler-core distal-runtime)
# TODO (rohany): We need to include some "hidden" header files for now to compile
#  everything the way that we want to for DISTAL. These files should be moved to
#  external facing headers for a final version.
target_include_directories(distal-compiler-jit PRIVATE ../src/)
target_include_directories(distal-compiler-jit PRIVATE "${CMAKE_BINARY_DIR}/DISTAL/jit/")
install(TARGETS distal-compiler-jit DESTINATION lib)
