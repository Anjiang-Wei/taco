cmake_policy(SET CMP0026 OLD)

# TODO (rohany): Build only DISTAL::Compiler::AOT if we can't find Legion.
find_package(Legion REQUIRED)
find_package(BLAS REQUIRED)

# Collect include paths and library locations for the DISTAL runtime library.
get_filename_component(DISTAL_RUNTIME_LIB_INCLUDE "../legion/include/" ABSOLUTE)
get_property(DISTAL_RUNTIME_LIB_LOCATION TARGET distal-runtime-lib PROPERTY LOCATION)
# The Legion::Legion target is composed of Legion::LegionRuntime and Legion::RealmRuntime.
# We can find the locations of each of these explicitly as the logic to iterate over lists
# etc within CMake is too annoying.
get_target_property(LEGION_INCLUDE_PATHS Legion::LegionRuntime INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(REALM_INCLUDE_PATHS Legion::RealmRuntime INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(LEGION_LIB_LOCATION Legion::LegionRuntime INTERFACE_LOCATION)
get_target_property(REALM_LIB_LOCATION Legion::RealmRuntime INTERFACE_LOCATION)
# Package up all of the include paths into properly formatted include paths for the
# call to the host compiler.
foreach(PATH IN LISTS LEGION_INCLUDE_PATHS REALM_INCLUDE_PATHS)
    list(APPEND DISTAL_JIT_INCLUDE_PATHS ${PATH})
endforeach()
list(APPEND DISTAL_JIT_INCLUDE_PATHS ${DISTAL_RUNTIME_LIB_INCLUDE} "${BLAS_INCLUDE_DIR}/include")
list(TRANSFORM DISTAL_JIT_INCLUDE_PATHS PREPEND "-I")
list(JOIN DISTAL_JIT_INCLUDE_PATHS " " DISTAL_JIT_INCLUDE_PATHS)
# Package up all of the individual libraries needed by JIT-ed code.
set(DISTAL_JIT_LINK_FLAGS "${LEGION_LIB_LOCATION} ${REALM_LIB_LOCATION} -lstdc++ ${BLAS_LIBRARIES} ${DISTAL_RUNTIME_LIB_LOCATION}")
# Using these defined variables, generate a header file preprocessed with the variables.
configure_file(jit/deps.h.in jit/deps.h @ONLY)

# DISTAL-Runtime
# add_library(distal-runtime)
# TODO (rohany): Handle CUDA generation of the library.
# TODO (rohany): I think that for a first pass we can keep the distal runtime library stationary,
#  include DISTAL after it, and then just reference it here.

# DISTAL-Compiler-Core contains common utilities for DISTAL compiler utilities.
file(GLOB DISTAL_CORE_SOURCE core/*.cpp)
add_library(distal-compiler-core include/distal-compiler-core.h ${DISTAL_CORE_SOURCE})
target_include_directories(distal-compiler-core PRIVATE include)
target_link_libraries(distal-compiler-core PUBLIC taco)
install(TARGETS distal-compiler-core DESTINATION lib)

# DISTAL-Compiler-AOT
file(GLOB DISTAL_COMPILER_AOT_SOURCE distal-compiler-aot.h aot/*.cpp)
add_library(distal-compiler-aot include/distal-compiler-aot.h ${DISTAL_COMPILER_AOT_SOURCE})
target_include_directories(distal-compiler-aot PRIVATE include)
# DISTAL's CMakeLists.txt should get included after the TACO library has been defined.
target_link_libraries(distal-compiler-aot PUBLIC taco distal-compiler-core)
# TODO (rohany): We need to include some "hidden" header files for now to compile
#  everything the way that we want to for DISTAL. These files should be moved to
#  external facing headers for a final version.
target_include_directories(distal-compiler-aot PRIVATE ../src/)
install(TARGETS distal-compiler-aot DESTINATION lib)

# DISTAL-Compiler-JIT
file(GLOB DISTAL_COMPILER_JIT_SOURCE jit/*.cpp)
add_library(distal-compiler-jit include/distal-compiler-jit.h ${DISTAL_COMPILER_JIT_SOURCE})
target_include_directories(distal-compiler-jit PRIVATE include)
target_link_libraries(distal-compiler-jit PUBLIC taco Legion::Legion distal-compiler-core)
# TODO (rohany): This is temporary hacks to link against the DISTAL runtime. Once the initial version
#  of the JIT lands, we will port the DISTAL runtime from legion/ into this directory.
target_include_directories(distal-compiler-jit PRIVATE ../legion/include)
target_link_libraries(distal-compiler-jit PUBLIC distal-runtime-lib)
# TODO (rohany): We need to include some "hidden" header files for now to compile
#  everything the way that we want to for DISTAL. These files should be moved to
#  external facing headers for a final version.
target_include_directories(distal-compiler-jit PRIVATE ../src/)
target_include_directories(distal-compiler-jit PRIVATE "${CMAKE_BINARY_DIR}/DISTAL/jit/")
install(TARGETS distal-compiler-jit DESTINATION lib)
