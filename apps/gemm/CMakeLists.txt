find_package(Legion REQUIRED)

add_executable(gemm main.cpp)
include_directories(${TACO_INCLUDE_DIR} ${DISTAL_RUNTIME_INCLUDE_DIRS})
target_link_libraries(gemm PUBLIC Legion::Legion distal-compiler-jit distal-runtime)
# TODO (rohany): The runtime library should compile with OpenMP and not require this
#  of the source program that uses the libraries.
if (Legion_USE_OpenMP)
  target_compile_options(gemm PRIVATE ${OpenMP_CXX_FLAGS})
endif()
