list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Legion REQUIRED)
if(OPENMP)
  find_package(OpenMP REQUIRED)
endif(OPENMP)

# For non-system provided installations, specify HDF5_ROOT or TBLIS_ROOT
# environment variables as appropriate.
find_package(HDF5 REQUIRED)
find_package(TBLIS REQUIRED)

# Find the BLAS library to use. We have to manually add the include path
# to the BLAS library because CMake doesn't set this for us.
find_package(BLAS REQUIRED)
find_path(BLAS_INCLUDE_DIR cblas.h ${CMAKE_PREFIX_PATH})
include_directories("${BLAS_INCLUDE_DIR}/include")

# Include the headers and supporting source files.
include_directories(include/ "${CMAKE_BINARY_DIR}/include" cub/)
file(GLOB LG_SOURCES src/*.cpp)
file(GLOB LG_CU_SOURCES src/*.cu)
if (Legion_USE_CUDA)
  if (NOT DEFINED TACO_CUDA_LIBS)
    message(ERROR "Please set TACO_CUDA_LIBS to the path to CUDA libraries")
  endif()
  link_directories(${TACO_CUDA_LIBS})
endif()

# Collect sources that can be also used by tools outside of TACO codes.
set(LG_EXTRA_SOURCES src/hdf5_utils.cpp src/string_utils.cpp src/error.cpp src/dummy_read.cpp)

# Optimize these codes.
SET(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")

option(TACO_USE_LOGGING_MAPPER OFF)
if (TACO_USE_LOGGING_MAPPER)
    add_definitions(-DTACO_USE_LOGGING_MAPPER)
endif()


# We're targeting GPUs with at least -sm_60 capability. We need this so that
# we can use the atomicAdd() function in leaf kernels.
# On Lassen, we need to target -sm_70.
if (DEFINED ENV{LLNL_COMPUTE_NODES})
  set(CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
else()
  set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
endif()

# Create two separate versions of the DISTAL runtime library, for whether
# or not we use CUDA.
add_library(distal-runtime-lib ${LG_SOURCES})
target_include_directories(distal-runtime-lib PRIVATE include)
target_link_libraries(distal-runtime-lib Legion::Legion)
if (OPENMP)
    target_compile_options(distal-runtime-lib PRIVATE ${OpenMP_CXX_FLAGS})
endif(OPENMP)

if (Legion_USE_CUDA)
  cuda_add_library(distal-runtime-lib-cuda ${LG_SOURCES} ${LG_CU_SOURCES})
  target_compile_definitions(distal-runtime-lib-cuda PUBLIC TACO_USE_CUDA)
  target_include_directories(distal-runtime-lib-cuda PRIVATE include)
  target_link_libraries(distal-runtime-lib-cuda Legion::Legion cublas cusparse)
  if (OPENMP)
      target_compile_options(distal-runtime-lib-cuda PRIVATE ${OpenMP_CXX_FLAGS})
  endif(OPENMP)
endif()

function(add_app_folder folder)
    add_executable("${folder}" "${folder}/main.cpp" "${folder}/taco-generated.cpp")
    target_include_directories("${folder}" PRIVATE "${folder}")
    # Unfortunately, older versions of CMake do not export BLAS::BLAS.
    target_link_libraries("${folder}" distal-runtime-lib Legion::Legion ${BLAS_LIBRARIES} TBLIS::TBLIS)
    if (OPENMP)
      target_compile_options("${folder}" PRIVATE ${OpenMP_CXX_FLAGS})
    endif(OPENMP)

    if (Legion_USE_CUDA)
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${folder}/taco-generated.cu")
            cuda_add_executable("${folder}-cuda" "${folder}/main.cpp" "${folder}/taco-generated.cu")
            target_link_libraries("${folder}-cuda" distal-runtime-lib-cuda Legion::Legion cublas cusparse -fopenmp)
            if (OPENMP)
                target_compile_options("${folder}-cuda" PRIVATE ${OpenMP_CXX_FLAGS})
            endif(OPENMP)
            target_compile_definitions("${folder}-cuda" PUBLIC TACO_USE_CUDA)
            set_property(TARGET "${folder}-cuda" PROPERTY CUDA_STANDARD 14)
        endif()
    endif()
endfunction()

set(TEST_BINARIES "")
macro(add_basic_legion_test binary tag)
    # TODO (rohany): See what more runtime debug flags I can place here.
    if (OPENMP)
        add_test(NAME "legion-${binary}-${tag}" COMMAND ${binary} -lg:partcheck -lg:safe_mapper -ll:ocpu 1 -ll:othr 1 -ll:csize 5000 ${ARGN})
    else()
        add_test(NAME "legion-${binary}-${tag}" COMMAND ${binary} -lg:partcheck -lg:safe_mapper -ll:cpu 1 -ll:csize 5000 ${ARGN})
    endif()
    # Collect all the test binaries that we depend on.
    list(APPEND TEST_BINARIES "${binary}")
endmacro()
macro(add_basic_legion_gpu_test binary tag)
    if (Legion_USE_CUDA)
        add_test(NAME "legion-${binary}-cuda-${tag}" COMMAND "${binary}-cuda" -lg:partcheck -lg:safe_mapper -ll:ocpu 1 -ll:othr 1 -ll:gpu 1 -ll:csize 5000 -ll:fsize 2048 ${ARGN})
        # Collect all the test binaries that we depend on.
        list(APPEND TEST_BINARIES "${binary}-cuda")
    endif()
endmacro()
macro(add_basic_legion_cpugpu_test binary tag)
    add_basic_legion_test(${binary} ${tag} ${ARGN})
    add_basic_legion_gpu_test(${binary} ${tag} ${ARGN})
endmacro()

add_app_folder(pummaMM)
add_app_folder(summaMM)
add_app_folder(cannonMM)
add_app_folder(chemTest)
add_app_folder(placement-test)
add_app_folder(johnsonMM)
add_app_folder(solomonikMM)
add_app_folder(cosma)
add_app_folder(ttv)
add_app_folder(mttkrp)
add_app_folder(ttmc)
add_app_folder(innerprod)
add_app_folder(matvec-adapt)

# Sparse codes.
add_app_folder(spmv)
add_app_folder(spttv)
add_app_folder(spmm)
add_app_folder(sddmm)
add_app_folder(spmttkrp)
add_app_folder(spadd3)
add_app_folder(spinnerprod)
add_app_folder(spvecadd)

# Add a variety of tests for the generated code.

# Tests for cannon.
add_basic_legion_cpugpu_test(cannonMM basic -gx 4 -gy 4 -n 2048)
add_basic_legion_cpugpu_test(cannonMM nondiv1 -gx 4 -gy 4 -n 2047)
add_basic_legion_cpugpu_test(cannonMM nondiv2 -gx 4 -gy 4 -n 2049)

# Tests for SUMMA.
add_basic_legion_test(summaMM basic -gx 4 -gy 4 -n 2048)
add_basic_legion_test(summaMM nondiv1 -gx 4 -gy 4 -n 2047)
add_basic_legion_test(summaMM nondiv2 -gx 4 -gy 4 -n 2049)

# Tests for PUMMA.
add_basic_legion_test(pummaMM basic -gx 4 -gy 4 -n 2048)

# Tests for Johnson.
add_basic_legion_test(johnsonMM basic -n 1024 -gdim 3)

# Tests for Solomonik-2.5D.
add_basic_legion_test(solomonikMM basic -n 1024 -rpoc 3 -c 2 -rpoc3 2)

# Tests for the matvec-adapt stuff. Run with and without a match.
add_basic_legion_test(matvec-adapt match -n 1024 -pieces 8 -match)
add_basic_legion_test(matvec-adapt nomatch -n 1024 -pieces 8)

# Deduplicate the list of test binaries so that we have all of our dependencies.
# All tests must be added above this line.
list(REMOVE_DUPLICATES TEST_BINARIES)
add_custom_target(test-legion COMMAND ctest -R legion DEPENDS ${TEST_BINARIES})

add_executable(memspeed memspeed/main.cpp)
target_link_libraries(memspeed Legion::Legion)

add_executable(hier-test hier-part-testing/main.cpp)
target_link_libraries(hier-test Legion::Legion)

# Add some hand written codes here.
if(Legion_USE_CUDA)
    cuda_add_executable(handwritten-spmv handwritten-spmv/handwritten.cpp handwritten-spmv/handwritten.cu ${LG_EXTRA_SOURCES})
    include_directories(handwritten-spmv PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(handwritten-spmv Legion::Legion ${HDF5_LIBRARIES})
    target_compile_options(handwritten-spmv PRIVATE ${OpenMP_CXX_FLAGS})
    target_compile_definitions(handwritten-spmv PUBLIC TACO_USE_CUDA)
else()
    add_executable(handwritten-spmv handwritten-spmv/handwritten.cpp ${LG_EXTRA_SOURCES})
    include_directories(handwritten-spmv PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(handwritten-spmv Legion::Legion ${HDF5_LIBRARIES})
    target_compile_options(handwritten-spmv PRIVATE ${OpenMP_CXX_FLAGS})
endif()

# Utility to preload tns files into hdf5 files for easier ingestion.
add_executable(tensor_to_hdf5 fileio/tensor_to_hdf5.cpp)
target_link_libraries(tensor_to_hdf5 distal-runtime-lib Legion::Legion ${HDF5_LIBRARIES})
include_directories(tensor_to_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})

# Utility to convert COO tensors into desired formats.
add_executable(format-converter format-converter/main.cpp format-converter/taco-generated.cpp format-converter/taco-generated.h)
target_link_libraries(format-converter distal-runtime-lib Legion::Legion ${HDF5_LIBRARIES})
include_directories(format-converter PRIVATE ${HDF5_INCLUDE_DIRS})

# Sparse tests.
set(SPARSE_TEST_BINARIES "")
macro(add_basic_legion_sparse_test binary tag)
    # TODO (rohany): See what more runtime debug flags I can place here.
    if (OPENMP)
        add_test(NAME "legion-sparse-${binary}-${tag}" COMMAND ${binary} -lg:partcheck -lg:safe_mapper -ll:ocpu 1 -ll:othr 1 -ll:csize 5000 -ll:nsize 5G -ll:ncsize 0 ${ARGN})
    else()
        add_test(NAME "legion-sparse-${binary}-${tag}" COMMAND ${binary} -lg:partcheck -lg:safe_mapper -ll:cpu 1 -ll:csize 5000 ${ARGN})
    endif()
    # Collect all the test binaries that we depend on.
    list(APPEND SPARSE_TEST_BINARIES "${binary}")
endmacro()
add_basic_legion_sparse_test(tensor_to_hdf5 basic -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.tns -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -h5dump)
add_basic_legion_sparse_test(tensor_to_hdf5 full-vec -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.tns -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.hdf5 -h5dump)
add_basic_legion_sparse_test(tensor_to_hdf5 sparse-vec -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-sparse.tns -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-sparse.hdf5 -h5dump)
add_basic_legion_sparse_test(tensor_to_hdf5 3-tensor -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.tns -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5 -h5dump)
add_basic_legion_sparse_test(format-converter basic -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -format ds -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter full-vec -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.hdf5 -format s -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.vec.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter sparse-vec -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-sparse.hdf5 -format s -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-sparse.vec.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter dcsr -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -format ss -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.dcsr.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter sd -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -format sd -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.sd.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter csc -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -format ds -mode_ordering 10 -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csc.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter 3-tensor-dss -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5 -format dss -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dss.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter 3-tensor-dds -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5 -format dds -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dds.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter 3-tensor-sss -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5 -format sss -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.sss.hdf5 -dump -h5dump)
add_basic_legion_sparse_test(format-converter 3-tensor-sds -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5 -format sds -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.sds.hdf5 -dump -h5dump)
# Make sure that the format-converter tests runs after tensor_to_hdf5.
set_tests_properties(legion-sparse-format-converter-basic PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-basic")
set_tests_properties(legion-sparse-format-converter-full-vec PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-full-vec")
set_tests_properties(legion-sparse-format-converter-sparse-vec PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-sparse-vec")
set_tests_properties(legion-sparse-format-converter-dcsr PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-basic")
set_tests_properties(legion-sparse-format-converter-sd PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-basic")
set_tests_properties(legion-sparse-format-converter-csc PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-basic")
set_tests_properties(legion-sparse-format-converter-3-tensor-dss PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-3-tensor")
set_tests_properties(legion-sparse-format-converter-3-tensor-dds PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-3-tensor")
set_tests_properties(legion-sparse-format-converter-3-tensor-sss PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-3-tensor")
set_tests_properties(legion-sparse-format-converter-3-tensor-sds PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-3-tensor")

add_basic_legion_sparse_test(spmv basic -csr ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -pieces 4 -dump)
add_basic_legion_sparse_test(spmv pos -csr ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -pieces 4 -pos -dump)
add_basic_legion_sparse_test(spmv pos-dcsr -dcsr ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.dcsr.hdf5 -pieces 4 -pos -dump)
add_basic_legion_sparse_test(spmv csc -csc ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csc.hdf5 -pieces 4 -spx ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.vec.hdf5 -dump)
add_basic_legion_sparse_test(spmv csc-sparse-gather -csc ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csc.hdf5 -pieces 4 -spx ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-sparse.vec.hdf5 -dump)
set_tests_properties(legion-sparse-spmv-basic PROPERTIES DEPENDS "legion-sparse-format-converter-basic")
set_tests_properties(legion-sparse-spmv-pos PROPERTIES DEPENDS "legion-sparse-format-converter-basic")
set_tests_properties(legion-sparse-spmv-pos-dcsr PROPERTIES DEPENDS "legion-sparse-format-converter-dcsr")
set_tests_properties(legion-sparse-spmv-csc PROPERTIES DEPENDS "legion-sparse-format-converter-csc")
set_tests_properties(legion-sparse-spmv-csc PROPERTIES DEPENDS "legion-sparse-format-converter-full-vec")
set_tests_properties(legion-sparse-spmv-csc-sparse-gather PROPERTIES DEPENDS "legion-sparse-format-converter-csc")
set_tests_properties(legion-sparse-spmv-csc-sparse-gather PROPERTIES DEPENDS "legion-sparse-format-converter-sparse-vec")

add_basic_legion_sparse_test(spmm basic -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -pieces 4 -dump)
set_tests_properties(legion-sparse-spmm-basic PROPERTIES DEPENDS "legion-sparse-format-converter-basic")

add_basic_legion_sparse_test(spttv basic -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dss.hdf5 -pieces 4 -dump)
add_basic_legion_sparse_test(spttv pos -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dss.hdf5 -pieces 4 -pos -dump)
add_basic_legion_sparse_test(spttv partial-pos -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dss.hdf5 -pieces 4 -partial_pos -dump)
set_tests_properties(legion-sparse-spttv-basic PROPERTIES DEPENDS "legion-sparse-format-converter-3-tensor-dss")
set_tests_properties(legion-sparse-spttv-pos PROPERTIES DEPENDS "legion-sparse-format-converter-3-tensor-dss")
set_tests_properties(legion-sparse-spttv-partial-pos PROPERTIES DEPENDS "legion-sparse-format-converter-3-tensor-dss")

add_basic_legion_sparse_test(sddmm basic -csr ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -pieces 4 -dump)
set_tests_properties(legion-sparse-sddmm-basic PROPERTIES DEPENDS "legion-sparse-format-converter-basic")

add_basic_legion_sparse_test(spmttkrp basic -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dss.hdf5 -pieces 4 -dump)
set_tests_properties(legion-sparse-spmttkrp-basic PROPERTIES DEPENDS "legion-sparse-format-converter-3-tensor-dss")

add_basic_legion_sparse_test(spadd3 basic -tensorB ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -tensorC ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -tensorD ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -pieces 4 -dump)
set_tests_properties(legion-sparse-spadd3-basic PROPERTIES DEPENDS "legion-sparse-format-converter-basic")

add_basic_legion_sparse_test(spinnerprod basic -tensorB ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dss.hdf5 -tensorC ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dss.hdf5 -pieces 4 -dump)
set_tests_properties(legion-sparse-spinnerprod-basic PROPERTIES DEPENDS "legion-sparse-format-converter-3-tensor-dss")

add_basic_legion_sparse_test(spvecadd basic -vecb ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-sparse.vec.hdf5 -vecc ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-sparse.vec.hdf5 -pieces 4 -dump)
set_tests_properties(legion-sparse-spvecadd-basic PROPERTIES DEPENDS "legion-sparse-format-converter-sparse-vec")

# Use GoogleTest to write some test cases that utilize Legion.
# First, create a library for GoogleTest itself.
file(GLOB GTEST_SOURCE ../test/gtest/gtest-all.cc)
file(GLOB GTEST_HEADER ../test/gtest/gtest.h)
add_library(distal-gtest ${GTEST_HEADER} ${GTEST_SOURCE})
target_include_directories(distal-gtest PRIVATE ../test/)
# Next, create a executable for DISTAL tests that links against GoogleTest.
file(GLOB DISTAL_TEST_HEADERS test/*.h)
file(GLOB DISTAL_TEST_SOURCES test/*.cpp)
if (Legion_USE_CUDA)
    cuda_add_executable(distal-runtime-test ${DISTAL_TEST_SOURCES} ${DISTAL_TEST_HEADERS})
    target_compile_definitions(distal-runtime-test PUBLIC TACO_USE_CUDA)
    target_link_libraries(distal-runtime-test distal-runtime-lib-cuda)
else()
    add_executable(distal-runtime-test ${DISTAL_TEST_SOURCES} ${DISTAL_TEST_HEADERS})
    target_link_libraries(distal-runtime-test distal-runtime-lib)
endif()
# Compile the runtime tests with low optimization and debug symbols.
target_compile_options(distal-runtime-test PRIVATE "-g3" "-O0" "-ggdb")
# Compile distal-runtime-test with OpenMP if we're configured with it.
if (OPENMP)
  target_compile_options(distal-runtime-test PRIVATE ${OpenMP_CXX_FLAGS})
endif(OPENMP)
target_include_directories(distal-runtime-test PRIVATE ../test/)
target_link_libraries(distal-runtime-test distal-gtest Legion::Legion ${HDF5_LIBRARIES})
# Tell CMake about all of the tests that GoogleTest will register.
include(GoogleTest)
gtest_add_tests(TARGET distal-runtime-test)

# Run all sparse tests and distal runtime tests.
list(REMOVE_DUPLICATES SPARSE_TEST_BINARIES)
add_custom_target(test-legion-sparse COMMAND ctest -R \"\(\(legion-sparse\)|\(DISTALRuntime\)\)\" DEPENDS ${SPARSE_TEST_BINARIES} distal-runtime-test)
