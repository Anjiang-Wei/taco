find_package(Legion REQUIRED)
if(OPENMP)
  find_package(OpenMP REQUIRED)
endif(OPENMP)

# We need to have HDF5_ROOT set here to do this if we're using
# a non-system provided HDF5 installation.
find_package(HDF5 REQUIRED)

# Include the headers and supporting source files.
include_directories(include/ OpenBLAS/install/include "${CMAKE_BINARY_DIR}/include")
link_directories(OpenBLAS/install/lib)
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

function(add_app_folder folder)
    file(GLOB SOURCES "${folder}/main.cpp" "${folder}/taco-generated.cpp" ${LG_SOURCES})
    add_executable("${folder}" ${SOURCES})
    target_include_directories("${folder}" PRIVATE "${folder}")
    target_link_libraries("${folder}" Legion::Legion openblas)
    if (OPENMP)
      target_compile_options("${folder}" PRIVATE ${OpenMP_CXX_FLAGS})
    endif(OPENMP)

    if (Legion_USE_CUDA)
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${folder}/taco-generated.cu")
            cuda_add_executable("${folder}-cuda" "${folder}/main.cpp" "${folder}/taco-generated.cu" ${LG_SOURCES} ${LG_CU_SOURCES})
            target_link_libraries("${folder}-cuda" Legion::Legion cublas -fopenmp)
	    # set_property(TARGET "${folder}-cuda" PROPERTY CUDA_ARCHITECTURES 60)
	    # target_compile_options("${folder}-cuda" PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60>)
	    # target_compile_options("${folder}-cuda" PUBLIC -gencode arch=compute_50,code=sm_50)
            if (OPENMP)
                target_link_libraries("${folder}-cuda" -fopenmp)
                target_compile_options("${folder}-cuda" PUBLIC -fopenmp)
            endif(OPENMP)
            target_compile_definitions("${folder}-cuda" PUBLIC TACO_USE_CUDA)
            set_property(TARGET "${folder}-cuda" PROPERTY CUDA_STANDARD 14)
        endif()
    endif()
endfunction()

function(add_cuda_folder folder)
    if (Legion_USE_CUDA)
        file(GLOB SOURCES "${folder}/*.cpp" "${folder}/*.cu" ${LG_SOURCES})
        cuda_add_executable("${folder}" ${SOURCES} ${LG_CU_SOURCES})
        target_link_libraries("${folder}" Legion::Legion)
        target_compile_definitions("${folder}" PUBLIC TACO_USE_CUDA)
        set_property(TARGET "${folder}" PROPERTY CUDA_STANDARD 14)
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
add_app_folder(placement-test)
add_app_folder(johnsonMM)
add_app_folder(solomonikMM)
add_app_folder(cosma)
add_app_folder(ttv)
add_app_folder(mttkrp)
add_app_folder(ttmc)
add_app_folder(innerprod)
add_app_folder(matvec-adapt)

add_cuda_folder(cuda-test)

# Sparse codes.
add_app_folder(spmv)
add_app_folder(spttv)
add_app_folder(spmm)
add_app_folder(sddmm)
add_app_folder(spmttkrp)

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
add_executable(tensor_to_hdf5 fileio/tensor_to_hdf5.cpp ${LG_EXTRA_SOURCES})
target_link_libraries(tensor_to_hdf5 Legion::Legion ${HDF5_LIBRARIES})
include_directories(tensor_to_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})

# Utility to convert COO tensors into desired formats.
add_executable(format-converter format-converter/main.cpp format-converter/taco-generated.cpp ${LG_SOURCES})
target_link_libraries(format-converter Legion::Legion ${HDF5_LIBRARIES})
include_directories(format-converter PRIVATE ${HDF5_INCLUDE_DIRS} format-converter)

# Sparse tests.
set(SPARSE_TEST_BINARIES "")
macro(add_basic_legion_sparse_test binary tag)
    # TODO (rohany): See what more runtime debug flags I can place here.
    if (OPENMP)
        add_test(NAME "legion-sparse-${binary}-${tag}" COMMAND ${binary} -lg:partcheck -lg:safe_mapper -ll:ocpu 1 -ll:othr 1 -ll:csize 5000 ${ARGN})
    else()
        add_test(NAME "legion-sparse-${binary}-${tag}" COMMAND ${binary} -lg:partcheck -lg:safe_mapper -ll:cpu 1 -ll:csize 5000 ${ARGN})
    endif()
    # Collect all the test binaries that we depend on.
    list(APPEND SPARSE_TEST_BINARIES "${binary}")
endmacro()
add_basic_legion_sparse_test(tensor_to_hdf5 basic -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.tns -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5)
add_basic_legion_sparse_test(tensor_to_hdf5 full-vec -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.tns -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.hdf5)
add_basic_legion_sparse_test(tensor_to_hdf5 3-tensor -tensor ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.tns -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5)
add_basic_legion_sparse_test(format-converter basic -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -format ds -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -dump)
add_basic_legion_sparse_test(format-converter full-vec -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.hdf5 -format s -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.vec.hdf5 -dump)
add_basic_legion_sparse_test(format-converter dcsr -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -format ss -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.dcsr.hdf5 -dump)
add_basic_legion_sparse_test(format-converter sd -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -format sd -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.sd.hdf5 -dump)
add_basic_legion_sparse_test(format-converter csc -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.hdf5 -format ds -mode_ordering 10 -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csc.hdf5 -dump)
add_basic_legion_sparse_test(format-converter 3-tensor-dss -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5 -format dss -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.dss.hdf5 -dump)
add_basic_legion_sparse_test(format-converter 3-tensor-sss -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5 -format sss -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.sss.hdf5 -dump)
add_basic_legion_sparse_test(format-converter 3-tensor-sds -coofile ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.hdf5 -format sds -o ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-3-tensor.sds.hdf5 -dump)
# Make sure that the format-converter tests runs after tensor_to_hdf5.
set_tests_properties(legion-sparse-format-converter-basic PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-basic")
set_tests_properties(legion-sparse-format-converter-full-vec PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-full-vec")
set_tests_properties(legion-sparse-format-converter-dcsr PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-basic")
set_tests_properties(legion-sparse-format-converter-sd PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-basic")
set_tests_properties(legion-sparse-format-converter-csc PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-basic")
set_tests_properties(legion-sparse-format-converter-3-tensor-dss PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-3-tensor")
set_tests_properties(legion-sparse-format-converter-3-tensor-sss PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-3-tensor")
set_tests_properties(legion-sparse-format-converter-3-tensor-sds PROPERTIES DEPENDS "legion-sparse-tensor_to_hdf5-3-tensor")

add_basic_legion_sparse_test(spmv basic -csr ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -pieces 4 -dump)
add_basic_legion_sparse_test(spmv pos -csr ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csr.hdf5 -pieces 4 -pos -dump)
add_basic_legion_sparse_test(spmv pos-dcsr -dcsr ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.dcsr.hdf5 -pieces 4 -pos -dump)
add_basic_legion_sparse_test(spmv csc -csc ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-mat-holes.csc.hdf5 -pieces 4 -spx ${CMAKE_CURRENT_SOURCE_DIR}/tensors/test-vec-full.vec.hdf5 -dump)
set_tests_properties(legion-sparse-spmv-basic PROPERTIES DEPENDS "legion-sparse-format-converter-basic")
set_tests_properties(legion-sparse-spmv-pos PROPERTIES DEPENDS "legion-sparse-format-converter-basic")
set_tests_properties(legion-sparse-spmv-pos-dcsr PROPERTIES DEPENDS "legion-sparse-format-converter-dcsr")
set_tests_properties(legion-sparse-spmv-csc PROPERTIES DEPENDS "legion-sparse-format-converter-csc")
set_tests_properties(legion-sparse-spmv-csc PROPERTIES DEPENDS "legion-sparse-format-converter-full-vec")

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


list(REMOVE_DUPLICATES SPARSE_TEST_BINARIES)
add_custom_target(test-legion-sparse COMMAND ctest -R legion-sparse DEPENDS ${SPARSE_TEST_BINARIES})
