cmake_minimum_required(VERSION 3.17.1)
project(TrilinosBenchmarks NONE)

set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Trilinos REQUIRED COMPONENTS Tpetra)

# Use the same compilers and flags as Trilinos.
set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
set(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )
set(CMAKE_CXX_FLAGS  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
set(CMAKE_Fortran_FLAGS  "${Trilinos_Fortran_COMPILER_FLAGS} ${CMAKE_Fortran_FLAGS}")

set(EXECUTABLE_OUTPUT_PATH "bin")

enable_language(CXX)

add_executable(benchmark benchmark.cpp)
target_include_directories(benchmark PRIVATE ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
target_link_libraries(benchmark ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
