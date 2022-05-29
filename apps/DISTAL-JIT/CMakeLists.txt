find_package(Legion REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
file(GLOB SOURCE_CODE *.cc)
add_executable(distal-jit ${SOURCE_CODE})

include_directories(${TACO_INCLUDE_DIR} ../../DISTAL/include ../../legion/include)
target_link_libraries(distal-jit LINK_PUBLIC taco Legion::Legion distal-compiler-jit)
