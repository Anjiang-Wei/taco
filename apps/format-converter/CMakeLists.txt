find_package(Legion REQUIRED)

add_executable(format-converter main.cpp)
target_include_directories(format-converter PRIVATE ${TACO_INCLUDE_DIR} ${DISTAL_RUNTIME_INCLUDE_DIRS})
target_link_libraries(format-converter PRIVATE Legion::Legion distal-compiler-jit distal-runtime)
