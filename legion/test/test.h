#ifndef LG_TEST_H
#define LG_TEST_H

#include "gtest/gtest.h"
#include "legion.h"
#include "mappers/default_mapper.h"
#include "task_ids.h"

// Nasty global variables to get access to argc and argv.
extern int my_argc;
extern char** my_argv;

// DISTALRuntime is a fixture that initializes Legion once for use
// within all tests.
class DISTALRuntime : public ::testing::Test {
public:
  static void SetUpTestCase();
  static void TearDownTestCase();
  static Legion::Context ctx;
  static Legion::Runtime* runtime;
};

// DISTALRuntimeTestMapper is a mapper that maps all regions in
// SYSTEM_MEMORY to allow for inline mappings performed by the
// implicit top level task.
class DISTALRuntimeTestMapper : public Legion::Mapping::DefaultMapper {
public:
  DISTALRuntimeTestMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine &machine, const Legion::Processor &local, const char* name);
  Legion::Memory default_policy_select_target_memory(Legion::Mapping::MapperContext ctx,
                                                     Legion::Processor target_proc,
                                                     const Legion::RegionRequirement &req,
                                                     Legion::MemoryConstraint mc = Legion::MemoryConstraint());
};

#endif // LG_TEST_H