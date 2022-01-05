#ifndef LG_TEST_H
#define LG_TEST_H

#include "gtest/gtest.h"
#include "legion.h"
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

#endif // LG_TEST_H