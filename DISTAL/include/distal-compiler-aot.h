#ifndef DISTAL_COMPILER_AOT_H
#define DISTAL_COMPILER_AOT_H

#include "taco.h"

namespace DISTAL {
namespace Compiler {
namespace AOT {

// Compiles a scheduled taco::IndexStmt to files in the output directory. It generates
// code for the compute statement itself, as well as data distribution code for each
// tensor in the statement.
void compile(taco::IndexStmt stmt, std::string directory);

} // namespace AOT
} // namespace Compiler
} // namespace DISTAL

#endif // DISTAL_COMPILER_AOT