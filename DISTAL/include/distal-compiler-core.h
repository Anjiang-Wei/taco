#ifndef DISTAL_CORE_H
#define DISTAL_CORE_H

#include "taco.h"

namespace DISTAL {
namespace Compiler {
namespace Core {

// getTensors returns all TensorVars present in an IndexStatement.
std::vector<taco::TensorVar> getTensors(const taco::IndexStmt stmt);

// stmtUsesGPU returns true if an IndexStmt utilizes GPUs.
bool stmtUsesGPU(const taco::IndexStmt stmt);

// stmtIsDistributed returns true if an IndexStmt is distributed.
bool stmtIsDistributed(const taco::IndexStmt stmt);

// stmtHasSparseLHS returns true if an IndexStmt assigns into a sparse tensor.
bool stmtHasSparseLHS(const taco::IndexStmt stmt);

} // namespace Core
} // namespace Compiler
} // namespace DISTAL

#endif // DISTAL_CORE_H