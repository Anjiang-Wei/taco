#include "distal-compiler-aot.h"
#include "distal-compiler-core.h"

#include "taco/lower/lower.h"

// "Hidden" headers from within the TACO source tree.
#include "codegen/codegen_legion_c.h"
#include "codegen/codegen_legion_cuda.h"

namespace DISTAL {
namespace Compiler {
namespace AOT {

void compile(taco::IndexStmt stmt, std::string directory) {
  std::vector<taco::ir::Stmt> stmts;
  // Compile each of the tensor distributions in the input statement.
  auto tensors = DISTAL::Compiler::Core::getTensors(stmt);
  for (auto t : tensors) {
    taco_iassert(!t.getDistribution().empty());
    auto trans = t.translateDistribution();
    // TODO (rohany): Standardize the names for these functions. Or the user could provide
    //  them to the compiler method?
    std::string funcName = "placeLegion" + t.getName();
    auto lowered = taco::lowerLegionSeparatePartitionCompute(trans, funcName);
    stmts.push_back(lowered);
  }
  stmts.push_back(taco::lowerLegionSeparatePartitionCompute(stmt, "computeLegion"));

  // TODO (rohany): This doesn't handle a case where the data is distributed into
  //  the GPUs but the code is distributed onto CPUs.
  // TODO (rohany): There isn't a way right now for the user to specify the names of
  //  the generated files.
  if (DISTAL::Compiler::Core::stmtUsesGPU(stmt)) {
    taco::ir::CodegenLegionCuda::compileToDirectory(directory, taco::ir::Block::make(stmts));
  } else {
    taco::ir::CodegenLegionC::compileToDirectory(directory, taco::ir::Block::make(stmts));
  }
}

} // namespace AOT
} // namespace Compiler
} // namespace DISTAL