#include "distal-compiler-core.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/util/collections.h"

namespace DISTAL {
namespace Compiler {
namespace Core {

std::vector<taco::TensorVar> getTensors(const taco::IndexStmt stmt) {
  // First find the assignment.
  struct AssignmentFinder : public taco::IndexNotationVisitor {
    void visit(const taco::AssignmentNode *node) {
      this->assign = node;
    }

    taco::Assignment assign;
  } assignmentFinder;
  stmt.accept(&assignmentFinder);

  std::vector<taco::TensorVar> result;
  std::set<taco::TensorVar> dedup;
  auto lhsTV = assignmentFinder.assign.getLhs().getTensorVar();
  result.push_back(lhsTV);
  dedup.insert(lhsTV);

  // Next, visit the RHS of the assignment to find the remaining TensorVars.
  struct TensorVarFinder : public taco::IndexNotationVisitor {
    TensorVarFinder(std::vector<taco::TensorVar> &result, std::set<taco::TensorVar> &dedup) : result(result),
                                                                                              dedup(dedup) {}

    void visit(const taco::AccessNode *node) {
      auto tv = node->tensorVar;
      if (!taco::util::contains(this->dedup, tv)) {
        this->result.push_back(tv);
        this->dedup.insert(tv);
      }
    }

    std::vector<taco::TensorVar> &result;
    std::set<taco::TensorVar> &dedup;
  } tensorVarFinder(result, dedup);
  assignmentFinder.assign.getRhs().accept(&tensorVarFinder);

  return result;
}

bool stmtUsesGPU(const taco::IndexStmt stmt) {
  struct ParallelUnitFinder : public taco::IndexNotationVisitor {
    void visit(const taco::ForallNode* node) {
      if (node->parallel_unit == taco::ParallelUnit::GPUBlock ||
          node->parallel_unit == taco::ParallelUnit::GPUThread ||
          node->parallel_unit == taco::ParallelUnit::GPUWarp ||
          node->parallel_unit == taco::ParallelUnit::DistributedGPU) {
        this->usesGPU = true;
      }
      node->stmt.accept(this);
    }
    bool usesGPU = false;
  } finder;
  stmt.accept(&finder);
  return finder.usesGPU;
}

} // namespace Core
} // namespace Compiler
} // namespace DISTAL