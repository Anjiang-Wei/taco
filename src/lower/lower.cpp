#include "taco/lower/lower.h"

#include <algorithm>
#include <vector>
#include <list>
#include <set>
#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"

#include "taco/ir/ir.h"
#include "taco/ir/simplify.h"
#include "ir/ir_generators.h"
#include "taco/ir/ir_printer.h"

#include "taco/lower/lowerer_impl.h"
#include "taco/lower/iterator.h"
#include "mode_access.h"

#include "error/error_checks.h"
#include "taco/error/error_messages.h"
#include "taco/util/name_generator.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

#include "taco/ir/ir_verifier.h"

using namespace std;
using namespace taco::ir;

namespace taco {


// class Lowerer
Lowerer::Lowerer() : impl(new LowererImpl()) {
}

Lowerer::Lowerer(LowererImpl* impl) : impl(impl) {
}

std::shared_ptr<LowererImpl> Lowerer::getLowererImpl() {
  return impl;
}

ir::Stmt lower(IndexStmt stmt, std::string name, 
               bool assemble, bool compute, bool pack, bool unpack,
               Lowerer lowerer) {
  string reason;
  taco_iassert(isLowerable(stmt, &reason))
      << "Not lowerable, because " << reason << ": " << stmt;
  ir::Stmt lowered = lowerer.getLowererImpl()->lower(stmt, name, assemble, compute, pack, unpack, true, true, false);

  // TODO: re-enable this
  // std::string messages;
  // verify(lowered, &messages);
  // if (!messages.empty()) {
  //   std::cerr << "Verifier messages:\n" << messages << "\n";
  // }
  
  return lowered;
}

ir::Stmt lowerNoWait(IndexStmt stmt, std::string name, Lowerer lowerer) {
  return lowerLegion(stmt, name, true /* partition */ , true /* compute */, false /* waitOnFutureMap */);
}

ir::Stmt lowerLegion(IndexStmt stmt, std::string name,
                     bool partition, bool compute, bool waitOnFuture, bool setPlacementPrivilege,
                     Lowerer lowerer) {
  string reason;
  taco_iassert(isLowerable(stmt, &reason))
      << "Not lowerable, because " << reason << ": " << stmt;
  return lowerer.getLowererImpl()->lower(
      stmt,
      name,
      false /* assemble */,
      compute /* compute */,
      false /* pack */,
      false /* unpack */,
      partition /* partition */,
      waitOnFuture /* waitOnFutureMap */,
      setPlacementPrivilege /* setPlacementPrivilege */
  );
}

ir::Stmt lowerLegionSeparatePartitionCompute(IndexStmt stmt, std::string name, bool waitOnFuture, bool setPlacementPrivilege) {
  auto part = lowerLegion(stmt, "partitionFor" + name, true /* partition */, false /* compute */, waitOnFuture);
  auto compute = lowerLegion(stmt, name, false /* partition */, true /* compute */, waitOnFuture, setPlacementPrivilege);
  return ir::Block::make({part, compute});
}

bool isLowerable(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);

  // Must be concrete index notation
  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "the index statement is not in concrete index notation, because "
            + r;
    return false;
  }

  // Check for transpositions
//  if (!error::containsTranspose(this->getFormat(), freeVars, indexExpr)) {
//    *reason = error::expr_transposition;
//  }

  return true;
}

}
