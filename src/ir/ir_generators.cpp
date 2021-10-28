#include "ir_generators.h"

#include "taco/ir/ir.h"
#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {
namespace ir {

Stmt compoundStore(Expr a, Expr i, Expr val, bool use_atomics, ParallelUnit atomic_parallel_unit) {
  Expr add = (val.type().getKind() == Datatype::Bool) 
             ? Or::make(Load::make(a, i), val)
             : Add::make(Load::make(a, i), val);
  return Store::make(a, i, add, use_atomics, atomic_parallel_unit);
}

Stmt compoundAssign(Expr a, Expr val, bool use_atomics, ParallelUnit atomic_parallel_unit) {
  Expr add = (val.type().getKind() == Datatype::Bool) 
             ? Or::make(a, val) : Add::make(a, val);
  return Assign::make(a, add, use_atomics, atomic_parallel_unit);
}

Expr conjunction(std::vector<Expr> exprs) {
  taco_iassert(exprs.size() > 0) << "No expressions to and";
  Expr conjunction = exprs[0];
  for (size_t i = 1; i < exprs.size(); i++) {
    conjunction = And::make(conjunction, exprs[i]);
  }
  return conjunction;
}

Stmt doubleSizeIfFull(Expr a, Expr size, Expr needed) {
  Stmt realloc = Allocate::make(a, Mul::make(size, 2), true, size);
  Stmt resize = Assign::make(size, Mul::make(size, 2));
  Stmt ifBody = Block::make({realloc, resize});
  return IfThenElse::make(Lte::make(size, needed), ifBody);
}

// TODO (rohany): This will probably need to do some management around accessors etc, or it
//  will need to be pulled into the more specific mode formats.
Stmt lgDoubleSizeIfFull(Expr reg, Expr size, Expr needed, Expr oldPhysicalReg, Expr fieldID) {
  auto realloc = ir::makeLegionRealloc(reg, Mul::make(size, 2), reg, oldPhysicalReg, fieldID);
  auto resize = Assign::make(size, Mul::make(size, 2));
  auto ifBody = Block::make({realloc, resize});
  return IfThenElse::make(Lte::make(size, needed), ifBody);
}

Stmt atLeastDoubleSizeIfFull(Expr a, Expr size, Expr needed) {
  Expr newSizeVar = Var::make(util::toString(a) + "_new_size", Int());
  Expr newSize = Max::make(Mul::make(size, 2), Add::make(needed, 1));
  Stmt computeNewSize = VarDecl::make(newSizeVar, newSize);
  Stmt realloc = Allocate::make(a, newSizeVar, true, size);
  Stmt updateSize = Assign::make(size, newSizeVar);
  Stmt ifBody = Block::make({computeNewSize, realloc, updateSize});
  return IfThenElse::make(Lte::make(size, needed), ifBody);
}

Stmt lgAtLeastDoubleSizeIfFull(Expr reg, Expr size, Expr needed, Expr oldPhysicalReg, Expr fieldID) {
  Expr newSizeVar = Var::make(util::toString(reg) + "_new_size", Int());
  Expr newSize = Max::make(Mul::make(size, 2), Add::make(needed, 1));
  Stmt computeNewSize = VarDecl::make(newSizeVar, newSize);
  Stmt realloc = makeLegionRealloc(reg, newSizeVar, reg, oldPhysicalReg, fieldID);
  Stmt updateSize = Assign::make(size, newSizeVar);
  Stmt ifBody = Block::make({computeNewSize, realloc, updateSize});
  return IfThenElse::make(Lte::make(size, needed), ifBody);
}

}}
