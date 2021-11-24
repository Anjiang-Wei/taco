#ifndef TACO_CODEGEN_LEGION_C_H
#define TACO_CODEGEN_LEGION_C_H

#include "codegen.h"
#include "codegen_c.h"
#include "codegen_legion.h"

namespace taco {
namespace ir {

class CodegenLegionC : public CodeGen_C, public CodegenLegion {
public:
  // We set simplify to false because there is an annoying bug here that
  // is a bit hard to fix due to intertwined behavior. In particular, we
  // do analysis like task argument packing on the IR before it is simplified.
  // However, the IRPrinter will attempt to simplify the IR right before it
  // starts to print, causing some drift in the analysis, as the simplification
  // pass might remove variables that we think exist etc.
  CodegenLegionC(std::ostream &dest, OutputKind outputKind, bool simplify=false);
  void compile(Stmt stmt, bool isFirst=false) override;

private:
  // TODO (rohany): It doesn't seem like I can override these.
  using IRPrinter::visit;
  void visit(const For* node) override;
  void visit(const Function* node) override;
  void visit(const PackTaskArgs* node) override;
  void visit(const UnpackTensorData* node) override;
  void visit(const DeclareStruct* node) override;
  void visit(const Allocate* node) override;
  void emitHeaders(std::ostream& o) override;
  // TODO (rohany): It also doesn't seem like I can avoid duplicating this class.
  class FindVars;
  Stmt stmt;
};

}
}

#endif //TACO_CODEGEN_LEGION_C_H
