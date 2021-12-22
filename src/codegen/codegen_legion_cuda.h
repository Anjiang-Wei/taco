#ifndef TACO_CODEGEN_LEGION_CUDA_H
#define TACO_CODEGEN_LEGION_CUDA_H

#include "codegen.h"
#include "codegen_cuda.h"
#include "codegen_legion.h"


namespace taco {
namespace ir {

class CodegenLegionCuda : public CodeGen_CUDA, public CodegenLegion {
public:
  // See the comment in codegen_legion_c.h about why simplify is set
  // to false here.
  CodegenLegionCuda(std::ostream &dest, OutputKind outputKind, bool simplify=false);
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
  void printDeviceFunctions(const Function* func) override;
  std::string procForTask(Stmt target, Stmt func) override;
  void emitHeaders(std::ostream& o) override;
  // TODO (rohany): It also doesn't seem like I can avoid duplicating this class.
  //  It seems like an artifact of how the code is organized.
  class FindVars;
  class DeviceFunctionCollector;
  Stmt stmt;
};

}
}

#endif //TACO_CODEGEN_LEGION_CUDA_H
