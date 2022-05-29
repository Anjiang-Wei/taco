#ifndef TACO_CODEGEN_H
#define TACO_CODEGEN_H

#include <memory>
#include <vector>
#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"

namespace taco {
namespace ir {


class CodeGen : public IRPrinter {
public:
  /// Kind of output: header, implementation, or standalone implementation with no header. Since there
  /// is some logical overlap between ImplementationGen and ImplementationNoHeaderGen, use the method
  /// isImplementationGen to check if an implementation is required from a CodeGen object.
  enum OutputKind { HeaderGen, ImplementationGen, ImplementationNoHeaderGen };
  enum CodeGenType { C, CUDA };

  CodeGen(std::ostream& stream, CodeGenType type) : IRPrinter(stream), codeGenType(type) {};
  CodeGen(std::ostream& stream, bool color, bool simplify, CodeGenType type) : IRPrinter(stream, color, simplify), codeGenType(type) {};
  /// Initialize the default code generator
  static std::shared_ptr<CodeGen> init_default(std::ostream &dest, OutputKind outputKind);

  /// Compile a lowered function
  virtual void compile(Stmt stmt, bool isFirst=false) =0;

protected:
  static bool checkForAlloc(const Function *func);
  static int countYields(const Function *func);

  static std::string printCType(Datatype type, bool is_ptr);
  static std::string printCUDAType(Datatype type, bool is_ptr);

  static std::string printCAlloc(std::string pointer, std::string size);
  static std::string printCUDAAlloc(std::string pointer, std::string size);
  std::string printAlloc(std::string pointer, std::string size);

  static std::string printCFree(std::string pointer);
  static std::string printCUDAFree(std::string pointer);
  std::string printFree(std::string pointer);

  std::string printType(Datatype type, bool is_ptr);
  // printTypeInName does the same thing as printType, but sanitizes the output
  // so that the type can appear in a name. For example, T1<T2> -> T1_T2.
  std::string printTypeInName(Datatype type, bool is_ptr);
  std::string printContextDeclAndInit(std::map<Expr, std::string, ExprCompare> varMap,
                                          std::vector<Expr> localVars, int labels,
                                          std::string funcName);
  std::string printDecls(std::map<Expr, std::string, ExprCompare> varMap,
                         std::vector<Expr> inputs, std::vector<Expr> outputs);
  std::string printPack(std::map<std::tuple<Expr, TensorProperty, int, int>,
          std::string> outputProperties, std::vector<Expr> outputs);
  std::string printCoroutineFinish(int numYields, std::string funcName);
  void printYield(const Yield* op, std::vector<Expr> localVars,
                         std::map<Expr, std::string, ExprCompare> varMap, int labelCount, std::string funcName);
  virtual std::string printFuncName(const Function *func,
          std::map<Expr, std::string, ExprCompare> inputMap={}, 
          std::map<Expr, std::string, ExprCompare> outputMap={});

  void resetUniqueNameCounters();
  std::string genUniqueName(std::string name);
  void doIndentStream(std::stringstream &stream);
  CodeGenType codeGenType;

  virtual std::string unpackTensorProperty(std::string varname, const GetProperty* op,
                                           bool is_output_prop);
  std::string printTensorProperty(std::string varname, const GetProperty* op, bool is_ptr);

  bool isImplementationGen(OutputKind kind);
private:
  virtual std::string restrictKeyword() const { return ""; }

  std::string packTensorProperty(std::string varname, Expr tnsr, TensorProperty property,
                            int mode, int index);
  std::string pointTensorProperty(std::string varname);
};

std::vector<const GetProperty*> sortProps(std::map<Expr, std::string, ExprCompare> map);


} // namespace ir
} // namespace taco
#endif
