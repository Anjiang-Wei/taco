#include "codegen_legion_c.h"
#include "codegen_c.h"
#include "taco/util/strings.h"
#include "taco/ir/ir_rewriter.h"
#include <algorithm>
#include <fstream>

namespace taco {
namespace ir {

// find variables for generating declarations
// generates a single var for each GetProperty
class CodegenLegionC::FindVars : public IRVisitor {
public:
  std::map<Expr, std::string, ExprCompare> varMap;

  // the variables for which we need to add declarations
  std::map<Expr, std::string, ExprCompare> varDecls;

  std::vector<Expr> localVars;

  // this maps from tensor, property, mode, index to the unique var
  std::map<std::tuple<Expr, TensorProperty, int, int>, std::string> canonicalPropertyVar;

  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  std::map<std::tuple<Expr, TensorProperty, int, int>, std::string> outputProperties;

  // TODO: should replace this with an unordered set
  std::vector<Expr> outputTensors;
  std::vector<Expr> inputTensors;

  CodegenLegionC *codeGen;

  // copy inputs and outputs into the map
  FindVars(std::vector<Expr> inputs, std::vector<Expr> outputs, CodegenLegionC *codeGen)
      : codeGen(codeGen) {
    for (auto v: inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate input found in codegen";
      inputTensors.push_back(v);
      varMap[var] = var->name;
    }
    for (auto v: outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var)==0) << "Duplicate output found in codegen";
      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
  }

protected:
  using IRVisitor::visit;

  virtual void visit(const Var *op) {
    if (varMap.count(op) == 0) {
      varMap[op] = op->is_ptr? op->name : codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl *op) {
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const PackTaskArgs* args) {
    auto func = this->codeGen->idToFunc.at(args->forTaskID).as<Function>();
    for (auto& e : this->codeGen->taskArgs[func]) {
      e.accept(this);
    }
  }

  virtual void visit(const For *op) {
    // Don't count the variables inside the task as being used.
    if (op->isTask) {
      return;
    }

    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }

  virtual void visit(const GetProperty *op) {
    // TODO (rohany): This might be needed.
//    if (!util::contains(inputTensors, op->tensor) &&
//        !util::contains(outputTensors, op->tensor)) {
//      // Don't create header unpacking code for temporaries
//      return;
//    }

    // For certain TensorProperties, we need to ensure that we emit
    // also collect the properties that they depend on.
    switch (op->property) {
      case TensorProperty::ValuesWriteAccessor:
      case TensorProperty::ValuesReadAccessor:
      case TensorProperty::ValuesReductionNonExclusiveAccessor:
      case TensorProperty::ValuesReductionAccessor: {
        // If we have a values accessor, then the values array for this
        // tensor needs to be included as well.
        auto values = ir::GetProperty::make(op->tensor, TensorProperty::Values);
        values.accept(this);
        break;
      }
      case TensorProperty::IndicesAccessor: {
        // Similar logic holds for an index accessor. However, we don't
        // need to make another property here since we are carrying around
        // the property that we are referencing.
        op->accessorArgs.regionAccessing.accept(this);
        break;
      }
      default:
        break;
    }

    if (varMap.count(op) == 0) {
      auto key =
          std::tuple<Expr,TensorProperty,int,int>(op->tensor,op->property,
                                             (size_t)op->mode,
                                             (size_t)op->index);
      if (canonicalPropertyVar.count(key) > 0) {
        varMap[op] = canonicalPropertyVar[key];
      } else {
        auto unique_name = codeGen->genUniqueName(op->name);
        canonicalPropertyVar[key] = unique_name;
        varMap[op] = unique_name;
        varDecls[op] = unique_name;
        if (util::contains(outputTensors, op->tensor)) {
          outputProperties[key] = unique_name;
        }
      }
    }
  }
};

CodegenLegionC::CodegenLegionC(std::ostream &dest, OutputKind outputKind, bool simplify)
  : CodeGen(dest, false, simplify, C), CodeGen_C(dest, outputKind, simplify), CodegenLegion(dest, C) {}

void CodegenLegionC::visit(const PackTaskArgs *node) {
  doIndent();

  auto func = this->idToFunc.at(node->forTaskID).as<Function>();
  auto taskFor = this->idToFor.at(node->forTaskID).as<For>();
  taco_iassert(func) << "must be func";
  taco_iassert(taskFor) << "must be for";

  // Use this information to look up what variables need to be packed into the struct.
  auto stname = taskArgsName(func->name);

  // Make a variable for the raw allocation of the arguments.
  auto tempVar = node->var.as<Var>()->name + "Raw";
  out << stname << " " << tempVar << ";\n";

  // First emit mandatory prefix arguments.
  for (size_t i = 0; i < node->prefixVars.size(); i++) {
    doIndent();
    out << tempVar << "." << node->prefixVars[i] << " = " << node->prefixExprs[i] << ";\n";
  }

  for (auto arg : this->taskArgs[func]) {
    doIndent();
    out << tempVar << "." << arg << " = " << arg << ";\n";
  }

  // Construct the actual TaskArgument from this packed data.
  doIndent();
  out << "TaskArgument " << node->var << " = TaskArgument(&" << tempVar << ", sizeof(" << stname << "));\n";
}

// This is a no-op because we pull this IR-node out and handle it specially when constructing
// the header of a task.
void CodegenLegionC::visit(const UnpackTensorData*) {}
// This operation is also a no-op for a similar reason.
void CodegenLegionC::visit(const DeclareStruct*) {}

void CodegenLegionC::compile(Stmt stmt, bool isFirst) {
  // If we're outputting a header, emit the necessary defines.
  if (this->outputKind == HeaderGen) {
    out << "#ifndef TACO_GENERATED_H\n";
    out << "#define TACO_GENERATED_H\n";
  }

  this->stmt = stmt;
  // Collect all of the individual functions that we need to generate code for.
  this->collectAllFunctions(stmt);
  // Rewrite the task ID's within each function so that they are all unique.
  this->rewriteFunctionTaskIDs();
  // Emit any needed headers.
  this->emitHeaders(out);

  // Emit field accessors. We don't need to emit accessors if we are
  // generating a header file, as these declarations are local to the
  // generated code.
  if (this->outputKind == ImplementationGen) {
    this->collectAndEmitAccessors(stmt, out);
  }
  this->analyzeAndCreateTasks(this->outputKind, out);

  for (auto& f : this->allFunctions) {
    for (auto func : this->functions[f]) {
      CodeGen_C::compile(func, isFirst);
    }
    CodeGen_C::compile(f, isFirst);
  }

  this->emitRegisterTasks(this->outputKind, out);

  // If we're outputting a header, emit the necessary defines.
  if (this->outputKind == HeaderGen) {
    out << "#endif // TACO_GENERATED_H\n";
  }
}

void CodegenLegionC::visit(const For* node) {
  if (node->isTask) {
    return;
  }
  CodeGen_C::visit(node);
}

void CodegenLegionC::emitHeaders(std::ostream &o) {
  CodegenLegion::emitHeaders(this->outputKind, o);
  if (this->outputKind == HeaderGen) {
    return;
  }

  // TODO (rohany): This is pretty hacky, but I don't want to plumb an
  //  interface down here about the name of the generated files right now.
  o << "#include \"taco-generated.h\"\n";

  struct BLASFinder : public IRVisitor {
    void visit(const Call* node) {
      if (node->func.find("blas") != std::string::npos) {
        this->usesBLAS = true;
      }
      if (node->func.find("mttkrp") != std::string::npos || node->func.find("ttv") != std::string::npos) {
        this->usesLeafKernels = true;
      }
    }
    bool usesBLAS = false;
    bool usesLeafKernels = false;
  };
  BLASFinder bs;
  this->stmt.accept(&bs);
  if (bs.usesBLAS) {
    o << "#include \"cblas.h\"\n";
  }
  if (bs.usesLeafKernels) {
    o << "#include \"leaf_kernels.h\"\n";
  }
}

// TODO (rohany): This is duplicating alot of code.
void CodegenLegionC::visit(const Function* func) {
  if (outputKind == HeaderGen && func->name.find("task") != std::string::npos) {
    // If we're generating a header, we don't want to emit these
    // internal task declarations to the end user.
    return;
  }

  int numYields = countYields(func);
  emittingCoroutine = (numYields > 0);
  funcName = func->name;
  labelCount = 0;

  resetUniqueNameCounters();
  FindVars inputVarFinder(func->inputs, {}, this);
  func->body.accept(&inputVarFinder);
  FindVars outputVarFinder({}, func->outputs, this);
  func->body.accept(&outputVarFinder);

  // output function declaration
  doIndent();
  out << printFuncName(func, inputVarFinder.varDecls, outputVarFinder.varDecls);

  // If we're just generating a header, this is all we need to do.
  if (outputKind == HeaderGen) {
    out << ";\n";
    return;
  }

  out << " {\n";

  indent++;

  // find all the vars that are not inputs or outputs and declare them
  resetUniqueNameCounters();
  FindVars varFinder(func->inputs, func->outputs, this);
  func->body.accept(&varFinder);
  varMap = varFinder.varMap;
  localVars = varFinder.localVars;

  // Find the first unpackTensorData IR node. If this is a task, it will be the
  // first Stmt in the task.
  // Find the unpackTensorData IR node in the task. There must be one.
  struct UnpackTensorDataFinder : public IRVisitor {
    void visit(const UnpackTensorData* op) {
      if (data == nullptr) {
        data = op;
      }
    }
    const UnpackTensorData* data = nullptr;
  } unpackTensorDataFinder;
  func->body.accept(&unpackTensorDataFinder);

  // For tasks, unpack the regions.
  if (func->name.find("task") != std::string::npos) {
    taco_iassert(unpackTensorDataFinder.data);
    for (size_t i = 0; i < unpackTensorDataFinder.data->regions.size(); i++) {
      doIndent();
      auto reg = unpackTensorDataFinder.data->regions[i];
      out << "PhysicalRegion " << reg << " = regions[" << i << "];\n";
      doIndent();
      auto regParent = unpackTensorDataFinder.data->regionParents[i];
      out << "LogicalRegion " << regParent << " = regions[" << i << "].get_logical_region();\n";
    }
    out << "\n";
  }

  // If this was a distributed for loop, emit the point as the loop index.
  // TODO (rohany): Hacky way to tell that this function was a task.
  if (func->name.find("task") != std::string::npos) {
    auto forL = this->funcToFor.at(func).as<For>();
    taco_iassert(forL) << "must be a for";
    if (distributedParallelUnit(forL->parallel_unit)) {
      doIndent();
      out << printType(forL->var.type(), false) << " " << forL->var << " = task->index_point[0];\n";
    }
  }

  // Unpack arguments.
  auto args = this->taskArgs[func];
  if (args.size() > 0) {
    doIndent();
    out << taskArgsName(func->name) << "* args = (" << taskArgsName(func->name) << "*)(task->args);\n";
    // Unpack arguments from the pack;
    for (auto arg : args) {
      doIndent();
      out << printType(getVarType(arg), false) << " " << arg << " = args->" << arg << ";\n";
    }

    out << "\n";
  }

  // TODO (rohany): Hacky way to tell that this function was a task.
  // Remove certain declarations from the head of tasks. In particular, we'll remove dimension
  // sizes, since we'll pass those down through task arguments, and we'll also drop the regions
  // themselves as we have a separate procedure for declaring them. We'll also drop region parents,
  // since those are recovered through the same process as regions themselves.
  if (func->name.find("task") != std::string::npos) {
    std::set<Expr> regionArgs;
    regionArgs.insert(unpackTensorDataFinder.data->regions.begin(), unpackTensorDataFinder.data->regions.end());
    std::vector<Expr> toRemove;
    for (const auto& it : varFinder.varDecls) {
      if (isa<GetProperty>(it.first)) {
        auto g = it.first.as<GetProperty>();
        if (g->property == TensorProperty::Dimension || util::contains(regionArgs, it.first) ||
            g->property == TensorProperty::ValuesParent || g->property == TensorProperty::IndicesParents ||
            g->property == TensorProperty::DenseLevelRun || g->property == TensorProperty::Indices ||
            g->property == TensorProperty::Values) {
          toRemove.push_back(g);
        }
      }
    }
    for (const auto& it : toRemove) {
      varFinder.varDecls.erase(it);
    }
  }

  // Print variable declarations
  out << printDecls(varFinder.varDecls, func->inputs, func->outputs) << std::endl;

  if (emittingCoroutine) {
    out << printContextDeclAndInit(varMap, localVars, numYields, func->name)
        << std::endl;
  }

  // output body
  print(func->body);

  // TODO (rohany): I don't think that we need to reassign the values to the regions,
  //  as we don't need to write out changes to LogicalRegions. However, we might need
  //  to do something here for LegionTensorT specific operations (maybe updating sizes
  //  of result tensors or something like that).
  // output repack only if we allocated memory
  // if (checkForAlloc(func))
  //   out << std::endl << printPack(varFinder.outputProperties, func->outputs);

  if (emittingCoroutine) {
    out << printCoroutineFinish(numYields, funcName);
  }

//  doIndent();
  indent--;

  doIndent();
  out << "}\n";
}

void CodegenLegionC::visit(const Allocate* op) {
  if (!op->pack.logicalRegion.defined()) {
    CodeGen_C::visit(op);
  } else {
    doIndent();
    op->var.accept(this);
    stream << " = ";
    if (op->is_realloc) {
      stream << "legionRealloc(ctx, runtime, ";
      op->pack.logicalRegion.accept(this);
      stream << ", ";
      op->old_elements.accept(this);
      stream << ", ";
      op->num_elements.accept(this);
      stream << ", ";
      op->pack.fieldID.accept(this);
      stream << ", ";
      op->pack.priv.accept(this);
    } else {
      stream << "legionMalloc(ctx, runtime, ";
      op->pack.logicalRegion.accept(this);
      stream << ", ";
      op->num_elements.accept(this);
      stream << ", ";
      op->pack.fieldID.accept(this);
      stream << ", ";
      op->pack.priv.accept(this);
    }
    stream << ");";
    stream << std::endl;
  }
}

void CodegenLegionC::compileToDirectory(std::string prefix, ir::Stmt stmt) {
  taco_iassert(!prefix.empty() && prefix.back() == '/');
  {
    auto header = prefix + "taco-generated.h";
    std::ofstream f(header);
    CodegenLegionC headerComp(f, HeaderGen);
    headerComp.compile(stmt);
    f.close();
  }
  {
    auto source = prefix + "taco-generated.cpp";
    std::ofstream f(source);
    CodegenLegionC headerComp(f, ImplementationGen);
    headerComp.compile(stmt);
    f.close();
  }
}

}
}
