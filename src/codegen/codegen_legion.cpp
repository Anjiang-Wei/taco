#include "codegen_legion.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/version.h"
#include "taco/util/strings.h"

namespace taco {
namespace ir {

std::string CodegenLegion::unpackTensorProperty(std::string varname, const GetProperty* op, bool is_output_prop) {
  std::stringstream ret;
  std::string tp;
  auto tensor = op->tensor.as<Var>();
  auto values = ir::GetProperty::make(ir::Expr(tensor), TensorProperty::Values);
  ret << "  ";
  if (op->property == TensorProperty::Dimension) {
//    tp = "int";
//    ret << tp << " " << varname << " = runtime->get_index_space_domain(get_index_space(" << tensor->name <<
//        ")).hi()[" << op->mode << "] + 1;\n";
    ret << "size_t " << varname << " = " << tensor->name << "->dims[" << op->mode << "];\n";
  } else if (op->property == TensorProperty::IndexSpace) {
    tp = "auto";
    ret << tp << " " << varname << " = get_index_space(" << tensor->name << ");\n";
  } else if (op->property == TensorProperty::ValuesReadAccessor || op->property == TensorProperty::ValuesWriteAccessor) {
    // We can't just use the tensor's name when constructing the value accessors, as for child
    // tasks the value array is not just the tensor's name.
    // TODO (rohany): This is a hack, as I don't want to thread more information through
    //  the GetProperty or unify how we set the FieldID for the query results.
    ir::Expr fid;
    if (varname.find("_nnz") != std::string::npos) {
      fid = ir::Symbol::make("FID_VAL");
    } else {
      fid = ir::GetProperty::make(tensor, TensorProperty::ValuesFieldID);
    }
    ret << "auto " << varname << " = createAccessor<" << accessorTypeString(op) << ">(" << values << ", " << fid << ");\n";
  } else if (op->property == TensorProperty::ValuesReductionAccessor || op->property == TensorProperty::ValuesReductionNonExclusiveAccessor) {
    // TODO (rohany): This is a hack, as I don't want to thread more information through
    //  the GetProperty or unify how we set the FieldID for the query results.
    ir::Expr fid;
    if (varname.find("_nnz") != std::string::npos) {
      fid = ir::Symbol::make("FID_VAL");
    } else {
      fid = ir::GetProperty::make(tensor, TensorProperty::ValuesFieldID);
    }
    ret << "auto " << varname << " = createAccessor<" << accessorTypeString(op) << ">(" << values << ", " << fid << ", " << LegionRedopString(op->type) << ");\n";
  } else if (op->property == TensorProperty::DenseLevelRun) {
    ret << "IndexSpace " << varname << " = " << tensor->name << "->denseLevelRuns[" << op->index << "];\n";
  } else if (op->property == TensorProperty::Values) {
    // TODO (rohany): This is a hack...
    if (varname.find("_nnz") != std::string::npos) {
      ret << "RegionWrapper " << varname << ";\n";
      return ret.str();
    }
    ret << "RegionWrapper " << varname << " = " << tensor->name << "->vals;\n";
  } else if (op->property == TensorProperty::ValuesParent) {
    ret << "auto " << varname << " = " << tensor->name << "->valsParent;\n";
  } else if (op->property == TensorProperty::Indices) {
    ret << "RegionWrapper " << varname << " = " << tensor->name << "->indices[" << op->mode << "][" << op->index << "];\n";
  } else if (op->property == TensorProperty::IndicesParents) {
    ret << "auto " << varname << " = " << tensor->name << "->indicesParents[" << op->mode << "][" << op->index
        << "];\n";
  } else if (op->property == TensorProperty::IndicesAccessor) {
    std::string fid;
    if (op->accessorArgs.field.as<ir::Symbol>()) {
      fid = op->accessorArgs.field.as<ir::Symbol>()->name;
    } else if (op->accessorArgs.field.as<ir::GetProperty>()) {
      fid = op->accessorArgs.field.as<ir::GetProperty>()->name;
    } else {
      taco_iassert(false);
    }
    auto regionAccessing = op->accessorArgs.regionAccessing;
    ret << "auto " << varname << " = createAccessor<" << accessorTypeString(op) << ">(" << regionAccessing << ", "
        << fid << ");\n";
  } else if (op->property == TensorProperty::ValuesFieldID) {
    ret << "auto " << varname << " = " << tensor->name << "->valsFieldID;\n";
  } else if (op->property == TensorProperty::IndicesFieldID) {
    ret << "auto " << varname << " = " << tensor->name << "->indicesFieldIDs[" << op->mode << "][" << op->index << "];\n";
  } else {
    return CodeGen::unpackTensorProperty(varname, op, is_output_prop);
  }
  return ret.str();
}

std::string CodegenLegion::printFuncName(const Function *func,
                                          std::map<Expr, std::string, ExprCompare> inputMap,
                                          std::map<Expr, std::string, ExprCompare> outputMap) {
  std::stringstream ret;

  bool isTask = func->name.find("task") != std::string::npos;

  // TODO (rohany): This is a hack.
  // When the parent function has a type, we give that type to all children functions
  // to handle reductions into scalars, as those tasks return values to accumulate
  // as the result of the scalar. However, since we also support upfront partitioning,
  // the PARTITION_ONLY functions also have a non-void return type, which confuses the
  // logic here. To be safe, we only let the tasks have a explicit return type if the
  // return type is a primitive type.
  if (func->returnType.getKind() == Datatype::Undefined || (isTask && func->returnType.getKind() == Datatype::CppType)) {
    ret << "void " << func->name << "(";
  } else {
    ret << func->returnType << " " << func->name << "(";
  }

  std::string delimiter;
//  const auto returnType = func->getReturnType();
//  if (returnType.second != Datatype()) {
//    ret << "void **" << ctxName << ", ";
//    ret << "char *" << coordsName << ", ";
//    ret << printType(returnType.second, true) << valName << ", ";
//    ret << "int32_t *" << bufCapacityName;
//    delimiter = ", ";
//  }

  if (!isTask) {
    // Add the context and runtime arguments.
    ret << "Legion::Context ctx, Legion::Runtime* runtime, ";
  }

  bool unfoldOutput = false;
  for (size_t i=0; i<func->outputs.size(); i++) {
    auto var = func->outputs[i].as<Var>();
    taco_iassert(var) << "Unable to convert output " << func->outputs[i]
                      << " to Var";
    if (var->is_parameter) {
      unfoldOutput = true;
      break;
    }

    if (var->is_tensor) {
      ret << delimiter << "LegionTensor* " << var->name;
    } else {
      auto tp = printType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    delimiter = ", ";
  }

  if (unfoldOutput) {
    for (auto prop : sortProps(outputMap)) {
      ret << delimiter << printTensorProperty(outputMap[prop], prop, true);
      delimiter = ", ";
    }
  }

  for (size_t i=0; i<func->inputs.size(); i++) {
    auto var = func->inputs[i].as<Var>();
    taco_iassert(var) << "Unable to convert output " << func->inputs[i]
                      << " to Var";
    if (var->is_tensor) {
      ret << delimiter << "LegionTensor* " << var->name;
    } else {
      auto tp = printType(var->type, var->is_ptr);
      ret << delimiter << tp << " " << var->name;
    }
    delimiter = ", ";
  }

  ret << ")";
  return ret.str();
}

// TODO (rohany): It's possible that this should be 2 calls -- collect and emit.
//  In that way, we can put all of the data structure collection up front.
void CodegenLegion::collectAndEmitAccessors(ir::Stmt stmt, std::ostream& out) {
  // Figure out what accessors we need to emit.
  struct AccessorCollector : public IRVisitor {
    void visit(const GetProperty* op) {
      switch (op->property) {
        case TensorProperty::ValuesReadAccessor:
        case TensorProperty::ValuesWriteAccessor:
        case TensorProperty::ValuesReductionAccessor:
        case TensorProperty::ValuesReductionNonExclusiveAccessor:
          this->accessors.insert(AccessorInfo{op->property, op->mode, op->type});
          break;
        case TensorProperty::IndicesAccessor:
          this->accessors.insert(AccessorInfo{op->property, op->accessorArgs.dim, op->accessorArgs.elemType, op->accessorArgs.priv});
          break;
        default:
          return;
      }
    }
    std::set<AccessorInfo> accessors;
  };
  AccessorCollector acol;
  stmt.accept(&acol);
  this->accessors = acol.accessors;

  // Emit a field accessor for each kind.
  for (auto info : this->accessors) {
    if (info.prop == TensorProperty::ValuesReductionAccessor) {
      out << "typedef ReductionAccessor<SumReduction<" << printType(info.typ, false)
          << ">,true," << info.dims << ",coord_t,Realm::AffineAccessor<" << printType(info.typ, false)
          << "," << info.dims << ",coord_t>> AccessorReduce" << printType(info.typ, false) << info.dims << ";\n";
    } else if (info.prop == TensorProperty::ValuesReductionNonExclusiveAccessor) {
        out << "typedef ReductionAccessor<SumReduction<" << printType(info.typ, false)
            << ">,false," << info.dims << ",coord_t,Realm::AffineAccessor<" << printType(info.typ, false)
            << "," << info.dims << ",coord_t>> AccessorReduceNonExcl" << printType(info.typ, false) << info.dims << ";\n";
    } else {
      std::string priv, suffix;
      if (info.prop == TensorProperty::ValuesWriteAccessor || info.priv == RW) {
        priv = "READ_WRITE";
        suffix = "RW";
      } else {
        priv = "READ_ONLY";
        suffix = "RO";
      }
      out << "typedef FieldAccessor<" << priv << "," << printType(info.typ, false) << ","
          << info.dims << ",coord_t,Realm::AffineAccessor<" << printType(info.typ, false) << ","
          << info.dims << ",coord_t>> Accessor" << suffix << printTypeInName(info.typ, false) << info.dims << ";\n";
    }
  }
  out << "\n";
}

void CodegenLegion::emitHeaders(OutputKind outputKind, std::ostream &out) {
  if (outputKind == HeaderGen) {
    out << "#include \"legion.h\"\n";
    out << "#include \"legion_tensor.h\"\n";
  } else {
    out << "#include \"taco_legion_header.h\"\n";
    out << "#include \"taco_mapper.h\"\n";
    out << "#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))\n";
    out << "#define TACO_MAX(_a,_b) ((_a) < (_b) ? (_b) : (_a))\n";
    out << "using namespace Legion;\n";
  }
  out << "\n";
}

void CodegenLegion::collectAllFunctions(ir::Stmt stmt) {
  struct FunctionFinder : public IRVisitor {
    void visit(const Function* func) {
      this->funcs.push_back(func);
    }
    std::vector<Stmt> funcs;
  };
  FunctionFinder ff;
  stmt.accept(&ff);
  this->allFunctions = ff.funcs;
}

void CodegenLegion::rewriteFunctionTaskIDs() {
  taco_iassert(this->allFunctions.size() > 0) << "must be called after collectAllFunctions()";
  // Rewrite task ID's using a scan-like algorithm.
  auto maxTaskID = 0;
  for (size_t i = 0; i < this->allFunctions.size(); i++) {
    // Increment all task ID's present in the function by the maxTaskID.
    struct TaskIDRewriter : public IRRewriter {
      void visit(const For* node) {
        auto body = rewrite(node->contents);
        if (node->isTask) {
          stmt = ir::For::make(node->var, node->start, node->end, node->increment, body, node->kind, node->parallel_unit, node->unrollFactor, node->vec_width, node->isTask, node->taskID + maxTaskID);
        } else {
          stmt = ir::For::make(node->var, node->start, node->end, node->increment, body, node->kind, node->parallel_unit, node->unrollFactor, node->vec_width, node->isTask, node->taskID);
        }
      }

      void visit(const Call* call) {
        if (call->func == "taskID") {
          auto oldTaskID = call->args[0].as<Literal>()->getValue<int>();
          expr = ir::Call::make("taskID", {oldTaskID + maxTaskID}, Auto);
        } else {
          // It's important that we don't change the Call node if nothing has changed,
          // because we rely on this equality in other parts of the program (like when
          // converting reduction accesses into operator <<=.
          IRRewriter::visit(call);
        }
      }

      void visit(const PackTaskArgs* p) {
        stmt = PackTaskArgs::make(p->var, p->forTaskID + this->maxTaskID, p->prefixVars, p->prefixExprs);
      }

      int maxTaskID;
    };
    TaskIDRewriter rw; rw.maxTaskID = maxTaskID;
    this->allFunctions[i] = rw.rewrite(this->allFunctions[i]);

    struct MaxTaskIDFinder : public IRVisitor {
      void visit(const For* node) {
        if (node->isTask) {
          this->maxTaskID = std::max(this->maxTaskID, node->taskID);
        }
        if (node->contents.defined()) {
          node->contents.accept(this);
        }
      }
      int maxTaskID;
    };
    MaxTaskIDFinder mf; mf.maxTaskID = maxTaskID;
    this->allFunctions[i].accept(&mf);
    maxTaskID = mf.maxTaskID;
  }
}

void CodegenLegion::analyzeAndCreateTasks(OutputKind outputKind, std::ostream& out) {
  for (auto ffunc : this->allFunctions) {
    struct TaskCollector : public IRVisitor {
      void visit(const For* node) {
        if (node->isTask) {
          std::stringstream funcName;
          funcName << "task_" << node->taskID;
          auto func = ir::Function::make(
              funcName.str(),
              {},
              {
                  // TODO (rohany): Marking these as is_parameter = false stops some weird behavior
                  //  in the rest of the code generator.
                  ir::Var::make("task", Task, true, false, false),
                  ir::Var::make("regions", PhysicalRegionVectorRef, false, false, false),
                  ir::Var::make("ctx", Context, false, false, false),
                  ir::Var::make("runtime", Runtime, true, false, false),
              },
              node->contents,
              this->returnType
          );
          this->functions.push_back(func);
          this->idToFor[node->taskID] = node;
          this->idToFunc[node->taskID] = func;
          this->funcToFor[func] = node;
        }
        node->contents.accept(this);
      }

      std::vector<Stmt> functions;

      std::map<int, Stmt> idToFor;
      std::map<int, Stmt> idToFunc;
      std::map<Stmt, Stmt> funcToFor;

      Datatype returnType;
    };
    TaskCollector tc;
    tc.returnType = ffunc.as<Function>()->returnType;
    ffunc.accept(&tc);
    for (auto f : util::reverse(tc.functions)) {
      this->functions[ffunc].push_back(f);
      this->funcToParentFunc[f] = ffunc;
    }
    this->idToFor.insert(tc.idToFor.begin(), tc.idToFor.end());
    this->idToFunc.insert(tc.idToFunc.begin(), tc.idToFunc.end());
    this->funcToFor.insert(tc.funcToFor.begin(), tc.funcToFor.end());

    // Collect the region arguments that each function needs.
    if (isa<Function>(ffunc)) {
      auto func = ffunc.as<Function>();
      for (auto& arg : func->outputs) {
        if (arg.as<Var>()->is_tensor) {
          this->regionArgs[func].push_back(arg);
        }
      }
      for (auto& arg : func->inputs) {
        if (arg.as<Var>()->is_tensor) {
          this->regionArgs[func].push_back(arg);
        }
      }
    }

    // Find variables used by each task in the task call hierarchy.
    struct VarsUsedByTask : public IRVisitor {
      void visit(const Var* v) {
        if (this->usedVars.size() == 0) {
          this->usedVars.push_back({});
        }
        if (v->type.getKind() != Datatype::CppType && !v->is_tensor) {
          this->usedVars.back().insert(v);
        }
      }

      // We don't want to visit the variables within GetProperty objects. In
      // this function, we also mark certain GetProperty objects as needed
      // when we see other GetProperties i.e. adding the ValuesField GetProperty
      // when we see Values.
      void visit(const GetProperty* g) {
        if (g->property == TensorProperty::Dimension ||
            g->property == TensorProperty::DenseLevelRun) {
          if (this->usedVars.size() == 0) {
            this->usedVars.push_back({});
          }
          this->usedVars.back().insert(g);
        }
        else if (g->property == TensorProperty::IndicesAccessor) {
          if (isa<GetProperty>(g->accessorArgs.field)) {
            if (this->usedVars.size() == 0) {
              this->usedVars.push_back({});
            }
            this->usedVars.back().insert(g->accessorArgs.field);
          }
        } else if ((g->property == TensorProperty::ValuesReadAccessor ||
                   g->property == TensorProperty::ValuesWriteAccessor ||
                   g->property == TensorProperty::ValuesReductionAccessor ||
                   g->property == TensorProperty::ValuesReductionNonExclusiveAccessor) &&
                   // TODO (rohany): This is a big hack, as we don't want to include generating
                   //  the values field access for query results. The real solution here is to
                   //  treat query results more like real tensors, that have get properties etc.
                   g->name.find("_nnz") == std::string::npos) {
          if (this->usedVars.size() == 0) {
            this->usedVars.push_back({});
          }
          this->usedVars.back().insert(ir::GetProperty::make(g->tensor, TensorProperty::ValuesFieldID));
        }
      }

      void visit(const VarDecl* v) {
        if (this->varsDeclared.size() == 0) {
          this->varsDeclared.push_back({});
        }
        this->varsDeclared.back().insert(v->var);
        v->rhs.accept(this);
      }

      void visit(const For* f) {
        if (f->isTask) {
          this->usedVars.push_back({});
          this->varsDeclared.push_back({});
        }
        // If f is a task, then it needs it's iteration variable passed down. So f is
        // a task, then we can treat it as _using_ the iteration variable. Otherwise,
        // the for loop declares its iterator variable. However, we only want to do
        // this if we are already in a task.
        if (!f->isTask && this->varsDeclared.size() > 0) {
          this->varsDeclared.back().insert(f->var);
        } else if (f->isTask && this->usedVars.size() > 0) {
          this->usedVars.back().insert(f->var);
        }

        f->start.accept(this);
        f->end.accept(this);
        f->increment.accept(this);
        f->contents.accept(this);
      }

      std::vector<std::set<Expr>> usedVars;
      std::vector<std::set<Expr>> varsDeclared;
    };
    VarsUsedByTask v;
    ffunc.accept(&v);

    // TODO (rohany): Clean up this code.
    auto funcIdx = 0;
    for (int i = v.usedVars.size() - 1; i > 0; i--) {
      auto func = this->functions[ffunc][funcIdx].as<Function>();
      taco_iassert(func) << "must be func";
      // Try to find the variables needed by a task. It's all the variables it uses that it doesn't
      // declare and are used by tasks above it.
      std::vector<Expr> uses;
      std::set_difference(v.usedVars[i].begin(), v.usedVars[i].end(), v.varsDeclared[i].begin(), v.varsDeclared[i].end(), std::back_inserter(uses));

      // TODO (rohany): For a distributed for loop, remove the iterator variable?
      auto forL = this->funcToFor.at(func).as<For>();
      if (distributedParallelUnit(forL->parallel_unit)) {
        auto matchedIdx = -1;
        for (size_t pos = 0; pos < uses.size(); pos++) {
          if (uses[pos] == forL->var) {
            matchedIdx = pos;
            break;
          }
        }
        if (matchedIdx != -1) {
          uses.erase(uses.begin() + matchedIdx);
        }
      }

      v.usedVars[i-1].insert(uses.begin(), uses.end());

      // Deduplicate any GetProperty uses so that they aren't emitted twice.
      std::set<GetProperty::Hashable> collected;
      std::vector<Expr> newUses;
      for (auto& e : uses) {
        if (isa<GetProperty>(e)) {
          // See if this GetProperty is already present.
          auto gp = e.as<GetProperty>();
          if (!util::contains(collected, gp->toHashable())) {
            newUses.push_back(e);
            collected.insert(gp->toHashable());
          }
        } else {
          newUses.push_back(e);
        }
      }
      uses = newUses;
      struct ExprSorter {
        bool operator() (Expr e1, Expr e2) {
          auto e1Name = getVarName(e1);
          auto e2Name = getVarName(e2);
          return e1Name < e2Name;
        }
      } exprSorter;
      std::sort(uses.begin(), uses.end(), exprSorter);

      // Find any included arguments from PackTaskArgs for this function.
      struct PackFinder : public IRVisitor {
        void visit(const PackTaskArgs* pack) {
          if (pack->forTaskID == taskID) {
            this->packVars = pack->prefixVars;
          }
        }

        std::vector<Expr> packVars;
        int taskID;
      };
      PackFinder pf; pf.taskID = forL->taskID;
      ffunc.accept(&pf);

      // We only need to generate these structs for the implementation,
      // as they are not user facing.
      if (outputKind == ImplementationGen) {
        out << "struct " << this->taskArgsName(func->name) << " {\n";
        this->indent++;
        for (auto& var : pf.packVars) {
          doIndent();
          out << printType(getVarType(var), false) << " " << var << ";\n";
        }
        for (auto& it : uses) {
          doIndent();
          out << printType(getVarType(it), false) << " " << it << ";\n";
        }
        this->indent--;
        out << "};\n\n";
      }

      this->taskArgs[func] = uses;

      funcIdx++;
    }

    // We also need to generate any structs declared by this function before we generate
    // code for the function. We only need to define these structs in the header file.
    if (outputKind == HeaderGen) {
      struct DeclareStructFinder : public IRVisitor {
        void visit(const DeclareStruct* ds) {
          this->declares.push_back(ds);
        }
        std::vector<Stmt> declares;
      };
      DeclareStructFinder df;
      ffunc.accept(&df);

      for (auto& declareExpr : df.declares) {
        auto declare = declareExpr.as<DeclareStruct>();
        out << "struct " << declare->name << " {\n";
        this->indent++;
        for (size_t i = 0; i < declare->fields.size(); i++) {
          doIndent();
          out << printType(declare->fieldTypes[i], false) << " " << declare->fields[i] << ";\n";
        }
        this->indent--;
        out << "};\n\n";
      }
    }
  }
}

std::string CodegenLegion::procForTask(Stmt, Stmt) {
  if (TACO_FEATURE_OPENMP) {
    return "Processor::OMP_PROC";
  }
  return "Processor::LOC_PROC";
}

void CodegenLegion::emitRegisterTasks(OutputKind outputKind, std::ostream &out) {
  // If we're emitting a header, just generate the stub.
  if (outputKind == HeaderGen) {
    out << "void registerTacoTasks();\n";
    return;
  }

  // Output a function performing all of the task registrations.
  out << "void registerTacoTasks() {\n";
  indent++;

  for (auto ffunc : this->allFunctions) {
    for (auto& f : this->functions[ffunc]) {
      auto func = f.as<Function>();
      auto forL = this->funcToFor.at(func).as<For>();

      // Tasks that launch no tasks are leaf tasks, so let Legion know about that.
      struct LeafTaskFinder : public IRVisitor {
        void visit(const For* node) {
          if (node->isTask) {
            this->isLeaf = false;
          }
          node->contents.accept(this);
        }
        void visit(const Call* node) {
          // Tasks that create partitions aren't leaf tasks either.
          if (node->func.find("create_index_partition") != std::string::npos) {
            this->isLeaf = false;
          }
        }
        bool isLeaf = true;
      };
      LeafTaskFinder finder;
      forL->contents.accept(&finder);

      doIndent();
      out << "{\n";
      indent++;

      doIndent();
      out << "TaskVariantRegistrar registrar(taskID(" << forL->taskID << "), \"" << func->name << "\");\n";

      // TODO (rohany): Make this delegation a virtual function that needs to be overridden.
      doIndent();
      std::string proc = this->procForTask(ffunc, func);
      out << "registrar.add_constraint(ProcessorConstraint(" << proc << "));\n";

      doIndent();
      if (finder.isLeaf) {
        out << "registrar.set_leaf();\n";
      } else {
        out << "registrar.set_inner();\n";
      }

      doIndent();
      // TODO (rohany): This is a hack.
      // When the parent function has a type, we give that type to all children functions
      // to handle reductions into scalars, as those tasks return values to accumulate
      // as the result of the scalar. However, since we also support upfront partitioning,
      // the PARTITION_ONLY functions also have a non-void return type, which confuses the
      // logic here. To be safe, we only let the tasks have a explicit return type if the
      // return type is a primitive type.
      if (func->returnType.getKind() != Datatype::Undefined && func->returnType.getKind() != Datatype::CppType) {
        out << "Runtime::preregister_task_variant<" << func->returnType << "," << func->name << ">(registrar, \"" <<  func->name << "\");\n";
      } else {
        out << "Runtime::preregister_task_variant<" << func->name << ">(registrar, \"" <<  func->name << "\");\n";
      }

      indent--;

      doIndent();
      out << "}\n";
    }
  }

  out << "}\n";
}

}
}
