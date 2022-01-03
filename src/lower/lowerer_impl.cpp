#include <taco/lower/mode_format_compressed.h>
#include "taco/lower/mode_format_rect_compressed.h"
#include "taco/lower/mode_format_dense.h"
#include "taco/lower/lowerer_impl.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/provenance_graph.h"
#include "taco/ir/ir.h"
#include "ir/ir_generators.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/simplify.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"
#include "mode_access.h"
#include "taco/util/collections.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/tensor.h"

using namespace std;
using namespace taco::ir;
using taco::util::combine;

namespace taco {

class LowererImpl::Visitor : public IndexNotationVisitorStrict {
public:
  Visitor(LowererImpl* impl) : impl(impl) {}
  Stmt lower(IndexStmt stmt) {
    this->stmt = Stmt();
    impl->accessibleIterators.scope();
    IndexStmtVisitorStrict::visit(stmt);
    impl->accessibleIterators.unscope();
    return this->stmt;
  }
  Expr lower(IndexExpr expr) {
    this->expr = Expr();
    IndexExprVisitorStrict::visit(expr);
    return this->expr;
  }
private:
  LowererImpl* impl;
  Expr expr;
  Stmt stmt;
  using IndexNotationVisitorStrict::visit;
  void visit(const AssignmentNode* node)    { stmt = impl->lowerAssignment(node); }
  void visit(const YieldNode* node)         { stmt = impl->lowerYield(node); }
  void visit(const ForallNode* node)        { stmt = impl->lowerForall(node); }
  void visit(const WhereNode* node)         { stmt = impl->lowerWhere(node); }
  void visit(const MultiNode* node)         { stmt = impl->lowerMulti(node); }
  void visit(const SuchThatNode* node)      { stmt = impl->lowerSuchThat(node); }
  void visit(const SequenceNode* node)      { stmt = impl->lowerSequence(node); }
  void visit(const AssembleNode* node)      { stmt = impl->lowerAssemble(node); }
  void visit(const AccessNode* node)        { expr = impl->lowerAccess(node); }
  void visit(const LiteralNode* node)       { expr = impl->lowerLiteral(node); }
  void visit(const NegNode* node)           { expr = impl->lowerNeg(node); }
  void visit(const AddNode* node)           { expr = impl->lowerAdd(node); }
  void visit(const SubNode* node)           { expr = impl->lowerSub(node); }
  void visit(const MulNode* node)           { expr = impl->lowerMul(node); }
  void visit(const DivNode* node)           { expr = impl->lowerDiv(node); }
  void visit(const SqrtNode* node)          { expr = impl->lowerSqrt(node); }
  void visit(const CastNode* node)          { expr = impl->lowerCast(node); }
  void visit(const CallIntrinsicNode* node) { expr = impl->lowerCallIntrinsic(node); }
  void visit(const ReductionNode* node)  {
    taco_ierror << "Reduction nodes not supported in concrete index notation";
  }
  void visit(const PlaceNode* node) { expr = impl->lower(node->expr); }
  void visit(const PartitionNode* node) { expr = impl->lower(node->expr); }
};

LowererImpl::LowererImpl() : visitor(new Visitor(this)) {
}


static void createCapacityVars(const map<TensorVar, Expr>& tensorVars,
                               map<Expr, Expr>* capacityVars) {
  for (auto& tensorVar : tensorVars) {
    Expr tensor = tensorVar.second;
    Expr capacityVar = Var::make(util::toString(tensor) + "_capacity", Int());
    capacityVars->insert({tensor, capacityVar});
  }
}

static void createReducedValueVars(const vector<Access>& inputAccesses,
                                   map<Access, Expr>* reducedValueVars) {
  for (const auto& access : inputAccesses) {
    const TensorVar inputTensor = access.getTensorVar();
    const std::string name = inputTensor.getName() + "_val";
    const Datatype type = inputTensor.getType().getDataType();
    reducedValueVars->insert({access, Var::make(name, type)});
  }
}

static void getDependentTensors(IndexStmt stmt, std::set<TensorVar>& tensors) {
  std::set<TensorVar> prev;
  do {
    prev = tensors;
    match(stmt,
      function<void(const AssignmentNode*, Matcher*)>([&](
          const AssignmentNode* n, Matcher* m) {
        if (util::contains(tensors, n->lhs.getTensorVar())) {
          const auto arguments = getArguments(Assignment(n));
          tensors.insert(arguments.begin(), arguments.end());
        }
      })
    );
  } while (prev != tensors);
}

static bool needComputeValues(IndexStmt stmt, TensorVar tensor) {
  if (tensor.getType().getDataType() != Bool) {
    return true;
  }

  struct ReturnsTrue : public IndexExprRewriterStrict {
    void visit(const AccessNode* op) {
      if (op->isAccessingStructure) {
        expr = op;
      }
    }

    void visit(const LiteralNode* op) {
      if (op->getDataType() == Bool && op->getVal<bool>()) {
        expr = op;
      }
    }

    void visit(const NegNode* op) {
      expr = rewrite(op->a);
    }

    void visit(const AddNode* op) {
      if (rewrite(op->a).defined() || rewrite(op->b).defined()) {
        expr = op;
      }
    }

    void visit(const MulNode* op) {
      if (rewrite(op->a).defined() && rewrite(op->b).defined()) {
        expr = op;
      }
    }

    void visit(const CastNode* op) {
      expr = rewrite(op->a);
    }

    void visit(const SqrtNode* op) {}
    void visit(const SubNode* op) {}
    void visit(const DivNode* op) {}
    void visit(const CallIntrinsicNode* op) {}
    void visit(const ReductionNode* op) {}
  };

  bool needComputeValue = false;
  match(stmt,
    function<void(const AssignmentNode*, Matcher*)>([&](
        const AssignmentNode* n, Matcher* m) {
      if (n->lhs.getTensorVar() == tensor &&
          !ReturnsTrue().rewrite(n->rhs).defined()) {
        needComputeValue = true;
      }
    })
  );

  return needComputeValue;
}

/// Returns true iff a result mode is assembled by inserting a sparse set of
/// result coordinates (e.g., compressed to dense).
static
bool hasSparseInserts(const std::vector<Iterator>& resultIterators,
                      const std::multimap<IndexVar, Iterator>& inputIterators) {
  for (const auto& resultIterator : resultIterators) {
    if (resultIterator.hasInsert()) {
      const auto indexVar = resultIterator.getIndexVar();
      const auto accessedInputs = inputIterators.equal_range(indexVar);
      for (auto inputIterator = accessedInputs.first;
           inputIterator != accessedInputs.second; ++inputIterator) {
        if (!inputIterator->second.isFull()) {
          return true;
        }
      }
    }
  }
  return false;
}

Stmt
LowererImpl::lower(IndexStmt stmt, string name,
                   bool assemble, bool compute,
                   bool pack, bool unpack,
                   bool partition, bool waitOnFutureMap, bool setPlacementPrivilege)
{
  this->funcName = name;
  this->assemble = assemble;
  this->compute = compute;
  this->legion = name.find("Legion") != std::string::npos;
  this->waitOnFutureMap = waitOnFutureMap;

  // Set up control over the privilege for placement operations.
  this->setPlacementPrivilege = setPlacementPrivilege;
  if (this->setPlacementPrivilege) {
    this->placementPrivilegeVar = ir::Var::make("priv", Datatype("Legion::PrivilegeMode"), false, false, true /* parameter */);
  }

  definedIndexVarsOrdered = {};
  definedIndexVars = {};
  definedIndexVarsExpanded = {};
  iterationSpacePointIdentifiers = {};

  // Figure out what sort of code we're supposed to be emitting.
  if (compute && partition) {
    this->legionLoweringKind = PARTITION_AND_COMPUTE;
  } else if (compute && !partition) {
    this->legionLoweringKind = COMPUTE_ONLY;
  } else if (!compute && partition) {
    this->legionLoweringKind = PARTITION_ONLY;
  } else {
    taco_uassert(false) << " invalid combination of compute/partition parameters";
  }

  // Hack: We still need to set compute to be true so that the rest of the machinery
  // works as expected if we are PARTITION_ONLY.
  if (this->legionLoweringKind == PARTITION_ONLY) {
    this->compute = true;
  }

  // Create result and parameter variables
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temporaries = getTemporaries(stmt);

  this->resultTensors.insert(results.begin(), results.end());
  // The set of LegionTensors is all of the results and arguments.
  this->legionTensors.insert(results.begin(), results.end());
  this->legionTensors.insert(arguments.begin(), arguments.end());

  needCompute = {};
  if (generateAssembleCode()) {
    const auto attrQueryResults = getAttrQueryResults(stmt);
    needCompute.insert(attrQueryResults.begin(), attrQueryResults.end());
  }
  if (generateComputeCode()) {
    needCompute.insert(results.begin(), results.end());
  }
  getDependentTensors(stmt, needCompute);

  assembledByUngroupedInsert = util::toSet(
      getAssembledByUngroupedInsertion(stmt));

  // Create datastructure needed for temporary workspace hoisting/reuse
  temporaryInitialization = getTemporaryLocations(stmt);

  // Convert tensor results and arguments IR variables
  map<TensorVar, Expr> resultVars;
  vector<Expr> resultsIR = createVars(results, &resultVars, unpack);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createVars(arguments, &tensorVars, pack);

  // Create variables for index sets on result tensors.
  vector<Expr> indexSetArgs;
  for (auto& access : getResultAccesses(stmt).first) {
    // Any accesses that have index sets will be added.
    if (access.hasIndexSetModes()) {
      for (size_t i = 0; i < access.getIndexVars().size(); i++) {
        if (access.isModeIndexSet(i)) {
          auto t = access.getModeIndexSetTensor(i);
          if (tensorVars.count(t) == 0) {
            ir::Expr irVar = ir::Var::make(t.getName(), t.getType().getDataType(), true, true, pack);
            tensorVars.insert({t, irVar});
            indexSetArgs.push_back(irVar);
          }
        }
      }
    }
  }
  argumentsIR.insert(argumentsIR.begin(), indexSetArgs.begin(), indexSetArgs.end());

  // Create the backwards map from Expr to TensorVar.
  for (auto it : this->tensorVars) {
    this->exprToTensorVar[it.second] = it.first;
  }

  // Figure out whether we are going to generate some GPU code.
  {
    struct GPULoopFinder : public IndexNotationVisitor {
      void visit(const ForallNode* node) {
        switch (node->parallel_unit) {
          case ParallelUnit::GPUBlock:
          case ParallelUnit::GPUWarp:
          case ParallelUnit::GPUThread:
            this->found = true;
            break;
          default:
            break;
        }
        node->stmt.accept(this);
      }
      bool found = false;
    } finder; stmt.accept(&finder);
    this->containsGPULoops = finder.found;
  }

  if (this->legion) {
    auto lookupTV = [&](ir::Expr e) {
      for (auto it : this->tensorVars) {
        if (it.second == e) {
          return it.first;
        }
      }
      taco_ierror << "couldn't reverse lookup tensor: " << e << " in: " << util::join(this->tensorVars) << std::endl;
      return TensorVar();
    };

    for (auto ir : resultsIR) {
      // Don't add scalars to the tensorVarOrdering.
      bool isScalar = false;
      for (auto it : this->scalars) {
        if (it.second == ir) {
          isScalar = true;
          // TODO (rohany): Assert that there is only at most one scalar result.
          this->performingScalarReduction = true;
          this->scalarReductionResult = tensorVars.at(it.first);
        }
      }
      if (!isScalar) {
        this->tensorVarOrdering.push_back(lookupTV(ir));
      }
    }
    for (auto ir : argumentsIR) {
      if (ir.as<Var>() && ir.as<Var>()->is_tensor) {
        this->tensorVarOrdering.push_back(lookupTV(ir));
      }
    }
  }

  // Create variables for temporaries.
  // TODO (rohany): I don't think that we ever use these variables
  //  in the generated code, but some code assumes that we have them
  //  here. Future work can remove this pass.
  for (auto& temp : temporaries) {
    ir::Expr irVar = ir::Var::make(temp.getName(), temp.getType().getDataType(),
                                   true, true);
    tensorVars.insert({temp, irVar});
  }

  // Create variables for keeping track of result values array capacity
  createCapacityVars(resultVars, &capacityVars);

  // Figure out whether we can preserve the non-zero structure
  // of an input tensor for the result tensor. However, to ensure
  // that we don't break normal TACO usage, we'll only do this
  // when Legion lowering is enabled. We also don't want to enable
  // this when we are explicitly assembling an output tensor.
  if (this->legion && !this->assemble) {
    this->preservesNonZeros = preservesNonZeroStructure(stmt, this->nonZeroAnalyzerResult);
  }

  // Create iterators.
  iterators = Iterators(stmt, tensorVars);

  std::vector<ir::Stmt> declareModeVars;
  for (auto tv : this->tensorVarOrdering) {
    for (int i = 0; i < tv.getOrder(); i++) {
      auto it = this->iterators.getLevelIteratorByLevel(tv, i);
      auto stmts = it.declareModeVariables();
      declareModeVars.push_back(stmts);
    }
  }

  provGraph = ProvenanceGraph(stmt);

  // Initialize the indexVarToExprMap.
  for (const IndexVar& indexVar : provGraph.getAllIndexVars()) {
    if (iterators.modeIterators().count(indexVar)) {
      indexVarToExprMap.insert({indexVar, iterators.modeIterators()[indexVar].getIteratorVar()});
    }
    else {
      indexVarToExprMap.insert({indexVar, Var::make(indexVar.getName(), Int())});
    }
  }
  // Also initialize the backwards map.
  for (auto it : this->indexVarToExprMap) {
    this->exprToIndexVarMap[it.second] = it.first;
  }

  vector<Access> inputAccesses, resultAccesses;
  set<Access> reducedAccesses;
  inputAccesses = getArgumentAccesses(stmt);
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(stmt);

  // Create variables that represent the reduced values of duplicated tensor
  // components
  createReducedValueVars(inputAccesses, &reducedValueVars);

  // Initialize the mapping from TensorVars to Accesses.
  if (this->legion) {
    auto addAccess = [&](Access& a) {
      // We don't want to count phantom accesses generated by Assemble().
      if (a.isAccessingStructure()) return;
      taco_iassert(!util::contains(this->tensorVarToAccess, a.getTensorVar()));
      this->tensorVarToAccess.insert({a.getTensorVar(), a});
    };
    for (auto a : inputAccesses) { addAccess(a); }
    for (auto a : resultAccesses) { addAccess(a); }
  }

  // Define and initialize dimension variables
  set<TensorVar> temporariesSet(temporaries.begin(), temporaries.end());
  vector<IndexVar> indexVars = getIndexVars(stmt);
  for (auto& indexVar : indexVars) {
    Expr dimension;
    // getDimension extracts an Expr that holds the dimension
    // of a particular tensor mode. This Expr should be used as a loop bound
    // when iterating over the dimension of the target tensor.
    auto getDimension = [&](const TensorVar& tv, const Access& a, int mode) {
      // If the tensor mode is windowed, then the dimension for iteration is the bounds
      // of the window. Otherwise, it is the actual dimension of the mode.
      if (a.isModeWindowed(mode)) {
        // The mode value used to access .levelIterator is 1-indexed, while
        // the mode input to getDimension is 0-indexed. So, we shift it up by 1.
        auto iter = iterators.getLevelIteratorByModeAccess(ModeAccess(a, mode + 1));
        return ir::Div::make(ir::Sub::make(iter.getWindowUpperBound(), iter.getWindowLowerBound()), iter.getStride());
      } else if (a.isModeIndexSet(mode)) {
        // If the mode has an index set, then the dimension is the size of
        // the index set.
        return ir::Literal::make(a.getIndexSet(mode).size());
      } else {
        return GetProperty::make(tensorVars.at(tv), TensorProperty::Dimension, mode);
      }
    };
    match(stmt,
      function<void(const AssignmentNode*, Matcher*)>([&](
          const AssignmentNode* n, Matcher* m) {
        m->match(n->rhs);
        if (!dimension.defined()) {
          auto ivars = n->lhs.getIndexVars();
          auto tv = n->lhs.getTensorVar();
          int loc = (int)distance(ivars.begin(),
                                  find(ivars.begin(),ivars.end(), indexVar));
          if(!util::contains(temporariesSet, tv)) {
            dimension = getDimension(tv, n->lhs, loc);
          }
        }
      }),
      function<void(const AccessNode*)>([&](const AccessNode* n) {
        auto indexVars = n->indexVars;
        if (util::contains(indexVars, indexVar) && !dimension.defined()) {
          int loc = (int)distance(indexVars.begin(),
                                  find(indexVars.begin(),indexVars.end(),
                                       indexVar));
          if(!util::contains(temporariesSet, n->tensorVar)) {
            dimension = getDimension(n->tensorVar, Access(n), loc);
          }
        }
      })
    );

    // TODO (rohany): Big Hack: If an index var is unbounded (can happen if we're generating
    //  data placement code over a processor grid that is higher dimensional than the tensor
    //  itself), then just substitute a dummy value for the dimension.
    if (!dimension.defined()) {
      dimension = ir::GetProperty::make(this->tensorVars.begin()->second, TensorProperty::Dimension, 0);
    }

    dimensions.insert({indexVar, dimension});
    underivedBounds.insert({indexVar, {ir::Literal::make(0), dimension}});
  }

  // Define and initialize scalar results and arguments
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(!util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        scalars.insert({result, tensorVars.at(result)});
        header.push_back(defineScalarVariable(result, true));
      }
    }
    for (auto& argument : arguments) {
      if (isScalar(argument.getType())) {
        taco_iassert(!util::contains(scalars, argument));
        taco_iassert(util::contains(tensorVars, argument));
        scalars.insert({argument, tensorVars.at(argument)});
        header.push_back(defineScalarVariable(argument, false));
      }
    }
  }

  // Allocate memory for scalar results
  if (generateAssembleCode()) {
    for (auto& result : results) {
      if (result.getOrder() == 0) {
        Expr resultIR = resultVars.at(result);
        Expr vals = GetProperty::make(resultIR, TensorProperty::Values);
        header.push_back(Allocate::make(vals, 1));
      }
    }
  }

  // Analyze each of the accesses to find out the dimensions of the values components of each tensor.
  match(stmt, std::function<void(const AccessNode*)>([&](const AccessNode* node) {
    Access acc(node);
    // Don't consider phantom accesses created by Assemble.
    if (acc.isAccessingStructure()) return;
    this->valuesAnalyzer.addAccess(acc, this->iterators, this->indexVarToExprMap);
  }), std::function<void(const AssignmentNode*)>([&](const AssignmentNode* node) {
    // Don't consider phantom accesses created by Assemble.
    if (node->lhs.isAccessingStructure()) return;
    this->valuesAnalyzer.addAccess(node->lhs, this->iterators, this->indexVarToExprMap);
  }));

  // If we're computing on a partition, then make variables for the partitions, and add
  // them to the function inputs.
  std::vector<TensorVar> computingOn;
  match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
    if (!node->computingOn.empty()) {
      computingOn = node->computingOn;
    }
  }));
  for (auto var : computingOn) {
    this->computingOnPartition[var] = ir::Var::make(var.getName() + "Partition", LogicalPartition);
    argumentsIR.push_back(this->computingOnPartition[var]);
  }

  // TODO (rohany): Delete this code around top level transfers, as there isn't really
  //  such a concept anymore.
  // If there are distributed loops, and no transfers present for an access, then that
  // transfer is occurring at the top level, so add it here.
  Stmt topLevelTransfers;
  bool foundDistributed = false;
  match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
    foundDistributed |= distributedParallelUnit(node->parallel_unit);
  }));
  if (foundDistributed) {
    // Collect all transfers in the index stmt.
    std::vector<TensorVar> transfers;
    match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
      for (auto& t : node->transfers) {
        transfers.push_back(t.getAccess().getTensorVar());
      }
      for (auto var : node->computingOn) {
        transfers.push_back(var);
      }
    }));

    auto hasTransfer = [&](Access a) {
      for (auto& t : transfers) {
        if (t == a.getTensorVar()) { return true; }
      }
      return false;
    };

    // For all accesses, see if they have transfers.
    std::vector<Access> accessesWithoutTransfers;
    match(stmt, function<void(const AccessNode*)>([&](const AccessNode* node) {
      Access a(node);
      if (!hasTransfer(a)) { accessesWithoutTransfers.push_back(a); }
    }), function<void(const AssignmentNode*)>([&](const AssignmentNode* node) {
      if (!hasTransfer(node->lhs)) { accessesWithoutTransfers.push_back(node->lhs); }
    }));

    // We should stop emitting this fake "top_level_transfer" thing.
    // std::vector<Stmt> stmts;
    // for (auto t : accessesWithoutTransfers) {
    //   auto v = ir::Var::make("tx", Datatype::Int32);
    //   auto tv = ir::Var::make(t.getTensorVar().getName(), Datatype::Int32);
    //   auto fcall = ir::Call::make("top_level_transfer", {tv}, Datatype::Int32);
    //   stmts.push_back(ir::Assign::make(v, fcall));
    // }
    // topLevelTransfers = ir::Block::make(stmts);
  }

  // Allocate and initialize append and insert mode indices
  // TODO (rohany): I don't think that I want this. Or at least, it needs to be changed
  //  to not write out of partitions.
  Stmt initializeResults = initResultArrays(resultAccesses, inputAccesses,
                                            reducedAccesses);
  if (this->legion && this->assemble == false) {
    initializeResults = ir::Block::make();
  }

  // BoundsInferenceVisitor infers the exact bounds (inclusive) that each tensor is accessed on.
  // In particular, the BoundsInferenceVisitor derives coordinate space bounds on each index
  // variable in a tensor's accessed. Separate methods must be used for position space bounds.
  struct BoundsInferenceVisitor : public IndexNotationVisitor {
    BoundsInferenceVisitor(std::map<TensorVar, Expr> &tvs, ProvenanceGraph &pg, Iterators &iterators,
                           std::map<IndexVar, std::vector<ir::Expr>>& underivedBounds, std::map<IndexVar, ir::Expr>& indexVarToExprMap,
                           std::set<IndexVar> presentIvars)
        : pg(pg), iterators(iterators), underivedBounds(underivedBounds), indexVarToExprMap(indexVarToExprMap),
          presentIvars(presentIvars) {
      for (auto &it : tvs) {
        this->inScopeVars[it.first] = {};
      }
      for (auto& it : indexVarToExprMap) {
        exprToIndexVarMap[it.second] = it.first;
      }
    }

    void inferBounds(IndexStmt stmt) {
      IndexStmtVisitorStrict::visit(stmt);
    }
    void inferBounds(IndexExpr expr) {
      IndexExprVisitorStrict::visit(expr);
    }

    void visit(const ForallNode* node) {
      if (node == this->trackingForall) {
        this->tracking = true;
      }

      // Add the forall variable to the scope for each tensorVar that hasn't
      // been requested yet.
      for (auto& it : this->inScopeVars) {
        if (!util::contains(this->requestedTensorVars, it.first)) {
          it.second.insert(node->indexVar);
          auto fused = this->pg.getMultiFusedParents(node->indexVar);
          it.second.insert(fused.begin(), fused.end());
        }
      }

      if (this->tracking || (this->trackingForall == nullptr)) {
        for (auto& t : node->transfers) {
          this->requestedTensorVars.insert(t.getAccess().getTensorVar());
        }
        for (auto var : node->computingOn) {
          this->requestedTensorVars.insert(var);
        }
      }

      // Recurse down the index statement.
      this->definedIndexVars.push_back(node->indexVar);
      this->forallDepth++;
      this->inferBounds(node->stmt);
    }

    void visit(const AssignmentNode* node) {
      this->inferBounds(node->lhs);
      this->inferBounds(node->rhs);
    }

    void visit(const AccessNode* node) {
      // For each variable of the access, find its bounds.
      for (auto& var : node->indexVars) {
        auto children = this->pg.getChildren(var);
        // If the index variable has no children, then it is a raw access.
        if (children.size() == 0) {
          // If the index variable is in scope for the request, then we will need to
          // just access that point of the index variable. Otherwise, we will access
          // the full bounds of that variable.
          if (util::contains(this->inScopeVars[node->tensorVar], var)) {
            auto expr = this->indexVarToExprMap[var];
            this->derivedBounds[node->tensorVar].push_back({expr, expr});
          } else {
            this->derivedBounds[node->tensorVar].push_back(this->pg.deriveIterBounds(var, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators));
          }
        } else {
          // If the index variable has children, then we need to recover how it accesses
          // the tensors in the expression based on how those children are made. We first
          // calculate how to recover the index variable.
          auto accessExpr = this->pg.recoverVariable(var, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators);

          // Next, we repeatedly replace variables the recovered expression until it
          // no longer changes. Exactly how the rewriting is done is detailed in the
          // BoundsInferenceExprRewriter.
          auto rwFn = [&](bool lower, ir::Expr bound) {
            BoundsInferenceExprRewriter rw(this->pg, this->iterators, this->underivedBounds, this->indexVarToExprMap,
                                           this->inScopeVars[node->tensorVar], this->exprToIndexVarMap,
                                           this->definedIndexVars, lower, this->presentIvars);
            do {
              rw.changed = false;
              bound = rw.rewrite(bound);
            } while(rw.changed);
            return bound;
          };
          auto lo = ir::simplify(rwFn(true, accessExpr));
          auto hi = ir::simplify(rwFn(false, accessExpr));
          this->derivedBounds[node->tensorVar].push_back({lo, hi});
        }
      }
    }

    ProvenanceGraph& pg;
    Iterators& iterators;
    std::map<IndexVar, std::vector<ir::Expr>>& underivedBounds;
    std::map<IndexVar, ir::Expr>& indexVarToExprMap;

    std::map<ir::Expr, IndexVar> exprToIndexVarMap;

    std::vector<IndexVar> definedIndexVars;
    std::map<TensorVar, std::set<IndexVar>> inScopeVars;
    std::set<TensorVar> requestedTensorVars;

    std::set<IndexVar> presentIvars;

    std::map<TensorVar, std::vector<std::vector<ir::Expr>>> derivedBounds;

    const ForallNode* trackingForall = nullptr;
    bool tracking = false;

    int forallDepth = 0;
  };

  match(stmt, function<void(const ForallNode*)>([&](const ForallNode* f) {
    auto fused = this->provGraph.getMultiFusedParents(f->indexVar);
    if (fused.size() > 0) {
      this->presentIvars.insert(fused.begin(), fused.end());
    } else {
      this->presentIvars.insert(f->indexVar);
    }
  }));

  match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
    // Want to derive bounds for each distributed forall. Can worry about how to
    // connect this all together later.
    auto f = Forall(node);
    if (f.isDistributed() || !f.getTransfers().empty()) {
      // Get bounds for this forall.
      BoundsInferenceVisitor bi(this->tensorVars, this->provGraph, this->iterators, this->underivedBounds, this->indexVarToExprMap, this->presentIvars);
      bi.trackingForall = node;
      bi.inferBounds(stmt);
      // std::cout << "Bounds for index var: " << f.getIndexVar() << " at forall: " << f << std::endl;
      // for (auto it : bi.derivedBounds) {
      //   cout << "Bounds for: " << it.first.getName() << endl;
      //   for (auto& bounds : it.second) {
      //     cout << util::join(bounds) << endl;
      //   }
      // }
      this->derivedBounds[f.getIndexVar()] = bi.derivedBounds;
    }
  }));

  // If we're going to COMPUTE_ONLY, create the top level partition pack.
  if (this->legionLoweringKind == COMPUTE_ONLY) {
    this->topLevelPartitionPack = ir::Var::make("partitionPack", this->getTopLevelTensorPartitionPackType(), true /* is_ptr */);
    argumentsIR.push_back(this->topLevelPartitionPack);
  }

  match(stmt, function<void(const PlaceNode*)>([&](const PlaceNode* node) {
    this->isPlacementCode = true;
    this->placements = node->placements;
  }), function<void(const PartitionNode*)>([&](const PartitionNode* node) {
    this->isPartitionCode = true;
  }));

  if (this->isPlacementCode) {
    // Set up Face() index launch bounds restrictions for any placement operations
    // that use Face().
    struct IndexVarFaceCollector : public IndexNotationVisitor {
      IndexVarFaceCollector(std::map<IndexVar, int>& indexVarFaces,
                            std::vector<std::pair<Grid, GridPlacement>>& placements,
                            ProvenanceGraph& pg)
        : indexVarFaces(indexVarFaces), placements(placements), pg(pg) {}

      void visit (const ForallNode* node) {
        if (distributedParallelUnit(node->parallel_unit)) {
          auto fused = this->pg.getMultiFusedParents(node->indexVar);
          if (fused.size() == 0) {
            fused = std::vector<IndexVar>({node->indexVar});
          }
          taco_iassert(fused.size()  > 0);
          auto placement = this->placements[distIndex].second;
          taco_iassert(fused.size() == placement.axes.size());
          // For all positions that are restricted to a Face of the processor grid,
          // override the iteration bounds of that variable to just that face of the
          // grid.
          for (size_t i = 0; i < placement.axes.size(); i++) {
            auto axis = placement.axes[i];
            if (axis.kind == GridPlacement::AxisMatch::Face) {
              this->indexVarFaces[fused[i]] = axis.face;
            }
          }
          distIndex++;
        }
        node->stmt.accept(this);
      }

      int distIndex = 0;
      std::map<IndexVar, int>& indexVarFaces;
      std::vector<std::pair<Grid, GridPlacement>>& placements;
      ProvenanceGraph& pg;
    };
    IndexVarFaceCollector fc(this->indexVarFaces, this->placements, this->provGraph);
    stmt.accept(&fc);
  }

  // Lower the index statement to compute and/or assemble
  Stmt body = lower(stmt);

  // Post-process result modes and allocate memory for values if necessary
  Stmt finalizeResults = finalizeResultArrays(resultAccesses);

  std::vector<ir::Stmt> mallocRhsRegions;
  if (this->legion) {
    // We may need to access some data in a region in the top level
    // task itself if we aren't launching any tasks. This pass adds
    // malloc's every region that has an accessor in the body if the
    // body does not launch any tasks.
    struct AccessorFinder : public IRVisitor {
      void visit(const For *node) {
        this->hasTasks |= node->isTask;
        node->contents.accept(this);
        node->start.accept(this);
        node->end.accept(this);
      }

      void visit(const GetProperty* node) {
        switch (node->property) {
          case TensorProperty::ValuesReductionNonExclusiveAccessor:
          case TensorProperty::ValuesReductionAccessor:
          case TensorProperty::ValuesReadAccessor:
          case TensorProperty::ValuesWriteAccessor:
          case TensorProperty::IndicesAccessor: {
            auto hashable = node->toHashable();
            if (!util::contains(this->gpSet, hashable)) {
              this->gpSet.insert(hashable);
              this->gps.push_back(node);
            }
            break;
          }
          default:
            break;
        }
      }

      bool hasTasks = false;
      std::set<ir::GetProperty::Hashable> gpSet;
      std::vector<const ir::GetProperty*> gps;
    };
    AccessorFinder accessorFinder; body.accept(&accessorFinder);
    if (!accessorFinder.hasTasks && !accessorFinder.gpSet.empty()) {
      // Malloc all RHS regions that are accessed by an accessor.
      for (auto acc : accessorFinder.gps) {
        auto tensorVar = this->exprToTensorVar[acc->tensor];
        if (!util::contains(arguments, tensorVar)) continue;
        if (acc->property == TensorProperty::ValuesReadAccessor) {
          auto tv = acc->tensor;
          auto values = ir::GetProperty::make(tv, TensorProperty::Values);
          auto valuesParent = ir::GetProperty::make(tv, TensorProperty::ValuesParent);
          auto alloc = ir::Call::make("legionMalloc", {ctx, runtime, values, valuesParent, fidVal, readOnly}, Auto);
          mallocRhsRegions.push_back(ir::Assign::make(values, alloc));
          mallocRhsRegions.push_back(ir::Assign::make(acc, ir::makeCreateAccessor(acc, values, fidVal)));
        } else if (acc->property == TensorProperty::IndicesAccessor) {
          auto region = acc->accessorArgs.regionAccessing;
          auto regionParent = acc->accessorArgs.regionParent;
          auto field = acc->accessorArgs.field;
          auto alloc = ir::Call::make("legionMalloc", {ctx, runtime, region, regionParent, field, readOnly}, Auto);
          mallocRhsRegions.push_back(ir::Assign::make(region, alloc));
          // Update the accessor.
          mallocRhsRegions.push_back(ir::Assign::make(acc, ir::makeCreateAccessor(acc, region, field)));
        } else {
          taco_iassert(false);
        }
      }
    }
  }
  // We also need to clean up after ourselves if we malloc or realloc any regions.
  std::vector<ir::Stmt> unmapAllocedRegions;
  if (this->legion) {
    struct legionAllocFinder : public IRVisitor {
      void visit(const Assign* node) {
        auto call = node->rhs.as<Call>();
        auto gp = node->lhs.as<GetProperty>();
        if (call && gp && (call->func == "legionMalloc" || call->func == "legionRealloc")) {
          auto key = std::tuple<std::string,Expr,TensorProperty,int,int>(gp->name, gp->tensor, gp->property, gp->mode, gp->index);
          this->allocations.insert(key);
        }
      }
      void visit(const Allocate* node) {
        if (node->pack.logicalRegion.defined()) {
          auto gp = node->var.as<GetProperty>();
          taco_iassert(gp);
          auto key = std::tuple<std::string,Expr,TensorProperty,int,int>(gp->name, gp->tensor, gp->property, gp->mode, gp->index);
          this->allocations.insert(key);
        }
      }
      // The set is ordered to have the string first so that a consistent ordering of statements
      // on the generated code is present. With the ir::Expr first, the ordering is not guaranteed.
      std::set<std::tuple<std::string, ir::Expr, ir::TensorProperty, int, int>> allocations;
    } allocFinder;

    // We'll visit the loop body as well as the set of statements that we manually malloced.
    if (initializeResults.defined()) {
      initializeResults.accept(&allocFinder);
    }
    body.accept(&allocFinder);
    if (!mallocRhsRegions.empty()) {
      ir::Block::make(mallocRhsRegions).accept(&allocFinder);
    }

    // Now, emit unmap operations for each of the GetProperties that we saw were allocated.
    for (auto key : allocFinder.allocations) {
      auto gp = ir::GetProperty::make(std::get<1>(key), std::get<2>(key), std::get<3>(key), std::get<4>(key), std::get<0>(key));
      unmapAllocedRegions.push_back(ir::SideEffect::make(ir::Call::make("runtime->unmap_region", {ctx, gp}, Auto)));
    }
  }

  // Collect an add any parameter variables to the function's inputs.
  struct ParameterFinder : public IRVisitor {
    void visit(const Var* node) {
      if (node->is_parameter) {
        if (!util::contains(this->collectedVars, node)) {
          vars.push_back(node);
          collectedVars.insert(node);
        }
      }
    }
    std::vector<ir::Expr> vars;
    std::set<ir::Expr> collectedVars;
  } pfinder; body.accept(&pfinder);

  Datatype returnType = Datatype::Undefined;
  // Store scalar stack variables back to results
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        if (this->legion) {
          taco_iassert(util::contains(scalars, result));
          returnType = result.getType().getDataType();
          Expr varValueIR = tensorVars.at(result);
          footer.push_back(ir::Return::make(varValueIR));
        } else {
          taco_iassert(util::contains(scalars, result));
          taco_iassert(util::contains(tensorVars, result));
          Expr resultIR = scalars.at(result);
          Expr varValueIR = tensorVars.at(result);
          Expr valuesArrIR = GetProperty::make(resultIR, TensorProperty::Values);
          footer.push_back(Store::make(valuesArrIR, 0, varValueIR, markAssignsAtomicDepth > 0, atomicParallelUnit));
        }
      }
    }
  }

  // If the desired lowering target is to create partitions only, then change
  // the return type to be a vector of LogicalPartitions as the result.
  if (this->legionLoweringKind == PARTITION_ONLY) {
    taco_iassert(returnType.getKind() == Datatype::Undefined);
    returnType = this->getTopLevelTensorPartitionPackType();
  } else if ((this->isPartitionCode || this->isPlacementCode) && this->legionLoweringKind != COMPUTE_ONLY) {
    // The result for partition and placement codes is a LogicalPartition.
    taco_iassert(returnType.getKind() == Datatype::Undefined);
    returnType = Datatype("LogicalPartition");
  }

  if (this->legion) {
    // Remove any scalars from the results IR for legion.
    std::vector<ir::Expr> newResultsIR;
    for (auto e : resultsIR) {
      auto isScalar = false;
      for (auto it : this->scalars) {
        if (it.second == e) {
          isScalar = true;
        }
      }
      if (!isScalar) {
        newResultsIR.push_back(e);
      }
    }
    resultsIR = newResultsIR;
  }

  // Create function
  return Function::make(name, resultsIR, util::combine(argumentsIR, pfinder.vars),
                        Block::blanks(
                                      // TODO (rohany): Does this need to go before or after the header?
                                      Block::make(mallocRhsRegions),
                                      Block::make(header),
                                      Block::make(this->taskHeader),
                                      ir::Block::make(declareModeVars),
                                      initializeResults,
                                      topLevelTransfers,
                                      body,
                                      finalizeResults,
                                      ir::Block::make(unmapAllocedRegions),
                                      Block::make(footer)), returnType);
}


Stmt LowererImpl::lowerAssignment(Assignment assignment)
{
  taco_iassert(generateAssembleCode() || generateComputeCode());

  Stmt computeStmt;
  TensorVar result = assignment.getLhs().getTensorVar();
  Expr var = getTensorVar(result);

  const bool needComputeAssign = util::contains(needCompute, result);

  Expr rhs;
  if (needComputeAssign) {
    rhs = lower(assignment.getRhs());
  }

  // Assignment to scalar variables.
  if (isScalar(result.getType())) {
    if (needComputeAssign) {
      if (!assignment.getOperator().defined()) {
        computeStmt = Assign::make(var, rhs);
      }
      else {
        taco_iassert(isa<taco::Add>(assignment.getOperator()));
        bool useAtomics = markAssignsAtomicDepth > 0 &&
                          !util::contains(whereTemps, result);
        // TODO (rohany): Might have to do a reduction assignment here?
        computeStmt = compoundAssign(var, rhs, useAtomics, atomicParallelUnit);
      }
    }
  }
  // Assignments to tensor variables (non-scalar).
  else {
    // We ask for non-exclusive accessor if we are supposed to do an atomic
    // assignment.
    Expr values = getValuesArray(result, !(markAssignsAtomicDepth > 0));
    Expr loc = generateValueLocExpr(assignment.getLhs());

    std::vector<Stmt> accessStmts;

    if (isAssembledByUngroupedInsertion(result)) {
      std::vector<Expr> coords;
      Expr prevPos = 0;
      size_t i = 0;
      const auto resultIterators = getIterators(assignment.getLhs());
      for (const auto& it : resultIterators) {
        // TODO: Should only assemble levels that can be assembled together
        //if (it == this->nextTopResultIterator) {
        //  break;
        //}

        coords.push_back(getCoordinateVar(it));

        const auto yieldPos = it.getYieldPos(prevPos, coords);
        accessStmts.push_back(yieldPos.compute());
        Expr pos = it.getPosVar();
        accessStmts.push_back(VarDecl::make(pos, yieldPos[0]));

        if (generateAssembleCode()) {
          accessStmts.push_back(it.getInsertCoord(prevPos, pos, coords));
        }

        prevPos = pos;
        ++i;
      }
    }

    if (needComputeAssign && values.defined()) {
      if (!assignment.getOperator().defined()) {
        computeStmt = Store::make(values, loc, rhs);
      }
      else {
        computeStmt = compoundStore(values, loc, rhs,
                                    markAssignsAtomicDepth > 0,
                                    atomicParallelUnit);
      }
      taco_iassert(computeStmt.defined());
    }

    if (!accessStmts.empty()) {
      accessStmts.push_back(computeStmt);
      computeStmt = Block::make(accessStmts);
    }
  }

  if (util::contains(guardedTemps, result) && result.getOrder() == 0) {
    Expr guard = tempToBitGuard[result];
    Stmt setGuard = Assign::make(guard, true, markAssignsAtomicDepth > 0,
                                 atomicParallelUnit);
    computeStmt = Block::make(computeStmt, setGuard);
  }

  Expr assembleGuard = generateAssembleGuard(assignment.getRhs());
  const bool assembleGuardTrivial = isa<ir::Literal>(assembleGuard);

  // TODO: If only assembling so defer allocating value memory to the end when
  //       we'll know exactly how much we need.
  bool temporaryWithSparseAcceleration = util::contains(tempToIndexList, result);
  if (generateComputeCode() && !temporaryWithSparseAcceleration) {
    taco_iassert(computeStmt.defined());
    return assembleGuardTrivial ? computeStmt : IfThenElse::make(assembleGuard,
                                                                 computeStmt);
  }

  if (temporaryWithSparseAcceleration) {
    taco_iassert(markAssignsAtomicDepth == 0)
      << "Parallel assembly of sparse accelerator not supported";

    Expr values = getValuesArray(result);
    Expr loc = generateValueLocExpr(assignment.getLhs());

    Expr bitGuardArr = tempToBitGuard.at(result);
    Expr indexList = tempToIndexList.at(result);
    Expr indexListSize = tempToIndexListSize.at(result);

    Stmt markBitGuardAsTrue = Store::make(bitGuardArr, loc, true);
    Stmt trackIndex = Store::make(indexList, indexListSize, loc);
    Expr incrementSize = ir::Add::make(indexListSize, 1);
    Stmt incrementStmt = Assign::make(indexListSize, incrementSize);

    Stmt firstWriteAtIndex = Block::make(trackIndex, markBitGuardAsTrue, incrementStmt);
    if (needComputeAssign && values.defined()) {
      Stmt initialStorage = computeStmt;
      if (assignment.getOperator().defined()) {
        // computeStmt is a compund stmt so we need to emit an initial store
        // into the temporary
        initialStorage =  Store::make(values, loc, rhs);
      }
      firstWriteAtIndex = Block::make(initialStorage, firstWriteAtIndex);
    }

    Expr readBitGuard = Load::make(bitGuardArr, loc);
    computeStmt = IfThenElse::make(ir::Neg::make(readBitGuard),
                                   firstWriteAtIndex, computeStmt);
  }

  return assembleGuardTrivial ? computeStmt : IfThenElse::make(assembleGuard,
                                                               computeStmt);
}


  Stmt LowererImpl::lowerYield(Yield yield) {
  std::vector<Expr> coords;
  for (auto& indexVar : yield.getIndexVars()) {
    coords.push_back(getCoordinateVar(indexVar));
  }
  Expr val = lower(yield.getExpr());
  return ir::Yield::make(coords, val);
}


pair<vector<Iterator>, vector<Iterator>>
LowererImpl::splitAppenderAndInserters(const vector<Iterator>& results) {
  vector<Iterator> appenders;
  vector<Iterator> inserters;

  // TODO: Choose insert when the current forall is nested inside a reduction
  for (auto& result : results) {
    if (isAssembledByUngroupedInsertion(result.getTensor())) {
      continue;
    }

    taco_iassert(result.hasAppend() || result.hasInsert())
        << "Results must support append or insert";

    if (result.hasAppend()) {
      appenders.push_back(result);
    } else {
      taco_iassert(result.hasInsert());
      inserters.push_back(result);
    }
  }

  return {appenders, inserters};
}


Stmt LowererImpl::lowerForall(Forall forall)
{
  bool hasExactBound = provGraph.hasExactBound(forall.getIndexVar());
  bool forallNeedsUnderivedGuards = !hasExactBound && emitUnderivedGuards;

  // TODO (rohany): The optimization that this method performs does not seem
  //  to be sound, and gcc is still able to vectorize the code fine.
  // if (!ignoreVectorize && forallNeedsUnderivedGuards &&
  //     (forall.getParallelUnit() == ParallelUnit::CPUVector ||
  //      forall.getUnrollFactor() > 0)) {
  //   return lowerForallCloned(forall);
  // }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel) {
    inParallelLoopDepth++;
  }

  // Record that we might have some fresh locators that need to be recovered.
  std::vector<Iterator> freshLocateIterators;

  // Recover any available parents that were not recoverable previously
  vector<Stmt> recoverySteps;
  for (const IndexVar& varToRecover : provGraph.newlyRecoverableParents(forall.getIndexVar(), definedIndexVars)) {
    // place pos guard
    if (forallNeedsUnderivedGuards && provGraph.isCoordVariable(varToRecover) &&
        provGraph.getChildren(varToRecover).size() == 1 &&
        provGraph.isPosVariable(provGraph.getChildren(varToRecover)[0])) {
      IndexVar posVar = provGraph.getChildren(varToRecover)[0];
      std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(posVar, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

      Expr minGuard = Lt::make(indexVarToExprMap[posVar], iterBounds[0]);
      Expr maxGuard = Gte::make(indexVarToExprMap[posVar], iterBounds[1]);
      Expr guardCondition = Or::make(minGuard, maxGuard);
      if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
        guardCondition = maxGuard;
      }
      ir::Stmt guard = ir::IfThenElse::make(guardCondition, ir::Continue::make());
      recoverySteps.push_back(guard);
    }

    Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    taco_iassert(indexVarToExprMap.count(varToRecover));
    recoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));

    // After we've recovered this index variable, some iterators are now
    // accessible for use when declaring locator access variables. So, generate
    // the accessors for those locator variables as part of the recovery process.
    // This is necessary after a fuse transformation, for example: If we fuse
    // two index variables (i, j) into f, then after we've generated the loop for
    // f, all locate accessors for i and j are now available for use. So, remember
    // that we have some new locate iterators that should be recovered.
    for (auto& iters : iterators.levelIterators()) {
      if (iters.second.getIndexVar() == varToRecover && iters.second.hasLocate()) {
        freshLocateIterators.push_back(iters.second);
      }
    }

    // place underived guard
    std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    if (forallNeedsUnderivedGuards && underivedBounds.count(varToRecover) &&
        !provGraph.hasPosDescendant(varToRecover)) {

      // FIXME: [Olivia] Check this with someone
      // Removed underived guard if indexVar is bounded is divisible by its split child indexVar
      vector<IndexVar> children = provGraph.getChildren(varToRecover);
      bool hasDirectDivBound = false;
      std::vector<ir::Expr> iterBoundsInner = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

        for (auto& c: children) {
          if (provGraph.hasExactBound(c) && provGraph.derivationPath(varToRecover, c).size() == 2) {
              std::vector<ir::Expr> iterBoundsUnderivedChild = provGraph.deriveIterBounds(c, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
              if (iterBoundsUnderivedChild[1].as<ir::Literal>()->getValue<int>() % iterBoundsInner[1].as<ir::Literal>()->getValue<int>() == 0)
              hasDirectDivBound = true;
              break;
          }
      }
      if (!hasDirectDivBound) {
          Stmt guard = IfThenElse::make(Gte::make(indexVarToExprMap[varToRecover], underivedBounds[varToRecover][1]),
                                        Continue::make());
          recoverySteps.push_back(guard);
      }
    }

    // TODO (rohany): Is there a way to pull this check into the loop guard?
    // If this index variable was divided into multiple equal chunks, then we
    // must add an extra guard to make sure that further scheduling operations
    // on descendent index variables exceed the bounds of each equal portion of
    // the loop. For a concrete example, consider a loop of size 10 that is divided
    // into two equal components -- 5 and 5. If the loop is then transformed
    // with .split(..., 3), each inner chunk of 5 will be split into chunks of
    // 3. Without an extra guard, the second chunk of 3 in the first group of 5
    // may attempt to perform an iteration for the second group of 5, which is
    // incorrect.
    if (this->provGraph.isDivided(varToRecover)) {
      // Collect the children iteration variables.
      auto children = this->provGraph.getChildren(varToRecover);
      auto outer = children[0];
      auto inner = children[1];
      // Find the iteration bounds of the inner variable -- that is the size
      // that the outer loop was broken into.
      auto bounds = this->provGraph.deriveIterBounds(inner, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
      auto parentBounds = this->provGraph.deriveIterBounds(varToRecover, this->definedIndexVarsOrdered, this->underivedBounds, this->indexVarToExprMap, this->iterators);
      // Use the difference between the bounds to find the size of the loop.
      auto dimLen = ir::Sub::make(bounds[1], bounds[0]);
      // For a variable f divided into into f1 and f2, the guard ensures that
      // for iteration f, f should be within f1 * dimLen and (f1 + 1) * dimLen.
      auto upperBound = ir::simplify(ir::Add::make(ir::Mul::make(ir::Add::make(this->indexVarToExprMap[outer], 1), dimLen), parentBounds[0]));
      auto guard = ir::Gte::make(this->indexVarToExprMap[varToRecover], upperBound);
      recoverySteps.push_back(IfThenElse::make(guard, ir::Continue::make()));
    }
    // If this is divided onto a partition, ensure that we don't go past the
    // initial bounds of the partition.
    auto dividedOntoPartition = this->provGraph.isDividedOntoPartition(varToRecover);
    if (dividedOntoPartition.first) {
      // Collect the children iteration variables.
      auto bound = dividedOntoPartition.second;
      // The variable shouldn't go past the upper bounds of the partition.
      auto guard = ir::Gt::make(this->indexVarToExprMap[varToRecover], bound);
      recoverySteps.push_back(IfThenElse::make(guard, ir::Continue::make()));
    }
  }
  Stmt recoveryStmt = Block::make(recoverySteps);

  taco_iassert(!definedIndexVars.count(forall.getIndexVar()));
  definedIndexVars.insert(forall.getIndexVar());
  definedIndexVarsOrdered.push_back(forall.getIndexVar());

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && !distributedParallelUnit(forall.getParallelUnit())) {
    taco_iassert(!parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(!parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars[forall.getParallelUnit()] = forall.getIndexVar();
    vector<Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    parallelUnitSizes[forall.getParallelUnit()] = ir::Sub::make(bounds[1], bounds[0]);
  }

  MergeLattice lattice = MergeLattice::make(forall, iterators, provGraph, definedIndexVars, whereTempsToResult);
  vector<Access> resultAccesses;
  set<Access> reducedAccesses;
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(forall);

  // Pre-allocate/initialize memory of value arrays that are full below this
  // loops index variable
  Stmt preInitValues = initResultArrays(forall.getIndexVar(), resultAccesses,
                                        getArgumentAccesses(forall),
                                        reducedAccesses);

  // Emit temporary initialization if forall is sequential and leads to a where statement
  vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
  auto temp = temporaryInitialization.find(forall);
  if (temp != temporaryInitialization.end() && forall.getParallelUnit() == ParallelUnit::NotParallel && !isScalar(temp->second.getTemporary().getType()))
    temporaryValuesInitFree = codeToInitializeTemporary(temp->second);

  Stmt loops;
  // Emit a loop that iterates over over a single iterator (optimization)
  if (lattice.iterators().size() == 1 && lattice.iterators()[0].isUnique()) {
    taco_iassert(lattice.points().size() == 1);

    MergePoint point = lattice.points()[0];
    Iterator iterator = lattice.iterators()[0];

    vector<Iterator> locators = point.locators();
    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(point.results());

    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(iterator.getIndexVar());
    IndexVar posDescendant;
    bool hasPosDescendant = false;
    if (!underivedAncestors.empty()) {
      hasPosDescendant = provGraph.getPosIteratorFullyDerivedDescendant(underivedAncestors[0], &posDescendant);
    }

    bool isWhereProducer = false;
    vector<Iterator> results = point.results();
    for (Iterator result : results) {
      for (auto it = tensorVars.begin(); it != tensorVars.end(); it++) {
        if (it->second == result.getTensor()) {
          if (whereTempsToResult.count(it->first)) {
            isWhereProducer = true;
            break;
          }
        }
      }
    }

    // For now, this only works when consuming a single workspace.
    //bool canAccelWithSparseIteration = inParallelLoopDepth == 0 && provGraph.isFullyDerived(iterator.getIndexVar()) &&
    //                                   iterator.isDimensionIterator() && locators.size() == 1;
    bool canAccelWithSparseIteration =
        provGraph.isFullyDerived(iterator.getIndexVar()) &&
        iterator.isDimensionIterator() && locators.size() == 1;
    if (canAccelWithSparseIteration) {
      bool indexListsExist = false;
      // We are iterating over a dimension and locating into a temporary with a tracker to keep indices. Instead, we
      // can just iterate over the indices and locate into the dense workspace.
      for (auto it = tensorVars.begin(); it != tensorVars.end(); ++it) {
        if (it->second == locators[0].getTensor() && util::contains(tempToIndexList, it->first)) {
          indexListsExist = true;
          break;
        }
      }
      canAccelWithSparseIteration &= indexListsExist;
    }

    if (!isWhereProducer && hasPosDescendant && underivedAncestors.size() > 1 && provGraph.isPosVariable(iterator.getIndexVar()) && posDescendant == forall.getIndexVar()) {
      loops = lowerForallFusedPosition(forall, iterator, locators,
                                         inserters, appenders, reducedAccesses, recoveryStmt);
    }
    else if (canAccelWithSparseIteration) {
      loops = lowerForallDenseAcceleration(forall, locators, inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit dimension coordinate iteration loop
    else if (iterator.isDimensionIterator()) {
      // A proper fix to #355. Adding information that those locate iterators are now ready is the
      // correct way to recover them, rather than blindly duplicating the emitted locators.
      auto locatorsCopy = std::vector<Iterator>(point.locators());
      for (auto it : freshLocateIterators) {
        if (!util::contains(locatorsCopy, it)) {
          locatorsCopy.push_back(it);
        }
      }
      loops = lowerForallDimension(forall, locatorsCopy,
                                   inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit position iteration loop
    else if (iterator.hasPosIter()) {
      loops = lowerForallPosition(forall, iterator, locators,
                                    inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit coordinate iteration loop
    else {
      taco_iassert(iterator.hasCoordIter());
//      taco_not_supported_yet
      loops = Stmt();
    }
  }
  // Emit general loops to merge multiple iterators
  else {
    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
    taco_iassert(underivedAncestors.size() == 1); // TODO: add support for fused coordinate of pos loop
    loops = lowerMergeLattice(lattice, underivedAncestors[0],
                              forall.getStmt(), reducedAccesses);
  }
//  taco_iassert(loops.defined());

  if (!generateComputeCode() && !hasStores(loops)) {
    // If assembly loop does not modify output arrays, then it can be safely
    // omitted.
    loops = Stmt();
  }
  definedIndexVars.erase(forall.getIndexVar());
  definedIndexVarsOrdered.pop_back();
  if (forall.getParallelUnit() != ParallelUnit::NotParallel && !distributedParallelUnit(forall.getParallelUnit())) {
    inParallelLoopDepth--;
    taco_iassert(parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars.erase(forall.getParallelUnit());
    parallelUnitSizes.erase(forall.getParallelUnit());
  }
  return Block::blanks(preInitValues,
                       temporaryValuesInitFree[0],
                       loops,
                       temporaryValuesInitFree[1]);
}

Stmt LowererImpl::lowerForallCloned(Forall forall) {
  // want to emit guards outside of loop to prevent unstructured loop exits

  // construct guard
  // underived or pos variables that have a descendant that has not been defined yet
  vector<IndexVar> varsWithGuard;
  for (auto var : provGraph.getAllIndexVars()) {
    if (provGraph.isRecoverable(var, definedIndexVars)) {
      continue; // already recovered
    }
    if (provGraph.isUnderived(var) && !provGraph.hasPosDescendant(var)) { // if there is pos descendant then will be guarded already
      varsWithGuard.push_back(var);
    }
    else if (provGraph.isPosVariable(var)) {
      // if parent is coord then this is variable that will be guarded when indexing into coord array
      if(provGraph.getParents(var).size() == 1 && provGraph.isCoordVariable(provGraph.getParents(var)[0])) {
        varsWithGuard.push_back(var);
      }
    }
  }

  // determine min and max values for vars given already defined variables.
  // we do a recovery where we fill in undefined variables with either 0's or the max of their iteration
  std::map<IndexVar, Expr> minVarValues;
  std::map<IndexVar, Expr> maxVarValues;
  set<IndexVar> definedForGuard = definedIndexVars;
  vector<Stmt> guardRecoverySteps;
  Expr maxOffset = 0;
  bool setMaxOffset = false;

  for (auto var : varsWithGuard) {
    std::vector<IndexVar> currentDefinedVarOrder = definedIndexVarsOrdered; // TODO: get defined vars at time of this recovery

    std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
    std::map<IndexVar, Expr> maxChildValues = indexVarToExprMap;

    for (auto child : provGraph.getFullyDerivedDescendants(var)) {
      if (!definedIndexVars.count(child)) {
        std::vector<ir::Expr> childBounds = provGraph.deriveIterBounds(child, currentDefinedVarOrder, underivedBounds, indexVarToExprMap, iterators);

        minChildValues[child] = childBounds[0];
        maxChildValues[child] = childBounds[1];

        // recover new parents
        for (const IndexVar& varToRecover : provGraph.newlyRecoverableParents(child, definedForGuard)) {
          Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                          minChildValues, iterators);
          Expr maxRecoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                             maxChildValues, iterators);
          if (!setMaxOffset) { // TODO: work on simplifying this
            maxOffset = ir::Add::make(maxOffset, ir::Sub::make(maxRecoveredValue, recoveredValue));
            setMaxOffset = true;
          }
          taco_iassert(indexVarToExprMap.count(varToRecover));

          guardRecoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
          definedForGuard.insert(varToRecover);
        }
        definedForGuard.insert(child);
      }
    }

    minVarValues[var] = provGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, minChildValues, iterators);
    maxVarValues[var] = provGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, maxChildValues, iterators);
  }

  // Build guards
  Expr guardCondition;
  for (auto var : varsWithGuard) {
    std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

    Expr minGuard = Lt::make(minVarValues[var], iterBounds[0]);
    Expr maxGuard = Gte::make(ir::Add::make(maxVarValues[var], ir::simplify(maxOffset)), iterBounds[1]);
    Expr guardConditionCurrent = Or::make(minGuard, maxGuard);

    if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
      guardConditionCurrent = maxGuard;
    }

    if (guardCondition.defined()) {
      guardCondition = Or::make(guardConditionCurrent, guardCondition);
    }
    else {
      guardCondition = guardConditionCurrent;
    }
  }

  Stmt unvectorizedLoop;

  taco_uassert(guardCondition.defined())
    << "Unable to vectorize or unroll loop over unbound variable " << forall.getIndexVar();

  // build loop with guards (not vectorized)
  if (!varsWithGuard.empty()) {
    ignoreVectorize = true;
    unvectorizedLoop = lowerForall(forall);
    ignoreVectorize = false;
  }

  // build loop without guards
  emitUnderivedGuards = false;
  Stmt vectorizedLoop = lowerForall(forall);
  emitUnderivedGuards = true;

  // return guarded loops
  return Block::make(Block::make(guardRecoverySteps), IfThenElse::make(guardCondition, unvectorizedLoop, vectorizedLoop));
}

Stmt LowererImpl::searchForFusedPositionStart(Forall forall, Iterator posIterator) {
  vector<Stmt> searchForUnderivedStart;
  vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
  ir::Expr last_block_start_temporary;
  for (int i = (int) underivedAncestors.size() - 2; i >= 0; i--) {
    Iterator posIteratorLevel = posIterator;
    for (int j = (int) underivedAncestors.size() - 2; j > i; j--) { // take parent of iterator enough times to get correct level
      posIteratorLevel = posIteratorLevel.getParent();
    }

    // TODO (rohany): If we're legion, we could do this by just looking up the size of the posIterator, rather
    //  than doing this loop.
    // Get the size of the pos array by walking down the iterator chain and building up the size.
    ir::Expr parentSize = 1;
    // First, walk up to the top to get the root.
    Iterator rootIterator = posIterator;
    while (!rootIterator.isRoot()) {
      rootIterator = rootIterator.getParent();
    }
    while (rootIterator.getChild() != posIteratorLevel) {
      rootIterator = rootIterator.getChild();
      if (rootIterator.hasAppend()) {
        parentSize = rootIterator.getSize(parentSize);
      } else if (rootIterator.hasInsert()) {
        parentSize = ir::Mul::make(parentSize, rootIterator.getWidth());
      }
    }

    // We're going to use this in multiple places, so record if our level
    // is indeed iteration over a Legion Sparse Format.
    std::shared_ptr<const RectCompressedModeFormat> legionSparseLevel = nullptr;
    if (posIteratorLevel.getMode().getModeFormat().is<RectCompressedModeFormat>()) {
      legionSparseLevel = posIteratorLevel.getMode().getModeFormat().as<RectCompressedModeFormat>();
    }

    // If we're operating on a sparse level, then we're probably operating
    // on a partition of that level as well. Extract the domain of the position
    // array so that we constrain the binary search to within that range.
    auto posReg = legionSparseLevel->getRegion(posIteratorLevel.getMode().getModePack(), RectCompressedModeFormat::POS);
    auto crdReg = legionSparseLevel->getRegion(posIteratorLevel.getMode().getModePack(), RectCompressedModeFormat::CRD);
    auto domVar = ir::Var::make(posIteratorLevel.getMode().getName() + "PosDomain", Domain(1));
    auto crdDomVar = ir::Var::make(posIteratorLevel.getMode().getName() + "CrdDomain", Domain(1));
    auto getDom = ir::Call::make("runtime->get_index_space_domain", {ctx, getIndexSpace(posReg)}, Auto);
    auto getCrdDom = ir::Call::make("runtime->get_index_space_domain", {ctx, getIndexSpace(crdReg)}, Auto);
    if (legionSparseLevel) {
      this->taskHeader.push_back(ir::VarDecl::make(domVar, getDom));
      this->taskHeader.push_back(ir::VarDecl::make(crdDomVar, getCrdDom));
    }

    // emit bounds search on cpu just bounds, on gpu search in blocks
    if (parallelUnitIndexVars.count(ParallelUnit::GPUBlock)) {
      Expr values_per_block;
      {
        // we do a recovery where we fill in undefined variables with 0's to get start target (just like for vector guards)
        std::map<IndexVar, Expr> zeroedChildValues = indexVarToExprMap;
        zeroedChildValues[parallelUnitIndexVars[ParallelUnit::GPUBlock]] = 1;
        set<IndexVar> zeroDefinedIndexVars = {parallelUnitIndexVars[ParallelUnit::GPUBlock]};
        for (IndexVar child : provGraph.getFullyDerivedDescendants(posIterator.getIndexVar())) {
          if (child != parallelUnitIndexVars[ParallelUnit::GPUBlock]) {
            zeroedChildValues[child] = 0;

            // recover new parents
            for (const IndexVar &varToRecover : provGraph.newlyRecoverableParents(child, zeroDefinedIndexVars)) {
              Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                              zeroedChildValues, iterators);
              taco_iassert(indexVarToExprMap.count(varToRecover));
              zeroedChildValues[varToRecover] = recoveredValue;
              zeroDefinedIndexVars.insert(varToRecover);
              if (varToRecover == posIterator.getIndexVar()) {
                break;
              }
            }
            zeroDefinedIndexVars.insert(child);
          }
        }
        values_per_block = zeroedChildValues[posIterator.getIndexVar()];
      }

      IndexVar underived = underivedAncestors[i];
      ir::Expr blockStarts_temporary = ir::Var::make(underived.getName() + "_blockStarts",
                                                     getCoordinateVar(underived).type(), true, false);
      // If we're lowering to Legion, then we'll use a DeferredBuffer for the allocation.
      if (this->legion) {
        // Make the DeferredBuffer backing the blockStarts_temporary pointer.
        auto bufTy = DeferredBuffer(blockStarts_temporary.type(), 1);
        auto rectTy = Rect(1);
        auto rect = ir::makeConstructor(rectTy, {0, parallelUnitSizes[ParallelUnit::GPUBlock]});
        auto buf = ir::Var::make("buf", bufTy);
        this->taskHeader.push_back(ir::VarDecl::make(buf, makeConstructor(bufTy, {rect, GPUFBMem})));
        // Now that the deferred buffer is constructed, extract the pointer out.
        this->taskHeader.push_back(ir::VarDecl::make(blockStarts_temporary, ir::MethodCall::make(buf, "ptr", {0}, false /* deref */, Auto)));
      } else {
        this->taskHeader.push_back(ir::VarDecl::make(blockStarts_temporary, 0));
        this->taskHeader.push_back(
            Allocate::make(blockStarts_temporary, ir::Add::make(parallelUnitSizes[ParallelUnit::GPUBlock], 1)));
        if (!this->legion) {
          footer.push_back(Free::make(blockStarts_temporary));
        }
      }

      Expr blockSize;
      if (parallelUnitSizes.count(ParallelUnit::GPUThread)) {
        blockSize = parallelUnitSizes[ParallelUnit::GPUThread];
        if (parallelUnitSizes.count(ParallelUnit::GPUWarp)) {
          blockSize = ir::Mul::make(blockSize, parallelUnitSizes[ParallelUnit::GPUWarp]);
        }
      } else {
        std::vector<IndexVar> definedIndexVarsMatched = definedIndexVarsOrdered;
        // find sub forall that tells us block size
        match(forall.getStmt(),
              function<void(const ForallNode *, Matcher *)>([&](
                      const ForallNode *n, Matcher *m) {
                if (n->parallel_unit == ParallelUnit::GPUThread) {
                  vector<Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsMatched,
                                                                   underivedBounds, indexVarToExprMap, iterators);
                  blockSize = ir::Sub::make(bounds[1], bounds[0]);
                }
                definedIndexVarsMatched.push_back(n->indexVar);
              })
        );
      }
      taco_iassert(blockSize.defined());

      if (i == (int) underivedAncestors.size() - 2) {
        std::vector<Expr> args;
        if (legionSparseLevel) {
          auto acc = legionSparseLevel->getAccessor(posIteratorLevel.getMode().getModePack(), RectCompressedModeFormat::POS);
          args = {
              acc, // array
              blockStarts_temporary, // results
              ir::FieldAccess::make(domVar, "bounds.lo", false /* deref */, Int()), // Search start.
              ir::FieldAccess::make(domVar, "bounds.hi", false /* deref */, Int()), // Search end.
              values_per_block, // values_per_block
              blockSize, // block_size
              parallelUnitSizes[ParallelUnit::GPUBlock], // num_blocks
              // We need this offset here because whatever partition of the crd region we are operating
              // on dictates what positions within the pos region we are actually looking for. If we don't
              // offset the positions we're searching for, then we're going to end up with wildly inaccurate
              // estimates, where all of the threads are are starting at the bottom of the positions array.
              ir::FieldAccess::make(crdDomVar, "bounds.lo", false /* deref */, Int()), // offset
          };
        } else {
          args = {
              posIteratorLevel.getMode().getModePack().getArray(0), // array
              blockStarts_temporary, // results
              ir::Literal::zero(posIteratorLevel.getBeginVar().type()), // arrayStart
              parentSize, // arrayEnd
              values_per_block, // values_per_block
              blockSize, // block_size
              parallelUnitSizes[ParallelUnit::GPUBlock] // num_blocks
          };
        }
        this->taskHeader.push_back(ir::SideEffect::make(ir::Call::make("taco_binarySearchBeforeBlockLaunch", args,
                                                                       getCoordinateVar(underived).type())));
      }
      else {
        taco_iassert(false) << "Unimplemented. Follow the methodology above to fix this case.";
        std::vector<Expr> args = {
                posIteratorLevel.getMode().getModePack().getArray(0), // array
                blockStarts_temporary, // results
                ir::Literal::zero(posIteratorLevel.getBeginVar().type()), // arrayStart
                parentSize, // arrayEnd
                last_block_start_temporary, // targets
                blockSize, // block_size
                parallelUnitSizes[ParallelUnit::GPUBlock] // num_blocks
        };
        this->taskHeader.push_back(
            ir::SideEffect::make(ir::Call::make("taco_binarySearchIndirectBeforeBlockLaunch", args,
                                                getCoordinateVar(underived).type())));
      }
      searchForUnderivedStart.push_back(VarDecl::make(posIteratorLevel.getBeginVar(),
                                                      ir::Load::make(blockStarts_temporary,
                                                                     indexVarToExprMap[parallelUnitIndexVars[ParallelUnit::GPUBlock]])));
      searchForUnderivedStart.push_back(VarDecl::make(posIteratorLevel.getEndVar(),
                                                      ir::Load::make(blockStarts_temporary, ir::Add::make(
                                                              indexVarToExprMap[parallelUnitIndexVars[ParallelUnit::GPUBlock]],
                                                              1))));
      last_block_start_temporary = blockStarts_temporary;
    } else {
      // If the posIterator is a Legion level, then use the bounds on the partition as the start
      // and end of the posIteration.
      if (legionSparseLevel) {
        this->taskHeader.push_back(VarDecl::make(posIteratorLevel.getBeginVar(), ir::FieldAccess::make(domVar, "bounds.lo", false /* deref */, Int())));
        this->taskHeader.push_back(VarDecl::make(posIteratorLevel.getEndVar(), ir::FieldAccess::make(domVar, "bounds.hi", false /* deref */, Int())));
      } else {
        this->taskHeader.push_back(VarDecl::make(posIteratorLevel.getBeginVar(), ir::Literal::zero(posIteratorLevel.getBeginVar().type())));
        this->taskHeader.push_back(VarDecl::make(posIteratorLevel.getEndVar(), parentSize));
      }
    }

    // We do a recovery where we fill in undefined variables with 0's to get start target (just like for vector guards).
    // This process ends up duplicating some variable declarations (but not in a mis-compilation way, it just looks
    // a little suspicious.
    // TODO (rohany): Change these declarations to use fresh variables?
    Expr underivedStartTarget;
    if (i == (int) underivedAncestors.size() - 2) {
      std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
      set<IndexVar> minDefinedIndexVars = definedIndexVars;
      minDefinedIndexVars.erase(forall.getIndexVar());

      for (IndexVar child : provGraph.getFullyDerivedDescendants(posIterator.getIndexVar())) {
        if (!minDefinedIndexVars.count(child)) {
          std::vector<ir::Expr> childBounds = provGraph.deriveIterBounds(child, definedIndexVarsOrdered,
                                                                         underivedBounds,
                                                                         indexVarToExprMap, iterators);
          minChildValues[child] = childBounds[0];

          // recover new parents
          for (const IndexVar &varToRecover : provGraph.newlyRecoverableParents(child, minDefinedIndexVars)) {
            Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                            minChildValues, iterators);
            taco_iassert(indexVarToExprMap.count(varToRecover));
            searchForUnderivedStart.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
            minDefinedIndexVars.insert(varToRecover);
            if (varToRecover == posIterator.getIndexVar()) {
              break;
            }
          }
          minDefinedIndexVars.insert(child);
        }
      }
      underivedStartTarget = indexVarToExprMap[posIterator.getIndexVar()];
    }
    else {
      underivedStartTarget = this->iterators.modeIterator(underivedAncestors[i+1]).getPosVar();
    }

    std::vector<Expr> binarySearchArgs;
    if (legionSparseLevel) {
      auto acc = legionSparseLevel->getAccessor(posIteratorLevel.getMode().getModePack(), RectCompressedModeFormat::POS);
      binarySearchArgs = {
          acc, // array
          posIteratorLevel.getBeginVar(), // arrayStart
          posIteratorLevel.getEndVar(), // arrayEnd
          underivedStartTarget // target
      };
    } else {
      binarySearchArgs = {
          posIteratorLevel.getMode().getModePack().getArray(0), // array
          posIteratorLevel.getBeginVar(), // arrayStart
          posIteratorLevel.getEndVar(), // arrayEnd
          underivedStartTarget // target
      };
    }
    Expr posVarUnknown = this->iterators.modeIterator(underivedAncestors[i]).getPosVar();
    searchForUnderivedStart.push_back(ir::VarDecl::make(posVarUnknown,
                                                        ir::Call::make("taco_binarySearchBefore", binarySearchArgs,
                                                                       getCoordinateVar(underivedAncestors[i]).type())));
    Stmt locateCoordVar;
    if (posIteratorLevel.getParent().hasPosIter()) {
      auto posIteratorParent = posIteratorLevel.getParent();
      if (posIteratorParent.getMode().getModeFormat().is<RectCompressedModeFormat>()) {
        auto rcmf = posIteratorParent.getMode().getModeFormat().as<RectCompressedModeFormat>();
        auto crdAcc = rcmf->getAccessor(posIteratorParent.getMode().getModePack(), RectCompressedModeFormat::CRD);
        locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], ir::Load::make(crdAcc, posVarUnknown));
      } else {
        locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], ir::Load::make(posIteratorLevel.getParent().getMode().getModePack().getArray(1), posVarUnknown));
      }
    } else {
      locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], posVarUnknown);
    }
    searchForUnderivedStart.push_back(locateCoordVar);
  }
  return ir::Block::make(searchForUnderivedStart);
}

// TODO (rohany): Replace this static incrementing ID with a pass during code
//  generation that collects all sharding functors and uniquely numbers them.
static int shardingFunctorID = 0;
Stmt LowererImpl::lowerForallDimension(Forall forall,
                                       vector<Iterator> locators,
                                       vector<Iterator> inserters,
                                       vector<Iterator> appenders,
                                       set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  // Figure out up front if we are doing a pos split or not. This
  // affects what kind of partitioning operations we are going to do.
  auto posRel = this->provGraph.getParentPosRel(forall.getIndexVar());
  bool inPosIter = posRel != nullptr;

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
    atomicParallelUnit = forall.getParallelUnit();
  }

  // TODO (rohany): Need some sort of stack mechanism to pop off the computing on
  //  var once (if) we support nested distributions.
  if (!forall.getComputingOn().empty()) {
    this->computingOnTensorVar = forall.getComputingOn();
  }
  if (forall.getOutputRaceStrategy() == OutputRaceStrategy::ParallelReduction && forall.isDistributed()) {
    this->performingLegionReduction = true;
  }

  // Loop dimension (for fused loops this could be multi-dimensional).
  auto dim = 1;
  // The index variables for this forall.
  std::vector<IndexVar> distIvars = {forall.getIndexVar()};
  // If this loop is a multi-dimensional distributed loop collapsed into
  // a single loop, then unpack thee actual variables that are being distributed.
  {
    auto fusedVars = this->provGraph.getMultiFusedParents(forall.getIndexVar());
    if (!fusedVars.empty()) {
      dim = fusedVars.size();
      distIvars = fusedVars;
    }
  }

  size_t originalDefinedIndexVarsLen = this->definedIndexVarsExpanded.size();
  // All of the distVars will be added to the defined list.
  for (auto& v : distIvars) {
    this->definedIndexVarsExpanded.push_back(v);
    this->pushIterationSpacePointIdentifier();
  }

  std::vector<ir::Stmt> pointIdentifierDecls;
  // Create the point identifiers for each level. The idea here is to uniquely
  // identify each iteration space point at the depth that it is, so that each
  // iteration space point at a particular level in the loop nest has a different
  // ID. We then use these ID's to name the partitions created at that level in
  // the loop nest. We apply a similar strategy that taco uses to generate locators
  // into dense loop nests. At each step i, the variable for i is defined by
  // var(i-1) * dim(i) + var(i).
  for (size_t i = originalDefinedIndexVarsLen; i < this->definedIndexVarsExpanded.size(); i++) {
    Expr rhs;
    if (i == 0) {
      // In the first level of the iteration, the iteration space identifier is
      // the variable itself.
      rhs = this->indexVarToExprMap[this->definedIndexVarsExpanded[i]];
    } else {
      // Otherwise, we construct the variable from the prior level iteration space point.
      auto var = this->indexVarToExprMap[this->definedIndexVarsExpanded[i]];
      auto bounds = this->provGraph.deriveIterBounds(this->definedIndexVarsExpanded[i], this->definedIndexVarsOrdered, this->underivedBounds, this->indexVarToExprMap, this->iterators);
      rhs = ir::Add::make(ir::Mul::make(this->iterationSpacePointIdentifiers[i - 1], bounds[1]), var);
    }
    pointIdentifierDecls.push_back(ir::VarDecl::make(this->iterationSpacePointIdentifiers[i], rhs));
  }

  auto prevDistVar = this->curDistVar;

  if (forall.isDistributed()) {
    this->curDistVar = forall.getIndexVar();
    this->distLoopDepth++;
  }

  this->varsInScope[this->curDistVar].insert(forall.getIndexVar());
  auto parents = this->provGraph.getMultiFusedParents(forall.getIndexVar());
  if (!parents.empty()) {
    for (auto i : parents) {
      this->varsInScope[this->curDistVar].insert(i);
    }
  }
  // Save the scope and replace it once we exit from recursion.
  auto savedScopeVars = std::set<IndexVar>(this->varsInScope[this->curDistVar]);

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  // After recursing, remove all of the dist vars.
  for (auto _ : distIvars) {
    this->definedIndexVarsExpanded.pop_back();
  }

  if (forall.isDistributed()) {
    this->curDistVar = forall.getIndexVar();
    this->distLoopDepth--;
  }
  this->varsInScope[this->curDistVar] = savedScopeVars;

  // Allocate a buffer onto the GPU for the reduction result.
  std::vector<ir::Stmt> gpuReductionPreamble, gpuReductionPostamble;
  if (forall.getParallelUnit() == taco::ParallelUnit::GPUBlock && this->performingScalarReduction) {
    auto initVal = ir::Var::make("init", this->scalarReductionResult.type());
    gpuReductionPreamble.push_back(ir::VarDecl::make(initVal, 0));
    // Make a buffer for the allocation.
    auto bufTy = DeferredBuffer(this->scalarReductionResult.type(), 1);
    auto domTy = Domain(1);
    auto rectTy = Rect(1);
    auto dom = ir::makeConstructor(domTy, {ir::makeConstructor(rectTy, {0, 0})});
    auto buf = ir::Var::make("buf", bufTy);
    auto addr = [](ir::Expr e) {
      return ir::Call::make("&", {e}, Auto);
    };
    gpuReductionPreamble.push_back(ir::VarDecl::make(buf, makeConstructor(bufTy, {GPUFBMem, dom, addr(initVal)})));
    auto bufPtr = ir::Var::make("bufPtr", Pointer(this->scalarReductionResult.type()));
    gpuReductionPreamble.push_back(ir::VarDecl::make(bufPtr, ir::MethodCall::make(buf, "ptr", {0}, false, Auto)));
    // Now, rewrite body so that writes into the scalar result are replaced
    // by writes into this buffer.
    struct ScalarReductionRewriter : public IRRewriter {
      void visit(const Assign* node) {
        if (node->lhs == this->target) {
          this->stmt = ir::Store::make(result, 0, node->rhs, true /* useAtomics */);
        } else {
          this->stmt = node;
        }
      }
      ir::Expr target;
      ir::Expr result;
    };
    ScalarReductionRewriter rw;
    rw.target = this->scalarReductionResult; rw.result = bufPtr;
    body = rw.rewrite(body);
    // Finally, issue a copy of the buffer's data back to the CPU to be scheduled after the kernel.
    auto copySize = ir::Call::make("sizeof", {this->scalarReductionResult}, Auto);
    auto direction = ir::Symbol::make("cudaMemcpyHostToDevice");
    // TODO (rohany): Wrap this in a error checking call.
    auto call = ir::Call::make("cudaMemcpy", {addr(this->scalarReductionResult), bufPtr, copySize, direction}, Auto);
    gpuReductionPostamble.push_back(ir::SideEffect::make(call));
  }

  // As a simple hack, don't emit code that actually performs the iteration within a placement node.
  // We just care about emitting the actual distributed loop to do the data placement, not waste
  // time iterating over the data within it. Placement can be nested though, so only exclude the
  // inner body for the deepest placement level.
  if (forall.isDistributed() && this->isPlacementCode && size_t(this->distLoopDepth + 1) == this->placements.size()) {
    body = ir::Block::make({});
  }
  // We do the same thing for partitioning code.
  if (forall.isDistributed() && this->isPartitionCode) {
    body = ir::Block::make({});
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  // Emit loop with preamble and postamble.
  std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

  Stmt declarePartitionBounds;
  // serializeOnPriorHeader is a header for a task that serializes
  // on the first future provided to the task. It is set only if
  // this behavior is needed.
  Stmt serializeOnPriorHeader;

  Stmt unpackTensorData;

  // taskHeaderStmt is a statement containing the header information for the current task.
  // It is set only if the current for loop is a task.
  Stmt taskHeaderStmt;

  auto isTask = forall.isDistributed() || (forall.getTransfers().size() > 0);
  auto taskID = -1;
  // transfers is the set of statements that will be the preamble for a particular loop.
  // It contains all of the statements necessary to create any partitions required for
  // the tasks that the loop may launch.
  std::vector<ir::Stmt> transfers;
  // partitionStmts is a collection of statements to be used when the lowering kind
  // is just to create a partition of a tensor (this is different than loweringKind == PARTITION_ONLY).
  std::vector<ir::Stmt> partitionStmts;
  // returnPartitionStatements is a collection of statements that return the partitions created by
  // a particular loop. This is used at depth 0 to return the top level partitions.
  std::vector<ir::Stmt> returnPartitionStatements;
  // partitionOnlyStmts is a set of statements that only perform the partitioning of tensors
  // involved in the computation, and omits the actual computation using this partitions. It
  // is used when legionLoweringKind == PARTITION_ONLY.
  std::vector<ir::Stmt> partitionOnlyStmts;
  if (isTask) {
    taskID = this->taskCounter;
    this->taskCounter++;

    // Declare some commonly used datatypes.
    auto dimT = Domain(dim);
    auto domain = ir::Var::make("domain", dimT);
    auto pointInDimT = PointInDomainIterator(dim);
    auto pointT = Point(dim);

    auto domainIter = ir::Var::make("itr", pointInDimT);

    // We need to emit accessing the partition for any child task that uses the partition.
    // TODO (rohany): A hack that doesn't scale to nested distributions.
    if (!forall.getComputingOn().empty()) {
      // Add a declaration of all the needed partition bounds variables.
      auto var = *forall.getComputingOn().begin();
      auto tensorIspace = ir::GetProperty::make(this->tensorVars[var], TensorProperty::IndexSpace);
      auto bounds = ir::Call::make("runtime->get_index_space_domain", {ctx, tensorIspace}, Auto);
      auto boundsVar = ir::Var::make(var.getName() + "PartitionBounds", Auto);
      std::vector<ir::Stmt> declareBlock;
      declareBlock.push_back(ir::VarDecl::make(boundsVar, bounds));
      for (auto tvItr : this->provGraph.getPartitionBounds()) {
        for (auto idxItr : tvItr.second) {
          auto lo = ir::Load::make(ir::MethodCall::make(boundsVar, "lo", {}, false, Int64), idxItr.first);
          auto hi = ir::Load::make(ir::MethodCall::make(boundsVar, "hi", {}, false, Int64), idxItr.first);
          declareBlock.push_back(ir::VarDecl::make(idxItr.second.first, lo));
          declareBlock.push_back(ir::VarDecl::make(idxItr.second.second, hi));
        }
      }
      declarePartitionBounds = ir::Block::make(declareBlock);
    }

    // Declare the launch domain for the current task launch.
    util::append(transfers, this->declareLaunchDomain(domain, forall, distIvars));

    // Figure out what tensors are being partitioned at this loop. We collect the statements
    // in a separate vector because depending on the lowering kind we may or may not want to
    // add them to the final set of statements. We still perform the operations to discover
    // which tensors are partitioned so that the appropriate subregions are selected in the
    // generated code.
    std::vector<ir::Stmt> partitioningStmts;
    std::map<TensorVar, std::map<int, std::vector<ir::Expr>>> tensorLogicalPartitions;
    std::map<TensorVar, std::map<int, ir::Expr>> tensorDenseRunPartitions;
    // Extract the region domains for each region in the transfer.
    std::map<TensorVar, ir::Expr> domains;
    if (inPosIter) {
      // If we are splitting the position space, then we we construct partitions in a bottom-up manner.
      // TODO (rohany): I'm unsure how it works when there are variables transferred at lower
      //  than the distributed loop for a position split. For now, let's require that we are
      //  communicating all of the tensors at this loop.
      taco_iassert(forall.getTransfers().size() == this->tensorVarOrdering.size());

      // Find the deepest level accessed in the tensor. That is the point where we will
      // take the initial partition.
      auto underiveds = this->provGraph.getUnderivedAncestors(forall.getIndexVar());
      auto posAccess = posRel->getAccess();
      auto posTensor = posAccess.getTensorVar();
      int posMode = -1;
      for (auto ivar : this->provGraph.getUnderivedAncestors(forall.getIndexVar())) {
        for (int i = 0; i < posTensor.getOrder(); i++) {
          if (posAccess.getIndexVars()[i] == ivar) {
            posMode = std::max(posMode, i);
          }
        }
      }
      taco_iassert(posMode >= 0);
      // Convert the posMode into a level.
      int posLevel = -1;
      for (size_t i = 0; i < posTensor.getFormat().getModeOrdering().size(); i++) {
        if (posTensor.getFormat().getModeOrdering()[i] == posMode) {
          posLevel = i;
        }
      }
      taco_iassert(posLevel >= 0);

      // Get the initial iterator to partition.
      auto posIter = this->iterators.getLevelIteratorByLevel(posTensor, posLevel);
      taco_iassert(posIter.defined());

      // We should have only 1 parent, the position variable.
      auto posParents = this->provGraph.getParents(forall.getIndexVar());
      taco_iassert(posParents.size() == 1);
      auto posVar = posParents[0];
      // Symbolically recover the position variable to derive bounds on it.
      auto recovered = this->provGraph.recoverVariable(posVar, this->definedIndexVarsOrdered, this->underivedBounds,
                                                       this->indexVarToExprMap, this->iterators);
      // Here, we copy some logic from the BoundsInferenceExprRewriter to get bounds on the position variable.
      auto rwFn = [&](bool lower, ir::Expr bound) {
        BoundsInferenceExprRewriter rw(this->provGraph, this->iterators, this->underivedBounds,
                                       this->indexVarToExprMap,
                                       this->definedIndexVars, this->exprToIndexVarMap,
                                       this->definedIndexVarsOrdered, lower, this->presentIvars);
        do {
          rw.changed = false;
          bound = rw.rewrite(bound);
        } while (rw.changed);
        return bound;
      };
      auto lower = ir::simplify(rwFn(true /* lower */, recovered));
      auto upper = ir::simplify(rwFn(false /* lower */, recovered));

      // Now, we use the meta-abstractions on the format to partition the target level. We
      // create and populate the target DomainPointColorings.
      partitioningStmts.push_back(posIter.getInitializePosColoring());
      std::vector<ir::Stmt> coloringLoopBody;
      coloringLoopBody.push_back(ir::VarDecl::make(this->indexVarToExprMap[forall.getIndexVar()],
                                                   ir::Load::make(ir::Deref::make(domainIter, pointT), 0)));
      coloringLoopBody.push_back(posIter.getCreatePosColoringEntry(ir::Deref::make(domainIter, Auto), lower, upper));
      auto coloringLoop = ir::For::make(
          domainIter,
          ir::Call::make(pointInDimT.getName(), {domain}, pointInDimT),
          ir::MethodCall::make(domainIter, "valid", {}, false /* deref */, Datatype::Bool),
          1 /* increment -- hack to get ++ */,
         ir::Block::make(coloringLoopBody)
      );
      partitioningStmts.push_back(coloringLoop);
      partitioningStmts.push_back(posIter.getFinalizePosColoring());

      // Now, we'll use the initial partition of the target level to create partitions
      // of the rest of the levels in the tensor.
      ir::Expr partitionColor;
      if (!this->definedIndexVarsExpanded.empty()) {
        partitionColor = this->getIterationSpacePointIdentifier();
      }
      // Maybe add on the iteration space point identifier if we have one.
      auto maybeAddPartColor = [&](std::vector<ir::Expr> args) {
        if (!partitionColor.defined()) {
          return args;
        }
        args.push_back(partitionColor);
        return args;
      };

      // Use the format meta-abstraction to create the initial partition. The
      // last 2 elements in the ModeFunction are the initial upwards and downwards
      // facing partitions.
      auto posPartitionFunc = posIter.getCreatePartitionWithPosColoring(domain, partitionColor);
      taco_iassert(posPartitionFunc.defined());
      partitioningStmts.push_back(posPartitionFunc.compute());
      for (size_t i = 0; i < posPartitionFunc.numResults() - 2; i++) {
        tensorLogicalPartitions[posTensor][posLevel].push_back(posPartitionFunc[i]);
      }
      auto upwardsPart = posPartitionFunc[posPartitionFunc.numResults() - 2];
      auto downwardsPart = posPartitionFunc[posPartitionFunc.numResults() - 1];

      // We also need to record an initial partition to use for the tensor
      // that has the same non-zero structure as the pos tensor. This initial
      // strategy only works because the output tensor will have dimension
      // less than or equal the pos tensor.
      if (this->preservesNonZeros) {
        taco_iassert(this->nonZeroAnalyzerResult.resultAccess->getTensorVar().getOrder() <= posTensor.getOrder());
      }
      ir::Expr nonZeroInitialPartition = downwardsPart;
      // We also apply the initial step below if we should switch to using the
      // upwards partition as an initial partition of the output tensor.
      if (this->preservesNonZeros && this->nonZeroAnalyzerResult.resultAccess->getTensorVar().getOrder() == posLevel) {
        nonZeroInitialPartition = upwardsPart;
      }

      // TODO (rohany): Handle partitioning the DenseFormatRuns like as in createIndexPartitions.
      // First, partition all of the levels below the posLevel in a downward pass.
      for (int level = posLevel + 1; level < posTensor.getOrder(); level++) {
        auto mode = posTensor.getFormat().getModeOrdering()[level];
        auto iter = this->iterators.getLevelIteratorByModeAccess(ModeAccess(posAccess, mode + 1));
        auto partFunc = iter.getPartitionFromParent(downwardsPart, partitionColor);
        if (partFunc.defined()) {
          partitioningStmts.push_back(partFunc.compute());
          downwardsPart = partFunc.getResults().back();
          // Remember all of the partition variables so that we can use them when
          // constructing region requirements later.
          for (size_t i = 0; i < partFunc.numResults() - 1; i++) {
            tensorLogicalPartitions[posTensor][level].push_back(partFunc[i]);
          }
        }
      }

      // Then, using the resulting partition from the downward partitioning pass,
      // partition the values region.
      auto posTensorExpr = this->tensorVars[posTensor];
      auto valsLogicalPart = ir::Var::make(posTensor.getName() + "ValsLogicalPart", LogicalPartition);
      auto valsLogicalPartition = ir::Call::make("copyPartition", maybeAddPartColor({ctx, runtime, downwardsPart, ir::GetProperty::make(posTensorExpr, ir::TensorProperty::Values)}), Auto);
      partitioningStmts.push_back(ir::VarDecl::make(valsLogicalPart, valsLogicalPartition));
      tensorLogicalPartitions[posTensor][posTensor.getOrder()] = {valsLogicalPart};

      // Next, partition the levels of the tensor above the posLevel in an upwards pass.
      // TODO (rohany): Handle partitioning the DenseFormatRuns like as in createIndexPartitions.
      for (int level = posLevel - 1; level >= 0; level--) {
        auto iter = this->iterators.getLevelIteratorByLevel(posTensor, level);
        auto modeFunc = iter.getPartitionFromChild(upwardsPart, partitionColor);
        if (modeFunc.defined()) {
          partitioningStmts.push_back(modeFunc.compute());
          upwardsPart = modeFunc.getResults().back();
          for (size_t i = 0; i < modeFunc.numResults() - 1; i++) {
            tensorLogicalPartitions[posTensor][level].push_back(modeFunc[i]);
          }
          // If we hit the bottom level of the result non-zero tensor, then mark
          // the initial partition for the output tensor.
          if (this->preservesNonZeros && this->nonZeroAnalyzerResult.resultAccess->getTensorVar().getOrder() <= level) {
            nonZeroInitialPartition = upwardsPart;
          }
        }
      }

      // TODO (rohany): Not sure how to assert some of the assumptions below.
      // At this point, we should end up with a partition of the top dense levels
      // of posTensor, so let's copy that partition into a partition of the first
      // dense run of posTensor. We only do this if there is a dense run to partition.
      auto denseRun = ir::GetProperty::makeDenseLevelRun(posTensorExpr, 0);
      auto denseRunPartition = ir::Var::make(posTensor.getName() + "DenseRun0Partition", IndexPartition);
      DenseFormatRuns posTensorFormatRuns(posAccess, this->iterators);
      if (!posTensorFormatRuns.runs.empty() && util::contains(posTensorFormatRuns.runs[0].levels, 0)) {
        partitioningStmts.push_back(ir::VarDecl::make(denseRunPartition, ir::Call::make("copyPartition", maybeAddPartColor({ctx, runtime, upwardsPart, denseRun}), IndexPartition)));
        tensorDenseRunPartitions[posTensor][0] = denseRunPartition;
      }

      // TODO (rohany): This doesn't do as much work as the position space split
      //  for the input tensor, but should be fine based on the restrictions that
      //  we apply here?
      // Copy the partitions of the pos tensor over to the output tensor if
      // the non-zero structure is preserved.
      if (this->preservesNonZeros) {
        // TODO (rohany): This doesn't do as much work as the partitioning pass on
        //  the tensor who's position space has been split. For now that's fine
        //  since we don't have any use cases that do something different. If
        //  the below assertion fails then this partitioning pass will need to
        //  be updated as well. The assertion checks the position split occurs
        //  at or below the lowest level in the output tensor.
        auto tv = this->nonZeroAnalyzerResult.resultAccess->getTensorVar();
        taco_iassert((posLevel + 1) >= tv.getOrder());
        auto vals = ir::GetProperty::make(this->tensorVars[tv], TensorProperty::Values);
        auto lPart = ir::Var::make(tv.getName() + "ValsLogicalPart", LogicalPartition);
        auto createLPart = ir::Call::make("copyPartition", maybeAddPartColor({ctx, runtime, nonZeroInitialPartition, vals}), LogicalPartition);
        partitioningStmts.push_back(ir::VarDecl::make(lPart, createLPart));
        tensorLogicalPartitions[tv][tv.getOrder()] = {lPart};
        // TODO (rohany): Is there a neat way of deduplicating this code?
        // Now do an upwards partitioning pass.
        ir::Expr currentPart = lPart;
        for (int level = tv.getOrder() - 1; level >= 0; level--) {
          auto iter = this->iterators.getLevelIteratorByLevel(tv, level);
          auto modeFunc = iter.getPartitionFromChild(currentPart, partitionColor);
          if (modeFunc.defined()) {
            partitioningStmts.push_back(modeFunc.compute());
            currentPart = modeFunc.getResults().back();
            for (size_t i = 0; i < modeFunc.numResults() - 1; i++) {
              tensorLogicalPartitions[tv][level].push_back(modeFunc[i]);
            }
          }
        }

        // TODO (rohany): Again, there's some duplication here...
        // Also copy the result partition to the first dense run of the result tensor, if the tensor has
        // a top level dense run.
        // TODO (rohany): Does it suffice to use the posTensorFormatRuns here, or do we need another
        //  one specific to the tensor being copied?
        if (!posTensorFormatRuns.runs.empty() && util::contains(posTensorFormatRuns.runs[0].levels, 0)) {
          auto denseRun = ir::GetProperty::makeDenseLevelRun(this->tensorVars[tv], 0);
          auto denseRunPartition = ir::Var::make(tv.getName() + "DenseRun0Partition", IndexPartition);
          partitioningStmts.push_back(ir::VarDecl::make(denseRunPartition, ir::Call::make("copyPartition", maybeAddPartColor({ctx, runtime, currentPart, denseRun}), IndexPartition)));
          tensorDenseRunPartitions[tv][0] = denseRunPartition;
        }
      }

      // Now, partition the rest of the tensors in the program using the
      // derived top level partition of the pos tensor.
      for (auto tv : this->tensorVarOrdering) {
        // Skip the tensor who's position space we are iterating over.
        if (tv == posTensor) continue;
        // Also skip the tensor that shares the same non-zero structure as the input tensor
        // (the one we are performing a position split over), as we'll partition that in
        // the same way as the input tensor.
        if (this->preservesNonZeros && this->nonZeroAnalyzerResult.resultAccess->getTensorVar() == tv) continue;
        auto tvExpr = this->tensorVars[tv];
        auto tvAccess = this->tensorVarToAccess.at(tv);
        auto projection = this->constructAffineProjection(posAccess, tvAccess);
        if (!projection.defined()) {
          // If we don't have a projection, then this tensor is fully replicated.
          continue;
        }
        // TODO (rohany): Deduplicate this out into a helper function used by the standard
        //  partitioning pass.
        // Otherwise, we'll use the projection to create a partition of the first
        // dense run of the tensor.
        auto runs = DenseFormatRuns(tvAccess, this->iterators);
        taco_iassert(runs.runs.size() == 1);
        auto firstDenseRun = ir::GetProperty::makeDenseLevelRun(tvExpr, 0);
        auto firstDenseRunPart = ir::Var::make(tv.getName() + "DenseRun0Partition", IndexPartition);
        auto createDenseRunPart = ir::MethodCall::make(projection, "apply", maybeAddPartColor({ctx, runtime, denseRunPartition, firstDenseRun}), false /* deref */, IndexPartition);
        partitioningStmts.push_back(ir::VarDecl::make(firstDenseRunPart, createDenseRunPart));
        tensorDenseRunPartitions[tv][0] = firstDenseRunPart;
        auto currentPart = firstDenseRunPart;
        // TODO (rohany): Handle partitioning the dense format runs like createIndexPartitions.
        for (int level = runs.runs[0].levels.back(); level < tv.getOrder(); level++) {
          auto mode = tvAccess.getTensorVar().getFormat().getModeOrdering()[level];
          auto iter = this->iterators.getLevelIteratorByModeAccess({tvAccess, mode + 1});
          auto modeFunc = iter.getPartitionFromChild(currentPart, partitionColor);
          if (modeFunc.defined()) {
            partitioningStmts.push_back(modeFunc.compute());
            currentPart = modeFunc.getResults().back();
            for (size_t i = 0; i < modeFunc.numResults() - 1; i++) {
              tensorLogicalPartitions[tv][level].push_back(modeFunc[i]);
            }
          }
        }
        // Finally, construct the values partition.
        auto partitionVals = ir::Call::make(
            "copyPartition",
            maybeAddPartColor(
                {ctx, runtime, currentPart, getLogicalRegion(ir::GetProperty::make(this->tensorVars[tv], TensorProperty::Values))}),
            Auto
        );
        auto valsPart = ir::Var::make(tv.getName() + "_vals_partition", Auto);
        partitioningStmts.push_back(ir::VarDecl::make(valsPart, partitionVals));
        tensorLogicalPartitions[tv][tv.getOrder()].push_back(valsPart);
      }
    } else {
      // If we aren't in a pos split, then we'll construct partitions in a standard top-down manner.
      for (auto& t : forall.getTransfers()) {
        auto tv = t.getAccess().getTensorVar();
        auto domain = ir::Var::make(tv.getName() + "Domain", Auto);
        // TODO (rohany): We'll assume for now that we want just the domains for the first level dense index space run.
        auto ispace = ir::GetProperty::makeDenseLevelRun(this->tensorVars[tv], 0);
        partitioningStmts.push_back(ir::VarDecl::make(domain, ir::Call::make("runtime->get_index_space_domain", {ctx, ispace}, Auto)));
        domains[t.getAccess().getTensorVar()] = domain;
      }

      // Make a coloring for each transfer.
      std::vector<Expr> colorings;
      for (auto &t : forall.getTransfers()) {
        auto c = ir::Var::make(t.getAccess().getTensorVar().getName() + "Coloring", DomainPointColoring);
        partitioningStmts.push_back(
            ir::VarDecl::make(c, ir::Call::make(DomainPointColoring.getName(), {}, DomainPointColoring)));
        colorings.push_back(c);
      }

      std::vector<Stmt> partStmts;
      for (size_t i = 0; i < distIvars.size(); i++) {
        auto ivar = distIvars[i];
        auto ivarExpr = this->indexVarToExprMap[ivar];
        partStmts.push_back(
            ir::VarDecl::make(ivarExpr, ir::Load::make(ir::Deref::make(domainIter, pointT), int32_t(i))));
      }

      // If operating on a partition, we need to get the bounds of the partition at each index point.
      if (!forall.getComputingOn().empty()) {
        util::append(partStmts, this->declarePartitionBoundsVars(domainIter, *forall.getComputingOn().begin()));
      }

      // Create colorings for each tensor being transferred.
      std::set<TensorVar> fullyReplicatedTensors;
      util::append(partStmts,
                   this->createDomainPointColorings(forall, domainIter, domains, fullyReplicatedTensors, colorings));
      // Construct a loop over the launch domain that colors the accessed subregion
      // of each tensor.
      auto l = ir::For::make(
          domainIter,
          ir::Call::make(pointInDimT.getName(), {domain}, pointInDimT),
          ir::MethodCall::make(domainIter, "valid", {}, false /* deref */, Datatype::Bool),
          1 /* increment -- hack to get ++ */,
          ir::Block::make(partStmts)
      );
      partitioningStmts.push_back(l);

      // Create IndexPartition objects from each of the colorings created.
      util::append(partitioningStmts,
                   this->createIndexPartitions(forall, domain,tensorLogicalPartitions,
                                               tensorDenseRunPartitions, fullyReplicatedTensors, colorings, distIvars));
    }

    // If we're not only performing compute statements, then include the statements that create
    // the necessary partitions.
    if (this->legionLoweringKind != COMPUTE_ONLY) {
      util::append(transfers, partitioningStmts);
    }

    // If we're emitting partitioning code, then this is all we care about. Package
    // up everything and add on a get_logical_partition call to return.
    if (this->isPartitionCode) {
      partitionStmts = transfers;
      // TODO (rohany): I'm asserting false here because this codepath uses the old (and deleted)
      //  partitionings variable, so I'm not sure what it does.
      taco_iassert(false);
      // auto pair = partitionings.begin();
      // auto region = this->tensorVars[pair->first];
      // auto part = pair->second;
      // partitionStmts.push_back(ir::Return::make(ir::Call::make("runtime->get_logical_partition", {ctx, getLogicalRegion(region), part}, Auto)));
    } else if (this->legionLoweringKind == PARTITION_ONLY && this->distLoopDepth == 0) {
      // TODO (rohany): Move this into a helper method?

      // This code is badly named / name overloaded, but we perform a
      // similar operation here if the desired output is code that just
      // partitions the tensors for computation.
      // Declare a result vector.

      // We'll build a struct used to hold onto all of the partitions within our tensor.
      // To define it, we need to collect the field names and field types that will be
      // present in the struct.
      auto structTy = this->getTopLevelTensorPartitionPackType();
      auto structName = util::toString(structTy);
      std::vector<std::string> fieldNames;
      std::vector<Datatype> fieldTypes;

      // First, instantiate an instance of the struct in a pointer.
      auto structPack = ir::Var::make("computePartitions", Auto);
      returnPartitionStatements.push_back(ir::VarDecl::make(structPack, ir::makeConstructor(structTy, {})));

      for (auto& t : this->tensorVarOrdering) {
        auto tensor = this->tensorVars[t];
        // If we create a top level partition for this tensor, then we need to
        // include it in the partitionPack. Otherwise, we don't do anything.
        if (util::contains(tensorLogicalPartitions, t)) {
          // Add information about partition to the fields.
          auto fieldName = t.getName() + "Partition";
          fieldNames.push_back(fieldName);
          fieldTypes.push_back(LegionTensorPartition);
          auto fieldAccess = ir::FieldAccess::make(structPack, fieldName, false /* isDeref */, Auto);

          // Initialize all fields of this LegionTensorPartition.
          auto indicesPartitions = ir::FieldAccess::make(fieldAccess, "indicesPartitions", false /* isDeref */, Auto);
          auto denseRuns = ir::FieldAccess::make(fieldAccess, "denseLevelRunPartitions", false /* isDeref */, Auto);
          returnPartitionStatements.push_back(ir::Assign::make(indicesPartitions, ir::makeConstructor(Vector(Vector(LogicalPartition)), {t.getOrder()})));
          // We need an access to use DenseFormatRuns. However, the number of dense format runs is
          // upper bounded by the order of the tensor, so we can lazily use that to initialize it.
          returnPartitionStatements.push_back(ir::Assign::make(denseRuns, ir::makeConstructor(Vector(IndexPartition), {t.getOrder()})));

          for (auto partLevels : tensorLogicalPartitions.at(t)) {
            // The final level is the values region.
            if (partLevels.first == t.getOrder()) {
              taco_iassert(partLevels.second.size() == 1);
              auto vals = ir::FieldAccess::make(fieldAccess, "valsPartition", false /* isDeref */, Auto);
              returnPartitionStatements.push_back(ir::Assign::make(vals, partLevels.second[0]));
            } else {
              // Otherwise it is a collection of partitions for the indices arrays.
              for (auto part : partLevels.second) {
                auto levelLoad = ir::Load::make(indicesPartitions, partLevels.first);
                returnPartitionStatements.push_back(ir::SideEffect::make(ir::MethodCall::make(levelLoad, "push_back", {part}, false /* isDeref */, Auto)));
              }
            }
          }
          if (util::contains(tensorDenseRunPartitions, t)) {
            for (auto densePart : tensorDenseRunPartitions.at(t)) {
              auto idxLoad = ir::Load::make(denseRuns, densePart.first);
              returnPartitionStatements.push_back(ir::Assign::make(idxLoad, densePart.second));
            }
          }
        }
      }
      returnPartitionStatements.push_back(ir::DeclareStruct::make(structName, fieldNames, fieldTypes));
      // We'll add this to the footer rather than returning right here so that any cleanup
      // that must happen before the partition task exits can occur.
      this->footer.push_back(ir::Return::make(structPack));
    }
    partitionOnlyStmts = transfers;

    // See which of the regions are accessed by the task body.
    std::set<ir::GetProperty::Hashable> regionsAccessedByTask;
    for (auto& it : this->tensorVarOrdering) {
      auto maybeAddReg = [&](ir::Expr reg) {
        taco_iassert(reg.as<GetProperty>() != nullptr);
        if (this->statementAccessesRegion(body, reg)) {
          regionsAccessedByTask.insert(reg.as<GetProperty>()->toHashable());
        }
      };
      // Check each of the regions that make up the tensor.
      for (int i = 0; i < it.getOrder(); i++) {
        auto levelIt = this->iterators.getLevelIteratorByLevel(it, i);
        for (auto reg : levelIt.getRegions()) {
          maybeAddReg(reg.region);
        }
      }
      // Check the values region of the tensor.
      maybeAddReg(ir::GetProperty::make(this->tensorVars[it], TensorProperty::Values));
    }

    // Lower the appropriate kind of task call depending on whether the forall
    // is distributed.
    if (forall.isDistributed()) {
      util::append(transfers, this->lowerIndexLaunch(forall, domain,tensorLogicalPartitions, regionsAccessedByTask, taskID, unpackTensorData));
    } else {
      // Lower a serial loop of task launches.
      util::append(transfers, this->lowerSerialTaskLoop(forall, domain, domainIter, pointT,tensorLogicalPartitions, regionsAccessedByTask, taskID, unpackTensorData));
    }

    // Take the statements currently in the task header that were filled in by recursive calls.
    taskHeaderStmt = ir::Block::make(this->taskHeader);
    this->taskHeader = {};
  }

  // If this forall is supposed to be replaced with a call to a leaf kernel,
  // do so and don't emit the surrounding loop and recovery statements.
  if (util::contains(this->calls, forall.getIndexVar()) && this->legionLoweringKind != PARTITION_ONLY) {
    return Block::make({taskHeaderStmt, unpackTensorData, serializeOnPriorHeader, declarePartitionBounds, this->calls[forall.getIndexVar()]->replaceValidStmt(
        forall,
        this->provGraph,
        this->tensorVars,
        this->performingLegionReduction,
        this->definedIndexVarsOrdered,
        this->underivedBounds,
        this->indexVarToExprMap,
        this->iterators
    )});
  }

  Stmt returnReduction;
  if (this->performingScalarReduction && isTask) {
    // Tasks need to return their reduction result.
    returnReduction = ir::Return::make(this->scalarReductionResult);
  }

  // Add some preambles and postambles to the loop body we're emitting.
  body = Block::make(taskHeaderStmt, unpackTensorData, serializeOnPriorHeader, recoveryStmt, ir::Block::make(pointIdentifierDecls), declarePartitionBounds, body, returnReduction);

  Stmt posAppend = generateAppendPositions(appenders);

  LoopKind kind = LoopKind::Serial;
  if (forall.isDistributed()) {
    kind = LoopKind::Distributed;
  } else if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  } else if (forall.getParallelUnit() != ParallelUnit::NotParallel
            && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    // Realm doesn't support runtime parallel loops yet, so use a static distribution.
    kind = LoopKind::Static_Chunked;
  }

  if (forall.isDistributed()) {
    this->curDistVar = prevDistVar;
  }

  // Return just the partitioning statements if we are generating partitioning code.
  if (this->isPartitionCode) {
    return Block::blanks(ir::Block::make(partitionStmts));
  }

  // When generating code that only partitions tensors in the computation, we do not
  // want to emit the actual compute loops. Therefore, we'll stop including the task
  // bodies when the launched tasks do not launch any more tasks.
  if (this->legionLoweringKind == PARTITION_ONLY) {
    struct TaskFinder : public IndexNotationVisitor {
      void visit(const ForallNode* node) {
        auto f = Forall(node);
        this->hasTasks |= (f.isDistributed() || (!f.getTransfers().empty()));
        node->stmt.accept(this);
      }
      bool hasTasks = false;
    };
    TaskFinder t; forall.getStmt().accept(&t);
    if (!t.hasTasks) {
      // If we're at depth 0, we also need to return the partitions.
      if (this->distLoopDepth == 0) {
        return Block::make(Block::make(partitionOnlyStmts), Block::make(returnPartitionStatements));
      } else {
        return Block::make(partitionOnlyStmts);
      }
    }
  }

  return Block::blanks(ir::Block::make(transfers), ir::Block::make(gpuReductionPreamble),
                       For::make(coordinate, bounds[0], bounds[1], 1, body,
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(),
                                 ignoreVectorize ? 0 : forall.getUnrollFactor(),
                                 // TODO (rohany): What do we do for vector width here?
                                 0,
                                 isTask, taskID),
                       ir::Block::make(gpuReductionPostamble),
                       posAppend,
                       ir::Block::make(returnPartitionStatements));
}

  Stmt LowererImpl::lowerForallDenseAcceleration(Forall forall,
                                                 vector<Iterator> locators,
                                                 vector<Iterator> inserters,
                                                 vector<Iterator> appenders,
                                                 set<Access> reducedAccesses,
                                                 ir::Stmt recoveryStmt)
  {
    taco_iassert(locators.size() == 1) << "Optimizing a dense workspace is only supported when the consumer is the only RHS tensor";
    taco_iassert(provGraph.isFullyDerived(forall.getIndexVar())) << "Sparsely accelerating a dense workspace only works with fully derived index vars";
    taco_iassert(forall.getParallelUnit() == ParallelUnit::NotParallel) << "Sparsely accelerating a dense workspace only works within serial loops";


    TensorVar var;
    for (auto it = tensorVars.begin(); it != tensorVars.end(); ++it) {
      if (it->second == locators[0].getTensor() && util::contains(tempToIndexList, it->first)) {
        var = it->first;
        break;
      }
    }

    Expr indexList = tempToIndexList.at(var);
    Expr indexListSize = tempToIndexListSize.at(var);
    Expr bitGuard = tempToBitGuard.at(var);
    Expr loopVar = ir::Var::make(var.getName() + "_index_locator", taco::Int32, false, false);
    Expr coordinate = getCoordinateVar(forall.getIndexVar());

    if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
      markAssignsAtomicDepth++;
      atomicParallelUnit = forall.getParallelUnit();
    }

    Stmt declareVar = VarDecl::make(coordinate, Load::make(indexList, loopVar));
    Stmt body = lowerForallBody(coordinate, forall.getStmt(), locators, inserters, appenders, reducedAccesses);
    Stmt resetGuard = ir::Store::make(bitGuard, coordinate, ir::Literal::make(false), markAssignsAtomicDepth > 0, atomicParallelUnit);

    if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
      markAssignsAtomicDepth--;
    }

    body = Block::make(declareVar, recoveryStmt, body, resetGuard);

    Stmt posAppend = generateAppendPositions(appenders);

    LoopKind kind = LoopKind::Serial;
    if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
      kind = LoopKind::Vectorized;
    }
    else if (forall.getParallelUnit() != ParallelUnit::NotParallel
             && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
      kind = LoopKind::Runtime;
    }

    return Block::blanks(For::make(loopVar, 0, indexListSize, 1, body, kind,
                                         ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(),
                                         ignoreVectorize ? 0 : forall.getUnrollFactor()),
                                         posAppend);
  }

Stmt LowererImpl::lowerForallCoordinate(Forall forall, Iterator iterator,
                                        vector<Iterator> locators,
                                        vector<Iterator> inserters,
                                        vector<Iterator> appenders,
                                        set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt) {
  taco_not_supported_yet;
  return Stmt();
}

Stmt LowererImpl::lowerForallPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders,
                                      set<Access> reducedAccesses,
                                      ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Stmt declareCoordinate = Stmt();
  Stmt strideGuard = Stmt();
  Stmt boundsGuard = Stmt();
  if (provGraph.isCoordVariable(forall.getIndexVar())) {
    Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                              coordinates(iterator)).getResults()[0];
    // If the iterator is windowed, we must recover the coordinate index
    // variable from the windowed space.
    if (iterator.isWindowed()) {
      if (iterator.isStrided()) {
        // In this case, we're iterating over a compressed level with a for
        // loop. Since the iterator variable will get incremented by the for
        // loop, the guard introduced for stride checking doesn't need to
        // increment the iterator variable.
        strideGuard = this->strideBoundsGuard(iterator, coordinateArray, false /* incrementPosVar */);
      }
      coordinateArray = this->projectWindowedPositionToCanonicalSpace(iterator, coordinateArray);
      // If this forall is being parallelized via CPU threads (OpenMP), then we can't
      // emit a `break` statement, since OpenMP doesn't support breaking out of a
      // parallel loop. Instead, we'll bound the top of the loop and omit the check.
      if (forall.getParallelUnit() != ParallelUnit::CPUThread) {
        boundsGuard = this->upperBoundGuardForWindowPosition(iterator, coordinate);
      }
    }
    declareCoordinate = VarDecl::make(coordinate, coordinateArray);
  }
  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  body = Block::make(recoveryStmt, body);

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  Expr parentPos = iterator.getParent().getPosVar();
  auto parentPositions = this->getAllNeededParentPositions(iterator);
  if (!provGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    ModeFunction bounds = iterator.posBounds(parentPositions);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
    // If we have a window on this iterator, then search for the start of
    // the window rather than starting at the beginning of the level.
    if (iterator.isWindowed()) {
      auto startBoundCopy = startBound;
      startBound = this->searchForStartOfWindowPosition(iterator, startBound, endBound);
      // As discussed above, if this position loop is parallelized over CPU
      // threads (OpenMP), then we need to have an explicit upper bound to
      // the for loop, instead of breaking out of the loop in the middle.
      if (forall.getParallelUnit() == ParallelUnit::CPUThread) {
        endBound = this->searchForEndOfWindowPosition(iterator, startBoundCopy, endBound);
      }
    }
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());
    // TODO (rohany): I'm not sure what is happening in this case, but I'm not really considering
    //  tensor formats that have duplicates etc.
    taco_iassert(parentPositions.size() == 1);

    // E.g. a compressed mode with duplicates. Apply iterator chaining
    Expr parentSegend = iterator.getParent().getSegendVar();
    ModeFunction startBounds = iterator.posBounds(parentPos);
    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
    startBound = startBounds[0];
    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  // TODO (rohany): This isn't needed right now.
//  if (forall.isDistributed()) {
//    std::cout << "marking forall as distributed position" << std::endl;
//    kind = LoopKind::Distributed;
//  } else
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  } else if (forall.getParallelUnit() != ParallelUnit::NotParallel
           && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

// Loop with preamble and postamble
  return Block::blanks(
                       boundsCompute,
                       For::make(iterator.getPosVar(), startBound, endBound, 1,
                                 Block::make(strideGuard, declareCoordinate, boundsGuard, body),
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                       posAppend);

}

Stmt LowererImpl::lowerForallFusedPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders,
                                      set<Access> reducedAccesses,
                                      ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Stmt declareCoordinate = Stmt();
  if (provGraph.isCoordVariable(forall.getIndexVar())) {
    Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                              coordinates(iterator)).getResults()[0];
    declareCoordinate = VarDecl::make(coordinate, coordinateArray);
  }

  // If we're generating legion code and in a distributed loop, then we want to break out if we
  // don't actually have any values to process in this task.
  if (this->legion && this->distLoopDepth > 0) {
    auto posRel = this->provGraph.getParentPosRel(forall.getIndexVar());
    taco_iassert(posRel);
    auto posTensor = posRel->getAccess().getTensorVar();
    auto valsReg = ir::GetProperty::make(this->tensorVars[posTensor], ir::TensorProperty::Values);
    auto valsDomain = ir::Var::make(posTensor.getName() + "ValsDomain", Domain(1));
    this->taskHeader.push_back(ir::VarDecl::make(valsDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, getIndexSpace(valsReg)}, Auto)));
    this->taskHeader.push_back(ir::IfThenElse::make(ir::MethodCall::make(valsDomain, "empty", {}, false /* deref */, Bool), ir::Return::make(Expr())));
  }

  // declare upper-level underived ancestors that will be tracked with while loops
  Expr writeResultCond;
  vector<Stmt> loopsToTrackUnderived;
  vector<Stmt> searchForUnderivedStart;
  std::vector<ir::Stmt> posVarRecoveryStmts;
  std::map<IndexVar, vector<Expr>> coordinateBounds = provGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
  vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());

  if (underivedAncestors.size() > 1) {
    // each underived ancestor is initialized to min coordinate bound
    IndexVar posIteratorVar;
#if TACO_ASSERTS
    bool hasIteratorAncestor = provGraph.getPosIteratorAncestor(
        iterator.getIndexVar(), &posIteratorVar);
    taco_iassert(hasIteratorAncestor);
#else /* !TACO_ASSERTS */
    provGraph.getPosIteratorAncestor(
        iterator.getIndexVar(), &posIteratorVar);
#endif /* TACO_ASSERTS */
    // get pos variable then search for leveliterators to find the corresponding iterator

    Iterator posIterator;
    auto iteratorMap = iterators.levelIterators();
    int modePos = -1; // select lowest level possible
    for (auto it = iteratorMap.begin(); it != iteratorMap.end(); it++) {
      if (it->second.getIndexVar() == posIteratorVar && (int) it->first.getModePos() > modePos) {
        posIterator = it->second;
        modePos = (int) it->first.getModePos();
      }
    }
    taco_iassert(posIterator.hasPosIter());

    if (inParallelLoopDepth == 0) {
      for (int i = 0; i < (int) underivedAncestors.size() - 1; i ++) {
        // TODO: only if level is sparse emit underived_pos
        header.push_back(VarDecl::make(this->iterators.modeIterator(underivedAncestors[i]).getPosVar(), 0)); // TODO: set to start position bound
        header.push_back(VarDecl::make(getCoordinateVar(underivedAncestors[i]), coordinateBounds[underivedAncestors[i]][0]));
      }
    } else {
      searchForUnderivedStart.push_back(searchForFusedPositionStart(forall, posIterator));
    }

    Expr parentPos = this->iterators.modeIterator(underivedAncestors[underivedAncestors.size() - 2]).getPosVar();
    ModeFunction posBounds = posIterator.posBounds(parentPos);
    writeResultCond = ir::Eq::make(ir::Add::make(indexVarToExprMap[posIterator.getIndexVar()], 1), posBounds[1]);

    Stmt loopToTrackUnderiveds; // to track next ancestor
    for (int i = 0; i < (int) underivedAncestors.size() - 1; i++) {
      Expr coordVarUnknown = getCoordinateVar(underivedAncestors[i]);
      Expr posVarKnown = this->iterators.modeIterator(underivedAncestors[i+1]).getPosVar();
      if (i == (int) underivedAncestors.size() - 2) {
        posVarKnown = indexVarToExprMap[posIterator.getIndexVar()];
      }
      Expr posVarUnknown = this->iterators.modeIterator(underivedAncestors[i]).getPosVar();

      Iterator posIteratorLevel = posIterator;
      for (int j = (int) underivedAncestors.size() - 2; j > i; j--) { // take parent of iterator enough times to get correct level
        posIteratorLevel = posIteratorLevel.getParent();
      }

      // Recover the position variable for the corresponding level in the tensor we are
      // iterating over.
      {
        Iterator currentPosIterator = posIteratorLevel;
        while (!currentPosIterator.isRoot()) {
          if (currentPosIterator.getIndexVar() == this->iterators.modeIterator(underivedAncestors[i])) {
            if (currentPosIterator.hasPosIter()) {
              posVarRecoveryStmts.push_back(ir::VarDecl::make(currentPosIterator.getPosVar(), posVarUnknown));
            }
            break;
          }
          currentPosIterator = currentPosIterator.getParent();
        }
      }

      ModeFunction posBoundsLevel = posIteratorLevel.posBounds(posVarUnknown);
      Expr loopcond;
      if (posIteratorLevel.getMode().getModeFormat().is<RectCompressedModeFormat>()) {
        // If we have a Legion mode format, then we to check if the current position
        // is contained within the current rectangle.
        auto format = posIteratorLevel.getMode().getModeFormat().as<RectCompressedModeFormat>();
        auto posAcc = format->getAccessor(posIteratorLevel.getMode().getModePack(), RectCompressedModeFormat::POS);
        auto posLoad = ir::Load::make(posAcc, posVarUnknown);
        auto contains = ir::MethodCall::make(posLoad, "contains", {posVarKnown}, false /* deref */, Bool);
        loopcond = ir::Call::make("!", {contains}, Bool);
      } else {
        loopcond = ir::Eq::make(posVarKnown, posBoundsLevel[1]);
      }

      Stmt locateCoordVar;
      if (posIteratorLevel.getParent().hasPosIter()) {
        auto posIteratorParent = posIteratorLevel.getParent();
        if (posIteratorParent.getMode().getModeFormat().is<RectCompressedModeFormat>()) {
          auto rcmf = posIteratorParent.getMode().getModeFormat().as<RectCompressedModeFormat>();
          auto crdAcc = rcmf->getAccessor(posIteratorParent.getMode().getModePack(), RectCompressedModeFormat::CRD);
          locateCoordVar = ir::Assign::make(coordVarUnknown, ir::Load::make(crdAcc, posVarUnknown));
        } else {
          locateCoordVar = ir::Assign::make(coordVarUnknown, ir::Load::make(posIteratorLevel.getParent().getMode().getModePack().getArray(1), posVarUnknown));
        }
      }
      else {
        locateCoordVar = ir::Assign::make(coordVarUnknown, posVarUnknown);
      }
      Stmt loopBody = ir::Block::make(compoundAssign(posVarUnknown, 1), locateCoordVar, loopToTrackUnderiveds);
      if (posIteratorLevel.getParent().hasPosIter()) { // TODO: if level is unique or not
        loopToTrackUnderiveds = IfThenElse::make(loopcond, loopBody);
      }
      else {
        loopToTrackUnderiveds = While::make(loopcond, loopBody);
      }
    }
    loopsToTrackUnderived.push_back(loopToTrackUnderiveds);
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  body = Block::make(recoveryStmt, Block::make(loopsToTrackUnderived), Block::make(posVarRecoveryStmts), body);

  // Code to write results if using temporary and reset temporary
  if (!whereConsumers.empty() && whereConsumers.back().defined()) {
    Expr temp = tensorVars.find(whereTemps.back())->second;
    Stmt writeResults = Block::make(whereConsumers.back(), ir::Assign::make(temp, ir::Literal::zero(temp.type())));
    body = Block::make(body, IfThenElse::make(writeResultCond, writeResults));
  }

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  if (!provGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    Expr parentPos = iterator.getParent().getPosVar();
    ModeFunction bounds = iterator.posBounds(parentPos);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

    // E.g. a compressed mode with duplicates. Apply iterator chaining
    Expr parentPos = iterator.getParent().getPosVar();
    Expr parentSegend = iterator.getParent().getSegendVar();
    ModeFunction startBounds = iterator.posBounds(parentPos);
    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
    startBound = startBounds[0];
    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != ParallelUnit::NotParallel
           && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

  // Loop with preamble and postamble
  return Block::blanks(boundsCompute,
                       Block::make(Block::make(searchForUnderivedStart),
                       For::make(indexVarToExprMap[iterator.getIndexVar()], startBound, endBound, 1,
                                 Block::make(declareCoordinate, body),
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor())),
                       posAppend);

}

Stmt LowererImpl::lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
                                    IndexStmt statement,
                                    const std::set<Access>& reducedAccesses)
{
  Expr coordinate = getCoordinateVar(coordinateVar);
  vector<Iterator> appenders = filter(lattice.results(),
                                      [](Iterator it){return it.hasAppend();});

  vector<Iterator> mergers = lattice.points()[0].mergers();
  Stmt iteratorVarInits = codeToInitializeIteratorVars(lattice.iterators(), lattice.points()[0].rangers(), mergers, coordinate, coordinateVar);

  // if modeiteratornonmerger then will be declared in codeToInitializeIteratorVars
  auto modeIteratorsNonMergers =
          filter(lattice.points()[0].iterators(), [mergers](Iterator it){
            bool isMerger = find(mergers.begin(), mergers.end(), it) != mergers.end();
            return it.isDimensionIterator() && !isMerger;
          });
  bool resolvedCoordDeclared = !modeIteratorsNonMergers.empty();

  vector<Stmt> mergeLoopsVec;
  for (MergePoint point : lattice.points()) {
    // Each iteration of this loop generates a while loop for one of the merge
    // points in the merge lattice.
    IndexStmt zeroedStmt = zero(statement, getExhaustedAccesses(point,lattice));
    MergeLattice sublattice = lattice.subLattice(point);
    Stmt mergeLoop = lowerMergePoint(sublattice, coordinate, coordinateVar, zeroedStmt, reducedAccesses, resolvedCoordDeclared);
    mergeLoopsVec.push_back(mergeLoop);
  }
  Stmt mergeLoops = Block::make(mergeLoopsVec);

  // Append position to the pos array
  Stmt appendPositions = generateAppendPositions(appenders);

  return Block::blanks(iteratorVarInits,
                       mergeLoops,
                       appendPositions);
}

Stmt LowererImpl::lowerMergePoint(MergeLattice pointLattice,
                                  ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                  const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared)
{
  MergePoint point = pointLattice.points().front();

  vector<Iterator> iterators = point.iterators();
  vector<Iterator> mergers = point.mergers();
  vector<Iterator> rangers = point.rangers();
  vector<Iterator> locators = point.locators();

  taco_iassert(iterators.size() > 0);
  taco_iassert(mergers.size() > 0);
  taco_iassert(rangers.size() > 0);

  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates = codeToLoadCoordinatesFromPosIterators(iterators, !resolvedCoordDeclared);

  // Any iterators with an index set have extra work to do at the header
  // of the merge point.
  std::vector<ir::Stmt> indexSetStmts;
  for (auto& iter : filter(iterators, [](Iterator it) { return it.hasIndexSet(); })) {
    // For each iterator A with an index set B, emit the following code:
    //   setMatch = min(A, B); // Check whether A matches its index set at this point.
    //   if (A == setMatch && B == setMatch) {
    //     // If there was a match, project down the values of the iterators
    //     // to be the position variable of the index set iterator. This has the
    //     // effect of remapping the index of A to be the i'th position of the set.
    //     A_coord = B_pos;
    //     B_coord = B_pos;
    //   } else {
    //     // Advance the iterator and it's index set iterator accordingly if
    //     // there wasn't a match.
    //     A_pos += (A == setMatch);
    //     B_pos += (B == setMatch);
    //     // We must continue so that we only proceed to the rest of the cases in
    //     // the merge if there actually is a point present for A.
    //     continue;
    //   }
    auto setMatch = ir::Var::make("setMatch", Int());
    auto indexSetIter = iter.getIndexSetIterator();
    indexSetStmts.push_back(ir::VarDecl::make(setMatch, ir::Min::make(this->coordinates({iter, indexSetIter}))));
    // Equality checks for each iterator.
    auto iterEq = ir::Eq::make(iter.getCoordVar(), setMatch);
    auto setEq = ir::Eq::make(indexSetIter.getCoordVar(), setMatch);
    // Code to shift down each iterator to the position space of the index set.
    auto shiftDown = ir::Block::make(
      ir::Assign::make(iter.getCoordVar(), indexSetIter.getPosVar()),
      ir::Assign::make(indexSetIter.getCoordVar(), indexSetIter.getPosVar())
    );
    // Code to increment both iterator variables.
    auto incr = ir::Block::make(
      compoundAssign(iter.getIteratorVar(), ir::Cast::make(Eq::make(iter.getCoordVar(), setMatch), iter.getIteratorVar().type())),
      compoundAssign(indexSetIter.getIteratorVar(), ir::Cast::make(Eq::make(indexSetIter.getCoordVar(), setMatch), indexSetIter.getIteratorVar().type())),
      ir::Continue::make()
    );
    // Code that uses the defined parts together in the if-then-else.
    indexSetStmts.push_back(ir::IfThenElse::make(ir::And::make(iterEq, setEq), shiftDown, incr));
  }

  // Merge iterator coordinate variables
  Stmt resolvedCoordinate = resolveCoordinate(mergers, coordinate, !resolvedCoordDeclared);

  // Locate positions
  Stmt loadLocatorPosVars = declLocatePosVars(locators);

  // Deduplication loops
  auto dupIters = filter(iterators, [](Iterator it){return !it.isUnique() &&
                                                           it.hasPosIter();});
  bool alwaysReduce = (mergers.size() == 1 && mergers[0].hasPosIter());
  Stmt deduplicationLoops = reduceDuplicateCoordinates(coordinate, dupIters,
                                                       alwaysReduce);

  // One case for each child lattice point lp
  Stmt caseStmts = lowerMergeCases(coordinate, coordinateVar, statement, pointLattice,
                                   reducedAccesses);

  // Increment iterator position variables
  Stmt incIteratorVarStmts = codeToIncIteratorVars(coordinate, coordinateVar, iterators, mergers);

  /// While loop over rangers
  return While::make(checkThatNoneAreExhausted(rangers),
                     Block::make(loadPosIterCoordinates,
                                 ir::Block::make(indexSetStmts),
                                 resolvedCoordinate,
                                 loadLocatorPosVars,
                                 deduplicationLoops,
                                 caseStmts,
                                 incIteratorVarStmts));
}

Stmt LowererImpl::resolveCoordinate(std::vector<Iterator> mergers, ir::Expr coordinate, bool emitVarDecl) {
  if (mergers.size() == 1) {
    Iterator merger = mergers[0];
    if (merger.hasPosIter()) {
      // Just one position iterator so it is the resolved coordinate
      ModeFunction posAccess = merger.posAccess(merger.getPosVar(),
                                                coordinates(merger));
      auto access = posAccess[0];
      auto windowVarDecl = Stmt();
      auto stride = Stmt();
      auto guard = Stmt();
      // If the iterator is windowed, we must recover the coordinate index
      // variable from the windowed space.
      if (merger.isWindowed()) {

        // If the iterator is strided, then we have to skip over coordinates
        // that don't match the stride. To do that, we insert a guard on the
        // access. We first extract the access into a temp to avoid emitting
        // a duplicate load on the _crd array.
        if (merger.isStrided()) {
          windowVarDecl = VarDecl::make(merger.getWindowVar(), access);
          access = merger.getWindowVar();
          // Since we're merging values from a compressed array (not iterating over it),
          // we need to advance the outer loop if the current coordinate is not
          // along the desired stride. So, we pass true to the incrementPosVar
          // argument of strideBoundsGuard.
          stride = this->strideBoundsGuard(merger, access, true /* incrementPosVar */);
        }

        access = this->projectWindowedPositionToCanonicalSpace(merger, access);
        guard = this->upperBoundGuardForWindowPosition(merger, coordinate);
      }
      Stmt resolution = emitVarDecl ? VarDecl::make(coordinate, access) : Assign::make(coordinate, access);
      return Block::make(posAccess.compute(),
                         windowVarDecl,
                         stride,
                         resolution,
                         guard);
    }
    else if (merger.hasCoordIter()) {
      taco_not_supported_yet;
      return Stmt();
    }
    else if (merger.isDimensionIterator()) {
      // Just one dimension iterator so resolved coordinate already exist and we
      // do nothing
      return Stmt();
    }
    else {
      taco_ierror << "Unexpected type of single iterator " << merger;
      return Stmt();
    }
  }
  else {
    // Multiple position iterators so the smallest is the resolved coordinate
    if (emitVarDecl) {
      return VarDecl::make(coordinate, Min::make(coordinates(mergers)));
    }
    else {
      return Assign::make(coordinate, Min::make(coordinates(mergers)));
    }
  }
}

Stmt LowererImpl::lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                  MergeLattice lattice,
                                  const std::set<Access>& reducedAccesses)
{
  vector<Stmt> result;

  vector<Iterator> appenders;
  vector<Iterator> inserters;
  tie(appenders, inserters) = splitAppenderAndInserters(lattice.results());

  // Just one iterator so no conditionals
  if (lattice.iterators().size() == 1) {
    Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                appenders, reducedAccesses);
    result.push_back(body);
  }
  else {
    vector<pair<Expr,Stmt>> cases;
    for (MergePoint point : lattice.points()) {

      // Construct case expression
      vector<Expr> coordComparisons;
      for (Iterator iterator : point.rangers()) {
        if (!(provGraph.isCoordVariable(iterator.getIndexVar()) && provGraph.isDerivedFrom(iterator.getIndexVar(), coordinateVar))) {
          coordComparisons.push_back(Eq::make(iterator.getCoordVar(), coordinate));
        }
      }

      // Construct case body
      IndexStmt zeroedStmt = zero(stmt, getExhaustedAccesses(point, lattice));
      Stmt body = lowerForallBody(coordinate, zeroedStmt, {},
                                  inserters, appenders, reducedAccesses);
      if (coordComparisons.empty()) {
        Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                    appenders, reducedAccesses);
        result.push_back(body);
        break;
      }
      cases.push_back({taco::ir::conjunction(coordComparisons), body});
    }
    result.push_back(Case::make(cases, lattice.exact()));
  }

  return Block::make(result);
}


Stmt LowererImpl::lowerForallBody(Expr coordinate, IndexStmt stmt,
                                  vector<Iterator> locators,
                                  vector<Iterator> inserters,
                                  vector<Iterator> appenders,
                                  const set<Access>& reducedAccesses) {
  // If we're performing the optimization that allows for the non-zero structure
  // of an input tensor to be maintained in the result tensor, then we need to
  // do some cleanup here before dealing with appenders and inserters. In particular,
  // the iterator for the result tensor will be an appender (as it is sparse), but
  // will not quite be a inserter or locator, as it is tracking the variables from
  // the input tensor it is aligned with. Therefore, we will just ignore generating
  // appender related code for the result tensor in this case.
  Iterator resultNonZeroIter;
  if (this->preservesNonZeros) {
    std::vector<Iterator> realAppenders;
    for (auto &iter : appenders) {
      if (iter.getTensor() != this->tensorVars[this->nonZeroAnalyzerResult.resultAccess->getTensorVar()]) {
        realAppenders.push_back(iter);
      } else {
        resultNonZeroIter = iter;
      }
    }
    appenders = realAppenders;
  }

  Stmt initVals = resizeAndInitValues(appenders, reducedAccesses);

  // There can be overlaps between the inserters and locators, which results in
  // duplicate emitting of variable declarations. We'll fix that here.
  std::vector<Iterator> itersWithLocators;
  for (auto it : inserters) {
    if (!util::contains(itersWithLocators, it)) { itersWithLocators.push_back(it); }
  }
  for (auto it : locators) {
    if (!util::contains(itersWithLocators, it)) { itersWithLocators.push_back(it); }
  }
  auto declPosVars = declLocatePosVars(itersWithLocators);

  if (captureNextLocatePos) {
    capturedLocatePos = declPosVars;
    captureNextLocatePos = false;
  }

  Stmt trackPosVars;
  if (this->preservesNonZeros && resultNonZeroIter.defined()) {
    if (resultNonZeroIter.hasPosIter()) {
      auto tracking = resultNonZeroIter.getTrackingIterator();
      taco_iassert(tracking.defined());
      trackPosVars = ir::VarDecl::make(resultNonZeroIter.getPosVar(), tracking.getPosVar());
    }
  }

  // Code of loop body statement
  Stmt body = lower(stmt);

  // Code to append coordinates
  Stmt appendCoords = appendCoordinate(appenders, coordinate);

  // TODO: Emit code to insert coordinates

  return Block::make(initVals,
                     declPosVars,
                     trackPosVars,
                     body,
                     appendCoords);
}

Expr LowererImpl::getTemporarySize(Where where) {
  TensorVar temporary = where.getTemporary();
  Dimension temporarySize = temporary.getType().getShape().getDimension(0);
  Access temporaryAccess = getResultAccesses(where.getProducer()).first[0];
  std::vector<IndexVar> indexVars = temporaryAccess.getIndexVars();

  if(util::all(indexVars, [&](const IndexVar& var) { return provGraph.isUnderived(var);})) {
    // All index vars underived then use tensor properties to get tensor size
    taco_iassert(util::contains(dimensions, indexVars[0])) << "Missing " << indexVars[0];
    ir::Expr size = dimensions.at(indexVars[0]);
    for(size_t i = 1; i < indexVars.size(); ++i) {
      taco_iassert(util::contains(dimensions, indexVars[i])) << "Missing " << indexVars[i];
      size = ir::Mul::make(size, dimensions.at(indexVars[i]));
    }
    return size;
  }

  if (temporarySize.isFixed()) {
    return ir::Literal::make(temporarySize.getSize());
  }

  if (temporarySize.isIndexVarSized()) {
    IndexVar var = temporarySize.getIndexVarSize();
    vector<Expr> bounds = provGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds,
                                                     indexVarToExprMap, iterators);
    return ir::Sub::make(bounds[1], bounds[0]);
  }

  taco_ierror; // TODO
  return Expr();
}

vector<Stmt> LowererImpl::codeToInitializeDenseAcceleratorArrays(Where where) {
  TensorVar temporary = where.getTemporary();

  // TODO: emit as uint64 and manually emit bit pack code
  const Datatype bitGuardType = taco::Bool;
  const std::string bitGuardName = temporary.getName() + "_already_set";
  const Expr bitGuardSize = getTemporarySize(where);
  const Expr alreadySetArr = ir::Var::make(bitGuardName,
                                           bitGuardType,
                                           true, false);

  // TODO: TACO should probably keep state on if it can use int32 or if it should switch to
  //       using int64 for indices. This assumption is made in other places of taco.
  const Datatype indexListType = taco::Int32;
  const std::string indexListName = temporary.getName() + "_index_list";
  const Expr indexListArr = ir::Var::make(indexListName,
                                          indexListType,
                                          true, false);

  // no decl for shared memory
  Stmt alreadySetDecl = Stmt();
  Stmt indexListDecl = Stmt();
  const Expr indexListSizeExpr = ir::Var::make(indexListName + "_size", taco::Int32, false, false);
  Stmt freeTemps = Block::make(Free::make(indexListArr), Free::make(alreadySetArr));
  if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0) || !this->loweringToGPU()) {
    alreadySetDecl = VarDecl::make(alreadySetArr, ir::Literal::make(0));
    indexListDecl = VarDecl::make(indexListArr, ir::Literal::make(0));
  }

  tempToIndexList[temporary] = indexListArr;
  tempToIndexListSize[temporary] = indexListSizeExpr;
  tempToBitGuard[temporary] = alreadySetArr;

  Stmt allocateIndexList = Allocate::make(indexListArr, bitGuardSize);
  if (this->loweringToGPU()) {
    Stmt allocateAlreadySet = Allocate::make(alreadySetArr, bitGuardSize);
    Expr p = Var::make("p" + temporary.getName(), Int());
    Stmt guardZeroInit = Store::make(alreadySetArr, p, ir::Literal::zero(bitGuardType));

    Stmt zeroInitLoop = For::make(p, 0, bitGuardSize, 1, guardZeroInit, LoopKind::Serial);
    Stmt inits = Block::make(alreadySetDecl, indexListDecl, allocateAlreadySet, allocateIndexList, zeroInitLoop);
    return {inits, freeTemps};
  } else {
    Expr sizeOfElt = Sizeof::make(bitGuardType);
    Expr callocAlreadySet = ir::Call::make("calloc", {bitGuardSize, sizeOfElt}, Int());
    Stmt allocateAlreadySet = VarDecl::make(alreadySetArr, callocAlreadySet);
    Stmt inits = Block::make(indexListDecl, allocateIndexList, allocateAlreadySet);
    return {inits, freeTemps};
  }

}

// Returns true if the following conditions are met:
// 1) The temporary is a dense vector
// 2) There is only one value on the right hand side of the consumer
//    -- We would need to handle sparse acceleration in the merge lattices for 
//       multiple operands on the RHS
// 3) The left hand side of the where consumer is sparse, if the consumer is an 
//    assignment
// 4) CPU Code is being generated (TEMPORARY - This should be removed)
//    -- The sorting calls and calloc call in lower where are CPU specific. We 
//       could map calloc to a cudaMalloc and use a library like CUB to emit 
//       the sort. CUB support is built into CUDA 11 but not prior versions of 
//       CUDA so in that case, we'd probably need to include the CUB headers in 
//       the generated code.
std::pair<bool,bool> LowererImpl::canAccelerateDenseTemp(Where where) {
  // TODO: TEMPORARY -- Needs to be removed
  if(this->loweringToGPU()) {
    return std::make_pair(false, false);
  }

  TensorVar temporary = where.getTemporary();
  // (1) Temporary is dense vector
  if(!isDense(temporary.getFormat()) || temporary.getOrder() != 1) {
    return std::make_pair(false, false);
  }

  // (2) Multiple operands in inputs (need lattice to reason about iteration)
  const auto inputAccesses = getArgumentAccesses(where.getConsumer());
  if(inputAccesses.size() > 1 || inputAccesses.empty()) {
    return std::make_pair(false, false);
  }

  // No or multiple results?
  const auto resultAccesses = getResultAccesses(where.getConsumer()).first;
  if(resultAccesses.size() > 1 || resultAccesses.empty()) {
    return std::make_pair(false, false);
  }

  // No check for size of tempVar since we enforced the temporary is a vector 
  // and if there is only one RHS value, it must (should?) be the temporary
  std::vector<IndexVar> tempVar = inputAccesses[0].getIndexVars();

  // Get index vars in result.
  std::vector<IndexVar> resultVars = resultAccesses[0].getIndexVars();
  auto it = std::find_if(resultVars.begin(), resultVars.end(),
      [&](const auto& resultVar) {
          return resultVar == tempVar[0] ||
                 provGraph.isDerivedFrom(tempVar[0], resultVar);
  });

  if (it == resultVars.end()) {
    return std::make_pair(true, false);
  }

  int index = (int)(it - resultVars.begin());
  TensorVar resultTensor = resultAccesses[0].getTensorVar();
  int modeIndex = resultTensor.getFormat().getModeOrdering()[index];
  ModeFormat varFmt = resultTensor.getFormat().getModeFormats()[modeIndex];
  // (3) Level of result is sparse
  if(varFmt.isFull()) {
    return std::make_pair(false, false);
  }

  // Only need to sort the workspace if the result needs to be ordered
  return std::make_pair(true, varFmt.isOrdered());
}

vector<Stmt> LowererImpl::codeToInitializeTemporary(Where where) {
  TensorVar temporary = where.getTemporary();

  const bool accelerateDense = canAccelerateDenseTemp(where).first;

  Stmt freeTemporary = Stmt();
  Stmt initializeTemporary = Stmt();
  if (isScalar(temporary.getType())) {
    initializeTemporary = defineScalarVariable(temporary, true);
    Expr tempSet = ir::Var::make(temporary.getName() + "_set", Datatype::Bool);
    Stmt initTempSet = VarDecl::make(tempSet, false);
    initializeTemporary = Block::make(initializeTemporary, initTempSet);
    tempToBitGuard[temporary] = tempSet;
  } else {
    // TODO: Need to support keeping track of initialized elements for
    //       temporaries that don't have sparse accelerator
    taco_iassert(!util::contains(guardedTemps, temporary) || accelerateDense);

    // When emitting code to accelerate dense workspaces with sparse iteration, we need the following arrays
    // to construct the result indices
    if(accelerateDense) {
      vector<Stmt> initAndFree = codeToInitializeDenseAcceleratorArrays(where);
      initializeTemporary = initAndFree[0];
      freeTemporary = initAndFree[1];
    }

    Expr values;
    if (util::contains(needCompute, temporary) &&
        needComputeValues(where, temporary)) {
      // We must re-use the variable defined for this temporary array
      // as the further lowering steps will get confused if we use a
      // a variable that hasn't been defined.
      if (util::contains(this->temporaryArrays, temporary)) {
        values = this->temporaryArrays[temporary].values;
      } else {
        values = ir::Var::make(temporary.getName(),
                               temporary.getType().getDataType(), true, false);
      }
      taco_iassert(temporary.getType().getOrder() == 1)
          << " Temporary order was " << temporary.getType().getOrder();  // TODO
      Expr size = getTemporarySize(where);

      auto decl = VarDecl::make(values, ir::Literal::make(0));
      Stmt allocate = Allocate::make(values, size);
      freeTemporary = Block::make(freeTemporary, Free::make(values));
      initializeTemporary = Block::make(decl, initializeTemporary, allocate);
    }

    /// Make a struct object that lowerAssignment and lowerAccess can read
    /// temporary value arrays from.
    TemporaryArrays arrays;
    arrays.values = values;
    this->temporaryArrays.insert({temporary, arrays});
  }
  return {initializeTemporary, freeTemporary};
}

Stmt LowererImpl::lowerWhere(Where where) {
  TensorVar temporary = where.getTemporary();
  bool accelerateDenseWorkSpace, sortAccelerator;
  std::tie(accelerateDenseWorkSpace, sortAccelerator) =
      canAccelerateDenseTemp(where);

  // Declare and initialize the where statement's temporary
  vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
  bool temporaryHoisted = false;
  for (auto it = temporaryInitialization.begin(); it != temporaryInitialization.end(); ++it) {
    if (it->second == where && it->first.getParallelUnit() == ParallelUnit::NotParallel && !isScalar(temporary.getType())) {
      temporaryHoisted = true;
    }
  }

  if (!temporaryHoisted) {
    temporaryValuesInitFree = codeToInitializeTemporary(where);
  }

  Stmt initializeTemporary = temporaryValuesInitFree[0];
  Stmt freeTemporary = temporaryValuesInitFree[1];

  match(where.getConsumer(),
        std::function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
            if (op->lhs.getTensorVar().getOrder() > 0) {
              whereTempsToResult[where.getTemporary()] = (const AccessNode *) op->lhs.ptr;
            }
        })
  );

  Stmt consumer = lower(where.getConsumer());
  if (accelerateDenseWorkSpace && sortAccelerator) {
    // We need to sort the indices array
    Expr listOfIndices = tempToIndexList.at(temporary);
    Expr listOfIndicesSize = tempToIndexListSize.at(temporary);
    Expr sizeOfElt = ir::Sizeof::make(listOfIndices.type());
    Stmt sortCall = ir::Sort::make({listOfIndices, listOfIndicesSize, sizeOfElt});
    consumer = Block::make(sortCall, consumer);
  }

  // TODO (rohany): If the workspace is in shared memory, then every thread can do a write to the
  //  workspace, rather than emitting a loop to do the initialization. However, this has a pretty
  //  negligible performance impact.
  // Now that temporary allocations are hoisted, we always need to emit an initialization loop before entering the
  // producer but only if there is no dense acceleration
  if (util::contains(needCompute, temporary) && !isScalar(temporary.getType()) && !accelerateDenseWorkSpace) {
    // TODO: We only actually need to do this if:
    //      1) We use the temporary multiple times
    //      2) The PRODUCER RHS is sparse(not full). (Guarantees that old values are overwritten before consuming)

    Expr p = Var::make("p" + temporary.getName(), Int());
    auto values = this->temporaryArrays.at(temporary).values;
    Expr size = getTemporarySize(where);
    Stmt zeroInit = Store::make(values, p, ir::Literal::zero(temporary.getType().getDataType()));
    Stmt loopInit = For::make(p, 0, size, 1, zeroInit, LoopKind::Serial);
    initializeTemporary = Block::make(initializeTemporary, loopInit);
  }

  whereConsumers.push_back(consumer);
  whereTemps.push_back(where.getTemporary());
  captureNextLocatePos = true;

  // don't apply atomics to producer TODO: mark specific assignments as atomic
  bool restoreAtomicDepth = false;
  if (markAssignsAtomicDepth > 0) {
    markAssignsAtomicDepth--;
    restoreAtomicDepth = true;
  }

  Stmt producer = lower(where.getProducer());
  if (accelerateDenseWorkSpace) {
    const Expr indexListSizeExpr = tempToIndexListSize.at(temporary);
    const Stmt indexListSizeDecl = VarDecl::make(indexListSizeExpr, ir::Literal::make(0));
    initializeTemporary = Block::make(indexListSizeDecl, initializeTemporary);
  }

  if (restoreAtomicDepth) {
    markAssignsAtomicDepth++;
  }

  whereConsumers.pop_back();
  whereTemps.pop_back();
  whereTempsToResult.erase(where.getTemporary());
  return Block::make(initializeTemporary, producer, markAssignsAtomicDepth > 0 ? capturedLocatePos : ir::Stmt(), consumer,  freeTemporary);
}


Stmt LowererImpl::lowerSequence(Sequence sequence) {
  Stmt definition = lower(sequence.getDefinition());
  Stmt mutation = lower(sequence.getMutation());
  return Block::make(definition, mutation);
}


Stmt LowererImpl::lowerAssemble(Assemble assemble) {
  Stmt queries, freeQueryResults;
  if (generateAssembleCode() && assemble.getQueries().defined()) {
    std::vector<Stmt> allocStmts, freeStmts;
    const auto queryAccesses = getResultAccesses(assemble.getQueries()).first;
    for (const auto& queryAccess : queryAccesses) {
      const auto queryResult = queryAccess.getTensorVar();
      Expr values = ir::Var::make(queryResult.getName(),
                                  queryResult.getType().getDataType(),
                                  true, false);

      TemporaryArrays arrays;
      arrays.values = values;
      this->temporaryArrays.insert({queryResult, arrays});

      // Compute size of query result
      const auto indexVars = queryAccess.getIndexVars();
      taco_iassert(util::all(indexVars,
          [&](const auto& var) { return provGraph.isUnderived(var); }));
      Expr size = 1;
      for (const auto& indexVar : indexVars) {
        size = ir::Mul::make(size, getDimension(indexVar));
      }

      multimap<IndexVar, Iterator> readIterators;
      for (auto& read : getArgumentAccesses(assemble.getQueries())) {
        for (auto& readIterator : getIterators(read)) {
          for (auto& underivedAncestor :
              provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
            readIterators.insert({underivedAncestor, readIterator});
          }
        }
      }
      const auto writeIterators = getIterators(queryAccess);
      const bool zeroInit = hasSparseInserts(writeIterators, readIterators);
      if (zeroInit) {
        Expr sizeOfElt = Sizeof::make(queryResult.getType().getDataType());
        auto ptrType = Pointer(queryResult.getType().getDataType());
        Expr callocValues = ir::Cast::make(ir::Call::make("calloc", {size, sizeOfElt},
                                           queryResult.getType().getDataType()), ptrType);
        Stmt allocResult = VarDecl::make(values, callocValues);
        allocStmts.push_back(allocResult);
      }
      else {
        Stmt declResult = VarDecl::make(values, 0);
        allocStmts.push_back(declResult);

        Stmt allocResult = Allocate::make(values, size);
        allocStmts.push_back(allocResult);
      }

      Stmt freeResult = Free::make(values);
      freeStmts.push_back(freeResult);
    }
    Stmt allocResults = Block::make(allocStmts);
    freeQueryResults = Block::make(freeStmts);

    queries = lower(assemble.getQueries());
    queries = Block::blanks(allocResults, queries);
  }

  const auto& queryResults = assemble.getAttrQueryResults();
  const auto resultAccesses = getResultAccesses(assemble.getCompute()).first;

  std::vector<Stmt> initAssembleStmts;
  for (const auto& resultAccess : resultAccesses) {
    Expr prevSize = 1;
    std::vector<Expr> coords;
    const auto resultIterators = getIterators(resultAccess);
    const auto resultTensor = resultAccess.getTensorVar();
    const auto resultTensorVar = getTensorVar(resultTensor);
    const auto resultModeOrdering = resultTensor.getFormat().getModeOrdering();
    for (const auto& resultIterator : resultIterators) {
      if (generateAssembleCode()) {
        const size_t resultLevel = resultIterator.getMode().getLevel() - 1;
        const auto queryResultVars = queryResults.at(resultTensor)[resultLevel];
        std::vector<AttrQueryResult> queryResults;
        for (const auto& queryResultVar : queryResultVars) {
          queryResults.emplace_back(getTensorVar(queryResultVar),
                                    getValuesArray(queryResultVar));
        }

        if (resultIterator.hasSeqInsertEdge()) {
          Stmt initEdges = resultIterator.getSeqInitEdges(prevSize, queryResults);
          initAssembleStmts.push_back(initEdges);

          Stmt insertEdgeLoop = resultIterator.getSeqInsertEdge(
              resultIterator.getParent().getPosVar(), coords, queryResults);
          auto locateCoords = coords;
          for (auto iter = resultIterator.getParent(); !iter.isRoot();
               iter = iter.getParent()) {
            if (iter.hasLocate()) {
              Expr dim = GetProperty::make(resultTensorVar,
                  TensorProperty::Dimension,
                  resultModeOrdering[iter.getMode().getLevel() - 1]);
              Expr pos = iter.getPosVar();
              Stmt initPos = VarDecl::make(pos, iter.locate(locateCoords)[0]);
              insertEdgeLoop = For::make(coords.back(), 0, dim, 1,
                                         Block::make(initPos, insertEdgeLoop));
            } else {
              taco_not_supported_yet;
            }
            locateCoords.pop_back();
          }
          initAssembleStmts.push_back(insertEdgeLoop);
        }

        Stmt initCoords = resultIterator.getInitCoords(prevSize, queryResults);
        initAssembleStmts.push_back(initCoords);
      }

      Stmt initYieldPos = resultIterator.getInitYieldPos(prevSize);
      initAssembleStmts.push_back(initYieldPos);

      prevSize = resultIterator.getAssembledSize(prevSize);
      coords.push_back(getCoordinateVar(resultIterator));
    }

    if (generateAssembleCode()) {
      // TODO: call calloc if not compact or not unpadded
      auto resultTensorExpr = this->getTensorVar(resultTensor);
      Expr valuesArr = getValuesArray(resultTensor);
      if (this->legion) {
        taco_iassert(this->valuesAnalyzer.getValuesDim(resultTensor) == 1);
        auto valuesReg = ir::GetProperty::make(resultTensorExpr, TensorProperty::Values);
        auto valuesParent = ir::GetProperty::make(resultTensorExpr, TensorProperty::ValuesParent);
        auto allocValues = ir::makeLegionMalloc(valuesReg, prevSize, valuesParent, fidVal, readWrite);
        auto newAcc = ir::makeCreateAccessor(valuesArr, valuesReg, fidVal);
        initAssembleStmts.push_back(allocValues);
        initAssembleStmts.push_back(ir::Assign::make(valuesArr, newAcc));
        // Finally, we need to perform the subregion cast for the values array here.
        auto field = ir::FieldAccess::make(resultTensorExpr, "vals", true /* isDeref*/, Auto);
        auto subreg = ir::Call::make("getSubRegion", {ir::ctx, ir::runtime, valuesParent,
                                                      ir::makeConstructor(Rect(1), {0, ir::Sub::make(prevSize, 1)})}, Auto);
        initAssembleStmts.push_back(ir::Assign::make(field, subreg));
      } else {
        Stmt initValues = Allocate::make(valuesArr, prevSize);
        initAssembleStmts.push_back(initValues);
      }
    }
  }
  Stmt initAssemble = Block::make(initAssembleStmts);

  guardedTemps = util::toSet(getTemporaries(assemble.getCompute()));
  Stmt compute = lower(assemble.getCompute());

  std::vector<Stmt> finalizeAssembleStmts;
  for (const auto& resultAccess : resultAccesses) {
    Expr prevSize = 1;
    const auto resultIterators = getIterators(resultAccess);
    for (const auto& resultIterator : resultIterators) {
      Stmt finalizeYieldPos = resultIterator.getFinalizeYieldPos(prevSize);
      finalizeAssembleStmts.push_back(finalizeYieldPos);

      prevSize = resultIterator.getAssembledSize(prevSize);
    }
  }
  Stmt finalizeAssemble = Block::make(finalizeAssembleStmts);

  return Block::blanks(queries,
                       initAssemble,
                       compute,
                       finalizeAssemble,
                       freeQueryResults);
}


Stmt LowererImpl::lowerMulti(Multi multi) {
  Stmt stmt1 = lower(multi.getStmt1());
  Stmt stmt2 = lower(multi.getStmt2());
  return Block::make(stmt1, stmt2);
}

Stmt LowererImpl::lowerSuchThat(SuchThat suchThat) {
  auto scalls = suchThat.getCalls();
  this->calls.insert(scalls.begin(), scalls.end());
  Stmt stmt = lower(suchThat.getStmt());
  return Block::make(stmt);
}


Expr LowererImpl::lowerAccess(Access access) {
  if (access.isAccessingStructure()) {
    return true;
  }

  TensorVar var = access.getTensorVar();

  if (isScalar(var.getType())) {
    return getTensorVar(var);
  }

  if (!getIterators(access).back().isUnique()) {
    return getReducedValueVar(access);
  }

  if (var.getType().getDataType() == Bool &&
      getIterators(access).back().isZeroless())  {
    return true;
  } 

  const auto vals = getValuesArray(var);
  if (!vals.defined()) {
    return true;
  }

  return Load::make(vals, generateValueLocExpr(access));
}


Expr LowererImpl::lowerLiteral(Literal literal) {
  switch (literal.getDataType().getKind()) {
    case Datatype::Bool:
      return ir::Literal::make(literal.getVal<bool>());
    case Datatype::UInt8:
      return ir::Literal::make((unsigned long long)literal.getVal<uint8_t>());
    case Datatype::UInt16:
      return ir::Literal::make((unsigned long long)literal.getVal<uint16_t>());
    case Datatype::UInt32:
      return ir::Literal::make((unsigned long long)literal.getVal<uint32_t>());
    case Datatype::UInt64:
      return ir::Literal::make((unsigned long long)literal.getVal<uint64_t>());
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      return ir::Literal::make((int)literal.getVal<int8_t>());
    case Datatype::Int16:
      return ir::Literal::make((int)literal.getVal<int16_t>());
    case Datatype::Int32:
      return ir::Literal::make((int)literal.getVal<int32_t>());
    case Datatype::Int64:
      return ir::Literal::make((long long)literal.getVal<int64_t>());
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      return ir::Literal::make(literal.getVal<float>());
    case Datatype::Float64:
      return ir::Literal::make(literal.getVal<double>());
    case Datatype::Complex64:
      return ir::Literal::make(literal.getVal<std::complex<float>>());
    case Datatype::Complex128:
      return ir::Literal::make(literal.getVal<std::complex<double>>());
    case Datatype::CppType:
      taco_unreachable;
      break;
    case Datatype::Undefined:
      taco_unreachable;
      break;
  }
  return ir::Expr();
}


Expr LowererImpl::lowerNeg(Neg neg) {
  return ir::Neg::make(lower(neg.getA()));
}


Expr LowererImpl::lowerAdd(Add add) {
  Expr a = lower(add.getA());
  Expr b = lower(add.getB());
  return (add.getDataType().getKind() == Datatype::Bool)
         ? ir::Or::make(a, b) : ir::Add::make(a, b);
}


Expr LowererImpl::lowerSub(Sub sub) {
  return ir::Sub::make(lower(sub.getA()), lower(sub.getB()));
}


Expr LowererImpl::lowerMul(Mul mul) {
  Expr a = lower(mul.getA());
  Expr b = lower(mul.getB());
  return (mul.getDataType().getKind() == Datatype::Bool)
         ? ir::And::make(a, b) : ir::Mul::make(a, b);
}


Expr LowererImpl::lowerDiv(Div div) {
  return ir::Div::make(lower(div.getA()), lower(div.getB()));
}


Expr LowererImpl::lowerSqrt(Sqrt sqrt) {
  return ir::Sqrt::make(lower(sqrt.getA()));
}


Expr LowererImpl::lowerCast(Cast cast) {
  return ir::Cast::make(lower(cast.getA()), cast.getDataType());
}


Expr LowererImpl::lowerCallIntrinsic(CallIntrinsic call) {
  std::vector<Expr> args;
  for (auto& arg : call.getArgs()) {
    args.push_back(lower(arg));
  }
  return call.getFunc().lower(args);
}


Stmt LowererImpl::lower(IndexStmt stmt) {
  return visitor->lower(stmt);
}


Expr LowererImpl::lower(IndexExpr expr) {
  return visitor->lower(expr);
}


bool LowererImpl::generateAssembleCode() const {
  return this->assemble;
}


bool LowererImpl::generateComputeCode() const {
  return this->compute;
}


Expr LowererImpl::getTensorVar(TensorVar tensorVar) const {
  taco_iassert(util::contains(this->tensorVars, tensorVar)) << tensorVar;
  return this->tensorVars.at(tensorVar);
}


Expr LowererImpl::getCapacityVar(Expr tensor) const {
  taco_iassert(util::contains(this->capacityVars, tensor)) << tensor;
  return this->capacityVars.at(tensor);
}


ir::Expr LowererImpl::getValuesArray(TensorVar var, bool exclusive) const
{
  if (this->legion && util::contains(this->legionTensors, var)) {
    auto valuesDim = this->valuesAnalyzer.getValuesDim(var);
    // TODO (rohany): Hackingly including the size as the mode here.
    if (util::contains(this->resultTensors, var)) {
      if (this->performingLegionReduction) {
        if (exclusive) {
          return GetProperty::make(getTensorVar(var), TensorProperty::ValuesReductionAccessor, valuesDim);
        } else {
          return GetProperty::make(getTensorVar(var), TensorProperty::ValuesReductionNonExclusiveAccessor, valuesDim);
        }
      }
      return GetProperty::make(getTensorVar(var), TensorProperty::ValuesWriteAccessor, valuesDim);
    } else {
      return GetProperty::make(getTensorVar(var), TensorProperty::ValuesReadAccessor, valuesDim);
    }
  } else {
    return (util::contains(temporaryArrays, var))
           ? temporaryArrays.at(var).values
           : GetProperty::make(getTensorVar(var), TensorProperty::Values);
  }
}


Expr LowererImpl::getDimension(IndexVar indexVar) const {
  taco_iassert(util::contains(this->dimensions, indexVar)) << indexVar;
  return this->dimensions.at(indexVar);
}


std::vector<Iterator> LowererImpl::getIterators(Access access) const {
  vector<Iterator> result;
  TensorVar tensor = access.getTensorVar();
  for (int i = 0; i < tensor.getOrder(); i++) {
    int mode = tensor.getFormat().getModeOrdering()[i];
    result.push_back(iterators.getLevelIteratorByModeAccess(ModeAccess(access, mode+1)));
  }
  return result;
}


set<Access> LowererImpl::getExhaustedAccesses(MergePoint point,
                                              MergeLattice lattice) const
{
  set<Access> exhaustedAccesses;
  for (auto& iterator : lattice.exhausted(point)) {
    exhaustedAccesses.insert(iterators.modeAccess(iterator).getAccess());
  }
  return exhaustedAccesses;
}


Expr LowererImpl::getReducedValueVar(Access access) const {
  return this->reducedValueVars.at(access);
}


Expr LowererImpl::getCoordinateVar(IndexVar indexVar) const {
  return this->iterators.modeIterator(indexVar).getCoordVar();
}


Expr LowererImpl::getCoordinateVar(Iterator iterator) const {
  if (iterator.isDimensionIterator()) {
    return iterator.getCoordVar();
  }
  return this->getCoordinateVar(iterator.getIndexVar());
}


vector<Expr> LowererImpl::coordinates(Iterator iterator) const {
  taco_iassert(iterator.defined());

  vector<Expr> coords;
  do {
    coords.push_back(getCoordinateVar(iterator));
    iterator = iterator.getParent();
  } while (!iterator.isRoot());
  auto reverse = util::reverse(coords);
  return vector<Expr>(reverse.begin(), reverse.end());
}

vector<Expr> LowererImpl::coordinates(vector<Iterator> iterators)
{
  taco_iassert(all(iterators, [](Iterator iter){ return iter.defined(); }));
  vector<Expr> result;
  for (auto& iterator : iterators) {
    result.push_back(iterator.getCoordVar());
  }
  return result;
}


Stmt LowererImpl::initResultArrays(vector<Access> writes,
                                   vector<Access> reads,
                                   set<Access> reducedAccesses) {
  multimap<IndexVar, Iterator> readIterators;
  for (auto& read : reads) {
    for (auto& readIterator : getIterators(read)) {
      for (auto& underivedAncestor : provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
        readIterators.insert({underivedAncestor, readIterator});
      }
    }
  }

  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0 ||
        isAssembledByUngroupedInsertion(write.getTensorVar())) {
      continue;
    }

    std::vector<Stmt> initArrays;

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());

    Expr tensor = getTensorVar(write.getTensorVar());
    Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
    Expr valuesParent = GetProperty::make(tensor, TensorProperty::ValuesParent);
    bool clearValuesAllocation = false;

    Expr parentSize = 1;
    if (generateAssembleCode()) {
      for (const auto& iterator : iterators) {
        Expr size;
        Stmt init;
        // Initialize data structures for storing levels
        if (iterator.hasAppend()) {
          size = 0;
          init = iterator.getAppendInitLevel(parentSize, size);
        } else if (iterator.hasInsert()) {
          size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
          init = iterator.getInsertInitLevel(parentSize, size);
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        initArrays.push_back(init);

        // Declare position variable of append modes that are not above a
        // branchless mode (if mode below is branchless, then can share same
        // position variable)
        if (iterator.hasAppend() && (iterator.isLeaf() ||
            !iterator.getChild().isBranchless())) {
          initArrays.push_back(VarDecl::make(iterator.getPosVar(), 0));
        }

        parentSize = size;
        // Writes into a windowed iterator require the allocation to be cleared.
        clearValuesAllocation |= (iterator.isWindowed() || iterator.hasIndexSet());
      }

      // Pre-allocate memory for the value array if computing while assembling
      if (generateComputeCode()) {
        taco_iassert(!iterators.empty());

        Expr capacityVar = getCapacityVar(tensor);
        Expr allocSize = isValue(parentSize, 0)
                         ? DEFAULT_ALLOC_SIZE : parentSize;
        initArrays.push_back(VarDecl::make(capacityVar, allocSize));
        if (this->legion) {
          // TODO (rohany): This allocation should be scaled so that it doesn't allocate too much
          //  memory for a multi-dimensional allocation.
          // Allocate a new values array, and update the accessor.
          initArrays.push_back(makeLegionMalloc(valuesArr, capacityVar, valuesParent, fidVal, readWrite));
          auto valsAcc = this->getValuesArray(write.getTensorVar());
          auto newValsAcc = ir::Call::make("createAccessor<" + ir::accessorTypeString(valsAcc) + ">", {valuesArr, fidVal}, Auto);
          initArrays.push_back(ir::Assign::make(valsAcc, newValsAcc));
        } else {
          initArrays.push_back(Allocate::make(valuesArr, capacityVar, false /* is_realloc */, Expr() /* old_elements */,
                                              clearValuesAllocation));
        }
      }

      taco_iassert(!initArrays.empty());
      result.push_back(Block::make(initArrays));
    }
    else if (generateComputeCode()) {
      Iterator lastAppendIterator;
      // Compute size of values array
      for (auto& iterator : iterators) {
        if (iterator.hasAppend()) {
          lastAppendIterator = iterator;
          parentSize = iterator.getSize(parentSize);
        } else if (iterator.hasInsert()) {
          parentSize = ir::Mul::make(parentSize, iterator.getWidth());
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        parentSize = simplify(parentSize);
      }

      // Declare position variable for the last append level
      if (lastAppendIterator.defined()) {
        result.push_back(VarDecl::make(lastAppendIterator.getPosVar(), 0));
      }
    }

    if (generateComputeCode() && iterators.back().hasInsert() &&
        !isValue(parentSize, 0) &&
        (hasSparseInserts(iterators, readIterators) ||
         util::contains(reducedAccesses, write))) {
      // Zero-initialize values array if size statically known and might not
      // assign to every element in values array during compute
      // TODO: Right now for scheduled code we check if any iterator is not full and then emit
      // a zero-initialization loop. We only actually need a zero-initialization loop if the combined
      // iteration of all the iterators is not full. We can check this by seeing if we can recover a
      // full iterator from our set of iterators.
      // TODO (rohany): This call to zeroInitValues will need to be updated once we support
      //  parallel assembly of sparse tensors (that have formats like {Sparse, Dense}. We
      //  currently don't allow parallel assembly of any sort of sparse tensors, so this
      //  doesn't need any change right now.
      Expr size = generateAssembleCode() ? getCapacityVar(tensor) : parentSize;
      result.push_back(zeroInitValues(tensor, 0, size));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}


ir::Stmt LowererImpl::finalizeResultArrays(std::vector<Access> writes) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  bool clearValuesAllocation = false;
  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0 ||
        isAssembledByUngroupedInsertion(write.getTensorVar())) {
      continue;
    }

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());

    Expr parentSize = 1;
    // Maintain the deepest sparse level (i.e. doesn't have insert). This will be used to
    // splice out the subregion for the values array, as the last sparse level will index
    // the first dimension of a multi-dimensional sparse values region.
    Iterator lastSparseLevel = Iterator();
    for (const auto& iterator : iterators) {
      Expr size;
      Stmt finalize;
      // Post-process data structures for storing levels
      if (iterator.hasAppend()) {
        size = iterator.getPosVar();
        auto lastSparseLevelPos = lastSparseLevel.defined() ? lastSparseLevel.getPosVar() : Expr();
        finalize = iterator.getAppendFinalizeLevel(lastSparseLevelPos, parentSize, size);
        lastSparseLevel = iterator;
      } else if (iterator.hasInsert()) {
        size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
        finalize = iterator.getInsertFinalizeLevel(parentSize, size);
      } else {
        taco_ierror << "Write iterator supports neither append nor insert";
      }
      result.push_back(finalize);
      parentSize = size;
      // Writes into a windowed iterator require the allocation to be cleared.
      clearValuesAllocation |= (iterator.isWindowed() || iterator.hasIndexSet());
    }
    // Set the values array to the subregion of the values.
    // TODO (rohany): There are definitely cases where we don't need to do this, but I'm not
    //  worried about them yet.
    Expr tensor = getTensorVar(write.getTensorVar());
    Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
    Expr valuesParent = GetProperty::make(tensor, TensorProperty::ValuesParent);
    if (this->legion) {
      auto field = ir::FieldAccess::make(this->tensorVars[write.getTensorVar()], "vals", true /* isDeref*/, Auto);
      // If the vals region is multi-dimensional, then the deepest sparse level will
      // index into the first dimension of it, so we always want to use that instead
      // of the multiplication-based parentSize.
      auto subreg = ir::Call::make("getSubRegion", {ir::ctx, ir::runtime, valuesParent,
                                                    ir::makeConstructor(Rect(1), {0, ir::Sub::make(lastSparseLevel.getPosVar(), 1)})}, Auto);
      result.push_back(ir::Assign::make(field, subreg));
    }

    if (!generateComputeCode()) {
      // Allocate memory for values array after assembly if not also computing
      if (this->legion) {
        result.push_back(makeLegionMalloc(valuesArr, parentSize, valuesArr, fidVal, readWrite));
      } else {
        result.push_back(Allocate::make(valuesArr, parentSize, false /* is_realloc */, Expr() /* old_elements */,
                                            clearValuesAllocation));
      }
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}

Stmt LowererImpl::defineScalarVariable(TensorVar var, bool zero) {
  Datatype type = var.getType().getDataType();
  Expr varValueIR = Var::make(var.getName() + "_val", type, false, false);
  Expr init = (zero) ? ir::Literal::zero(type)
                     : Load::make(GetProperty::make(tensorVars.at(var),
                                                    TensorProperty::Values));
  tensorVars.find(var)->second = varValueIR;
  return VarDecl::make(varValueIR, init);
}

static
vector<Iterator> getIteratorsFrom(IndexVar var,
                                  const vector<Iterator>& iterators) {
  vector<Iterator> result;
  bool found = false;
  for (Iterator iterator : iterators) {
    if (var == iterator.getIndexVar()) found = true;
    if (found) {
      result.push_back(iterator);
    }
  }
  return result;
}


Stmt LowererImpl::initResultArrays(IndexVar var, vector<Access> writes,
                                   vector<Access> reads,
                                   set<Access> reducedAccesses) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  multimap<IndexVar, Iterator> readIterators;
  for (auto& read : reads) {
    for (auto& readIterator : getIteratorsFrom(var, getIterators(read))) {
      for (auto& underivedAncestor : provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
        readIterators.insert({underivedAncestor, readIterator});
      }
    }
  }

  vector<Stmt> result;
  for (auto& write : writes) {
    Expr tensor = getTensorVar(write.getTensorVar());
    Expr values = GetProperty::make(tensor, TensorProperty::Values);
    Expr valuesParent = GetProperty::make(tensor, TensorProperty::ValuesParent);

    vector<Iterator> iterators = getIteratorsFrom(var, getIterators(write));

    if (iterators.empty()) {
      continue;
    }

    Iterator resultIterator = iterators.front();

    // Initialize begin var.
    // TODO (rohany): This should not be generated in all situations (in particular
    //  it seems to be incorrect when there is a transpose in the input) for code
    //  that assembles sparse tensors.
    if (resultIterator.hasAppend() && !resultIterator.isBranchless()) {
      Expr begin = resultIterator.getBeginVar();
      result.push_back(VarDecl::make(begin, resultIterator.getPosVar()));
    }

    const bool isTopLevel = (iterators.size() == write.getIndexVars().size());
    if (resultIterator.getParent().hasAppend() || isTopLevel) {
      Expr resultParentPos = resultIterator.getParent().getPosVar();
      Expr resultParentPosNext = simplify(ir::Add::make(resultParentPos, 1));
      Expr initBegin = resultParentPos;
      Expr initEnd = resultParentPosNext;
      Expr stride = 1;

      Iterator initIterator;
      for (Iterator iterator : iterators) {
        if (!iterator.hasInsert()) {
          initIterator = iterator;
          break;
        }

        stride = simplify(ir::Mul::make(stride, iterator.getWidth()));
        initBegin = simplify(ir::Mul::make(resultParentPos, stride));
        initEnd = simplify(ir::Mul::make(resultParentPosNext, stride));

        // Initialize data structures for storing insert mode
        result.push_back(iterator.getInsertInitCoords(initBegin, initEnd));
      }

      if (initIterator.defined()) {
        // Initialize data structures for storing edges of next append mode
        taco_iassert(initIterator.hasAppend());
        result.push_back(initIterator.getAppendInitEdges(resultParentPos, initBegin, initEnd));
      } else if (generateComputeCode() && !isTopLevel) {
        if (isa<ir::Mul>(stride)) {
          Expr strideVar = Var::make(util::toString(tensor) + "_stride", Int());
          result.push_back(VarDecl::make(strideVar, stride));
          stride = strideVar;
        }

        // Resize values array if not large enough
        Expr capacityVar = getCapacityVar(tensor);
        Expr size = simplify(ir::Mul::make(resultParentPosNext, stride));

        if (this->legion) {
          auto valsAcc = this->getValuesArray(write.getTensorVar());
          auto newValsAcc = ir::Call::make("createAccessor<" + ir::accessorTypeString(valsAcc) + ">", {values, fidVal}, Auto);
          // resultParentPos is the position variable of the deepest sparse level
          // above the values array. This position will always be the first index
          // into the values array, so we use that instead of the multiplication-based
          // size variable.
          result.push_back(lgAtLeastDoubleSizeIfFull(values, capacityVar, resultParentPos, valuesParent, values, fidVal, ir::Assign::make(valsAcc, newValsAcc), readWrite));
        } else {
          result.push_back(atLeastDoubleSizeIfFull(values, capacityVar, size));
        }

        if (hasSparseInserts(iterators, readIterators) ||
            util::contains(reducedAccesses, write)) {
          // Zero-initialize values array if might not assign to every element
          // in values array during compute
          if (this->legion) {
            result.push_back(this->lgZeroInitValues(write));
          } else {
            result.push_back(zeroInitValues(tensor, resultParentPos, stride));
          }
        }
      }
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::resizeAndInitValues(const std::vector<Iterator>& appenders,
                                      const std::set<Access>& reducedAccesses) {
  if (!generateComputeCode()) {
    return Stmt();
  }

  std::function<Expr(Access)> getTensor = [&](Access access) {
    return getTensorVar(access.getTensorVar());
  };
  const auto reducedTensors = util::map(reducedAccesses, getTensor);

  std::vector<Stmt> result;

  for (auto& appender : appenders) {
    if (!appender.isLeaf()) {
      continue;
    }

    Expr tensor = appender.getTensor();
    // Get the tensorVar from a reverse map lookup.
    TensorVar tv;
    for (auto it : this->tensorVars) {
      if (it.second == tensor) {
        tv = it.first;
        break;
      }
    }
    taco_iassert(tv.defined());
    Expr values = GetProperty::make(tensor, TensorProperty::Values);
    Expr valuesParent = GetProperty::make(tensor, TensorProperty::ValuesParent);
    Expr capacity = getCapacityVar(appender.getTensor());
    Expr pos = appender.getIteratorVar();

    if (generateAssembleCode()) {
      if (this->legion) {
        auto valsAcc = this->getValuesArray(tv);
        auto newValsAcc = ir::Call::make("createAccessor<" + ir::accessorTypeString(valsAcc) + ">", {values, fidVal}, Auto);
        result.push_back(lgDoubleSizeIfFull(values, capacity, pos, valuesParent, values, fidVal, ir::Assign::make(valsAcc, newValsAcc), readWrite));
      } else {
        result.push_back(doubleSizeIfFull(values, capacity, pos));
      }
    }

    if (util::contains(reducedTensors, tensor)) {
      Expr zero = ir::Literal::zero(tensor.type());
      result.push_back(Store::make(values, pos, zero));
    }
  }

  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::zeroInitValues(Expr tensor, Expr begin, Expr size) {
  Expr lower = simplify(ir::Mul::make(begin, size));
  Expr upper = simplify(ir::Mul::make(ir::Add::make(begin, 1), size));
  Expr p = Var::make("p" + util::toString(tensor), Int());
  Expr values = GetProperty::make(tensor, TensorProperty::Values);
  Stmt zeroInit = Store::make(values, p, ir::Literal::zero(tensor.type()));
  LoopKind parallel = (isa<ir::Literal>(size) &&
                       to<ir::Literal>(size)->getIntValue() < (1 << 10))
                      ? LoopKind::Serial : LoopKind::Static_Chunked;
  if (this->loweringToGPU() && util::contains(parallelUnitSizes, ParallelUnit::GPUBlock)) {
    return ir::VarDecl::make(ir::Var::make("status", Int()),
                                    ir::Call::make("cudaMemset", {values, ir::Literal::make(0, Int()), ir::Mul::make(ir::Sub::make(upper, lower), ir::Literal::make(values.type().getNumBytes()))}, Int()));
  }
  return For::make(p, lower, upper, 1, zeroInit, parallel);
}

Stmt LowererImpl::lgZeroInitValues(const Access& acc) {
  auto tv = acc.getTensorVar();
  auto tvIR = this->getTensorVar(tv);
  auto accessVars = this->valuesAnalyzer.valuesAccess.at(acc);
  // TODO (rohany): I could avoid having this code by getting the value regions
  //  bounds in a header and then referencing that variable later.
  auto accessWidths = this->valuesAnalyzer.valuesAccessWidths.at(acc);
  // Skip the first variable, as that is fixed and depends on the position in
  // the sparse level above us (if exists). We need to initialize
  // all of the values for a particular value of accessVars[0].
  auto values = this->getValuesArray(tv);
  auto body = ir::Store::make(values, this->valuesAnalyzer.getAccessPoint(acc), ir::Literal::zero(tvIR.type()));
  for (size_t i = accessVars.size() - 1; i >= 1; i--) {
    body = ir::For::make(accessVars[i], 0, accessWidths[i], 1, body);
  }
  return body;
}

std::vector<IndexVar> getIndexVarFamily(const Iterator& it) {
  if (it.isRoot() || it.getMode().getLevel() == 1) {
    return {it.getIndexVar()};
  }
//  std::vector<IndexVar> result;
  auto rcall = getIndexVarFamily(it.getParent());
  rcall.push_back(it.getIndexVar());
  return rcall;
}

Stmt LowererImpl::declLocatePosVars(vector<Iterator> locators) {
  vector<Stmt> result;
  for (Iterator& locator : locators) {
    accessibleIterators.insert(locator);

    bool doLocate = true;
    for (Iterator ancestorIterator = locator.getParent();
         !ancestorIterator.isRoot() && ancestorIterator.hasLocate();
         ancestorIterator = ancestorIterator.getParent()) {
      if (!accessibleIterators.contains(ancestorIterator)) {
        doLocate = false;
      }
    }

    if (doLocate) {
      Iterator locateIterator = locator;
      if (locateIterator.hasPosIter()) {
        taco_iassert(!provGraph.isUnderived(locateIterator.getIndexVar()));
        continue; // these will be recovered with separate procedure
      }
      do {
        auto coords = coordinates(locateIterator);
        // If this dimension iterator operates over a window, then it needs
        // to be projected up to the window's iteration space.
        if (locateIterator.isWindowed()) {
          auto expr = coords[coords.size() - 1];
          coords[coords.size() - 1] = this->projectCanonicalSpaceToWindowedPosition(locateIterator, expr);
        } else if (locateIterator.hasIndexSet()) {
          // If this dimension iterator operates over an index set, follow the
          // indirection by using the locator access the index set's crd array.
          // The resulting value is where we should locate into the actual tensor.
          auto expr = coords[coords.size() - 1];
          auto indexSetIterator = locateIterator.getIndexSetIterator();
          auto coordArray = indexSetIterator.posAccess(expr, coordinates(indexSetIterator)).getResults()[0];
          coords[coords.size() - 1] = coordArray;
        }
        ModeFunction locate = locateIterator.locate(coords);
        taco_iassert(isValue(locate.getResults()[1], true));
        Stmt declarePosVar = VarDecl::make(locateIterator.getPosVar(),
                                           locate.getResults()[0]);
        result.push_back(declarePosVar);

        if (locateIterator.isLeaf()) {
          break;
        }
        locateIterator = locateIterator.getChild();
      } while (accessibleIterators.contains(locateIterator));
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::reduceDuplicateCoordinates(Expr coordinate,
                                             vector<Iterator> iterators,
                                             bool alwaysReduce) {
  vector<Stmt> result;
  for (Iterator& iterator : iterators) {
    taco_iassert(!iterator.isUnique() && iterator.hasPosIter());

    Access access = this->iterators.modeAccess(iterator).getAccess();
    Expr iterVar = iterator.getIteratorVar();
    Expr segendVar = iterator.getSegendVar();
    Expr reducedVal = iterator.isLeaf() ? getReducedValueVar(access) : Expr();
    Expr tensorVar = getTensorVar(access.getTensorVar());
    Expr tensorVals = GetProperty::make(tensorVar, TensorProperty::Values);

    // Initialize variable storing reduced component value.
    if (reducedVal.defined()) {
      Expr reducedValInit = alwaysReduce
                          ? Load::make(tensorVals, iterVar)
                          : ir::Literal::zero(reducedVal.type());
      result.push_back(VarDecl::make(reducedVal, reducedValInit));
    }

    if (iterator.isLeaf()) {
      // If iterator is over bottommost coordinate hierarchy level and will
      // always advance (i.e., not merging with another iterator), then we don't
      // need a separate segend variable.
      segendVar = iterVar;
      if (alwaysReduce) {
        result.push_back(compoundAssign(segendVar, 1));
      }
    } else {
      Expr segendInit = alwaysReduce ? ir::Add::make(iterVar, 1) : iterVar;
      result.push_back(VarDecl::make(segendVar, segendInit));
    }

    vector<Stmt> dedupStmts;
    if (reducedVal.defined()) {
      Expr partialVal = Load::make(tensorVals, segendVar);
      dedupStmts.push_back(compoundAssign(reducedVal, partialVal));
    }
    dedupStmts.push_back(compoundAssign(segendVar, 1));
    Stmt dedupBody = Block::make(dedupStmts);

    ModeFunction posAccess = iterator.posAccess(segendVar,
                                                coordinates(iterator));
    // TODO: Support access functions that perform additional computations
    //       and/or might access invalid positions.
    taco_iassert(!posAccess.compute().defined());
    taco_iassert(to<ir::Literal>(posAccess.getResults()[1])->getBoolValue());
    Expr nextCoord = posAccess.getResults()[0];
    Expr withinBounds = Lt::make(segendVar, iterator.getEndVar());
    Expr isDuplicate = Eq::make(posAccess.getResults()[0], coordinate);
    result.push_back(While::make(And::make(withinBounds, isDuplicate),
                                 Block::make(dedupStmts)));
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImpl::codeToInitializeIteratorVar(Iterator iterator, vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> result;
  taco_iassert(iterator.hasPosIter() || iterator.hasCoordIter() ||
               iterator.isDimensionIterator());

  Expr iterVar = iterator.getIteratorVar();
  Expr endVar = iterator.getEndVar();
  if (iterator.hasPosIter()) {
    auto parentPositions = this->getAllNeededParentPositions(iterator);
    auto parentPos = parentPositions.back();
    if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
      // E.g. a compressed mode without duplicates
      ModeFunction bounds = iterator.posBounds(parentPositions);
      result.push_back(bounds.compute());
      // if has a coordinate ranger then need to binary search
      if (any(rangers,
              [](Iterator it){ return it.isDimensionIterator(); })) {

        Expr binarySearchTarget = provGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[coordinateVar][0];
        if (binarySearchTarget != underivedBounds[coordinateVar][0]) {
          // If we have a window, then we need to project up the binary search target
          // into the window rather than the beginning of the level.
          if (iterator.isWindowed()) {
            binarySearchTarget = this->projectCanonicalSpaceToWindowedPosition(iterator, binarySearchTarget);
          }
          result.push_back(VarDecl::make(iterator.getBeginVar(), binarySearchTarget));

          vector<Expr> binarySearchArgs = {
                  iterator.getMode().getModePack().getArray(1), // array
                  bounds[0], // arrayStart
                  bounds[1], // arrayEnd
                  iterator.getBeginVar() // target
          };
          result.push_back(
                  VarDecl::make(iterVar, Call::make("taco_binarySearchAfter", binarySearchArgs, iterVar.type())));
        }
        else {
          result.push_back(VarDecl::make(iterVar, bounds[0]));
        }
      }
      else {
        auto bound = bounds[0];
        // If we have a window on this iterator, then search for the start of
        // the window rather than starting at the beginning of the level.
        if (iterator.isWindowed()) {
            bound = this->searchForStartOfWindowPosition(iterator, bounds[0], bounds[1]);
        }
        result.push_back(VarDecl::make(iterVar, bound));
      }

      result.push_back(VarDecl::make(endVar, bounds[1]));
    } else {
      taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
      taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

      // E.g. a compressed mode with duplicates. Apply iterator chaining
      Expr parentSegend = iterator.getParent().getSegendVar();
      ModeFunction startBounds = iterator.posBounds(parentPos);
      ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
      result.push_back(startBounds.compute());
      result.push_back(VarDecl::make(iterVar, startBounds[0]));
      result.push_back(endBounds.compute());
      result.push_back(VarDecl::make(endVar, endBounds[1]));
    }
  }
  else if (iterator.hasCoordIter()) {
    // E.g. a hasmap mode
    vector<Expr> coords = coordinates(iterator);
    coords.erase(coords.begin());
    ModeFunction bounds = iterator.coordBounds(coords);
    result.push_back(bounds.compute());
    result.push_back(VarDecl::make(iterVar, bounds[0]));
    result.push_back(VarDecl::make(endVar, bounds[1]));
  }
  else if (iterator.isDimensionIterator()) {
    // A dimension
    // If a merger then initialize to 0
    // If not then get first coord value like doing normal merge

    // If derived then need to recoverchild from this coord value
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr coord = coordinates(vector<Iterator>({iterator}))[0];
      result.push_back(VarDecl::make(coord, 0));
    }
    else {
      result.push_back(codeToLoadCoordinatesFromPosIterators(iterators, true));

      Stmt stmt = resolveCoordinate(mergers, coordinate, true);
      taco_iassert(stmt != Stmt());
      result.push_back(stmt);
      result.push_back(codeToRecoverDerivedIndexVar(coordinateVar, iterator.getIndexVar(), true));

      // emit bound for ranger too
      vector<Expr> startBounds;
      vector<Expr> endBounds;
      for (Iterator merger : mergers) {
        ModeFunction coordBounds = merger.coordBounds(merger.getParent().getPosVar());
        startBounds.push_back(coordBounds[0]);
        endBounds.push_back(coordBounds[1]);
      }
      //TODO: maybe needed after split reorder? underivedBounds[coordinateVar] = {ir::Max::make(startBounds), ir::Min::make(endBounds)};
      Stmt end_decl = VarDecl::make(iterator.getEndVar(), provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[1]);
      result.push_back(end_decl);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImpl::codeToInitializeIteratorVars(vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> results;
  // initialize mergers first (can't depend on initializing rangers)
  for (Iterator iterator : mergers) {
    results.push_back(codeToInitializeIteratorVar(iterator, iterators, rangers, mergers, coordinate, coordinateVar));
  }

  for (Iterator iterator : rangers) {
      if (find(mergers.begin(), mergers.end(), iterator) == mergers.end()) {
        results.push_back(codeToInitializeIteratorVar(iterator, iterators, rangers, mergers, coordinate, coordinateVar));
      }
  }
  return results.empty() ? Stmt() : Block::make(results);
}

Stmt LowererImpl::codeToRecoverDerivedIndexVar(IndexVar underived, IndexVar indexVar, bool emitVarDecl) {
  if(underived != indexVar) {
    // iterator indexVar must be derived from coordinateVar
    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(indexVar);
    taco_iassert(find(underivedAncestors.begin(), underivedAncestors.end(), underived) != underivedAncestors.end());

    vector<Stmt> recoverySteps;
    for (const IndexVar& varToRecover : provGraph.derivationPath(underived, indexVar)) {
      if(varToRecover == underived) continue;
      recoverySteps.push_back(provGraph.recoverChild(varToRecover, indexVarToExprMap, emitVarDecl, iterators));
    }
    return Block::make(recoverySteps);
  }
  return Stmt();
}

Stmt LowererImpl::codeToIncIteratorVars(Expr coordinate, IndexVar coordinateVar, vector<Iterator> iterators, vector<Iterator> mergers) {
  if (iterators.size() == 1) {
    Expr ivar = iterators[0].getIteratorVar();

    if (iterators[0].isUnique()) {
      return compoundAssign(ivar, 1);
    }

    // If iterator is over bottommost coordinate hierarchy level with
    // duplicates and iterator will always advance (i.e., not merging with
    // another iterator), then deduplication loop will take care of
    // incrementing iterator variable.
    return iterators[0].isLeaf()
           ? Stmt()
           : Assign::make(ivar, iterators[0].getSegendVar());
  }

  vector<Stmt> result;

  // We emit the level iterators before the mode iterator because the coordinate
  // of the mode iterator is used to conditionally advance the level iterators.

  auto levelIterators =
      filter(iterators, [](Iterator it){return !it.isDimensionIterator();});
  for (auto& iterator : levelIterators) {
    Expr ivar = iterator.getIteratorVar();
    if (iterator.isUnique()) {
      Expr increment = iterator.isFull()
                     ? 1
                     : ir::Cast::make(Eq::make(iterator.getCoordVar(),
                                               coordinate),
                                      ivar.type());
      result.push_back(compoundAssign(ivar, increment));
    } else if (!iterator.isLeaf()) {
      result.push_back(Assign::make(ivar, iterator.getSegendVar()));
    }
  }

  auto modeIterators =
      filter(iterators, [](Iterator it){return it.isDimensionIterator();});
  for (auto& iterator : modeIterators) {
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr ivar = iterator.getIteratorVar();
      result.push_back(compoundAssign(ivar, 1));
    }
    else {
      result.push_back(codeToLoadCoordinatesFromPosIterators(iterators, false));
      Stmt stmt = resolveCoordinate(mergers, coordinate, false);
      taco_iassert(stmt != Stmt());
      result.push_back(stmt);
      result.push_back(codeToRecoverDerivedIndexVar(coordinateVar, iterator.getIndexVar(), false));
    }
  }

  return Block::make(result);
}

Stmt LowererImpl::codeToLoadCoordinatesFromPosIterators(vector<Iterator> iterators, bool declVars) {
  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates;
  if (iterators.size() > 1) {
    vector<Stmt> loadPosIterCoordinateStmts;
    auto posIters = filter(iterators, [](Iterator it){return it.hasPosIter();});
    for (auto& posIter : posIters) {
      taco_tassert(posIter.hasPosIter());
      ModeFunction posAccess = posIter.posAccess(posIter.getPosVar(),
                                                 coordinates(posIter));
      loadPosIterCoordinateStmts.push_back(posAccess.compute());
      auto access = posAccess[0];
      // If this iterator is windowed, then it needs to be projected down to
      // recover the coordinate variable.
      // TODO (rohany): Would be cleaner to have this logic be moved into the
      //  ModeFunction, rather than having to check in some places?
      if (posIter.isWindowed()) {

        // If the iterator is strided, then we have to skip over coordinates
        // that don't match the stride. To do that, we insert a guard on the
        // access. We first extract the access into a temp to avoid emitting
        // a duplicate load on the _crd array.
        if (posIter.isStrided()) {
          loadPosIterCoordinateStmts.push_back(VarDecl::make(posIter.getWindowVar(), access));
          access = posIter.getWindowVar();
          // Since we're locating into a compressed array (not iterating over it),
          // we need to advance the outer loop if the current coordinate is not
          // along the desired stride. So, we pass true to the incrementPosVar
          // argument of strideBoundsGuard.
          loadPosIterCoordinateStmts.push_back(this->strideBoundsGuard(posIter, access, true /* incrementPosVar */));
        }

        access = this->projectWindowedPositionToCanonicalSpace(posIter, access);
      }
      if (declVars) {
        loadPosIterCoordinateStmts.push_back(VarDecl::make(posIter.getCoordVar(), access));
      }
      else {
        loadPosIterCoordinateStmts.push_back(Assign::make(posIter.getCoordVar(), access));
      }
      if (posIter.isWindowed()) {
        loadPosIterCoordinateStmts.push_back(this->upperBoundGuardForWindowPosition(posIter, posIter.getCoordVar()));
      }
    }
    loadPosIterCoordinates = Block::make(loadPosIterCoordinateStmts);
  }
  return loadPosIterCoordinates;
}


static
bool isLastAppender(Iterator iter) {
  taco_iassert(iter.hasAppend());
  while (!iter.isLeaf()) {
    iter = iter.getChild();
    if (iter.hasAppend()) {
      return false;
    }
  }
  return true;
}


Stmt LowererImpl::appendCoordinate(vector<Iterator> appenders, Expr coord) {
  vector<Stmt> result;
  for (auto& appender : appenders) {
    Expr pos = appender.getPosVar();
    Iterator appenderChild = appender.getChild();

    if (appenderChild.defined() && appenderChild.isBranchless()) {
      // Already emitted assembly code for current level when handling
      // branchless child level, so don't emit code again.
      continue;
    }

    vector<Stmt> appendStmts;

    if (generateAssembleCode()) {
      appendStmts.push_back(appender.getAppendCoord(pos, coord));
      while (!appender.isRoot() && appender.isBranchless()) {
        // Need to append result coordinate to parent level as well if child
        // level is branchless (so child coordinates will have unique parents).
        appender = appender.getParent();
        if (!appender.isRoot()) {
          taco_iassert(appender.hasAppend()) << "Parent level of branchless, "
              << "append-capable level must also be append-capable";
          taco_iassert(!appender.isUnique()) << "Need to be able to insert "
              << "duplicate coordinates to level, but level is declared unique";

          Expr coord = getCoordinateVar(appender);
          appendStmts.push_back(appender.getAppendCoord(pos, coord));
        }
      }
    }

    if (generateAssembleCode() || isLastAppender(appender)) {
      appendStmts.push_back(compoundAssign(pos, 1));

      Stmt appendCode = Block::make(appendStmts);
      if (appenderChild.defined() && appenderChild.hasAppend()) {
        // Emit guard to avoid appending empty slices to result.
        // TODO: Users should be able to configure whether to append zeroes.
        Expr shouldAppend = Lt::make(appenderChild.getBeginVar(),
                                     appenderChild.getPosVar());
        appendCode = IfThenElse::make(shouldAppend, appendCode);
      }
      result.push_back(appendCode);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::generateAppendPositions(vector<Iterator> appenders) {
  vector<Stmt> result;
  if (generateAssembleCode()) {
    for (Iterator appender : appenders) {
      if (appender.isBranchless() || 
          isAssembledByUngroupedInsertion(appender.getTensor())) {
        continue;
      }

      Expr pos = [](Iterator appender) {
        // Get the position variable associated with the appender. If a mode
        // is above a branchless mode, then the two modes can share the same
        // position variable.
        while (!appender.isLeaf() && appender.getChild().isBranchless()) {
          appender = appender.getChild();
        }
        return appender.getPosVar();
      }(appender);
      Expr beginPos = appender.getBeginVar();
      auto parentPositions = this->getAllNeededParentPositions(appender);
      result.push_back(appender.getAppendEdges(parentPositions, beginPos, pos));
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Expr LowererImpl::generateValueLocExpr(Access access) const {
  if (isScalar(access.getTensorVar().getType())) {
    return ir::Literal::make(0);
  }
  // If using legion, return the PointT<...> accessor.
  if (this->legion && util::contains(this->legionTensors, access.getTensorVar())) {
    return this->valuesAnalyzer.getAccessPoint(access);
  }

  Iterator it = getIterators(access).back();

  // to make indexing temporary arrays with index var work correctly
  if (!provGraph.isUnderived(it.getIndexVar()) && !access.getIndexVars().empty() &&
      util::contains(indexVarToExprMap, access.getIndexVars().front()) &&
      !it.hasPosIter() && access.getIndexVars().front() == it.getIndexVar()) {
    return indexVarToExprMap.at(access.getIndexVars().front());
  }

  return it.getPosVar();
}


Expr LowererImpl::checkThatNoneAreExhausted(std::vector<Iterator> iterators)
{
  taco_iassert(!iterators.empty());
  if (iterators.size() == 1 && iterators[0].isFull()) {
    std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(iterators[0].getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators);
    Expr guards = Lt::make(iterators[0].getIteratorVar(), bounds[1]);
    if (bounds[0] != ir::Literal::make(0)) {
      guards = And::make(guards, Gte::make(iterators[0].getIteratorVar(), bounds[0]));
    }
    return guards;
  }

  vector<Expr> result;
  for (const auto& iterator : iterators) {
    taco_iassert(!iterator.isFull()) << iterator
        << " - full iterators do not need to partake in merge loop bounds";
    Expr iterUnexhausted = Lt::make(iterator.getIteratorVar(),
                                    iterator.getEndVar());
    result.push_back(iterUnexhausted);
  }

  return (!result.empty())
         ? taco::ir::conjunction(result)
         : Lt::make(iterators[0].getIteratorVar(), iterators[0].getEndVar());
}


Expr LowererImpl::generateAssembleGuard(IndexExpr expr) {
  class GenerateGuard : public IndexExprVisitorStrict {
  public:
    GenerateGuard(const std::set<TensorVar>& guardedTemps,
                  const std::map<TensorVar,Expr>& tempToGuard)
        : guardedTemps(guardedTemps), tempToGuard(tempToGuard) {}

    Expr lower(IndexExpr expr) {
      this->expr = Expr();
      IndexExprVisitorStrict::visit(expr);
      return this->expr;
    }

  private:
    Expr expr;
    const std::set<TensorVar>& guardedTemps;
    const std::map<TensorVar,Expr>& tempToGuard;

    using IndexExprVisitorStrict::visit;

    void visit(const AccessNode* node) {
      expr = (util::contains(guardedTemps, node->tensorVar) &&
              node->tensorVar.getOrder() == 0)
             ? tempToGuard.at(node->tensorVar) : true;
    }

    void visit(const LiteralNode* node) {
      expr = true;
    }

    void visit(const NegNode* node) {
      expr = lower(node->a);
    }

    void visit(const AddNode* node) {
      expr = Or::make(lower(node->a), lower(node->b));
    }

    void visit(const SubNode* node) {
      expr = Or::make(lower(node->a), lower(node->b));
    }

    void visit(const MulNode* node) {
      expr = And::make(lower(node->a), lower(node->b));
    }

    void visit(const DivNode* node) {
      expr = And::make(lower(node->a), lower(node->b));
    }

    void visit(const SqrtNode* node) {
      expr = lower(node->a);
    }

    void visit(const CastNode* node) {
      expr = lower(node->a);
    }

    void visit(const CallIntrinsicNode* node) {
      Expr ret = false;
      for (const auto& arg : node->args) {
        ret = Or::make(ret, lower(arg));
      }
      expr = ret;
    }

    void visit(const ReductionNode* node) {
      taco_ierror
          << "Reduction nodes not supported in concrete index notation";
    }
  };

  return ir::simplify(GenerateGuard(guardedTemps, tempToBitGuard).lower(expr));
}


bool LowererImpl::isAssembledByUngroupedInsertion(TensorVar result) {
  return util::contains(assembledByUngroupedInsert, result);
}


bool LowererImpl::isAssembledByUngroupedInsertion(Expr result) {
  for (const auto& tensor : assembledByUngroupedInsert) {
    if (getTensorVar(tensor) == result) {
      return true;
    }
  }
  return false;
}


bool LowererImpl::hasStores(Stmt stmt) {
  if (!stmt.defined()) {
    return false;
  }

  struct FindStores : IRVisitor {
    bool hasStore;
    const std::map<TensorVar, Expr>& tensorVars;
    const std::map<TensorVar, Expr>& tempToBitGuard;

    using IRVisitor::visit;

    FindStores(const std::map<TensorVar, Expr>& tensorVars,
               const std::map<TensorVar, Expr>& tempToBitGuard)
        : tensorVars(tensorVars), tempToBitGuard(tempToBitGuard) {}

    void visit(const Store* stmt) {
      hasStore = true;
    }

    void visit(const Assign* stmt) {
      for (const auto& tensorVar : tensorVars) {
        if (stmt->lhs == tensorVar.second) {
          hasStore = true;
          break;
        }
      }
      if (hasStore) {
        return;
      }
      for (const auto& bitGuard : tempToBitGuard) {
        if (stmt->lhs == bitGuard.second) {
          hasStore = true;
          break;
        }
      }
    }

    bool hasStores(Stmt stmt) {
      hasStore = false;
      stmt.accept(this);
      return hasStore;
    }
  };
  return FindStores(tensorVars, tempToBitGuard).hasStores(stmt);
}


Expr LowererImpl::searchForStartOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end) {
    taco_iassert(iterator.isWindowed());
    vector<Expr> args = {
            // Search over the `crd` array of the level,
            iterator.getMode().getModePack().getArray(1),
            // between the start and end position,
            start, end,
            // for the beginning of the window.
            iterator.getWindowLowerBound(),
    };
    return Call::make("taco_binarySearchAfter", args, Datatype::UInt64);
}


Expr LowererImpl::searchForEndOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end) {
    taco_iassert(iterator.isWindowed());
    vector<Expr> args = {
            // Search over the `crd` array of the level,
            iterator.getMode().getModePack().getArray(1),
            // between the start and end position,
            start, end,
            // for the end of the window.
            iterator.getWindowUpperBound(),
    };
    return Call::make("taco_binarySearchAfter", args, Datatype::UInt64);
}


Stmt LowererImpl::upperBoundGuardForWindowPosition(Iterator iterator, ir::Expr access) {
  taco_iassert(iterator.isWindowed());
  return ir::IfThenElse::make(
    ir::Gte::make(access, ir::Div::make(ir::Sub::make(iterator.getWindowUpperBound(), iterator.getWindowLowerBound()), iterator.getStride())),
    ir::Break::make()
  );
}


Stmt LowererImpl::strideBoundsGuard(Iterator iterator, ir::Expr access, bool incrementPosVar) {
  Stmt cont = ir::Continue::make();
  // If requested to increment the iterator's position variable, add the increment
  // before the continue statement.
  if (incrementPosVar) {
    cont = ir::Block::make({
                               ir::Assign::make(iterator.getPosVar(),
                                                ir::Add::make(iterator.getPosVar(), ir::Literal::make(1))),
                               cont
                           });
  }
  // The guard makes sure that the coordinate being accessed is along the stride.
  return ir::IfThenElse::make(
      ir::Neq::make(ir::Rem::make(ir::Sub::make(access, iterator.getWindowLowerBound()), iterator.getStride()), ir::Literal::make(0)),
      cont
  );
}


Expr LowererImpl::projectWindowedPositionToCanonicalSpace(Iterator iterator, ir::Expr expr) {
  return ir::Div::make(ir::Sub::make(expr, iterator.getWindowLowerBound()), iterator.getStride());
}


Expr LowererImpl::projectCanonicalSpaceToWindowedPosition(Iterator iterator, ir::Expr expr) {
  return ir::Add::make(ir::Mul::make(expr, iterator.getStride()), iterator.getWindowLowerBound());
}

bool LowererImpl::anyParentInSet(IndexVar var, std::set<IndexVar>& s) {
  auto children = this->provGraph.getChildren(var);
  for (auto c : children) {
    if (util::contains(s, c)) {
      return true;
    }
    if (anyParentInSet(c, s)) {
      return true;
    }
  }
  return false;
}

std::vector<ir::Stmt> LowererImpl::declareLaunchDomain(ir::Expr domain, Forall forall, const std::vector<IndexVar>& distVars) {
  std::vector<ir::Stmt> result;
  auto dim = distVars.size();
  auto pointT = Point(dim);
  auto indexSpaceT = IndexSpaceT(dim);
  auto rectT = Rect(dim);
  // If we're computing on a tensor, then use the domain of the partition as the
  // launch domain for the task launch.
  if (!forall.getComputingOn().empty()) {
    auto getDomain = ir::Call::make("runtime->get_index_partition_color_space", {ctx, ir::Call::make("get_index_partition", {this->computingOnPartition[*forall.getComputingOn().begin()]}, Auto)}, Auto);
    result.push_back(ir::VarDecl::make(domain, getDomain));
  } else {
    // Otherwise, construct the launch domain from the distribution variables.
    auto varIspace = ir::Var::make(forall.getIndexVar().getName() + "IndexSpace", Auto);
    auto lowerBound = ir::Var::make("lowerBound", pointT);
    auto upperBound = ir::Var::make("upperBound", pointT);
    std::vector<ir::Expr> lowerBoundExprs;
    std::vector<ir::Expr> upperBoundExprs;
    for (auto it : distVars) {
      // If the bounds of an index variable have been overridden for placement code
      // use those bounds instead of the ones derived from the Provenance Graph.
      if (this->isPlacementCode && util::contains(this->indexVarFaces, it)) {
        auto face = this->indexVarFaces[it];
        lowerBoundExprs.push_back(face);
        upperBoundExprs.push_back(face);
      } else {
        auto bounds = provGraph.deriveIterBounds(it, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
        lowerBoundExprs.push_back(bounds[0]);
        upperBoundExprs.push_back(ir::Sub::make(bounds[1], 1));
      }
    }
    result.push_back(ir::VarDecl::make(lowerBound, makeConstructor(pointT, lowerBoundExprs)));
    result.push_back(ir::VarDecl::make(upperBound, makeConstructor(pointT, upperBoundExprs)));

    auto makeIspace = ir::Call::make(
        "runtime->create_index_space",
        {ctx, makeConstructor(rectT, {lowerBound, upperBound})},
        Auto
    );
    result.push_back(ir::VarDecl::make(varIspace, makeIspace));
    auto makeDomain = ir::Call::make(
        "runtime->get_index_space_domain",
        {ctx, makeConstructor(indexSpaceT, {varIspace})},
        domain.type()
    );
    result.push_back(ir::VarDecl::make(domain, makeDomain));
  }
  return result;
}

bool LowererImpl::statementAccessesRegion(ir::Stmt stmt, ir::Expr target) {
  taco_iassert(target.as<GetProperty>() != nullptr);

  struct AccessFinder : public IRVisitor {
    void doesTargetMatch(ir::Expr regIR) {
      auto reg = regIR.as<GetProperty>();
      if (!reg) return;
      if (reg->tensor != this->targetGP->tensor) return;
      // If the target of the load is an accessor of the target
      // getProperty, then mark that the current task reads the target.
      switch (this->targetGP->property) {
        // If the target is a values region, then access into any sort of
        // accessor of the same tensor is what we are looking for.
        case TensorProperty::Values: {
          switch (reg->property) {
            case ir::TensorProperty::ValuesReductionAccessor:
            case ir::TensorProperty::ValuesWriteAccessor:
            case ir::TensorProperty::ValuesReadAccessor:
            case ir::TensorProperty::ValuesReductionNonExclusiveAccessor:
              this->readsRegion = true;
              break;
            default:
              return;
          }
          break;
        }
        // If the target is an indices region, then we are looking for an accessor
        // that accesses the same index and mode as the original property.
        case TensorProperty::Indices: {
          if (reg->property == TensorProperty::IndicesAccessor && reg->index == this->targetGP->index &&
              reg->mode == this->targetGP->mode) {
            this->readsRegion = true;
          }
          break;
        }
        default:
          break;
      }
    }

    void visit(const Load* load) {
      this->doesTargetMatch(load->arr);
    }

    void visit(const Store* store) {
      this->doesTargetMatch(store->arr);
      store->data.accept(this);
    }

    void visit(const For* node) {
      if (node->isTask) { return; }
      node->contents.accept(this);
      // We also look into the loop guards because those may access regions.
      node->start.accept(this);
      node->end.accept(this);
      node->increment.accept(this);
    }

    const GetProperty* targetGP;
    bool readsRegion = false;
  };

  AccessFinder finder;
  finder.targetGP = target.as<GetProperty>();
  stmt.accept(&finder);
  return finder.readsRegion;
}

std::vector<ir::Stmt> LowererImpl::declarePartitionBoundsVars(ir::Expr domainIter, TensorVar tensor) {
  std::vector<ir::Stmt> result;
  auto point = ir::Var::make("domPoint", Datatype("DomainPoint"));
  result.push_back(ir::VarDecl::make(point, ir::Deref::make(domainIter, Auto)));
  auto part = tensor;
  auto partVar = ir::Var::make(part.getName() + "PartitionBounds", Auto);
  auto subreg = ir::Call::make("runtime->get_logical_subregion_by_color", {ctx, this->computingOnPartition[part], point}, Auto);
  auto subregispace = ir::MethodCall::make(subreg, "get_index_space", {}, false, Auto);
  auto bounds = ir::Call::make("runtime->get_index_space_domain", {subregispace}, Auto);
  result.push_back(ir::VarDecl::make(partVar, bounds));
  // Declare all of the bounds variables here.
  for (auto tvItr : this->provGraph.getPartitionBounds()) {
    for (auto idxItr : tvItr.second) {
      auto lo = ir::Load::make(ir::MethodCall::make(partVar, "lo", {}, false, Int64), idxItr.first);
      auto hi = ir::Load::make(ir::MethodCall::make(partVar, "hi", {}, false, Int64), idxItr.first);
      result.push_back(ir::VarDecl::make(idxItr.second.first, lo));
      result.push_back(ir::VarDecl::make(idxItr.second.second, hi));
    }
  }
  return result;
}

std::vector<ir::Stmt> LowererImpl::createDomainPointColorings(
    Forall forall,
    ir::Expr domainIter,
    std::map<TensorVar, ir::Expr> domains,
    std::set<TensorVar>& fullyReplicatedTensors,
    std::vector<ir::Expr> colorings
) {
  taco_iassert(colorings.size() == forall.getTransfers().size());

  std::vector<ir::Stmt> result;
  // Add a dummy partition object for each transfer.
  for (size_t idx = 0; idx < forall.getTransfers().size(); idx++) {
    auto& t = forall.getTransfers()[idx];
    auto& tv = t.getAccess().getTensorVar();
    auto n = tv.getName();

    // If this tensor isn't partitioned by any variables in the current loop,
    // then the full thing is going to be replicated. In this case, it's better
    // to pass the region directly to each child task, rather than a full aliasing
    // partition.
    bool hasPartitioningVar = false;
    for (auto ivar : t.getAccess().getIndexVars()) {
      if (this->anyParentInSet(ivar, this->varsInScope[this->curDistVar])) {
        hasPartitioningVar = true;
      }
    }
    if (!hasPartitioningVar && !(this->isPlacementCode || this->isPartitionCode)) {
      fullyReplicatedTensors.insert(t.getAccess().getTensorVar());
      continue;
    }

    // TODO (rohany): Fully understand how to pick which dimensions to do the partitioning on.
    // TODO (rohany): For now, we assume that we're only going to create the initial partitions on
    //  dense runs, and on the first run only. We'll leave it up to later to figure out how to implement
    //  partitioning a run that nested in other tensors.

    auto runs = DenseFormatRuns(t.getAccess(), this->iterators);
    taco_iassert(runs.runs.size() > 0);
    // Assume that we're partitioning the first run.
    auto run = runs.runs[0];
    // TODO (rohany): Rename these variables later.
    // The point we're making matches the dimensionality of the run.
    auto tensorDim = run.modes.size();
    auto txPoint = Point(tensorDim);
    auto txRect = Rect(tensorDim);
    // We need to only partition the modes that are part of the run.
    auto denseRunIndexSpace = ir::GetProperty::makeDenseLevelRun(this->tensorVars[tv], 0);
    auto tbounds = this->derivedBounds[forall.getIndexVar()][tv];
    std::vector<Expr> los, his;
    for (auto modeIdx : run.modes) {
      los.push_back(tbounds[modeIdx][0]);
      auto partUpper = ir::Load::make(ir::MethodCall::make(domains[tv], "hi", {}, false, Int64), int(modeIdx));
      auto upper = ir::Min::make(tbounds[modeIdx][1], partUpper);
      his.push_back(upper);
    }
    auto start = ir::Var::make(n + "Start", txPoint);
    auto end = ir::Var::make(n + "End", txPoint);
    result.push_back(ir::VarDecl::make(start, makeConstructor(txPoint, los)));
    result.push_back(ir::VarDecl::make(end, makeConstructor(txPoint, his)));
    auto rect = ir::Var::make(n + "Rect", txRect);
    result.push_back(ir::VarDecl::make(rect, makeConstructor(txRect, {start, end})));

    // It's possible that this partitioning makes a rectangle that goes out of bounds
    // of the tensor's index space. If so, replace the rectangle with an empty Rect.
    auto lb = ir::MethodCall::make(domains[tv], "contains", {ir::FieldAccess::make(rect, "lo", false, Auto)}, false, Bool);
    auto hb = ir::MethodCall::make(domains[tv], "contains", {ir::FieldAccess::make(rect, "hi", false, Auto)}, false, Bool);
    auto guard = ir::Or::make(ir::Neg::make(lb), ir::Neg::make(hb));
    result.push_back(ir::IfThenElse::make(guard, ir::Block::make(ir::Assign::make(rect, ir::MethodCall::make(rect, "make_empty", {}, false, Auto)))));
    result.push_back(ir::Assign::make(ir::Load::make(colorings[idx], ir::Deref::make(domainIter, Auto)), rect));
  }

  return result;
}

std::vector<ir::Stmt> LowererImpl::createIndexPartitions(
    Forall forall,
    ir::Expr domain,
    std::map<TensorVar, std::map<int, std::vector<ir::Expr>>>& tensorLogicalPartitions,
    std::map<TensorVar, std::map<int, ir::Expr>>& tensorDenseRunPartitions,
    std::set<TensorVar> fullyReplicatedTensors,
    std::vector<Expr> colorings,
    const std::vector<IndexVar>& distIvars) {
  taco_iassert(colorings.size() == forall.getTransfers().size());

  std::vector<ir::Stmt> result;

  for (size_t idx = 0; idx < forall.getTransfers().size(); idx++) {
    auto& t = forall.getTransfers()[idx];
    auto& tv = t.getAccess().getTensorVar();

    // Skip fully replicated tensors.
    if (util::contains(fullyReplicatedTensors, tv)) {
      continue;
    }

    // TODO (rohany): These heuristics for distjoint / aliased will have to be updated
    //  for sparse tensors. However, if we have an initial partitioning pass, then I'm not
    //  sure it really matters how well we guess disjoint vs aliased.

    auto coloring = colorings[idx];
    auto part = ir::Var::make(tv.getName() + "_dense_run_0_Partition", Auto);
    // TODO (rohany): Make this LEGION_COMPUTE_KIND since it is happening
    //  off of the critical path.
    auto partKind = disjointPart;
    // Figure out how many axes of the tensor are not being partitioned in order
    // to figure out how many axes of the tensor are being partitioned. If
    // the tensor is being partitioned in as many ways as the target loop is
    // distributed, then the partition is disjoint. If there are unpartitioned
    // axes and more distribution variables, then the tensor is likely aliased.
    size_t aliasingVarsCount = 0;
    for (auto ivar : t.getAccess().getIndexVars()) {
      if (!this->anyParentInSet(ivar, this->varsInScope[this->curDistVar])) {
        aliasingVarsCount++;
      }
    }
    assert(aliasingVarsCount <= t.getAccess().getIndexVars().size());
    size_t partitionedVars = t.getAccess().getIndexVars().size() - aliasingVarsCount;
    if (partitionedVars < distIvars.size()) {
      partKind = aliasedPart;
    }

    // If none of the variables in the access are changing in this loop, then we're
    // most likely operating on an aliased partition as well.
    std::set<IndexVar> curLoopVars;
    curLoopVars.insert(forall.getIndexVar());
    auto mfp = this->provGraph.getMultiFusedParents(forall.getIndexVar());
    curLoopVars.insert(mfp.begin(), mfp.end());
    bool accessIter = false;
    for (auto var : t.getAccess().getIndexVars()) {
      if (this->anyParentInSet(var, curLoopVars)) {
        accessIter = true;
      }
    }
    if (!accessIter) {
      partKind = aliasedPart;
    }

    // If we're doing a reduction, we're most likely not operating on a disjoint partition.
    // So, fall back to an aliased partition.
    if (forall.getOutputRaceStrategy() == OutputRaceStrategy::ParallelReduction) {
      partKind = aliasedPart;
    }

    // If we're lowering placement code, it's too hard to figure this out (for now).
    if (this->isPlacementCode) {
      partKind = computePart;
    }

    // Pure partitioning code always results in disjoint partitions.
    if (this->isPartitionCode) {
      partKind = disjointPart;
    }

    // TODO (rohany): We need to communicate this data structure of the partitions of each level
    //  to later passes that lower index launches / serial task launches. An example data structure
    //  for this can be as follows:
    //  * Map<TensorVar, map<int, vector<Expr>>>, where we maintain a map from each TensorVar to
    //    a map from each level to a list of partition objects for that level. This list corresponds
    //    to the regions contained in that level. The partition of the values will be at level = TensorVar.order().

    // We'll do the initial partition of the first dense run.
    // TODO (rohany): Figure out how to take partitions of not the first dense run too.
    auto runs = DenseFormatRuns(t.getAccess(), this->iterators);
    taco_iassert(!runs.runs.empty());
    auto denseRunIndexSpace = ir::GetProperty::makeDenseLevelRun(this->tensorVars[tv], 0);
    // Partition the dense run using the coloring. Record the partition of the dense
    // index space that we created.
    tensorDenseRunPartitions[tv][0] = part;

    // If we have some index variables defined already, then this isn't a top level partition,
    // meaning that we need to tag it with an ID.
    ir::Expr partitionColor;
    if (!this->definedIndexVarsExpanded.empty()) {
      partitionColor = this->getIterationSpacePointIdentifier();
    }

    // Maybe add on the iteration space point identifier if we have one.
    auto maybeAddPartColor = [&](std::vector<ir::Expr> args) {
      if (!partitionColor.defined()) {
        return args;
      }
      args.push_back(partitionColor);
      return args;
    };

    // TODO (rohany): Extract this into a helper method?

    // We don't need to mark the index partitions of the denseRuns with colors, since
    // we aren't going to actually look up the denseLevelRun partitions directly.
    auto partcall = ir::Call::make("runtime->create_index_partition", {ctx, denseRunIndexSpace, domain, coloring, partKind}, Auto);
    result.push_back(ir::VarDecl::make(part, partcall));
    // Using this initial partition, partition the rest of the tensor.
    Expr currentLevelPartition = part;

    // We need to partition each mode in the tensor, but partition all of the dense
    // modes in a dense mode run in an aggregate manner. We do this by maintaining
    // an index of the dense run that is currently being considered, and skipping
    // modes within the run once we've performed the partitioning operation for
    // the dense run.
    size_t currentDenseRunIndex = 0;
    for (int level = 0; level < tv.getOrder(); level++) {
      // Check if the current level is the start of the current dense run. If it is,
      // then our job is to just partition the dense run by copying the current partition.
      // This is sound because the first sparse region (or values region) after the dense
      // run shares the same index space dimensions as the dense run itself. We also skip
      // emitting a partition for level 0 since that has already been done by the initial
      // partitioning step.
      if (currentDenseRunIndex < runs.runs.size() && runs.runs[currentDenseRunIndex].levels.front() == level && level != 0) {
        auto denseRun = ir::GetProperty::makeDenseLevelRun(this->tensorVars[tv], currentDenseRunIndex);
        auto runPart = ir::Var::make(tv.getName() + "_dense_run_" + util::toString(currentDenseRunIndex) + "_Partition", Auto);
        auto partitionCall = ir::Call::make(
            "copyPartition",
            {ctx, runtime, currentLevelPartition, denseRun},
            Auto
        );
        result.push_back(ir::VarDecl::make(runPart, partitionCall));
        currentLevelPartition = runPart;
        tensorDenseRunPartitions[tv][currentDenseRunIndex] = runPart;
      }

      // If the current level is not within a dense run, then we can partition it using
      // the current partition.
      if (currentDenseRunIndex >= runs.runs.size() || !util::contains(runs.runs[currentDenseRunIndex].levels, level)) {
        auto mode = t.getAccess().getTensorVar().getFormat().getModeOrdering()[level];
        auto iter = this->iterators.getLevelIteratorByModeAccess({t.getAccess(), mode + 1});
        auto partFunc = iter.getPartitionFromParent(currentLevelPartition, partitionColor);
        if (partFunc.defined()) {
          result.push_back(partFunc.compute());
          currentLevelPartition = partFunc.getResults().back();
          // Remember all of the partition variables so that we can use them when
          // constructing region requirements later.
          for (size_t i = 0; i < partFunc.numResults() - 1; i++) {
            tensorLogicalPartitions[tv][level].push_back(partFunc[i]);
          }
        }
      }

      // Finally, if we've hit the back of a dense run, bump the currentDenseRunIndex
      // onto the next dense run.
      if (currentDenseRunIndex < runs.runs.size() && runs.runs[currentDenseRunIndex].levels.back() <= level) {
        currentDenseRunIndex++;
      }
    }

    auto partitionVals = ir::Call::make(
        "copyPartition",
        maybeAddPartColor(
            {ctx, runtime, currentLevelPartition, getLogicalRegion(ir::GetProperty::make(this->tensorVars[tv], TensorProperty::Values))}),
        Auto
    );
    auto valsPart = ir::Var::make(tv.getName() + "_vals_partition", Auto);
    result.push_back(ir::VarDecl::make(valsPart, partitionVals));
    tensorLogicalPartitions[tv][tv.getOrder()].push_back(valsPart);
  }
  return result;
}

std::pair<ir::Expr, ir::Expr> LowererImpl::getPrivilegeForTensor(Forall forall, const TensorVar& tv) {
  // If we are supposed to allow control over the generated privilege for placement codes,
  // return the variable then.
  if (this->isPlacementCode && this->setPlacementPrivilege) {
    return std::make_pair(this->placementPrivilegeVar, exclusive);
  }
  // Creating partitions should only need read-only, virtual instances of the data.
  if (this->legionLoweringKind == PARTITION_ONLY) {
    return std::make_pair(readOnly, exclusive);
  }

  // TODO (rohany): Assuming that all tensors have the same type right now.
  auto reduce = ir::Symbol::make(LegionRedopString(tv.getType().getDataType()));
  if (util::contains(this->resultTensors, tv)) {
    // If we're already reducing, we can't go up the lattice to read_write
    // so stay at reduction.
    if (forall.getOutputRaceStrategy() == OutputRaceStrategy::ParallelReduction || this->performingLegionReduction) {
      return std::make_pair(reduce, simultaneous);
    }
    return std::make_pair(readWrite, exclusive);
  }
  return std::make_pair(readOnly, exclusive);
}

std::vector<ir::Stmt> LowererImpl::lowerIndexLaunch(
    Forall forall,
    ir::Expr domain,
    std::map<TensorVar, std::map<int, std::vector<ir::Expr>>>& tensorLogicalPartitions,
    std::set<ir::GetProperty::Hashable> regionsAccessed,
    int taskID,
    Stmt& unpackTensorData
) {
  std::vector<ir::Stmt> result;
  std::vector<Stmt> itlStmts;

  std::vector<ir::Expr> packedTensorData;
  std::vector<ir::Expr> packedTensorDataParents;

  // Small helper routine to append a tag argument.
  auto maybeAddTag = [](std::vector<ir::Expr> args, ir::Expr tag) {
    if (tag.defined()) {
      args.push_back(tag);
    }
    return args;
  };

  // Construct the region requirements for each tensor argument.
  std::vector<Expr> regionReqs;
  bool taskReadsAnyVars = false;
  for (auto& it : this->tensorVarOrdering) {
    auto tv = it;
    auto tvIR = this->tensorVars[tv];
    auto priv = this->getPrivilegeForTensor(forall, tv);
    std::vector<Expr> regionReqArgs;
    // Since we only currently support writing to tensors with the same non-zero
    // structure as an input, we still are only going to be reading the internal
    // regions of a tensor, and doing writes only to the values.
    auto indicesPriv = readOnly;
    auto indicesAccessMode = exclusive;

    // TODO (rohany): It seems like we're going to want to wrap up this stuff into a helper method
    //  that lets us nicely apply a function to each level (+ dense run) in a tensor.
    // Let's get all of the regions involved with this tensor.
    for (int level = 0; level < tv.getOrder(); level++) {
      auto iter = this->iterators.getLevelIteratorByLevel(tv, level);
      // Get the regions for this level. If there aren't any regions, continue
      // because there's nothing for us to do then.
      auto regions = iter.getRegions();
      if (regions.empty()) {
        continue;
      }

      // Construct RegionRequirements for each region. If there is a
      // partition present for the region, use it, otherwise pass the
      // entire region down to the children.
      ir::Expr req;
      for (size_t i = 0; i < regions.size(); i++) {
        // If the task being launched doesn't access the target region, then we can
        // virtually map the region. Or, for placement code, we don't want to virtually
        // map a region that the leaf placement tasks use.
        taco_iassert(regions[i].region.as<GetProperty>() != nullptr);
        ir::Expr tag;
        if (!util::contains(regionsAccessed, regions[i].region.as<GetProperty>()->toHashable()) && !(this->isPlacementCode && size_t(this->distLoopDepth + 1) == this->placements.size())) {
          tag = virtualMap;
          taskReadsAnyVars |= true;
        }
        if (util::contains(tensorLogicalPartitions, tv)) {
          // Logic to select the partition that we should use for the index launch.
          auto getPartition = [&](size_t idx) {
            // The proper partition depends on what the LegionLoweringKind is.
            if (this->legionLoweringKind == COMPUTE_ONLY) {
              // If we're in COMPUTE_ONLY, then we've already created all the partitions
              // that we need. So, we need to look up the partition with the right identifier.
              if (this->definedIndexVarsExpanded.empty()) {
                // If we're at the top level, then we need to look in the pack for this partition.
                auto partField = this->getTopLevelTensorPartition(tv);
                auto indicesPartitions = ir::FieldAccess::make(partField, "indicesPartitions", false /* isDeref */, Auto);
                // We need to look at the indices field here of the partition.
                auto loadLevel = ir::Load::make(indicesPartitions, level);
                return ir::Load::make(loadLevel, int32_t(idx));
              } else {
                // Otherwise, we look up the partition on the target LogicalRegion.
                return ir::Call::make("runtime->get_logical_partition_by_color", {ctx, getLogicalRegion(regions[i].region), this->getIterationSpacePointIdentifier()}, LogicalPartition);
              }
            } else {
              // If we aren't in COMPUTE_ONLY, then we should have defined these partitioning
              // objects to actually use. So, accessing the names is valid.
              taco_iassert(util::contains(tensorLogicalPartitions[tv], level));
              auto partitions = tensorLogicalPartitions[tv][level];
              taco_iassert(partitions.size() == regions.size());
              return partitions[idx];
            }
          };

          regionReqArgs = {
            getPartition(i),
            0,
            indicesPriv,
            indicesAccessMode,
            getLogicalRegion(regions[i].regionParent),
          };
        } else {
          regionReqArgs = {
            getLogicalRegion(regions[i].region),
            indicesPriv,
            indicesAccessMode,
            regions[i].regionParent,
          };
        }
        auto req = ir::makeConstructor(RegionRequirement, maybeAddTag(regionReqArgs, tag));
        req = ir::MethodCall::make(req, "add_field", {regions[i].field}, false /* deref */, Auto);
        regionReqs.push_back(req);

        // Remember the order in which we are packing regions.
        packedTensorData.push_back(regions[i].region);
        packedTensorDataParents.push_back(regions[i].regionParent);
      }
    }

    ir::Expr req;
    // Perform a similar analysis as above for the values region.
    auto valsReg = ir::GetProperty::make(tvIR, TensorProperty::Values);
    auto valsParent = ir::GetProperty::make(tvIR, TensorProperty::ValuesParent);
    ir::Expr tag;
    if (!util::contains(regionsAccessed, valsReg.as<GetProperty>()->toHashable()) && !(this->isPlacementCode && size_t(this->distLoopDepth + 1) == this->placements.size())) {
      tag = virtualMap;
      taskReadsAnyVars |= true;
    }
    if (util::contains(tensorLogicalPartitions, tv)) {
      ir::Expr valsPartition;
      // The logic to select valsPartition is similar to the logic above to select the partitions
      // for the indices arrays.
      if (this->legionLoweringKind == COMPUTE_ONLY) {
        if (this->definedIndexVarsExpanded.empty()) {
          auto partField = this->getTopLevelTensorPartition(tv);
          valsPartition = ir::FieldAccess::make(partField, "valsPartition", false /* isDeref */, Auto);
        } else {
          valsPartition = ir::Call::make("runtime->get_logical_partition_by_color", {ctx, getLogicalRegion(valsReg), this->getIterationSpacePointIdentifier()}, LogicalPartition);
        }
      } else {
        taco_iassert(util::contains(tensorLogicalPartitions[tv], tv.getOrder()));
        taco_iassert(tensorLogicalPartitions[tv][tv.getOrder()].size() == 1);
        valsPartition = tensorLogicalPartitions[tv][tv.getOrder()][0];
      }

      // Now add the region requirement for the values.
      regionReqArgs = {
        valsPartition,
        0,
        priv.first,
        priv.second,
        // TODO (rohany): Should this be region or region parent?
        valsParent,
      };
    } else {
      regionReqArgs = {
        getLogicalRegion(ir::GetProperty::make(tvIR, TensorProperty::Values)),
        priv.first,
        priv.second,
        ir::GetProperty::make(tvIR, TensorProperty::ValuesParent),
      };
    }
    req = ir::makeConstructor(RegionRequirement, maybeAddTag(regionReqArgs, tag));
    req = ir::MethodCall::make(req, "add_field", {fidVal}, false /* deref */, Auto);
    regionReqs.push_back(req);

    // Remember the order in which we are packing regions.
    packedTensorData.push_back(ir::GetProperty::make(tvIR, TensorProperty::Values));
    packedTensorDataParents.push_back(ir::GetProperty::make(tvIR, TensorProperty::ValuesParent));
  }

  // These args have to be for each of the subtasks.
  auto args = ir::Var::make("taskArgs", Auto);
  bool unpackFaceArgs = false;
  // We only generate code for control replicated placement if the distribution
  // is done at the top level.
  auto useCtrlRep = this->distLoopDepth == 0;
  if (this->isPlacementCode) {
    auto placementGrid = this->placements[this->distLoopDepth].first;
    auto placement = this->placements[this->distLoopDepth].second;

    // Count the number of Face() axes placements.
    int count = 0;
    for (auto axis : placement.axes) {
      if (axis.kind == GridPlacement::AxisMatch::Face) {
        count++;
      }
    }
    if (count > 0) {
      std::vector<Expr> prefixVars, prefixExprs;
      if (useCtrlRep) {
        // If we are using control replication, we'll need to do some extra
        // work to set up a sharding functor so that index tasks are sharded to
        // the right positions. To do so, we'll need to add a sharding functor ID
        // to the argument pack. Next, we need to register the sharding functor
        // to the runtime system, rather than letting the mapper handle it.
        int sfID = shardingFunctorID++;
        prefixVars.push_back(ir::Var::make("sfID", Int32));
        prefixExprs.push_back(ir::Call::make("shardingID", {sfID}, Int32));

        // Create the vector of dimensions.
        auto vecty = Datatype("std::vector<int>");
        auto dimVec = ir::Var::make("dims", vecty);
        itlStmts.push_back(ir::VarDecl::make(dimVec, ir::makeConstructor(vecty, {})));
        for (int i = 0; i < placementGrid.getDim(); i++) {
          itlStmts.push_back(ir::SideEffect::make(
              ir::MethodCall::make(dimVec, "push_back", {placementGrid.getDimSize(i)}, false /* deref */, Auto)));
        }
        itlStmts.push_back(
            ir::SideEffect::make(
                ir::Call::make(
                    "registerPlacementShardingFunctor",
                    {ctx, runtime, ir::Call::make("shardingID", {sfID}, Int32), dimVec},
                    Auto
                )
            )
        );
      } else {
        // If we are directed to place a tensor onto a Face of the placement
        // grid, then we need to package up the full dimensions of the placement
        // grid into the task's arguments so that the mapper can extract it.
        for (int i = 0; i < placementGrid.getDim(); i++) {
          std::stringstream varname;
          varname << "dim" << i;
          auto var = ir::Var::make(varname.str(), Int32);
          prefixVars.push_back(var); prefixExprs.push_back(placementGrid.getDimSize(i));
        }
      }
      itlStmts.push_back(ir::PackTaskArgs::make(args, taskID, prefixVars, prefixExprs));
      unpackFaceArgs = true;
    } else {
      itlStmts.push_back(ir::PackTaskArgs::make(args, taskID, {}, {}));
    }
  } else {
    itlStmts.push_back(ir::PackTaskArgs::make(args, taskID, {}, {}));
  }

  auto launcher = ir::Var::make("launcher", IndexLauncher);
  auto launcherMake = ir::Call::make(
      IndexLauncher.getName(),
      {
          ir::Call::make("taskID", {taskID}, Datatype::Int32),
          domain,
          args,
          ir::Call::make(ArgumentMap.getName(), {}, ArgumentMap),
      },
      IndexLauncher
  );
  itlStmts.push_back(ir::VarDecl::make(launcher, launcherMake));
  for (auto& req : regionReqs) {
    auto mcall = ir::MethodCall::make(launcher, "add_region_requirement", {req}, false /* deref */, Auto);
    itlStmts.push_back(ir::SideEffect::make(mcall));
  }
  if (unpackFaceArgs) {
    auto tag = placementMap;
    if (useCtrlRep) {
      tag = placementShard;
    }
    auto addTag = ir::Assign::make(ir::FieldAccess::make(launcher, "tag", false, Auto), tag);
    itlStmts.push_back(addTag);
  }
  // If this task reads the regions explicitly, then give a chance to the
  // mapper to potentially garbage collect these instances.
  if (taskReadsAnyVars && !this->isPlacementCode) {
    auto tag = ir::FieldAccess::make(launcher, "tag", false, Auto);
    itlStmts.push_back(ir::Assign::make(tag, ir::BitOr::make(tag, untrackValidRegions)));
  }

  // If this is a nested distribution, keep it on the same node.
  if (this->distLoopDepth > 0) {
    auto tag = ir::FieldAccess::make(launcher, "tag", false, Auto);
    auto addTag = ir::Assign::make(tag, ir::BitOr::make(tag, sameAddressSpace));
    itlStmts.push_back(addTag);
  }

  auto fm = ir::Var::make("fm", Auto);
  auto fmCall = ir::Call::make(
      "runtime->execute_index_space",
      {ctx, launcher},
      Auto
  );
  if (this->performingScalarReduction) {
    // Use a different overload of execute_index_space that does the reduction for us.
    auto redop = ir::Symbol::make(LegionRedopString(this->scalarReductionResult.type()));
    auto call = ir::Call::make(
        "runtime->execute_index_space",
        {ctx, launcher, redop},
        Auto
    );
    std::stringstream funcName;
    funcName << "get<" << this->scalarReductionResult.type() << ">";
    // Wait on the result of the index launch reduction.
    auto reduced = ir::MethodCall::make(call, funcName.str(), {}, false, Auto);
    itlStmts.push_back(ir::Assign::make(this->scalarReductionResult, reduced));
  } else if (this->distLoopDepth == 0 && this->waitOnFutureMap) {
    itlStmts.push_back(ir::VarDecl::make(fm, fmCall));
    itlStmts.push_back(ir::SideEffect::make(ir::MethodCall::make(fm, "wait_all_results", {}, false, Auto)));
  } else {
    itlStmts.push_back(ir::SideEffect::make(fmCall));
  }

  // Placement code should return the LogicalPartition for the top level partition.
  if (this->isPlacementCode && this->distLoopDepth == 0 && this->legionLoweringKind == PARTITION_AND_COMPUTE) {
    auto tv = this->tensorVars.begin()->first;
    auto tvIR = this->tensorVars.begin()->second;
    taco_iassert(false); // Asserting false here because the code uses the now-defunct partitionings variable.
    // auto call = ir::Call::make("runtime->get_logical_partition", {ctx, getLogicalRegion(tvIR), partitionings.at(tv)}, LogicalPartition);
    // itlStmts.push_back(ir::Return::make(call));
  }

  // Set the returned unpackTensorData.
  unpackTensorData = ir::UnpackTensorData::make(packedTensorData, packedTensorDataParents);

  result.push_back(ir::Block::make(itlStmts));
  return result;
}


std::vector<ir::Stmt> LowererImpl::lowerSerialTaskLoop(
    Forall forall,
    ir::Expr domain,
    ir::Expr domainIter,
    Datatype pointT,
    std::map<TensorVar, std::map<int, std::vector<ir::Expr>>>& tensorLogicalPartitions,
    std::set<ir::GetProperty::Hashable> regionsAccessed,
    int taskID,
    Stmt& unpackTensorData
) {
  std::vector<ir::Stmt> result;
  // Extract the index variable for this loop.
  auto point = this->indexVarToExprMap[forall.getIndexVar()];

  // Create a loop that launches instances of the task.
  std::vector<Stmt> taskCallStmts;
  taskCallStmts.push_back(ir::VarDecl::make(point, ir::Deref::make(domainIter, pointT)));
  std::vector<ir::Expr> packedTensorData, packedTensorDataParents;

  // Small helper routine to append a tag argument.
  auto maybeAddTag = [](std::vector<ir::Expr> args, ir::Expr tag) {
    if (tag.defined()) {
      args.push_back(tag);
    }
    return args;
  };

  // Construct the region requirements for each tensor.
  std::vector<Expr> regionReqs;
  bool taskReadsAnyVars = false;
  for (auto& it : this->tensorVarOrdering) {
    auto tv = it;
    auto tvIR = this->tensorVars[tv];
    auto priv = this->getPrivilegeForTensor(forall, tv);
    // Since we only currently support writing to tensors with the same non-zero
    // structure as an input, we still are only going to be reading the internal
    // regions of a tensor, and doing writes only to the values.
    auto indicesPriv = readOnly;
    auto indicesAccessMode = exclusive;
    std::vector<Expr> regionReqArgs;

    // Begin region requirement construction for sparse tensors.

    for (int level = 0; level < tv.getOrder(); level++) {
      auto iter = this->iterators.getLevelIteratorByLevel(tv, level);
      // Get the regions for this level. If there aren't any regions, continue
      // because there's nothing for us to do then.
      auto regions = iter.getRegions();
      if (regions.empty()) {
        continue;
      }

      // Construct region requirements for this region. If there is a partition
      // present for the region, then get the subregion for this domain point,
      // otherwise pass the entire region down to the children.
      // TODO (rohany): Is there a chance to perform some deduplication here
      //  between lowerIndexLaunch and lowerSerialTaskLoop?
      ir::Expr req;
      for (size_t i = 0; i < regions.size(); i++) {
        // If the task being launched doesn't access the target region, then we can
        // virtually map the region. Or, for placement code, we don't want to virtually
        // map a region that the leaf placement tasks use.
        taco_iassert(regions[i].region.as<GetProperty>() != nullptr);
        ir::Expr tag;
        if (!util::contains(regionsAccessed, regions[i].region.as<GetProperty>()->toHashable()) && !(this->isPlacementCode && size_t(this->distLoopDepth + 1) == this->placements.size())) {
          tag = virtualMap;
          taskReadsAnyVars |= true;
        }

        if (util::contains(tensorLogicalPartitions, tv)) {
          // Logic to select the partition expr we should use for the index launch.
          ir::Expr logicalPart;
          // The proper partition depends on what the LegionLoweringKind is.
          if (this->legionLoweringKind == COMPUTE_ONLY) {
            // If we're in COMPUTE_ONLY, then we've already created all the partitions
            // that we need. So, we need to look up the partition with the right identifier.
            if (this->definedIndexVarsExpanded.empty()) {
              // If we're at the top level, then we need to look in the pack for this partition.
              auto partField = this->getTopLevelTensorPartition(tv);
              auto indicesPartitions = ir::FieldAccess::make(partField, "indicesPartitions", false /* isDeref */, Auto);
              // We need to look at the indices field here of the partition.
              auto loadLevel = ir::Load::make(indicesPartitions, level);
              logicalPart = ir::Load::make(loadLevel, int32_t(i));
            } else {
              // Otherwise, we look up the partition on the target LogicalRegion.
              logicalPart = ir::Call::make("runtime->get_logical_partition_by_color", {ctx, getLogicalRegion(regions[i].region), this->getIterationSpacePointIdentifier()}, LogicalPartition);
            }
          } else {
            // If we aren't in COMPUTE_ONLY, then we should have defined these partitioning
            // objects to actually use. So, accessing the names is valid.
            taco_iassert(util::contains(tensorLogicalPartitions[tv], level));
            auto partitions = tensorLogicalPartitions[tv][level];
            taco_iassert(partitions.size() == regions.size());
            logicalPart = partitions[i];
          }
          // Use the logical partition to get the target subregion.
          auto subreg = ir::Call::make("runtime->get_logical_subregion_by_color", {ctx, logicalPart, point}, Auto);
          regionReqArgs = {
            getLogicalRegion(subreg),
            indicesPriv,
            indicesAccessMode,
            regions[i].regionParent,
          };
        } else {
          regionReqArgs = {
            getLogicalRegion(regions[i].region),
            indicesPriv,
            indicesAccessMode,
            regions[i].regionParent,
          };
        }
        auto req = ir::makeConstructor(RegionRequirement, maybeAddTag(regionReqArgs, tag));
        req = ir::MethodCall::make(req, "add_field", {regions[i].field}, false /* deref */, Auto);
        regionReqs.push_back(req);

        // Remember the order in which we are packing regions.
        packedTensorData.push_back(regions[i].region);
        packedTensorDataParents.push_back(regions[i].regionParent);
      }
    }

    // Do the same analysis for the values.
    ir::Expr req;
    auto valsReg = ir::GetProperty::make(tvIR, TensorProperty::Values);
    auto valsParent = ir::GetProperty::make(tvIR, TensorProperty::ValuesParent);
    auto tag = ir::Expr(0);
    if (!util::contains(regionsAccessed, valsReg.as<GetProperty>()->toHashable()) && !(this->isPlacementCode && size_t(this->distLoopDepth + 1) == this->placements.size())) {
      tag = ir::BitOr::make(tag, virtualMap);
      taskReadsAnyVars |= true;
    }
    if (util::contains(tensorLogicalPartitions, tv)) {
      ir::Expr valsPartition;
      // The logic to select valsPartition is similar to the logic above to select the partitions
      // for the indices arrays.
      if (this->legionLoweringKind == COMPUTE_ONLY) {
        if (this->definedIndexVarsExpanded.empty()) {
          auto partField = this->getTopLevelTensorPartition(tv);
          valsPartition = ir::FieldAccess::make(partField, "valsPartition", false /* isDeref */, Auto);
        } else {
          valsPartition = ir::Call::make("runtime->get_logical_partition_by_color", {ctx, getLogicalRegion(valsReg), this->getIterationSpacePointIdentifier()}, LogicalPartition);
        }
      } else {
        taco_iassert(util::contains(tensorLogicalPartitions[tv], tv.getOrder()));
        taco_iassert(tensorLogicalPartitions[tv][tv.getOrder()].size() == 1);
        valsPartition = tensorLogicalPartitions[tv][tv.getOrder()][0];
      }
      // Get the proper subregion out of the valsPartition.
      auto subreg = ir::Call::make("runtime->get_logical_subregion_by_color", {ctx, valsPartition, point}, Auto);
      // Now add the region requirement for the values.
      regionReqArgs = {
        getLogicalRegion(subreg),
        priv.first,
        priv.second,
        valsParent,
      };
    } else {
      regionReqArgs = {
        getLogicalRegion(ir::GetProperty::make(tvIR, TensorProperty::Values)),
        priv.first,
        priv.second,
        ir::GetProperty::make(tvIR, TensorProperty::ValuesParent),
      };
    }
    req = ir::makeConstructor(RegionRequirement, maybeAddTag(regionReqArgs, tag));
    req = ir::MethodCall::make(req, "add_field", {fidVal}, false /* deref */, Auto);
    regionReqs.push_back(req);

    // Remember the order in which we are packing regions.
    packedTensorData.push_back(ir::GetProperty::make(tvIR, TensorProperty::Values));
    packedTensorDataParents.push_back(ir::GetProperty::make(tvIR, TensorProperty::ValuesParent));
  }

  auto args = ir::Var::make("taskArgs", Auto);
  taskCallStmts.push_back(ir::PackTaskArgs::make(args, taskID, {}, {}));

  auto launcher = ir::Var::make("launcher", TaskLauncher);
  auto launcherMake = ir::Call::make(
      TaskLauncher.getName(),
      {
          ir::Call::make("taskID", {taskID}, Datatype::Int32),
          args,
      },
      TaskLauncher
  );
  taskCallStmts.push_back(ir::VarDecl::make(launcher, launcherMake));
  for (auto& req : regionReqs) {
    auto mcall = ir::MethodCall::make(launcher, "add_region_requirement", {req}, false /* deref */, Auto);
    taskCallStmts.push_back(ir::SideEffect::make(mcall));
  }

  // Extract the task's tag.
  auto tag = ir::FieldAccess::make(launcher, "tag", false, Auto);

  // If this task reads the regions explicitly, then give a chance to the
  // mapper to potentially garbage collect these instances.
  if (taskReadsAnyVars && !this->isPlacementCode) {
    taskCallStmts.push_back(ir::Assign::make(tag, ir::BitOr::make(tag, untrackValidRegions)));
  }

  // We give the option to all looped task launches to backpressure task execution.
  // In the case of reductions, this is done so that the reductions don't all occur
  // at the same time and OOM the processor. For read/write tasks, we sometimes need
  // backpressure on to avoid deadlocks around deferred resource allocation.
  taskCallStmts.push_back(ir::Assign::make(tag, ir::BitOr::make(tag, backpressureTask)));

  // The actual task call.
  auto tcall = ir::Call::make("runtime->execute_task", {ctx, launcher}, Auto);
  taskCallStmts.push_back(ir::SideEffect::make(tcall));

  // Finally, wrap everything within a for loop that launches the tasks.
  auto tcallLoop = ir::For::make(
      domainIter,
      ir::Call::make(domainIter.type().getName(), {domain}, domainIter.type()),
      ir::MethodCall::make(domainIter, "valid", {}, false /* deref */, Datatype::Bool),
      1 /* increment -- hack to get ++ */,
      ir::Block::make(taskCallStmts)
  );

  // Set the returned unpackTensorData.
  unpackTensorData = ir::UnpackTensorData::make(packedTensorData, packedTensorDataParents);

  result.push_back(tcallLoop);
  return result;
}

LowererImpl::DenseFormatRuns::DenseFormatRuns(const Access& a, const Iterators& iterators) {
  bool inDenseRun = false;
  auto tv = a.getTensorVar();
  auto format = tv.getFormat();
  for (int level = 0; level < tv.getOrder(); level++) {
    auto modeNum = format.getModeOrdering()[level];
    auto iter = iterators.getLevelIteratorByModeAccess({a, modeNum + 1});
    if (iter.isDense()) {
      // If we aren't in a run, start the run.
      if (!inDenseRun) {
        inDenseRun = true;
        this->runs.push_back(DenseRun{});
      }
      // Accumulate data about the run.
      this->runs.back().modes.push_back(modeNum);
      this->runs.back().levels.push_back(level);
    } else {
      // End any runs in progress.
      inDenseRun = false;
    }
  }
}

void LowererImpl::ValuesAnalyzer::addAccess(const Access& access, const Iterators &iterators, const std::map<IndexVar, ir::Expr>& indexVarToExprMap) {
  // TODO (rohany): What do we want to know / how do we want to organize it.
  //  * Information about each tensorvar -- the dimensionality of the values
  //    array. That can be stored in a map<tensorvar, {info}>.
  //  * Information about how to actually access the values array, this is specific
  //    to each access, as the access point depends on the variables in the access.

  auto tv = access.getTensorVar();
  DenseFormatRuns runs(access, iterators);
  auto getIter = [&](int level) {
    auto mode = tv.getFormat().getModeOrdering()[level];
    return iterators.getLevelIteratorByModeAccess({access, mode + 1});
  };

  // Figure out the dimensionality of the values array.
  if (!util::contains(this->valuesDims, tv)) {
    // If there aren't any dense runs in the tensor, then the values array can
    // only be one-dimensional.
    if (runs.runs.empty()) {
      this->valuesDims[tv] = 1;
    } else {
      // Get the last run.
      auto lastRun = runs.runs.back();
      // TODO (rohany): It might be good to add some of this logic into the DenseFormatRun
      //  object itself, as this might be useful when doing multi-dimensional sparse levels too.
      // If the last run contains the final level of the tensor, then the values
      // array has dimensionality equal to that of the dense run. Otherwise,
      // there are sparse levels in between the dense run and the values array,
      // so the values array is one-dimensional.
      if (util::contains(lastRun.levels, tv.getOrder() - 1)) {
        // If the dense run contains level zero, the resulting dimensionality
        // is the number of levels in the run. Otherwise, it has 1 extra dimension
        // for the hook dimension in from the parent level.
        if (util::contains(lastRun.levels, 0)) {
          this->valuesDims[tv] = lastRun.levels.size();
        } else {
          this->valuesDims[tv] = lastRun.levels.size() + 1;
        }
      } else {
        this->valuesDims[tv] = 1;
      }
    }
  }

  if (!util::contains(this->valuesAccess, access)) {
    // If there aren't any dense runs in the tensor, then we use the index
    // variable corresponding to the last level in the tensor.
    if (runs.runs.empty()) {
      // Accesses from a sparse level need to use the posVar().
      auto iter = getIter(tv.getOrder() - 1);
      this->valuesAccess[access] = {iter.getPosVar()};
      this->valuesAccessWidths[access] = {iter.getWidth()};
    } else {
      // Get the last run.
      auto lastRun = runs.runs.back();
      // This logic mirrors the logic above for constructing the
      // dimensionality of the values array.
      if (util::contains(lastRun.levels, tv.getOrder() - 1)) {
        // Get the index variables for each level in the run.
        std::vector<ir::Expr> targetVars, targetWidths;
        if (!util::contains(lastRun.levels, 0)) {
          // Get the position of the level before this dense run if the run
          // doesn't start from the root of the tensor.
          auto iter = getIter(lastRun.levels.front() - 1);
          targetVars.push_back(iter.getPosVar());
          targetWidths.push_back(iter.getWidth());
        }

        // Add the locator variables for each of the remaining dimensions in the run.
        for (auto level : lastRun.levels) {
          auto iter = getIter(level);
          targetVars.push_back(indexVarToExprMap.at(iter.getIndexVar()));
          targetWidths.push_back(iter.getWidth());
        }
        this->valuesAccess[access] = targetVars;
        this->valuesAccessWidths[access] = targetWidths;
      } else {
        // Accesses from a sparse level need to use the posVar().
        auto iter = getIter(tv.getOrder() - 1);
        this->valuesAccess[access] = {iter.getPosVar()};
        this->valuesAccessWidths[access] = {iter.getWidth()};
      }
    }
  }
}

int LowererImpl::ValuesAnalyzer::getValuesDim(const TensorVar &tv) const {
  return this->valuesDims.at(tv);
}

ir::Expr LowererImpl::ValuesAnalyzer::getAccessPoint(const Access& access) const {
  taco_iassert(this->valuesAccess.at(access).size() == size_t(this->valuesDims.at(access.getTensorVar())));
  auto pointArgs = this->valuesAccess.at(access);
  auto pointTy = Point(pointArgs.size());
  return ir::makeConstructor(pointTy, pointArgs);
}

std::vector<ir::Expr> LowererImpl::getAllNeededParentPositions(Iterator &iter) {
  taco_iassert(iter.hasPosIter());
  // If we aren't an LgSparse level, then we only need the parent position. Also if we are the root
  // iterator or our parent is the root iterator then we should short circuit as well.
  if (!iter.isLgSparse() || iter.isRoot() || (iter.getParent().isRoot())) {
    return {iter.getParent().getPosVar()};
  }
  // If we are an LgSparse level, the pos level has dimensionality equal to
  // the number of Dense levels above us, +1 if we run into a level that isn't Dense.
  std::vector<ir::Expr> positions;
  auto curIter = iter;
  while (!curIter.isRoot()) {
    auto parent = curIter.getParent();
    // We don't want to include the root iterator's contribution here.
    if (!parent.isRoot()) {
      Expr pos;
      if (parent.isDense()) {
        // If the parent is dense, we just want that iterator's index variable's current value.
        pos = this->indexVarToExprMap[parent.getIndexVar()];
      } else {
        // Otherwise, we want the actual computed position.
        pos = parent.getPosVar();
      }
      positions.push_back(pos);
    }
    if (!parent.isDense()) {
      break;
    }
    curIter = parent;
  }
  if (positions.size() == 0) {
    taco_iassert(false);
  }
  auto rev = util::reverse(positions);
  return {rev.begin(), rev.end()};
}

ir::Expr LowererImpl::constructAffineProjection(Access &from, Access &to) {
  auto fromDenseRuns = DenseFormatRuns(from, this->iterators);
  auto toDenseRuns = DenseFormatRuns(to, this->iterators);

  // If the target tensor doesn't have any dense runs then we won't be able to
  // partition it.
  if (toDenseRuns.runs.empty()) {
    return Expr();
  }
  auto toDenseRun = toDenseRuns.runs[0];

  // If the `from` tensor doesn't have a top level dense run then
  // resulting partition does not allow us to restrict the accessed
  // index spaces of any dimensions, so we just return empty.
  if (fromDenseRuns.runs.empty() || !util::contains(fromDenseRuns.runs[0].levels, 0)) {
    return Expr();
  }

  // Get the top level dense run of `from`. We'll use this set of dense levels to understand
  // what index variables are partitioned, and will be the basis of the projection that we construct.
  taco_iassert(!fromDenseRuns.runs.empty());
  auto fromFirstDenseRun = fromDenseRuns.runs[0];
  taco_iassert(util::contains(fromFirstDenseRun.levels, 0));

  std::vector<ir::Expr> projectionArgs(fromFirstDenseRun.modes.size());
  // For each mode in the dense run, we need to find the mode of `to` that is partitioned
  // by the the mode in the dense run. This amounts to finding the position of the index
  // variable in the access, and then using that information to construct the projection.
  for (size_t i = 0; i < fromFirstDenseRun.modes.size(); i++) {
    auto fromMode = fromFirstDenseRun.modes[i];
    auto fromLevel = fromFirstDenseRun.levels[i];
    auto fromIndexVar = from.getIndexVars()[fromMode];
    // Now look up what mode in `to` this IndexVar accesses.
    auto toIndexVarModePos = std::find(to.getIndexVars().begin(), to.getIndexVars().end(), fromIndexVar);
    // If we didn't find the index variable, then this index in the projection is \bot.
    if (toIndexVarModePos == to.getIndexVars().end()) {
      projectionArgs[fromLevel] = AffineProjectionBot;
    } else {
      auto toIndexVarMode = toIndexVarModePos - to.getIndexVars().begin();
      // Find the level that this mode corresponds to.
      auto toIndexVarLevel = -1;
      for (size_t j = 0; j < toDenseRun.modes.size(); j++) {
        if (toDenseRun.modes[j] == toIndexVarMode) {
          toIndexVarLevel = toDenseRun.levels[j];
        }
      }
      taco_iassert(toIndexVarLevel != -1);
      projectionArgs[fromLevel] = toIndexVarLevel;
    }
  }
  for (auto expr : projectionArgs) {
    taco_iassert(expr.defined());
  }
  // If we aren't performing a projection on any dimensions, then the target
  // tensor is not being partitioned. In this case, return an empty Expr.
  if (util::all(projectionArgs, [](ir::Expr e) { return e == AffineProjectionBot; })) {
    return Expr();
  }
  return makeConstructor(AffineProjection, projectionArgs);
}

LowererImpl::BoundsInferenceExprRewriter::BoundsInferenceExprRewriter(ProvenanceGraph &pg, Iterators &iterators,
                              std::map<IndexVar, std::vector<ir::Expr>> &underivedBounds,
                              std::map<IndexVar, ir::Expr> &indexVarToExprMap,
                              std::set<IndexVar> &inScopeVars,
                              std::map<ir::Expr, IndexVar>& exprToIndexVarMap,
                              std::vector<IndexVar>& definedIndexVars,
                              bool lower,
                              std::set<IndexVar> presentIvars)
      : pg(pg), iterators(iterators), underivedBounds(underivedBounds), indexVarToExprMap(indexVarToExprMap),
        definedIndexVars(definedIndexVars), exprToIndexVarMap(exprToIndexVarMap), inScopeVars(inScopeVars), lower(lower),
        presentIvars(presentIvars) {}

void LowererImpl::BoundsInferenceExprRewriter::visit(const Var* var) {
  // If there is a var that isn't an index variable (like a partition bounds var),
  // then just return.
  if (this->exprToIndexVarMap.count(var) == 0) {
    expr = var;
    return;
  }
  auto ivar = this->exprToIndexVarMap.at(var);
  if (util::contains(this->inScopeVars, ivar)) {
    // If this ivar is in scope of the request, then access along it is fixed.
    expr = var;
  } else {

    // If a variable being derived is not even going to be present in the loop
    // (i.e. a variable that we split again), then we might want to expand it
    // into the variables that derive it. However, if neither of those variables
    // are in scope, then the bounds the provenance graph provides us for the
    // suspect variable are the ones we should take.
    if (!util::contains(this->presentIvars, ivar)) {
      struct InscopeVarVisitor : public IRVisitor {
        InscopeVarVisitor(ProvenanceGraph& pg) : pg(pg) {}
        void visit(const Var* var) {
          auto ivar = this->exprToIndexVarMap[var];
          if (util::contains(this->inScopeVars, ivar)) {
            this->anyInScope = true;
            return;
          }
          // There's a special case here for staggered variables. These variables
          // aren't really the subject of a parent-child relationship, so we flatten
          // that relationship here when looking at bounds.
          auto res = this->pg.getStaggeredVar(ivar);
          if (res.first) {
            if (util::contains(this->inScopeVars, res.second)) {
              this->anyInScope = true;
              return;
            }
          }
        }
        std::set<IndexVar> inScopeVars;
        std::map<Expr, IndexVar> exprToIndexVarMap;
        bool anyInScope = false;
        ProvenanceGraph& pg;
      };
      InscopeVarVisitor isv(this->pg); isv.inScopeVars = this->inScopeVars; isv.exprToIndexVarMap = this->exprToIndexVarMap;
      auto recovered = this->pg.recoverVariable(ivar, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators);
      auto staggered = this->pg.getStaggeredVar(ivar);
      if (staggered.first && !util::contains(this->inScopeVars, staggered.second)) {
        // If this variable is staggered and the staggered variable isn't in scope,
        // the derived bounds are not correct, because they are like (in + jn % gridX)
        // which doesn't make any sense. So, we flatten the relationship here and substitute
        // the upper/lower bound of the variable itself.
        auto bounds = this->pg.deriveIterBounds(ivar, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators);
        auto idx = lower ? 0 : 1;
        this->changed = true;
        this->expr = ir::Sub::make(bounds[idx], ir::Literal::make(idx));
        return;
      }
      recovered.accept(&isv);
      // If there are some variables in scope, use this as the rewritten expression.
      // A future call to the rewriter will expand the resulting variables.
      if (isv.anyInScope) {
        this->expr = recovered;
        this->changed = true;
        return;
      }
    }

    // Otherwise, the full bounds of this ivar will be accessed. So, derive the
    // bounds. Depending on whether we are deriving a lower or upper bound, use the
    // appropriate one.
    auto bounds = this->pg.deriveIterBounds(ivar, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators);
    auto idx = lower ? 0 : 1;
    this->changed = true;
    // If we are deriving an upper bound, we substitute an inclusive
    // bound here. This ensures that we calculate indices for only the
    // exact locations we access, and will map cleanly to Legion partitioning.
    expr = ir::Sub::make(bounds[idx], ir::Literal::make(idx));
  }
}

void LowererImpl::BoundsInferenceExprRewriter::visit(const GetProperty* gp) {
  // TODO (rohany): For some reason, I need to have this empty visit method
  //  for GetProperty here.
  expr = gp;
}

bool LowererImpl::loweringToGPU() {
  return should_use_CUDA_codegen() || this->containsGPULoops;
}

}
