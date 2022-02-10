#include "taco/index_notation/transformations.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/error/error_messages.h"
#include "taco/util/collections.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"
#include "taco/lower/mode.h"
#include "taco/lower/mode_format_impl.h"
#include "taco/lower/mode_format_rect_compressed.h"

#include <iostream>
#include <algorithm>
#include <limits>

using namespace std;

namespace taco {

// class Transformation
Transformation::Transformation(Reorder reorder)
    : transformation(new Reorder(reorder)) {
}

Transformation::Transformation(Precompute precompute)
    : transformation(new Precompute(precompute)) {
}

Transformation::Transformation(ForAllReplace forallreplace)
        : transformation(new ForAllReplace(forallreplace)) {
}

Transformation::Transformation(Parallelize parallelize)
        : transformation(new Parallelize(parallelize)) {
}

Transformation::Transformation(AddSuchThatPredicates addsuchthatpredicates)
        : transformation(new AddSuchThatPredicates(addsuchthatpredicates)) {
}

IndexStmt Transformation::apply(IndexStmt stmt, string* reason) const {
  return transformation->apply(stmt, reason);
}

std::ostream& operator<<(std::ostream& os, const Transformation& t) {
  t.transformation->print(os);
  return os;
}


// class Reorder
struct Reorder::Content {
  std::vector<IndexVar> replacePattern;
  bool pattern_ordered; // In case of Reorder(i, j) need to change replacePattern ordering to actually reorder
};

Reorder::Reorder(IndexVar i, IndexVar j) : content(new Content) {
  content->replacePattern = {i, j};
  content->pattern_ordered = false;
}

Reorder::Reorder(std::vector<taco::IndexVar> replacePattern) : content(new Content) {
  content->replacePattern = replacePattern;
  content->pattern_ordered = true;
}

IndexVar Reorder::geti() const {
  return content->replacePattern[0];
}

IndexVar Reorder::getj() const {
  if (content->replacePattern.size() == 1) {
    return geti();
  }
  return content->replacePattern[1];
}

const std::vector<IndexVar>& Reorder::getreplacepattern() const {
  return content->replacePattern;
}

IndexStmt Reorder::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  // collect current ordering of IndexVars
  bool startedMatch = false;
  std::vector<IndexVar> currentOrdering;
  bool matchFailed = false;

  match(stmt,
        std::function<void(const ForallNode*)>([&](const ForallNode* op) {
          bool matches = std::find (getreplacepattern().begin(), getreplacepattern().end(), op->indexVar) != getreplacepattern().end();
          if (matches) {
            currentOrdering.push_back(op->indexVar);
            startedMatch = true;
          }
          else if (startedMatch && currentOrdering.size() != getreplacepattern().size()) {
            matchFailed = true;
          }
        })
  );

  if (!content->pattern_ordered && currentOrdering == getreplacepattern()) {
    taco_iassert(getreplacepattern().size() == 2);
    content->replacePattern = {getreplacepattern()[1], getreplacepattern()[0]};
  }

  if (matchFailed || currentOrdering.size() != getreplacepattern().size()) {
    *reason = "The foralls of reorder pattern: " + util::join(getreplacepattern()) + " were not directly nested.";
    return IndexStmt();
  }
  return ForAllReplace(currentOrdering, getreplacepattern()).apply(stmt, reason);
}

void Reorder::print(std::ostream& os) const {
  os << "reorder(" << util::join(getreplacepattern()) << ")";
}

std::ostream& operator<<(std::ostream& os, const Reorder& reorder) {
  reorder.print(os);
  return os;
}


// class Precompute
struct Precompute::Content {
  IndexExpr expr;
  IndexVar i;
  IndexVar iw;
  TensorVar workspace;
};

Precompute::Precompute() : content(nullptr) {
}

Precompute::Precompute(IndexExpr expr, IndexVar i, IndexVar iw,
                     TensorVar workspace) : content(new Content) {
  content->expr = expr;
  content->i = i;
  content->iw = iw;
  content->workspace = workspace;
}

IndexExpr Precompute::getExpr() const {
  return content->expr;
}

IndexVar Precompute::geti() const {
  return content->i;
}

IndexVar Precompute::getiw() const {
  return content->iw;
}

TensorVar Precompute::getWorkspace() const {
  return content->workspace;
}

static bool containsExpr(Assignment assignment, IndexExpr expr) {
   struct ContainsVisitor : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    IndexExpr expr;
    bool contains = false;

    void visit(const UnaryExprNode* node) {
      if (equals(IndexExpr(node), expr)) {
        contains = true;
      }
      else {
        IndexNotationVisitor::visit(node);
      }
    }

    void visit(const BinaryExprNode* node) {
      if (equals(IndexExpr(node), expr)) {
        contains = true;
      }
      else {
        IndexNotationVisitor::visit(node);
      }
    }

    void visit(const ReductionNode* node) {
      taco_ierror << "Reduction node in concrete index notation.";
    }
  };

  ContainsVisitor visitor;
  visitor.expr = expr;
  visitor.visit(assignment);
  return visitor.contains;
}

static Assignment getAssignmentContainingExpr(IndexStmt stmt, IndexExpr expr) {
  Assignment assignment;
  match(stmt,
        function<void(const AssignmentNode*,Matcher*)>([&assignment, &expr](
            const AssignmentNode* node, Matcher* ctx) {
          if (containsExpr(node, expr)) {
            assignment = node;
          }
        })
  );
  return assignment;
}

IndexStmt Precompute::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  // Precondition: The expr to precompute is not in `stmt`
  Assignment assignment = getAssignmentContainingExpr(stmt, getExpr());
  if (!assignment.defined()) {
    *reason = "The expression (" + util::toString(getExpr()) + ") " +
              "is not in " + util::toString(stmt);
    return IndexStmt();
  }

  struct PrecomputeRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Precompute precompute;

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = precompute.geti();

      if (foralli.getIndexVar() == i) {
        IndexStmt s = foralli.getStmt();
        TensorVar ws = precompute.getWorkspace();
        IndexExpr e = precompute.getExpr();
        IndexVar iw = precompute.getiw();

        IndexStmt consumer = forall(i, replace(s, {{e, ws(i)}}));
        IndexStmt producer = forall(iw, ws(iw) = replace(e, {{i,iw}}));
        Where where(consumer, producer);

        stmt = where;
        return;
      }
      IndexNotationRewriter::visit(node);
    }

  };
  PrecomputeRewriter rewriter;
  rewriter.precompute = *this;
  return rewriter.rewrite(stmt);
}

void Precompute::print(std::ostream& os) const {
  os << "precompute(" << getExpr() << ", " << geti() << ", "
     << getiw() << ", " << getWorkspace() << ")";
}

bool Precompute::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Precompute& precompute) {
  precompute.print(os);
  return os;
}

// class ForAllReplace
struct ForAllReplace::Content {
  std::vector<IndexVar> pattern;
  std::vector<IndexVar> replacement;
};

ForAllReplace::ForAllReplace() : content(nullptr) {
}

ForAllReplace::ForAllReplace(std::vector<IndexVar> pattern, std::vector<IndexVar> replacement) : content(new Content) {
  taco_iassert(!pattern.empty());
  content->pattern = pattern;
  content->replacement = replacement;
}

std::vector<IndexVar> ForAllReplace::getPattern() const {
  return content->pattern;
}

std::vector<IndexVar> ForAllReplace::getReplacement() const {
  return content->replacement;
}

IndexStmt ForAllReplace::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  /// Since all IndexVars can only appear once, assume replacement will work and error if it doesn't
  struct ForAllReplaceRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    ForAllReplace transformation;
    string* reason;
    int elementsMatched = 0;
    ForAllReplaceRewriter(ForAllReplace transformation, string* reason)
            : transformation(transformation), reason(reason) {}

    IndexStmt forallreplace(IndexStmt stmt) {
      IndexStmt replaced = rewrite(stmt);

      // Precondition: Did not find pattern
      if (replaced == stmt || elementsMatched == -1) {
        *reason = "The pattern of ForAlls: " +
                  util::join(transformation.getPattern()) +
                  " was not found while attempting to replace with: " +
                  util::join(transformation.getReplacement());
        return IndexStmt();
      }
      return replaced;
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      vector<IndexVar> pattern = transformation.getPattern();
      if (elementsMatched == -1) {
        return; // pattern did not match
      }

      if(elementsMatched >= (int) pattern.size()) {
        IndexNotationRewriter::visit(node);
        return;
      }

      if (foralli.getIndexVar() == pattern[elementsMatched]) {
        if (elementsMatched + 1 < (int) pattern.size() && !isa<Forall>(foralli.getStmt())) {
          // child is not a forallnode (not directly nested)
          elementsMatched = -1;
          return;
        }
        // assume rest of pattern matches
        vector<IndexVar> replacement = transformation.getReplacement();
        bool firstMatch = (elementsMatched == 0);
        elementsMatched++;
        stmt = rewrite(foralli.getStmt());
        if (firstMatch) {
          // add replacement nodes and cut out this node
          for (auto i = replacement.rbegin(); i != replacement.rend(); ++i ) {
            stmt = forall(*i, stmt);
          }
        }
        // else cut out this node
        return;
      }
      else if (elementsMatched > 0) {
        elementsMatched = -1; // pattern did not match
        return;
      }
      // before pattern match
      IndexNotationRewriter::visit(node);
    }
  };
  return ForAllReplaceRewriter(*this, reason).forallreplace(stmt);
}

void ForAllReplace::print(std::ostream& os) const {
  os << "forallreplace(" << util::join(getPattern()) << ", " << util::join(getReplacement()) << ")";
}

std::ostream& operator<<(std::ostream& os, const ForAllReplace& forallreplace) {
  forallreplace.print(os);
  return os;
}

// class AddSuchThatRels
struct AddSuchThatPredicates::Content {
  std::vector<IndexVarRel> predicates;
  std::map<IndexVar, std::shared_ptr<LeafCallInterface>> calls;
};

AddSuchThatPredicates::AddSuchThatPredicates() : content(nullptr) {
}

AddSuchThatPredicates::AddSuchThatPredicates(std::vector<IndexVarRel> predicates, std::map<IndexVar, std::shared_ptr<LeafCallInterface>> calls) : content(new Content) {
//  taco_iassert(!predicates.empty());
  content->predicates = predicates;
  content->calls = calls;
}

std::vector<IndexVarRel> AddSuchThatPredicates::getPredicates() const {
  return content->predicates;
}

IndexStmt AddSuchThatPredicates::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  if (isa<SuchThat>(stmt)) {
    SuchThat suchThat = to<SuchThat>(stmt);
    vector<IndexVarRel> predicate = suchThat.getPredicate();
    vector<IndexVarRel> predicates = getPredicates();
    predicate.insert(predicate.end(), predicates.begin(), predicates.end());
    auto call = suchThat.getCalls();
    auto calls = this->content->calls;
    call.insert(calls.begin(), calls.end());
    return SuchThat(suchThat.getStmt(), predicate, call);
  }
  else{
    return SuchThat(stmt, content->predicates, this->content->calls);
  }
}

void AddSuchThatPredicates::print(std::ostream& os) const {
  os << "addsuchthatpredicates(" << util::join(getPredicates()) << ")";
}

std::ostream& operator<<(std::ostream& os, const AddSuchThatPredicates& addSuchThatPredicates) {
  addSuchThatPredicates.print(os);
  return os;
}

struct ReplaceReductionExpr : public IndexNotationRewriter {
  const std::map<Access,Access>& substitutions;
  ReplaceReductionExpr(const std::map<Access,Access>& substitutions)
          : substitutions(substitutions) {}
  using IndexNotationRewriter::visit;
  void visit(const AssignmentNode* node) {
    if (util::contains(substitutions, node->lhs)) {
      stmt = Assignment(substitutions.at(node->lhs), rewrite(node->rhs), node->op);
    }
    else {
      IndexNotationRewriter::visit(node);
    }
  }
};


IndexStmt scalarPromote(IndexStmt stmt, ProvenanceGraph provGraph, 
                        bool isWholeStmt, bool promoteScalar);

// class Parallelize
struct Parallelize::Content {
  IndexVar i;
  ParallelUnit  parallel_unit;
  OutputRaceStrategy output_race_strategy;
  TensorVar assembling;
};


Parallelize::Parallelize() : content(nullptr) {
}

Parallelize::Parallelize(IndexVar i) : Parallelize(i, ParallelUnit::DefaultUnit, OutputRaceStrategy::NoRaces) {}

Parallelize::Parallelize(IndexVar i, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy) : Parallelize(i, parallel_unit, output_race_strategy, TensorVar()) {}

Parallelize::Parallelize(IndexVar i, ParallelUnit parallel_unit, OutputRaceStrategy output_race_strategy, TensorVar assembling) : content(new Content) {
  content->i = i;
  content->parallel_unit = parallel_unit;
  content->output_race_strategy = output_race_strategy;
  content->assembling = assembling;
}


IndexVar Parallelize::geti() const {
  return content->i;
}

ParallelUnit Parallelize::getParallelUnit() const {
  return content->parallel_unit;
}

OutputRaceStrategy Parallelize::getOutputRaceStrategy() const {
  return content->output_race_strategy;
}

IndexStmt Parallelize::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  struct ParallelizeRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Parallelize parallelize;
    ProvenanceGraph provGraph;
    map<TensorVar,ir::Expr> tensorVars;
    vector<ir::Expr> assembledByUngroupedInsert;
    set<IndexVar> definedIndexVars;
    set<ParallelUnit> parentParallelUnits;
    std::string reason = "";
    bool preservesNonZeroOutputStructure = false;

    IndexStmt rewriteParallel(IndexStmt stmt) {
      provGraph = ProvenanceGraph(stmt);
      tensorVars = createIRTensorVars(stmt);
      assembledByUngroupedInsert.clear();
      for (const auto& result : getAssembledByUngroupedInsertion(stmt)) {
        assembledByUngroupedInsert.push_back(tensorVars[result]);
      }
      {
        NonZeroAnalyzerResult res;
        this->preservesNonZeroOutputStructure = preservesNonZeroStructure(stmt, res);
      }
      if (this->parallelize.content->assembling.defined()) {
        assembledByUngroupedInsert.push_back(tensorVars[this->parallelize.content->assembling]);
      }
      return rewrite(stmt);
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = parallelize.geti();

      Iterators iterators(foralli, tensorVars);
      definedIndexVars.insert(foralli.getIndexVar());
      MergeLattice lattice = MergeLattice::make(foralli, iterators, provGraph, definedIndexVars);
      // Precondition 3: No parallelization of variables under a reduction
      // variable (ie MergePoint has at least 1 result iterators)
      if (parallelize.getOutputRaceStrategy() == OutputRaceStrategy::NoRaces && lattice.results().empty()
          && lattice != MergeLattice({MergePoint({iterators.modeIterator(foralli.getIndexVar())}, {}, {})})) {
        reason = "Precondition failed: Free variables cannot be dominated by reduction variables in the iteration graph, "
                 "as this causes scatter behavior and we do not yet emit parallel synchronization constructs";
        return;
      }

      if (foralli.getIndexVar() == i) {
        // Precondition 1: No coiteration of node (ie Merge Lattice has only 1 iterator)
        if (lattice.iterators().size() != 1) {
          reason = "Precondition failed: The loop must not merge tensor dimensions, that is, it must be a for loop;";
          return;
        }

        // Precondition 2: Every result iterator must have insert capability,
        // or the result tensor preserves the same non-zero structure as the input.
        if (!this->preservesNonZeroOutputStructure) {
          for (Iterator iterator : lattice.results()) {
            if (util::contains(assembledByUngroupedInsert, iterator.getTensor())) {
              for (Iterator it = iterator; !it.isRoot(); it = it.getParent()) {
                if (it.hasInsertCoord() || !it.isYieldPosPure()) {
                  reason = "Precondition failed: The output tensor does not "
                           "support parallelized inserts";
                  return;
                }
              }
            } else {
              while (true) {
                if (!iterator.hasInsert()) {
                  reason = "Precondition failed: The output tensor must allow inserts";
                  return;
                }
                if (iterator.isLeaf()) {
                  break;
                }
                iterator = iterator.getChild();
              }
            }
          }
        }

        vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(i);
        IndexVar underivedAncestor = underivedAncestors.back();

        // get lattice that corresponds to underived ancestor. This is bottom-most loop that shares underived ancestor
        Forall underivedForall = foralli;
        match(foralli.getStmt(),
              function<void(const ForallNode*)>([&](const ForallNode* node) {
                vector<IndexVar> nodeUnderivedAncestors = provGraph.getUnderivedAncestors(node->indexVar);
                if (underivedAncestor == nodeUnderivedAncestors.back()) {
                  underivedForall = Forall(node);
                }
              })
        );
        MergeLattice underivedLattice = MergeLattice::make(underivedForall, iterators, provGraph, definedIndexVars);


        if(underivedLattice.results().empty() && parallelize.getOutputRaceStrategy() == OutputRaceStrategy::Temporary) {
          // Need to precompute reduction

          // Find all occurrences of reduction in expression
          vector<const AssignmentNode *> precomputeAssignments;
          match(foralli.getStmt(),
                function<void(const AssignmentNode*)>([&](const AssignmentNode* node) {
                  for (auto underivedVar : underivedAncestors) {
                    vector<IndexVar> reductionVars = Assignment(node).getReductionVars();
                    bool reducedByI =
                            find(reductionVars.begin(), reductionVars.end(), underivedVar) != reductionVars.end();
                    if (reducedByI) {
                      precomputeAssignments.push_back(node);
                      break;
                    }
                  }
                })
          );
          taco_iassert(!precomputeAssignments.empty());

          IndexStmt precomputed_stmt = forall(i, foralli.getStmt(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), node->transfers, node->computingOn, foralli.getUnrollFactor());
          for (auto assignment : precomputeAssignments) {
            // Construct temporary of correct type and size of outer loop
            TensorVar w(string("w_") + ParallelUnit_NAMES[(int) parallelize.getParallelUnit()], Type(assignment->lhs.getDataType(), {Dimension(i)}), taco::dense);

            // rewrite producer to write to temporary, mark producer as parallel
            IndexStmt producer = ReplaceReductionExpr(map<Access, Access>({{assignment->lhs, w(i)}})).rewrite(precomputed_stmt);
            taco_iassert(isa<Forall>(producer));
            Forall producer_forall = to<Forall>(producer);
            producer = forall(producer_forall.getIndexVar(), producer_forall.getStmt(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), node->transfers, node->computingOn, foralli.getUnrollFactor());

            // build consumer that writes from temporary to output, mark consumer as parallel reduction
            ParallelUnit reductionUnit = ParallelUnit::CPUThreadGroupReduction;
            if (parentParallelUnits.count(ParallelUnit::GPUWarp)) {
              reductionUnit = ParallelUnit::GPUWarpReduction;
            }
            else {
              reductionUnit = ParallelUnit::GPUBlockReduction;
            }
            IndexStmt consumer = forall(i, Assignment(assignment->lhs, w(i), assignment->op), reductionUnit, OutputRaceStrategy::ParallelReduction, node->transfers, node->computingOn);
            precomputed_stmt = where(consumer, producer);
          }
          stmt = precomputed_stmt;
          return;
        }

        if (parallelize.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
          // want to avoid extra atomics by accumulating variable and then 
          // reducing at end
          IndexStmt body = scalarPromote(foralli.getStmt(), provGraph, 
                                         false, true);
          stmt = forall(i, body, parallelize.getParallelUnit(), 
                        parallelize.getOutputRaceStrategy(), 
                        node->transfers, node->computingOn, foralli.getUnrollFactor());
          return;
        }


        stmt = forall(i, foralli.getStmt(), parallelize.getParallelUnit(), parallelize.getOutputRaceStrategy(), node->transfers, node->computingOn, foralli.getUnrollFactor());
        return;
      }

      if (foralli.getParallelUnit() != ParallelUnit::NotParallel) {
        parentParallelUnits.insert(foralli.getParallelUnit());
      }
      IndexNotationRewriter::visit(node);
    }

    void visit(const AssembleNode* op) {
      IndexVar i = parallelize.geti();
      IndexStmt queries = util::contains(op->queries.getIndexVars(), i) 
                        ? rewrite(op->queries) : op->queries;
      IndexStmt compute = util::contains(op->compute.getIndexVars(), i) 
                        ? rewrite(op->compute) : op->compute;
      if (queries == op->queries && compute == op->compute) {
        stmt = op;
      }
      else {
        stmt = new AssembleNode(queries, compute, op->results);
      }
    }

    void visit(const WhereNode* op) {
      IndexVar i = parallelize.geti();
      IndexStmt producer = util::contains(op->producer.getIndexVars(), i) 
                        ? rewrite(op->producer) : op->producer;
      IndexStmt consumer = util::contains(op->consumer.getIndexVars(), i) 
                        ? rewrite(op->consumer) : op->consumer;
      if (producer == op->producer && consumer == op->consumer) {
        stmt = op;
      }
      else {
        stmt = new WhereNode(consumer, producer);
      }
    }
  };

  ParallelizeRewriter rewriter;
  rewriter.parallelize = *this;
  IndexStmt rewritten = rewriter.rewriteParallel(stmt);
  if (!rewriter.reason.empty()) {
    *reason = rewriter.reason;
    return IndexStmt();
  }
  return rewritten;
}


void Parallelize::print(std::ostream& os) const {
  os << "parallelize(" << geti() << ")";
}


std::ostream& operator<<(std::ostream& os, const Parallelize& parallelize) {
  parallelize.print(os);
  return os;
}

// Distribution transformation related code.

struct Distribute::Content {
  std::vector<IndexVar> original;
  std::vector<IndexVar> distVars;
  std::vector<IndexVar> innerVars;
  // The parallel unit to distribute over.
  ParallelUnit parUnit;
  // Only one of onto and grid is set.
  Grid grid;
  std::vector<Access> onto;
};

Distribute::Distribute() : content(nullptr) {}

Distribute::Distribute(std::vector<IndexVar> original, std::vector<IndexVar> distVars, std::vector<IndexVar> innerVars,
                       Grid &g, ParallelUnit parUnit) : content(new Content) {
  // TODO (rohany): Should assert many things here: g.dims == original.size(),
  //  all index var vectors have the same size etc.
  this->content->original = original;
  this->content->distVars = distVars;
  this->content->innerVars = innerVars;
  this->content->grid = g;
  this->content->parUnit = parUnit;
}

Distribute::Distribute(std::vector<IndexVar> original, std::vector<IndexVar> distVars, std::vector<IndexVar> innerVars,
                       Access onto, ParallelUnit parUnit) : Distribute(original, distVars, innerVars, std::vector<Access>{onto}, parUnit) {}

Distribute::Distribute(std::vector<IndexVar> original, std::vector<IndexVar> distVars, std::vector<IndexVar> innerVars,
                       std::vector<Access> onto, ParallelUnit parUnit) : content(new Content) {
  // TODO (rohany): Should assert many things here: g.dims == original.size(),
  //  all index var vectors have the same size etc.
  this->content->original = original;
  this->content->distVars = distVars;
  this->content->innerVars = innerVars;
  this->content->parUnit = parUnit;
  this->content->onto = onto;
}

IndexStmt Distribute::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  // Sanity check some user-provided input sizes.
  taco_iassert(this->content->original.size() == this->content->distVars.size());
  taco_iassert(this->content->original.size() == this->content->innerVars.size());
  taco_iassert(this->content->original.size() == size_t(this->content->grid.getDim()));

  ProvenanceGraph pg(stmt);

  // TODO (rohany): Make sure that we can either insert into all the tensors
  //  or it preserves the non-zero structure of the input.

  OutputRaceStrategy raceStrategy = OutputRaceStrategy::NoRaces;
  // For each variable being distributed, see if there is a reduction occuring.
  // TODO (rohany): See how this differs from the parallelize check?
  for (size_t i = 0; i < this->content->original.size(); i++) {
    Forall target;
    std::set<IndexVar> definedIndexVars;
    match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
      if (!target.defined()) {
        definedIndexVars.insert(node->indexVar);
      }
      if (node->indexVar == this->content->original[i]) {
        target = node;
      }
    }));

    Iterators iterators(target);
    auto lattice = MergeLattice::make(target, iterators, pg, definedIndexVars);
    if (lattice.results().empty() && lattice != MergeLattice({MergePoint({iterators.modeIterator(target.getIndexVar())}, {}, {})})) {
      // We've found a reduction that we're attempting to parallelize over.
      raceStrategy = OutputRaceStrategy::ParallelReduction;
    }
    // If the forall has a fused position space, then it's probably going to require a reduction as well.
    if (pg.getParentPosRel(target.getIndexVar()) != nullptr) {
      auto posrel = pg.getParentPosRel(target.getIndexVar());
      auto underiveds = pg.getUnderivedAncestors(target.getIndexVar());
      auto resultAccesses = getResultAccesses(stmt).first;
      taco_iassert(resultAccesses.size() == 1);
      auto result = resultAccesses[0];
      taco_iassert(posrel->getAccess().getTensorVar() != result.getTensorVar());
      // If the result access has all of the same index variables as the input
      // tensor who's position space we are iterating over, then we don't need
      // to do a reduction, as each non-zero will be processed by a unique processor.
      // Another way of saying this is that each output position will be handled
      // by a unique processor if the input and output are accessed by the same fused
      // index variables.
      for (auto var : underiveds) {
        if (!util::contains(result.getIndexVars(), var) || !util::contains(posrel->getAccess().getIndexVars(), var)){
          raceStrategy = OutputRaceStrategy::ParallelReduction;
        }
      }
    }
  }

  // Initial implementation:
  // For each original variable, divide the loop into dimension of the grid pieces.
  // Then reorder the loops so that it's distVars -> innerVars.

  if (this->content->grid.defined()) {
    for (size_t i = 0; i < this->content->original.size(); i++) {
      stmt = stmt.divide(this->content->original[i], this->content->distVars[i], this->content->innerVars[i],
                         this->content->grid.getDimSize(i));
    }
  } else {
    taco_iassert(!this->content->onto.empty());
    auto& part = *this->content->onto.begin();
    for (size_t i = 0; i < this->content->original.size(); i++) {
      // Find the target variable.
      auto accessIdx = -1;
      for (size_t j = 0; j < part.getIndexVars().size(); j++) {
        if (this->content->original[i] == part.getIndexVars()[j]) {
          accessIdx = j;
        }
      }
      assert(accessIdx != -1);

      auto rel = IndexVarRel(new DivideOntoPartition(this->content->original[i], this->content->distVars[i], this->content->innerVars[i], part, accessIdx));
      stmt = Transformation(AddSuchThatPredicates({rel})).apply(stmt, reason);
      if (!stmt.defined()) {
        taco_uerror << reason;
      }
      stmt = Transformation(ForAllReplace({this->content->original[i]}, {this->content->distVars[i], this->content->innerVars[i]})).apply(stmt, reason);
      if (!stmt.defined()) {
        taco_uerror << reason;
      }
    }
  }

  // Note that reorder drops parallel annotations on loops, so add the annotations later.
  std::vector<IndexVar> order;
  order.insert(order.end(), this->content->distVars.begin(), this->content->distVars.end());
  order.insert(order.end(), this->content->innerVars.begin(), this->content->innerVars.end());
  stmt = stmt.reorder(order);

  struct distNameCounter : public IndexNotationVisitor {
    void visit(const ForallNode* node) {
      if (node->indexVar.getName().find("distFused") != std::string::npos) {
        this->count++;
      }
      node->stmt.accept(this);
    }
    int count = 0;
  } counter; stmt.accept(&counter);

  std::stringstream varname;
  varname << "distFused";
  if (counter.count != 0) {
    varname << counter.count;
  }

  IndexVar distFused(varname.str());
  if (this->content->distVars.size() > 1) {
    IndexVarRel rel = IndexVarRel(new MultiFuseRelNode(distFused, this->content->distVars));
    stmt = Transformation(AddSuchThatPredicates({rel})).apply(stmt, reason);
    if (!stmt.defined()) {
      taco_uerror << reason;
    }
    stmt = Transformation(ForAllReplace(this->content->distVars, {distFused})).apply(stmt, reason);
    if (!stmt.defined()) {
      taco_uerror << reason;
    }
  }

  // Mark for loops over the distributed variables as actually distributed.
  struct DistributedForallMarker : public IndexNotationRewriter {
    void visit(const ForallNode* node) {
      // TODO (rohany): Also need to mark the fused var.
      if (util::contains(this->distVars, node->indexVar) || node->indexVar == this->distFused) {
        stmt = forall(node->indexVar, rewrite(node->stmt), this->parUnit, this->raceStrategy, node->transfers, this->computingOn, node->unrollFactor);
      } else {
        IndexNotationRewriter::visit(node);
      }
    }
    std::set<IndexVar> distVars;
    IndexVar distFused;
    std::vector<TensorVar> computingOn;
    OutputRaceStrategy raceStrategy;
    ParallelUnit parUnit;
  };
  DistributedForallMarker m; m.distFused = distFused; m.raceStrategy = raceStrategy; m.parUnit = this->content->parUnit;
  if (!this->content->onto.empty()) {
    for (auto a : this->content->onto) {
      m.computingOn.push_back(a.getTensorVar());
    }
  }
  m.distVars.insert(this->content->distVars.begin(), this->content->distVars.end());
  stmt = m.rewrite(stmt);

  return stmt;
}

void Distribute::print(std::ostream& os) const {
  os << "distribute(" << util::join(this->content->original) << ")";
}

// class SetAssembleStrategy 

struct SetAssembleStrategy::Content {
  TensorVar result;
  AssembleStrategy strategy;
};

SetAssembleStrategy::SetAssembleStrategy(TensorVar result, 
                                         AssembleStrategy strategy) : 
    content(new Content) {
  content->result = result;
  content->strategy = strategy;
}

TensorVar SetAssembleStrategy::getResult() const {
  return content->result;
}

AssembleStrategy SetAssembleStrategy::getAssembleStrategy() const {
  return content->strategy;
}

IndexStmt SetAssembleStrategy::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  if (getAssembleStrategy() == AssembleStrategy::Append) {
    return stmt;
  }

  bool hasSeqInsertEdge = false;
  bool hasInsertCoord = false;
  bool hasNonpureYieldPosForUniqueMode = false;
  for (const auto& modeFormat : getResult().getFormat().getModeFormats()) {
    if (hasSeqInsertEdge) {
      if (modeFormat.hasSeqInsertEdge()) {
        *reason = "Precondition failed: The output tensor does not support "
                  "ungrouped insertion (cannot have multiple modes requiring "
                  "non-trivial edge insertion)";
        return IndexStmt();
      }
    } else {
      hasSeqInsertEdge = (hasSeqInsertEdge || modeFormat.hasSeqInsertEdge());
      if (modeFormat.hasSeqInsertEdge()) {
        if (hasInsertCoord) {
          *reason = "Precondition failed: The output tensor does not support "
                    "ungrouped insertion (cannot have mode requiring "
                    "non-trivial coordinate insertion above mode requiring "
                    "non-trivial edge insertion)";
          return IndexStmt();
        }
        hasSeqInsertEdge = true;
      }
      hasInsertCoord = (hasInsertCoord || modeFormat.hasInsertCoord());
    }
    if (hasNonpureYieldPosForUniqueMode) {
      *reason = "Precondition failed: The output tensor does not support "
                "ungrouped insertion (only last mode can have non-pure "
                "implementation of yield_pos and be unique)";
      return IndexStmt();
    } else if (!modeFormat.isYieldPosPure() && modeFormat.isUnique()) {
      hasNonpureYieldPosForUniqueMode = true;
    }
  }

  std::map<IndexVar,IndexVar> ivReplacements;
  for (const auto& indexVar : getIndexVars(stmt)) {
    ivReplacements[indexVar] = IndexVar("q" + indexVar.getName());
  }
  IndexStmt loweredQueries = replace(stmt, ivReplacements);

  // FIXME: Unneeded if scalar promotion is made default when concretizing
  loweredQueries = scalarPromote(loweredQueries);

  // Tracks all tensors that correspond to attribute query results or that are 
  // used to compute attribute queries
  std::set<TensorVar> insertedResults;  

  // Lower attribute queries to canonical forms using same schedule as 
  // actual computation
  Assemble::AttrQueryResults queryResults;
  struct LowerAttrQuery : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    TensorVar result;
    Assemble::AttrQueryResults& queryResults;
    std::set<TensorVar>& insertedResults;
    std::vector<TensorVar> arguments;
    std::vector<TensorVar> temps;
    std::map<TensorVar,TensorVar> tempReplacements;
    IndexStmt epilog;
    std::string reason = "";

    LowerAttrQuery(TensorVar result, Assemble::AttrQueryResults& queryResults, 
                   std::set<TensorVar>& insertedResults) : 
        result(result), queryResults(queryResults), 
        insertedResults(insertedResults) {}

    IndexStmt lower(IndexStmt stmt) {
      arguments = getArguments(stmt);
      temps = getTemporaries(stmt);
      for (const auto& tmp : temps) {
        tempReplacements[tmp] = TensorVar("q" + tmp.getName(), 
                                          Type(Bool, tmp.getType().getShape()), 
                                          tmp.getFormat());
      }

      queryResults = Assemble::AttrQueryResults();
      epilog = IndexStmt();
      stmt = IndexNotationRewriter::rewrite(stmt);
      if (epilog.defined()) {
        stmt = Where(epilog, stmt);
      }
      return stmt;
    }

    void visit(const ForallNode* op) {
      IndexStmt s = rewrite(op->stmt);
      if (s == op->stmt) {
        stmt = op;
      } else if (s.defined()) {
        stmt = new ForallNode(op->indexVar, s, op->parallel_unit, 
                              op->output_race_strategy, op->transfers, op->computingOn, op->unrollFactor);
      } else {
        stmt = IndexStmt();
      }
    }

    void visit(const AssignmentNode* op) {
      IndexExpr rhs = rewrite(op->rhs);

      const auto resultAccess = op->lhs;
      const auto resultTensor = resultAccess.getTensorVar();
      
      if (resultTensor != result) {
        Access lhs = to<Access>(rewrite(op->lhs));
        stmt = (rhs != op->rhs) ? Assignment(lhs, rhs, op->op) : op;
        return;
      }

      if (op->op.defined()) {
        reason = "Precondition failed: Ungrouped insertion not support for "
                 "output tensors that are scattered into";
        return;
      }

      queryResults[resultTensor] = 
          std::vector<std::vector<TensorVar>>(resultTensor.getOrder());

      const auto indices = resultAccess.getIndexVars();
      const auto modeFormats = resultTensor.getFormat().getModeFormats();
      const auto modeOrdering = resultTensor.getFormat().getModeOrdering();

      std::vector<IndexVar> parentCoords;
      std::vector<IndexVar> childCoords;
      for (size_t i = 0; i < indices.size(); ++i) {
        childCoords.push_back(indices[modeOrdering[i]]);
      }

      for (size_t i = 0; i < indices.size(); ++i) {
        const auto modeName = resultTensor.getName() + std::to_string(i + 1);

        parentCoords.push_back(indices[modeOrdering[i]]);
        childCoords.erase(childCoords.begin());

        for (const auto& query: 
            modeFormats[i].getAttrQueries(parentCoords, childCoords)) {
          const auto& groupBy = query.getGroupBy();

          // TODO: support multiple aggregations in single query
          taco_iassert(query.getAttrs().size() == 1); 

          std::vector<Dimension> queryDims;
          for (const auto& coord : groupBy) {
            const auto pos = std::find(groupBy.begin(), groupBy.end(), coord) 
                           - groupBy.begin();
            const auto dim = resultTensor.getType().getShape().getDimension(pos);
            queryDims.push_back(dim);
          }

          for (const auto& attr : query.getAttrs()) {
            switch (attr.aggr) {
              case AttrQuery::COUNT:
              {
                std::vector<IndexVar> dedupCoords = groupBy;
                dedupCoords.insert(dedupCoords.end(), attr.params.begin(), 
                                   attr.params.end());
                std::vector<Dimension> dedupDims(dedupCoords.size());
                TensorVar dedupTmp(modeName + "_dedup", Type(Bool, dedupDims));
                stmt = Assignment(dedupTmp(dedupCoords), rhs, Add());
                insertedResults.insert(dedupTmp);

                const auto resultName = modeName + "_" + attr.label;
                TensorVar queryResult(resultName, Type(Int(), queryDims));
                epilog = Assignment(queryResult(groupBy), 
                                    Cast(dedupTmp(dedupCoords), Int()), Add());
                for (const auto& coord : util::reverse(dedupCoords)) {
                  epilog = forall(coord, epilog);
                }
                insertedResults.insert(queryResult);

                queryResults[resultTensor][i] = {queryResult};
                return;
              }
              case AttrQuery::IDENTITY:
              case AttrQuery::MIN:
              case AttrQuery::MAX:
              default:
                taco_not_supported_yet;
                break;
            }
          }
        }
      }

      stmt = IndexStmt();
    }

    void visit(const AccessNode* op) {
      if (util::contains(arguments, op->tensorVar)) {
        expr = Access(op->tensorVar, op->indexVars, op->packageModifiers(),
                      true);
        return;
      } else if (util::contains(temps, op->tensorVar)) {
        expr = Access(tempReplacements[op->tensorVar], op->indexVars,
                      op->packageModifiers());
        return;
      }

      expr = op;
    }
  };
  LowerAttrQuery queryLowerer(getResult(), queryResults, insertedResults);
  loweredQueries = queryLowerer.lower(loweredQueries);

  if (!queryLowerer.reason.empty()) {
    *reason = queryLowerer.reason;
    return IndexStmt();
  }

  // Convert redundant reductions to assignments
  struct ReduceToAssign : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::set<TensorVar>& insertedResults;
    std::set<IndexVar> availableVars;

    ReduceToAssign(const std::set<TensorVar>& insertedResults) :
        insertedResults(insertedResults) {}

    void visit(const ForallNode* op) {
      availableVars.insert(op->indexVar);
      IndexNotationRewriter::visit(op);
      availableVars.erase(op->indexVar);
    }
    
    void visit(const AssignmentNode* op) {
      std::set<IndexVar> accessVars;
      for (const auto& index : op->lhs.getIndexVars()) {
        accessVars.insert(index);
      }

      if (op->op.defined() && accessVars == availableVars && 
          util::contains(insertedResults, op->lhs.getTensorVar())) {
        stmt = new AssignmentNode(op->lhs, op->rhs, IndexExpr());
        return;
      }

      stmt = op;
    }
  };
  loweredQueries = ReduceToAssign(insertedResults).rewrite(loweredQueries);

  // Inline definitions of temporaries into their corresponding uses, as long 
  // as the temporaries are not the results of reductions
  std::set<TensorVar> inlinedResults;
  struct InlineTemporaries : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::set<TensorVar>& insertedResults;
    std::set<TensorVar>& inlinedResults;
    std::map<TensorVar,std::pair<IndexExpr,Assignment>> tmpUse;

    InlineTemporaries(const std::set<TensorVar>& insertedResults,
                      std::set<TensorVar>& inlinedResults) :
        insertedResults(insertedResults), inlinedResults(inlinedResults) {}

    void visit(const WhereNode* op) {
      IndexStmt consumer = rewrite(op->consumer);
      IndexStmt producer = rewrite(op->producer);
      if (producer == op->producer && consumer == op->consumer) {
        stmt = op;
      } else {
        stmt = new WhereNode(consumer, producer);
      }
    }

    void visit(const AssignmentNode* op) {
      const auto lhsTensor = op->lhs.getTensorVar();
      if (util::contains(tmpUse, lhsTensor) && !op->op.defined()) {
        std::map<IndexVar,IndexVar> indexMap;
        const auto& oldIndices = 
            to<Access>(tmpUse[lhsTensor].first).getIndexVars();
        const auto& newIndices = op->lhs.getIndexVars();
        for (const auto& mapping : util::zip(oldIndices, newIndices)) {
          indexMap[mapping.first] = mapping.second;
        }

        std::vector<IndexVar> newCoords;
        const auto& oldCoords = 
            tmpUse[lhsTensor].second.getLhs().getIndexVars();
        for (const auto& oldCoord : oldCoords) {
          newCoords.push_back(indexMap.at(oldCoord));
        }

        IndexExpr reduceOp = tmpUse[lhsTensor].second.getOperator();
        TensorVar queryResult = 
            tmpUse[lhsTensor].second.getLhs().getTensorVar();
        IndexExpr rhs = op->rhs;
        if (rhs.getDataType() != queryResult.getType().getDataType()) {
          rhs = Cast(rhs, queryResult.getType().getDataType());
        }
        stmt = Assignment(queryResult(newCoords), rhs, reduceOp);
        inlinedResults.insert(queryResult);
        return;
      } 

      const Access rhsAccess = isa<Access>(op->rhs) ? to<Access>(op->rhs)
          : (isa<Cast>(op->rhs) && isa<Access>(to<Cast>(op->rhs).getA()))
            ? to<Access>(to<Cast>(op->rhs).getA()) : Access();
      if (rhsAccess.defined()) {
        const auto rhsTensor = rhsAccess.getTensorVar();
        if (util::contains(insertedResults, rhsTensor)) {
          tmpUse[rhsTensor] = std::make_pair(rhsAccess, Assignment(op));
        }
      }
      stmt = op;
    }
  };
  loweredQueries = InlineTemporaries(insertedResults, 
                                     inlinedResults).rewrite(loweredQueries);

  // Eliminate computation of redundant temporaries
  struct EliminateRedundantTemps : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::set<TensorVar>& inlinedResults;

    EliminateRedundantTemps(const std::set<TensorVar>& inlinedResults) :
        inlinedResults(inlinedResults) {}

    void visit(const ForallNode* op) {
      IndexStmt s = rewrite(op->stmt);
      if (s == op->stmt) {
        stmt = op;
      } else if (s.defined()) {
        stmt = new ForallNode(op->indexVar, s, op->parallel_unit, 
                              op->output_race_strategy, op->transfers, op->computingOn, op->unrollFactor);
      } else {
        stmt = IndexStmt();
      }
    }

    void visit(const WhereNode* op) {
      IndexStmt consumer = rewrite(op->consumer);
      if (consumer == op->consumer) {
        stmt = op;
      } else if (consumer.defined()) {
        stmt = new WhereNode(consumer, op->producer);
      } else {
        stmt = op->producer;
      }
    }

    void visit(const AssignmentNode* op) {
      const auto lhsTensor = op->lhs.getTensorVar();
      if (util::contains(inlinedResults, lhsTensor)) {
        stmt = IndexStmt();
      } else {
        stmt = op;
      }
    }
  };
  loweredQueries = 
      EliminateRedundantTemps(inlinedResults).rewrite(loweredQueries);

  return Assemble(loweredQueries, stmt, queryResults);
}

void SetAssembleStrategy::print(std::ostream& os) const {
  os << "assemble(" << getResult() << ", " 
     << AssembleStrategy_NAMES[(int)getAssembleStrategy()] << ")";
}

std::ostream& operator<<(std::ostream& os, 
                         const SetAssembleStrategy& assemble) {
  assemble.print(os);
  return os;
}

static bool compare(std::vector<IndexVar> vars1, std::vector<IndexVar> vars2) {
  return vars1 == vars2;
}

IndexVar getRootParent(ProvenanceGraph pg, IndexVar var) {
  auto parents = pg.getParents(var);
  if (parents.empty()) {
    return var;
  } else if (parents.size() == 1) {
    return getRootParent(pg, parents[0]);
  } else {
    taco_ierror << "can we get here?" << std::endl;
    return IndexVar();
  }
}

struct GEMM::Content {
  std::vector<TensorVar> tensorVars;
  std::vector<IndexVar> ivars;
  IndexVar rootIvar;
};

GEMM::GEMM() : content(new Content) {}

void GEMM::print(std::ostream &os) const {
  os << "GEMM";
}

IndexVar GEMM::getRootIvar() const {
  return this->content->rootIvar;
}

void GEMM::canApply(IndexStmt stmt, ProvenanceGraph pg, IndexVar root, std::string *reason) const {
  INIT_REASON(reason);

  // Find the target set of loops we want to replace with a GEMM call.
  struct Finder : IndexNotationVisitor {
    void visit(const ForallNode* node) {
      if (node->indexVar == this->target) {
        this->root = node;
      }
      node->stmt.accept(this);
    }

    IndexStmt root;
    IndexVar target;
  };
  Finder f; f.target = root;
  stmt.accept(&f);

  auto rootStmt = f.root;

  struct IVarCollector : IndexNotationVisitor {
    void visit(const ForallNode* node) {
      this->vars.push_back(node->indexVar);
      node->stmt.accept(this);
    }
    std::vector<IndexVar> vars;
  };
  IVarCollector c;
  rootStmt.accept(&c);

  auto indexVars = c.vars;

  if (indexVars.size() != 3) {
    taco_uerror << "vars to replace must be 3 nested loops.";
  }

  std::vector<IndexVar> rootVars;
  for (auto var : indexVars) {
    rootVars.push_back(getRootParent(pg, var));
  }

  // Get out the assignment.
  taco_iassert(isa<Forall>(rootStmt)); auto iloop = to<Forall>(rootStmt);
  taco_iassert(isa<Forall>(iloop.getStmt())); auto jloop = to<Forall>(iloop.getStmt());
  taco_iassert(isa<Forall>(jloop.getStmt())); auto kloop = to<Forall>(jloop.getStmt());
  taco_iassert(isa<Assignment>(kloop.getStmt())); auto assign = to<Assignment>(kloop.getStmt());
  // Extract out the tensors.
  taco_iassert(isa<Access>(assign.getLhs())); auto A = to<Access>(assign.getLhs());
  taco_iassert(isa<Mul>(assign.getRhs())); auto mul = to<Mul>(assign.getRhs());
  taco_iassert(isa<Access>(mul.getA())); auto B = to<Access>(mul.getA());
  taco_iassert(isa<Access>(mul.getB())); auto C = to<Access>(mul.getB());
  // Ensure that the variables are set up right.
  taco_iassert(compare(A.getIndexVars(), {rootVars[0], rootVars[1]}));
  taco_iassert(compare(B.getIndexVars(), {rootVars[0], rootVars[2]}));
  taco_iassert(compare(C.getIndexVars(), {rootVars[2], rootVars[1]}));

  taco_iassert(A.getTensorVar().getType().getDataType().isFloat());
  taco_iassert(A.getTensorVar().getType().getDataType() == B.getTensorVar().getType().getDataType());
  taco_iassert(B.getTensorVar().getType().getDataType() == C.getTensorVar().getType().getDataType());

  // At this point, we've found a matrix multiply that we can use.
  this->content->ivars = indexVars;
  this->content->tensorVars = {A.getTensorVar(), B.getTensorVar(), C.getTensorVar()};
  this->content->rootIvar = root;
}

ir::Stmt GEMM::replaceValidStmt(IndexStmt stmt,
                          ProvenanceGraph pg,
                          std::map<TensorVar, ir::Expr> tensorVars,
                          bool inReduction,
                          std::vector<IndexVar> definedVarOrder,
                          std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                          std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                          Iterators iterators
) const {
  // TODO (rohany): We could walk the statement again here and make sure that it's
  //  the same as the one we verified etc.

  std::vector<ir::Stmt> results;
  auto ctx = ir::Symbol::make("ctx");
  auto rowMajor = ir::Symbol::make("CblasRowMajor");
  auto noTrans = ir::Symbol::make("CblasNoTrans");

  std::vector<ir::Expr> tvars;
  for (auto var : this->content->tensorVars) {
    tvars.push_back(tensorVars[var]);
  }

  auto aIndexSpace = ir::GetProperty::make(tvars[0], ir::TensorProperty::IndexSpace);
  auto bIndexSpace = ir::GetProperty::make(tvars[1], ir::TensorProperty::IndexSpace);
  auto cIndexSpace = ir::GetProperty::make(tvars[2], ir::TensorProperty::IndexSpace);

  // Unpack the domains for each variable.
  auto aDomain = ir::Var::make("aDomain", Auto);
  auto bDomain = ir::Var::make("bDomain", Auto);
  auto cDomain = ir::Var::make("cDomain", Auto);
  results.push_back(ir::VarDecl::make(aDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, aIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(bDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, bIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(cDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, cIndexSpace}, Auto)));

  // Break out if either domain is empty.
  auto bVolZero = ir::Eq::make(ir::MethodCall::make(bDomain, "get_volume", {}, false, Auto), 0);
  auto cVolZero = ir::Eq::make(ir::MethodCall::make(cDomain, "get_volume", {}, false, Auto), 0);
  auto guard = ir::Or::make(bVolZero, cVolZero);
  results.push_back(ir::IfThenElse::make(guard, ir::Return::make(ir::Expr())));

  auto aAccess = ir::GetProperty::make(tvars[0], ir::TensorProperty::ValuesWriteAccessor, this->content->tensorVars[0].getOrder());
  if (inReduction) {
    aAccess = ir::GetProperty::make(tvars[0], ir::TensorProperty::ValuesReductionAccessor, this->content->tensorVars[0].getOrder());
  }
  auto bAccess = ir::GetProperty::make(tvars[1], ir::TensorProperty::ValuesReadAccessor, this->content->tensorVars[1].getOrder());
  auto cAccess = ir::GetProperty::make(tvars[2], ir::TensorProperty::ValuesReadAccessor, this->content->tensorVars[2].getOrder());

  auto getBounds = [] (ir::Expr e, std::string func) {
    return ir::MethodCall::make(e, func, {}, false, Int());
  };

  auto type = Type(this->content->tensorVars[0].getType().getDataType());
  auto ldA = ir::Div::make(
    ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(aAccess, "accessor", false, Auto), "strides", false, Int()), 0),
    ir::Sizeof::make(type)
  );
  auto ldB = ir::Div::make(
    ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(bAccess, "accessor", false, Auto), "strides", false, Int()), 0),
    ir::Sizeof::make(type)
  );
  auto ldC = ir::Div::make(
    ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(cAccess, "accessor", false, Auto), "strides", false, Int()), 0),
    ir::Sizeof::make(type)
  );

  std::vector<ir::Expr> args = {
      rowMajor,
      noTrans,
      noTrans,
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(bDomain, "hi"), 0), ir::Load::make(getBounds(bDomain, "lo"), 0))),
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(cDomain, "hi"), 1), ir::Load::make(getBounds(cDomain, "lo"), 1))),
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(cDomain, "hi"), 0), ir::Load::make(getBounds(cDomain, "lo"), 0))),
      1.f,
      ir::MethodCall::make(bAccess, "ptr", {getBounds(bDomain, "lo")}, false /* deref */, Auto),
      ldB,
      ir::MethodCall::make(cAccess, "ptr", {getBounds(cDomain, "lo")}, false /* deref */, Auto),
      ldC,
      1.f,
      ir::MethodCall::make(aAccess, "ptr", {getBounds(aDomain, "lo")}, false /* deref */, Auto),
      ldA,
  };

  // TODO (rohany): Pick the right call between double and float here.
  results.push_back(ir::SideEffect::make(ir::Call::make("cblas_dgemm", args, Auto)));
  return ir::Block::make(results);
}

void CuGEMM::print(std::ostream &os) const {
  os << "CuGEMM";
}

ir::Stmt CuGEMM::replaceValidStmt(IndexStmt stmt, ProvenanceGraph pg, std::map<TensorVar, ir::Expr> tensorVars,
                              bool inReduction, std::vector<IndexVar> definedVarOrder,
                              std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                              std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators) const {

  // TODO (rohany): We could walk the statement again here and make sure that it's
  //  the same as the one we verified etc.

  std::vector<ir::Stmt> stmts;
  auto ctx = ir::Symbol::make("ctx");
  auto noTrans = ir::Symbol::make("CUBLAS_OP_N");

  std::vector<ir::Expr> tvars;
  for (auto var : this->content->tensorVars) {
    tvars.push_back(tensorVars[var]);
  }

  auto aIndexSpace = ir::GetProperty::make(tvars[0], ir::TensorProperty::IndexSpace);
  auto bIndexSpace = ir::GetProperty::make(tvars[1], ir::TensorProperty::IndexSpace);
  auto cIndexSpace = ir::GetProperty::make(tvars[2], ir::TensorProperty::IndexSpace);

  // Unpack the domains for each variable.
  auto aDomain = ir::Var::make("aDomain", Auto);
  auto bDomain = ir::Var::make("bDomain", Auto);
  auto cDomain = ir::Var::make("cDomain", Auto);
  stmts.push_back(ir::VarDecl::make(aDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, aIndexSpace}, Auto)));
  stmts.push_back(ir::VarDecl::make(bDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, bIndexSpace}, Auto)));
  stmts.push_back(ir::VarDecl::make(cDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, cIndexSpace}, Auto)));

  // Break out if either domain is empty.
  auto bVolZero = ir::Eq::make(ir::MethodCall::make(bDomain, "get_volume", {}, false, Auto), 0);
  auto cVolZero = ir::Eq::make(ir::MethodCall::make(cDomain, "get_volume", {}, false, Auto), 0);
  auto guard = ir::Or::make(bVolZero, cVolZero);
  stmts.push_back(ir::IfThenElse::make(guard, ir::Return::make(ir::Expr())));

  auto aAccess = ir::GetProperty::make(tvars[0], ir::TensorProperty::ValuesWriteAccessor, this->content->tensorVars[0].getOrder());
  if (inReduction) {
    aAccess = ir::GetProperty::make(tvars[0], ir::TensorProperty::ValuesReductionAccessor, this->content->tensorVars[0].getOrder());
  }
  auto bAccess = ir::GetProperty::make(tvars[1], ir::TensorProperty::ValuesReadAccessor, this->content->tensorVars[1].getOrder());
  auto cAccess = ir::GetProperty::make(tvars[2], ir::TensorProperty::ValuesReadAccessor, this->content->tensorVars[2].getOrder());

  auto getBounds = [] (ir::Expr e, std::string func) {
    return ir::MethodCall::make(e, func, {}, false, Int());
  };

  auto type = Type(this->content->tensorVars[0].getType().getDataType());
  auto ldA = ir::Div::make(
      ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(aAccess, "accessor", false, Auto), "strides", false, Int()), 0),
      ir::Sizeof::make(type)
  );
  auto ldB = ir::Div::make(
      ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(bAccess, "accessor", false, Auto), "strides", false, Int()), 0),
      ir::Sizeof::make(type)
  );
  auto ldC = ir::Div::make(
      ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(cAccess, "accessor", false, Auto), "strides", false, Int()), 0),
      ir::Sizeof::make(type)
  );

  auto addr = [](ir::Expr e) {
    return ir::Call::make("&", {e}, Auto);
  };
  auto check = [](ir::Expr e) {
    return ir::Call::make("CHECK_CUBLAS", {e}, Auto);
  };

  // There's some extra book-keeping we have to do for CuBLAS that doesn't
  // have to happen for BLAS.
  auto alpha = ir::Var::make("alpha", Float64);
  stmts.push_back(ir::VarDecl::make(alpha, ir::Literal::make((double)1, Float64)));
  auto handleTy = Datatype("cublasHandle_t");
  auto streamTy = Datatype("cudaStream_t");
  // Get the CuBLAS handle.
  auto handle = ir::Var::make("handle", handleTy);
  stmts.push_back(ir::VarDecl::make(handle, ir::Call::make("getCuBLAS", {}, handleTy)));
  // Create a CUDA stream to launch the kernel on.
  auto stream = ir::Var::make("taskStream", streamTy);
  stmts.push_back(ir::VarDecl::make(stream, ir::makeConstructor(streamTy, {})));
  stmts.push_back(ir::SideEffect::make(ir::Call::make("cudaStreamCreate", {addr(stream)}, Auto)));
  // Attach the handle to the stream.
  stmts.push_back(ir::SideEffect::make(check(ir::Call::make("cublasSetStream", {handle, stream}, Auto))));
  // CuBLAS doesn't support row-major matrices. So, we can trick it by reversing
  // the order of the matrices in the call, and telling CuBLAS that the matrices
  // are in column-major format.
  std::vector<ir::Expr> args = {
      handle,
      noTrans,
      noTrans,
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(cDomain, "hi"), 1), ir::Load::make(getBounds(cDomain, "lo"), 1))),
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(bDomain, "hi"), 0), ir::Load::make(getBounds(bDomain, "lo"), 0))),
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(cDomain, "hi"), 0), ir::Load::make(getBounds(cDomain, "lo"), 0))),
      addr(alpha),
      ir::MethodCall::make(cAccess, "ptr", {getBounds(cDomain, "lo")}, false /* deref */, Auto),
      ldC,
      ir::MethodCall::make(bAccess, "ptr", {getBounds(bDomain, "lo")}, false /* deref */, Auto),
      ldB,
      addr(alpha),
      ir::MethodCall::make(aAccess, "ptr", {getBounds(aDomain, "lo")}, false /* deref */, Auto),
      ldA,
  };
  stmts.push_back(ir::SideEffect::make(check(ir::Call::make("cublasDgemm", args, Auto))));
  return ir::Block::make(stmts);
}

struct MTTKRP::Content {
  IndexVar root;
  TensorVar A;
  TensorVar B;
  TensorVar C;
  TensorVar D;
};

MTTKRP::MTTKRP() : content(new Content) {}

// TODO (rohany): Implement the MTTKRP validation.
void MTTKRP::canApply(IndexStmt stmt, ProvenanceGraph pg, IndexVar root, std::string *reason) const {
  // Find the target set of loops we want to replace with a MTTKRP call.
  struct Finder : IndexNotationVisitor {
    void visit(const ForallNode* node) {
      if (node->indexVar == this->target) {
        this->root = node;
      }
      node->stmt.accept(this);
    }

    IndexStmt root;
    IndexVar target;
  };
  Finder f; f.target = root;
  stmt.accept(&f);

  // TODO (rohany): Do validation here later.

  IndexStmt rootStmt = f.root;
  auto iLoop = to<Forall>(rootStmt);
  auto jLoop = to<Forall>(iLoop.getStmt());
  auto kLoop = to<Forall>(jLoop.getStmt());
  auto lLoop = to<Forall>(kLoop.getStmt());
  auto assign = to<Assignment>(lLoop.getStmt());
  auto A = to<Access>(assign.getLhs());
  auto mul = to<Mul>(assign.getRhs());
  auto mul2 = to<Mul>(mul.getA());
  auto B = to<Access>(mul2.getA());
  auto C = to<Access>(mul2.getB());
  auto D = to<Access>(mul.getB());

  this->content->root = root;
  this->content->A = A.getTensorVar();
  this->content->B = B.getTensorVar();
  this->content->C = C.getTensorVar();
  this->content->D = D.getTensorVar();
}

ir::Stmt MTTKRP::replaceValidStmt(IndexStmt stmt, ProvenanceGraph pg, std::map<TensorVar, ir::Expr> tensorVars,
                                  bool inReduction, std::vector<IndexVar> definedVarOrder,
                                  std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                                  std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators) const {
  auto A = tensorVars[this->content->A];
  auto B = tensorVars[this->content->B];
  auto C = tensorVars[this->content->C];
  auto D = tensorVars[this->content->D];

  std::vector<ir::Stmt> results;
  auto ctx = ir::Symbol::make("ctx");

  auto aIndexSpace = ir::GetProperty::make(A, ir::TensorProperty::IndexSpace);
  auto bIndexSpace = ir::GetProperty::make(B, ir::TensorProperty::IndexSpace);
  auto cIndexSpace = ir::GetProperty::make(C, ir::TensorProperty::IndexSpace);
  auto dIndexSpace = ir::GetProperty::make(D, ir::TensorProperty::IndexSpace);

  // Unpack domains for each variable.
  auto aDomain = ir::Var::make("aDomain", Auto);
  auto bDomain = ir::Var::make("bDomain", Auto);
  auto cDomain = ir::Var::make("cDomain", Auto);
  auto dDomain = ir::Var::make("dDomain", Auto);
  results.push_back(ir::VarDecl::make(aDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, aIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(bDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, bIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(cDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, cIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(dDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, dIndexSpace}, Auto)));

  // Break out if any domains are empty.
  auto bVolZero = ir::Eq::make(ir::MethodCall::make(bDomain, "get_volume", {}, false, Auto), 0);
  auto cVolZero = ir::Eq::make(ir::MethodCall::make(cDomain, "get_volume", {}, false, Auto), 0);
  auto dVolZero = ir::Eq::make(ir::MethodCall::make(dDomain, "get_volume", {}, false, Auto), 0);
  auto guard = ir::Or::make(ir::Or::make(bVolZero, cVolZero), dVolZero);
  results.push_back(ir::IfThenElse::make(guard, ir::Return::make(ir::Expr())));

  auto aAccess = ir::GetProperty::make(A, ir::TensorProperty::ValuesWriteAccessor, this->content->A.getOrder());
  if (inReduction) {
    aAccess = ir::GetProperty::make(A, ir::TensorProperty::ValuesReductionAccessor, this->content->A.getOrder());
  }
  auto bAccess = ir::GetProperty::make(B, ir::TensorProperty::ValuesReadAccessor, this->content->B.getOrder());
  auto cAccess = ir::GetProperty::make(C, ir::TensorProperty::ValuesReadAccessor, this->content->C.getOrder());
  auto dAccess = ir::GetProperty::make(D, ir::TensorProperty::ValuesReadAccessor, this->content->D.getOrder());

  // Set up the argument pack.
  auto packTy = Datatype("MTTKRPPack");
  auto pack = ir::Var::make("pack", packTy);
  results.push_back(ir::VarDecl::make(pack, ir::makeConstructor(packTy, {})));

  auto getBounds = [] (ir::Expr e, std::string func) {
    return ir::MethodCall::make(e, func, {}, false, Int());
  };
  auto ld = [] (ir::Expr e, int idx) {
    return ir::Load::make(e, idx);
  };
  auto bDomainHi = getBounds(bDomain, "hi");
  auto bDomainLo = getBounds(bDomain, "lo");
  auto iDim = ir::Add::make(1, ir::Sub::make(ld(bDomainHi, 0), ld(bDomainLo, 0)));
  auto jDim = ir::Add::make(1, ir::Sub::make(ld(bDomainHi, 1), ld(bDomainLo, 1)));
  auto kDim = ir::Add::make(1, ir::Sub::make(ld(bDomainHi, 2), ld(bDomainLo, 2)));
  auto lDim = ir::Add::make(1, ir::Sub::make(ld(getBounds(aDomain, "hi"), 1), ld(getBounds(aDomain, "lo"), 1)));
  auto type = Type(this->content->A.getType().getDataType());
  auto getLD = [&] (ir::Expr e, int idx) {
    return ir::Div::make(
        ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(e, "accessor", false, Auto), "strides", false, Int()), idx),
        ir::Sizeof::make(type)
    );
  };
  auto ldA = getLD(aAccess, 0);
  auto ldC = getLD(cAccess, 0);
  auto ldD = getLD(dAccess, 0);
  auto ldB3 = getLD(bAccess, 1);
  auto ldB2 = getLD(bAccess, 0);
  auto ldB1 = getLD(bAccess, 0);

  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "iDim", false, Auto), iDim));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "jDim", false, Auto), jDim));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "kDim", false, Auto), kDim));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "lDim", false, Auto), lDim));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldA", false, Auto), ldA));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldC", false, Auto), ldC));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldD", false, Auto), ldD));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldB1", false, Auto), ldB1));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldB2", false, Auto), ir::Div::make(ldB2, ldB3)));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldB3", false, Auto), ldB3));

  std::stringstream funcName;
  funcName << this->funcName << "<" << type.getDataType() << ">";

  results.push_back(ir::SideEffect::make(ir::Call::make(
      funcName.str(),
     {
       pack,
       ir::MethodCall::make(aAccess, "ptr", {getBounds(aDomain, "lo")}, false /* deref */, Auto),
       ir::MethodCall::make(bAccess, "ptr", {getBounds(bDomain, "lo")}, false /* deref */, Auto),
       ir::MethodCall::make(cAccess, "ptr", {getBounds(cDomain, "lo")}, false /* deref */, Auto),
       ir::MethodCall::make(dAccess, "ptr", {getBounds(dDomain, "lo")}, false /* deref */, Auto),
     },
     Auto
  )));
  return ir::Block::make(results);
}

IndexVar MTTKRP::getRootIvar() const {
  return this->content->root;
}

void MTTKRP::print(std::ostream &os) const {
  os << "MTTKRP";
}

struct TTMC::Content {
  IndexVar root;
  TensorVar A;
  TensorVar B;
  TensorVar C;
};

TTMC::TTMC() : content(new Content) {}

void TTMC::canApply(IndexStmt stmt, ProvenanceGraph pg, IndexVar root, std::string *reason) const {
  // TODO (rohany): Deduplicate this into a method on the LeafCallInterface.
  struct Finder : IndexNotationVisitor {
    void visit(const ForallNode* node) {
      if (node->indexVar == this->target) {
        this->root = node;
      }
      node->stmt.accept(this);
    }

    IndexStmt root;
    IndexVar target;
  };
  Finder f; f.target = root;
  stmt.accept(&f);

  // TODO (rohany): Do validation here later.

  IndexStmt rootStmt = f.root;
  auto iLoop = to<Forall>(rootStmt);
  auto jLoop = to<Forall>(iLoop.getStmt());
  auto kLoop = to<Forall>(jLoop.getStmt());
  auto lLoop = to<Forall>(kLoop.getStmt());
  auto assign = to<Assignment>(lLoop.getStmt());
  auto A = to<Access>(assign.getLhs());
  auto mul = to<Mul>(assign.getRhs());
  auto B = to<Access>(mul.getA());
  auto C = to<Access>(mul.getB());

  this->content->root = root;
  this->content->A = A.getTensorVar();
  this->content->B = B.getTensorVar();
  this->content->C = C.getTensorVar();
}

ir::Stmt TTMC::replaceValidStmt(IndexStmt stmt, ProvenanceGraph pg, std::map<TensorVar, ir::Expr> tensorVars,
                                bool inReduction, std::vector<IndexVar> definedVarOrder,
                                std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                                std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators) const {
  auto A = tensorVars[this->content->A];
  auto B = tensorVars[this->content->B];
  auto C = tensorVars[this->content->C];

  std::vector<ir::Stmt> results;
  auto ctx = ir::Symbol::make("ctx");
  auto rowMajor = ir::Symbol::make("CblasRowMajor");
  auto noTrans = ir::Symbol::make("CblasNoTrans");

  auto aIndexSpace = ir::GetProperty::make(A, ir::TensorProperty::IndexSpace);
  auto bIndexSpace = ir::GetProperty::make(B, ir::TensorProperty::IndexSpace);
  auto cIndexSpace = ir::GetProperty::make(C, ir::TensorProperty::IndexSpace);

  // Unpack domains for each variable.
  auto aDomain = ir::Var::make("aDomain", Auto);
  auto bDomain = ir::Var::make("bDomain", Auto);
  auto cDomain = ir::Var::make("cDomain", Auto);
  results.push_back(ir::VarDecl::make(aDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, aIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(bDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, bIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(cDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, cIndexSpace}, Auto)));

  // Break out if any domains are empty.
  auto bVolZero = ir::Eq::make(ir::MethodCall::make(bDomain, "get_volume", {}, false, Auto), 0);
  auto cVolZero = ir::Eq::make(ir::MethodCall::make(cDomain, "get_volume", {}, false, Auto), 0);
  auto guard = ir::Or::make(bVolZero, cVolZero);
  results.push_back(ir::IfThenElse::make(guard, ir::Return::make(ir::Expr())));

  auto aAccess = ir::GetProperty::make(A, ir::TensorProperty::ValuesWriteAccessor, this->content->A.getOrder());
  if (inReduction) {
    aAccess = ir::GetProperty::make(A, ir::TensorProperty::ValuesReductionAccessor, this->content->A.getOrder());
  }
  auto bAccess = ir::GetProperty::make(B, ir::TensorProperty::ValuesReadAccessor, this->content->B.getOrder());
  auto cAccess = ir::GetProperty::make(C, ir::TensorProperty::ValuesReadAccessor, this->content->C.getOrder());
  auto type = Type(this->content->A.getType().getDataType());

  // We'll call BLAS DGEMM in a loop.
  std::vector<ir::Stmt> loopStmts;
  auto loopVar = ir::Var::make("loopIdx", Int());

  auto getBounds = [] (ir::Expr e, std::string func) {
    return ir::MethodCall::make(e, func, {}, false, Int64);
  };

  auto getLD = [&] (ir::Expr e, int idx) {
    return ir::Div::make(
        ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(e, "accessor", false, Auto), "strides", false, Int64), idx),
        ir::Sizeof::make(type)
    );
  };

  auto getBasePtr = [&] (ir::Expr e, ir::Expr domain) {
    auto ptr = ir::MethodCall::make(e, "ptr", {getBounds(domain, "lo")}, false /* deref */, Int64);
    auto offset = ir::Mul::make(getLD(e, 0), loopVar);
    return ir::Add::make(ptr, offset);
  };

  std::vector<ir::Expr> args {
    rowMajor,
    noTrans,
    noTrans,
    ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(bDomain, "hi"), 1), ir::Load::make(getBounds(bDomain, "lo"), 1))),
    ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(cDomain, "hi"), 1), ir::Load::make(getBounds(cDomain, "lo"), 1))),
    ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(cDomain, "hi"), 0), ir::Load::make(getBounds(cDomain, "lo"), 0))),
    1.f,
    getBasePtr(bAccess, bDomain),
    getLD(bAccess, 1),
    ir::MethodCall::make(cAccess, "ptr", {getBounds(cDomain, "lo")}, false /* deref */, Auto),
    getLD(cAccess, 0),
    1.f,
    getBasePtr(aAccess, aDomain),
    getLD(aAccess, 1),
  };
  loopStmts.push_back(ir::SideEffect::make(ir::Call::make("cblas_dgemm", args, Auto)));

  // The loop bounds are from 0 to aBounds.hi()[0] - aBounds.lo()[0].
  auto bounds = ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(aDomain, "hi"), 0), ir::Load::make(getBounds(aDomain, "lo"), 0)));
  results.push_back(ir::For::make(loopVar, 0, bounds, 1, ir::Block::make(loopStmts)));
  return ir::Block::make(results);
}

IndexVar TTMC::getRootIvar() const {
  return this->content->root;
}

void TTMC::print(std::ostream &os) const {
  os << "TTMC";
}

// CuTTMC's replace valid statement is nearly identical to the normal TTMC,
// except that we make CuBLAS calls instead of BLAS calls.
ir::Stmt CuTTMC::replaceValidStmt(IndexStmt stmt, ProvenanceGraph pg, std::map<TensorVar, ir::Expr> tensorVars,
                                  bool inReduction, std::vector<IndexVar> definedVarOrder,
                                  std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                                  std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators) const {
  auto A = tensorVars[this->content->A];
  auto B = tensorVars[this->content->B];
  auto C = tensorVars[this->content->C];

  std::vector<ir::Stmt> results;
  auto ctx = ir::Symbol::make("ctx");
  auto noTrans = ir::Symbol::make("CUBLAS_OP_N");

  auto aIndexSpace = ir::GetProperty::make(A, ir::TensorProperty::IndexSpace);
  auto bIndexSpace = ir::GetProperty::make(B, ir::TensorProperty::IndexSpace);
  auto cIndexSpace = ir::GetProperty::make(C, ir::TensorProperty::IndexSpace);

  // Unpack domains for each variable.
  auto aDomain = ir::Var::make("aDomain", Auto);
  auto bDomain = ir::Var::make("bDomain", Auto);
  auto cDomain = ir::Var::make("cDomain", Auto);
  results.push_back(ir::VarDecl::make(aDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, aIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(bDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, bIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(cDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, cIndexSpace}, Auto)));

  // Break out if any domains are empty.
  auto bVolZero = ir::Eq::make(ir::MethodCall::make(bDomain, "get_volume", {}, false, Auto), 0);
  auto cVolZero = ir::Eq::make(ir::MethodCall::make(cDomain, "get_volume", {}, false, Auto), 0);
  auto guard = ir::Or::make(bVolZero, cVolZero);
  results.push_back(ir::IfThenElse::make(guard, ir::Return::make(ir::Expr())));

  auto aAccess = ir::GetProperty::make(A, ir::TensorProperty::ValuesWriteAccessor, this->content->A.getOrder());
  if (inReduction) {
    aAccess = ir::GetProperty::make(A, ir::TensorProperty::ValuesReductionAccessor, this->content->A.getOrder());
  }
  auto bAccess = ir::GetProperty::make(B, ir::TensorProperty::ValuesReadAccessor, this->content->B.getOrder());
  auto cAccess = ir::GetProperty::make(C, ir::TensorProperty::ValuesReadAccessor, this->content->C.getOrder());
  auto type = Type(this->content->A.getType().getDataType());

  auto addr = [](ir::Expr e) {
    return ir::Call::make("&", {e}, Auto);
  };
  auto check = [](ir::Expr e) {
    return ir::Call::make("CHECK_CUBLAS", {e}, Auto);
  };

  // Add the CuBLAS utility functions to the results.
  auto alpha = ir::Var::make("alpha", Float64);
  results.push_back(ir::VarDecl::make(alpha, ir::Literal::make((double)1, Float64)));
  auto handleTy = Datatype("cublasHandle_t");
  auto streamTy = Datatype("cudaStream_t");
  // Get the CuBLAS handle.
  auto handle = ir::Var::make("handle", handleTy);
  results.push_back(ir::VarDecl::make(handle, ir::Call::make("getCuBLAS", {}, handleTy)));
  // Create a CUDA stream to launch the kernel on.
  auto stream = ir::Var::make("taskStream", streamTy);
  results.push_back(ir::VarDecl::make(stream, ir::makeConstructor(streamTy, {})));
  results.push_back(ir::SideEffect::make(ir::Call::make("cudaStreamCreate", {addr(stream)}, Auto)));
  // Attach the handle to the stream.
  results.push_back(ir::SideEffect::make(check(ir::Call::make("cublasSetStream", {handle, stream}, Auto))));

  // We'll call BLAS DGEMM in a loop.
  std::vector<ir::Stmt> loopStmts;
  auto loopVar = ir::Var::make("loopIdx", Int());

  auto getBounds = [] (ir::Expr e, std::string func) {
    return ir::MethodCall::make(e, func, {}, false, Int64);
  };

  auto getLD = [&] (ir::Expr e, int idx) {
    return ir::Div::make(
        ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(e, "accessor", false, Auto), "strides", false, Int64), idx),
        ir::Sizeof::make(type)
    );
  };

  auto getBasePtr = [&] (ir::Expr e, ir::Expr domain) {
    auto ptr = ir::MethodCall::make(e, "ptr", {getBounds(domain, "lo")}, false /* deref */, Int64);
    auto offset = ir::Mul::make(getLD(e, 0), loopVar);
    return ir::Add::make(ptr, offset);
  };

  // CuBLAS doesn't support row-major matrices. So, we can trick it by reversing
  // the order of the matrices in the call, and telling CuBLAS that the matrices
  // are in column-major format.
  std::vector<ir::Expr> args {
      handle,
      noTrans,
      noTrans,
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(cDomain, "hi"), 1), ir::Load::make(getBounds(cDomain, "lo"), 1))),
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(bDomain, "hi"), 1), ir::Load::make(getBounds(bDomain, "lo"), 1))),
      ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(cDomain, "hi"), 0), ir::Load::make(getBounds(cDomain, "lo"), 0))),
      addr(alpha),
      ir::MethodCall::make(cAccess, "ptr", {getBounds(cDomain, "lo")}, false /* deref */, Auto),
      getLD(cAccess, 0),
      getBasePtr(bAccess, bDomain),
      getLD(bAccess, 1),
      addr(alpha),
      getBasePtr(aAccess, aDomain),
      getLD(aAccess, 1),
  };
  loopStmts.push_back(ir::SideEffect::make(check(ir::Call::make("cublasDgemm", args, Auto))));

  // The loop bounds are from 0 to aBounds.hi()[0] - aBounds.lo()[0].
  auto bounds = ir::Add::make(1, ir::Sub::make(ir::Load::make(getBounds(aDomain, "hi"), 0), ir::Load::make(getBounds(aDomain, "lo"), 0)));
  results.push_back(ir::For::make(loopVar, 0, bounds, 1, ir::Block::make(loopStmts)));
  return ir::Block::make(results);
}

void CuTTMC::print(std::ostream &os) const {
  os << "CuTTMC";
}

struct TTV::Content {
  IndexVar root;
  TensorVar A;
  TensorVar B;
  TensorVar C;
};

TTV::TTV() : content(new Content) {}

void TTV::canApply(IndexStmt stmt, ProvenanceGraph pg, IndexVar root, std::string *reason) const {
  // TODO (rohany): Deduplicate this into a method on the LeafCallInterface.
  struct Finder : IndexNotationVisitor {
    void visit(const ForallNode* node) {
      if (node->indexVar == this->target) {
        this->root = node;
      }
      node->stmt.accept(this);
    }

    IndexStmt root;
    IndexVar target;
  };
  Finder f; f.target = root;
  stmt.accept(&f);

  // TODO (rohany): Do validation here later.

  IndexStmt rootStmt = f.root;
  auto iLoop = to<Forall>(rootStmt);
  auto jLoop = to<Forall>(iLoop.getStmt());
  auto kLoop = to<Forall>(jLoop.getStmt());
  auto assign = to<Assignment>(kLoop.getStmt());
  auto A = to<Access>(assign.getLhs());
  auto mul = to<Mul>(assign.getRhs());
  auto B = to<Access>(mul.getA());
  auto C = to<Access>(mul.getB());

  this->content->root = root;
  this->content->A = A.getTensorVar();
  this->content->B = B.getTensorVar();
  this->content->C = C.getTensorVar();
}

IndexVar TTV::getRootIvar() const {
  return this->content->root;
}

void TTV::print(std::ostream &os) const {
  os << "TTV";
}

ir::Stmt TTV::replaceValidStmt(IndexStmt stmt, ProvenanceGraph pg, std::map<TensorVar, ir::Expr> tensorVars,
                               bool inReduction, std::vector<IndexVar> definedVarOrder,
                               std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                               std::map<taco::IndexVar, taco::ir::Expr> variableNames, Iterators iterators) const {
  auto A = tensorVars[this->content->A];
  auto B = tensorVars[this->content->B];
  auto C = tensorVars[this->content->C];

  std::vector<ir::Stmt> results;
  auto ctx = ir::Symbol::make("ctx");

  auto aIndexSpace = ir::GetProperty::make(A, ir::TensorProperty::IndexSpace);
  auto bIndexSpace = ir::GetProperty::make(B, ir::TensorProperty::IndexSpace);
  auto cIndexSpace = ir::GetProperty::make(C, ir::TensorProperty::IndexSpace);

  // Unpack domains for each variable.
  auto aDomain = ir::Var::make("aDomain", Auto);
  auto bDomain = ir::Var::make("bDomain", Auto);
  auto cDomain = ir::Var::make("cDomain", Auto);
  results.push_back(ir::VarDecl::make(aDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, aIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(bDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, bIndexSpace}, Auto)));
  results.push_back(ir::VarDecl::make(cDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, cIndexSpace}, Auto)));

  // Break out if any domains are empty.
  auto bVolZero = ir::Eq::make(ir::MethodCall::make(bDomain, "get_volume", {}, false, Auto), 0);
  auto cVolZero = ir::Eq::make(ir::MethodCall::make(cDomain, "get_volume", {}, false, Auto), 0);
  auto guard = ir::Or::make(bVolZero, cVolZero);
  results.push_back(ir::IfThenElse::make(guard, ir::Return::make(ir::Expr())));

  auto aAccess = ir::GetProperty::make(A, ir::TensorProperty::ValuesWriteAccessor, this->content->A.getOrder());
  if (inReduction) {
    aAccess = ir::GetProperty::make(A, ir::TensorProperty::ValuesReductionAccessor, this->content->A.getOrder());
  }
  auto bAccess = ir::GetProperty::make(B, ir::TensorProperty::ValuesReadAccessor, this->content->B.getOrder());
  auto cAccess = ir::GetProperty::make(C, ir::TensorProperty::ValuesReadAccessor, this->content->C.getOrder());

  // Set up the argument pack.
  auto packTy = Datatype("TTVPack");
  auto pack = ir::Var::make("pack", packTy);
  results.push_back(ir::VarDecl::make(pack, ir::makeConstructor(packTy, {})));

  auto getBounds = [] (ir::Expr e, std::string func) {
    return ir::MethodCall::make(e, func, {}, false, Int64);
  };
  auto ld = [] (ir::Expr e, int idx) {
    return ir::Load::make(e, idx);
  };
  auto bDomainHi = getBounds(bDomain, "hi");
  auto bDomainLo = getBounds(bDomain, "lo");
  auto iDim = ir::Add::make(1, ir::Sub::make(ld(bDomainHi, 0), ld(bDomainLo, 0)));
  auto jDim = ir::Add::make(1, ir::Sub::make(ld(bDomainHi, 1), ld(bDomainLo, 1)));
  auto kDim = ir::Add::make(1, ir::Sub::make(ld(bDomainHi, 2), ld(bDomainLo, 2)));
  auto type = Type(this->content->A.getType().getDataType());
  auto getLD = [&] (ir::Expr e, int idx) {
    return ir::Div::make(
        ir::Load::make(ir::FieldAccess::make(ir::FieldAccess::make(e, "accessor", false, Auto), "strides", false, Int64), idx),
        ir::Sizeof::make(type)
    );
  };
  auto ldA = getLD(aAccess, 0);
  auto ldB3 = getLD(bAccess, 1);
  auto ldB2 = getLD(bAccess, 0);

  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "iDim", false, Auto), iDim));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "jDim", false, Auto), jDim));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "kDim", false, Auto), kDim));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldA", false, Auto), ldA));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldB2", false, Auto), ir::Div::make(ldB2, ldB3)));
  results.push_back(ir::Assign::make(ir::FieldAccess::make(pack, "ldB3", false, Auto), ldB3));

  std::stringstream funcName;
  funcName << this->funcName << "<" << type.getDataType() << ">";

  results.push_back(ir::SideEffect::make(ir::Call::make(
      funcName.str(),
      {
          pack,
          ir::MethodCall::make(aAccess, "ptr", {getBounds(aDomain, "lo")}, false /* deref */, Auto),
          ir::MethodCall::make(bAccess, "ptr", {getBounds(bDomain, "lo")}, false /* deref */, Auto),
          ir::MethodCall::make(cAccess, "ptr", {getBounds(cDomain, "lo")}, false /* deref */, Auto),
      },
      Auto
  )));
  return ir::Block::make(results);
}

struct CuSparseSpMV::Content {
  IndexVar root;
  TensorVar a;
  TensorVar B;
  TensorVar c;
};

CuSparseSpMV::CuSparseSpMV() : content(new Content) {}

void CuSparseSpMV::canApply(IndexStmt stmt, ProvenanceGraph pg, IndexVar root, std::string *reason) const {
  // TODO (rohany): Implement validation.

  // TODO (rohany): Deduplicate this into a method on the LeafCallInterface.
  struct Finder : IndexNotationVisitor {
    void visit(const ForallNode* node) {
      if (node->indexVar == this->target) {
        this->root = node;
      }
      node->stmt.accept(this);
    }

    IndexStmt root;
    IndexVar target;
  };
  Finder f; f.target = root;
  stmt.accept(&f);

  IndexStmt rootStmt = f.root;
  auto iloop = to<Forall>(rootStmt);
  auto jloop = to<Forall>(iloop.getStmt());
  auto assign = to<Assignment>(jloop.getStmt());
  auto a = to<Access>(assign.getLhs());
  auto mul = to<Mul>(assign.getRhs());
  auto B = to<Access>(mul.getA());
  auto c = to<Access>(mul.getB());

  this->content->root = root;
  this->content->a = a.getTensorVar();
  this->content->B = B.getTensorVar();
  this->content->c = c.getTensorVar();
}

IndexVar CuSparseSpMV::getRootIvar() const {
  return this->content->root;
}

ir::Stmt CuSparseSpMV::replaceValidStmt(IndexStmt stmt, ProvenanceGraph pg, std::map<TensorVar, ir::Expr> tensorVars,
                                        bool inReduction, std::vector<IndexVar> definedVarOrder,
                                        std::map<IndexVar, std::vector<ir::Expr>> underivedBounds,
                                        std::map<taco::IndexVar, taco::ir::Expr> variableNames,
                                        Iterators iterators) const {
  auto a = tensorVars[this->content->a];
  auto B = tensorVars[this->content->B];
  auto c = tensorVars[this->content->c];

  std::vector<ir::Stmt> results;
  auto ctx = ir::Symbol::make("ctx");
  auto nonTrans = ir::Symbol::make("CUSPARSE_OPERATION_NON_TRANSPOSE");
  auto cusparseReal = ir::Symbol::make("CUDA_R_64F");
  auto defaultAlg = ir::Symbol::make("CUSPARSE_ALG_MERGE_PATH");
  auto index0 = ir::Symbol::make("CUSPARSE_INDEX_BASE_ZERO");
  auto generalMat = ir::Symbol::make("CUSPARSE_MATRIX_TYPE_GENERAL");

  auto B2Iter = iterators.getLevelIteratorByLevel(this->content->B, 1);
  auto pack = B2Iter.getMode().getModePack();
  auto B2 = B2Iter.getMode().getModeFormat().as<RectCompressedModeFormat>();

  auto pos = B2->getRegion(pack, RectCompressedModeFormat::POS);
  auto crd = B2->getRegion(pack, RectCompressedModeFormat::CRD);
  auto aVals = ir::GetProperty::make(a, ir::TensorProperty::Values);
  auto BVals = ir::GetProperty::make(B, ir::TensorProperty::Values);
  auto cVals = ir::GetProperty::make(c, ir::TensorProperty::Values);
  auto aValsAcc = ir::GetProperty::make(a, ir::TensorProperty::ValuesWriteAccessor, 1);
  auto BValsAcc = ir::GetProperty::make(B, ir::TensorProperty::ValuesReadAccessor, 1);
  auto cValsAcc = ir::GetProperty::make(c, ir::TensorProperty::ValuesReadAccessor, 1);

  auto aValsDomain = ir::Var::make("aValsDomain", Auto);
  auto cValsDomain = ir::Var::make("cValsDomain", Auto);
  auto posDomain = ir::Var::make("posDomain", Auto);
  auto crdDomain = ir::Var::make("crdDomain", Auto);
  auto BValsDomain = ir::Var::make("BValsDomain", Auto);
  results.push_back(ir::VarDecl::make(posDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, getIndexSpace(pos)}, Auto)));
  results.push_back(ir::VarDecl::make(crdDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, getIndexSpace(crd)}, Auto)));
  results.push_back(ir::VarDecl::make(aValsDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, getIndexSpace(aVals)}, Auto)));
  results.push_back(ir::VarDecl::make(BValsDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, getIndexSpace(BVals)}, Auto)));
  results.push_back(ir::VarDecl::make(cValsDomain, ir::Call::make("runtime->get_index_space_domain", {ctx, getIndexSpace(cVals)}, Auto)));

  auto addr = [](ir::Expr e) {
    return ir::Call::make("&", {e}, Auto);
  };
  auto check = [](ir::Expr e) {
    return ir::Call::make("CHECK_CUSPARSE", {e}, Auto);
  };
  auto getBounds = [](ir::Expr e, std::string func) {
    return ir::MethodCall::make(e, func, {}, false, Int64);
  };
  auto noConst = [](ir::Expr e, Datatype type) {
    return ir::Call::make("const_cast<" + util::toString(type) + "*>", {e}, Auto);
  };

  auto alpha = ir::Var::make("alpha", Float64);
  results.push_back(ir::VarDecl::make(alpha, ir::Literal::make((double)1, Float64)));
  auto handleTy = Datatype("cusparseHandle_t");
  auto streamTy = Datatype("cudaStream_t");
  // Get the CuSparse handle.
  auto handle = ir::Var::make("handle", handleTy);
  results.push_back(ir::VarDecl::make(handle, ir::Call::make("getCuSparse", {}, handleTy)));
  // Create a CUDA stream to launch the kernel on.
  auto stream = ir::Var::make("taskStream", streamTy);
  results.push_back(ir::VarDecl::make(stream, ir::makeConstructor(streamTy, {})));
  results.push_back(ir::SideEffect::make(ir::Call::make("cudaStreamCreate", {addr(stream)}, Auto)));
  // Attach the handle to the stream.
  results.push_back(ir::SideEffect::make(check(ir::Call::make("cusparseSetStream", {handle, stream}, Auto))));

  auto numRows = ir::Sub::make(ir::MethodCall::make(posDomain, "get_volume", {}, false /* deref */, Int()), 1);
  auto B2Dim = ir::GetProperty::make(B, ir::TensorProperty::Dimension, 1);

  // Set up properties about our matrix.
  auto matTy = Datatype("cusparseMatDescr_t");
  auto BMat = ir::Var::make("mat", matTy);
  results.push_back(ir::VarDecl::make(BMat, ir::makeConstructor(matTy, {})));
  results.push_back(ir::SideEffect::make(check(ir::Call::make("cusparseCreateMatDescr", {addr(BMat)}, Auto))));
  results.push_back(ir::SideEffect::make(check(ir::Call::make("cusparseSetMatIndexBase", {BMat, index0}, Auto))));
  results.push_back(ir::SideEffect::make(check(ir::Call::make("cusparseSetMatType", {BMat, generalMat}, Auto))));

  // A special pos pointer that is actually int32_t's instead of Rect<1>'s.
  auto B2Pos = ir::GetProperty::makeIndicesAccessor(
      B,
      B2Iter.getMode().getName() + "_pos",
      1 /* mode */,
      0 /* index */,
      ir::GetProperty::AccessorArgs(
          1 /* dim */,
          Int32 /* elemType */,
          ir::Symbol::make("FID_RECT_1") /* field */,
          pos /* regionAccessing */,
          pos /* regionParent */,
          ir::RO /* priv */
      )
  );

  auto sizet = UInt64;
  auto bufferSize = ir::Var::make("bufferSize", sizet);
  results.push_back(ir::VarDecl::make(bufferSize, 0));
  results.push_back(ir::SideEffect::make(check(ir::Call::make(
    "cusparseCsrmvEx_bufferSize",
    {
      handle,
      defaultAlg,
      nonTrans,
      numRows,
      B2Dim,
      ir::MethodCall::make(crdDomain, "get_volume", {}, false /* deref */, Auto),
      addr(alpha),
      cusparseReal,
      BMat,
      ir::MethodCall::make(BValsAcc, "ptr", {getBounds(BValsDomain, "lo")}, false /* deref */, Auto),
      cusparseReal,
      ir::MethodCall::make(B2Pos, "ptr", {getBounds(posDomain, "lo")}, false /* deref */, Auto),
      ir::MethodCall::make(B2->getAccessor(pack, RectCompressedModeFormat::CRD), "ptr", {getBounds(crdDomain, "lo")}, false /* deref */, Auto),
      ir::MethodCall::make(cValsAcc, "ptr", {getBounds(cValsDomain, "lo")}, false /* deref */, Auto),
      cusparseReal,
      addr(alpha),
      cusparseReal,
      ir::MethodCall::make(aValsAcc, "ptr", {getBounds(aValsDomain, "lo")}, false /* deref */, Auto),
      cusparseReal,
      cusparseReal,
      addr(bufferSize)
    },
    Auto
  ))));

  // Allocate the necessary space.
  auto workspacePtr = ir::Var::make("workspacePtr", Datatype("void*"));
  results.push_back(ir::VarDecl::make(workspacePtr, ir::Symbol::make("NULL")));
  auto defBufTy = DeferredBuffer(Datatype("char"), 1);
  auto defBuf = ir::Var::make("buf", defBufTy);
  results.push_back(ir::IfThenElse::make(
      ir::Gt::make(bufferSize, 0),
      ir::Block::make(
          ir::VarDecl::make(defBuf, ir::makeConstructor(defBufTy, {ir::makeConstructor(Rect(1), {0, ir::Sub::make(bufferSize, 1)}), ir::Symbol::make("Memory::Kind::GPU_FB_MEM")})),
          ir::Assign::make(workspacePtr, ir::MethodCall::make(defBuf, "ptr", {0}, false /* deref */, Auto))
      )
  ));
  results.push_back(ir::SideEffect::make(check(ir::Call::make(
    "cusparseCsrmvEx",
    {
      handle,
      defaultAlg,
      nonTrans,
      numRows,
      B2Dim,
      ir::MethodCall::make(crdDomain, "get_volume", {}, false /* deref */, Auto),
      addr(alpha),
      cusparseReal,
      BMat,
      ir::MethodCall::make(BValsAcc, "ptr", {getBounds(BValsDomain, "lo")}, false /* deref */, Auto),
      cusparseReal,
      ir::MethodCall::make(B2Pos, "ptr", {getBounds(posDomain, "lo")}, false /* deref */, Auto),
      ir::MethodCall::make(B2->getAccessor(pack, RectCompressedModeFormat::CRD), "ptr", {getBounds(crdDomain, "lo")}, false /* deref */, Auto),
      ir::MethodCall::make(cValsAcc, "ptr", {getBounds(cValsDomain, "lo")}, false /* deref */, Auto),
      cusparseReal,
      addr(alpha),
      cusparseReal,
      ir::MethodCall::make(aValsAcc, "ptr", {getBounds(aValsDomain, "lo")}, false /* deref */, Auto),
      cusparseReal,
      cusparseReal,
      workspacePtr,
    },
    Auto
  ))));
  return ir::Block::make(results);
}

void CuSparseSpMV::print(std::ostream &os) const {
  os << "CuSparseSpMV";
}

// Autoscheduling functions

IndexStmt parallelizeOuterLoop(IndexStmt stmt) {
  // get outer ForAll
  Forall forall;
  bool matched = false;
  match(stmt,
        function<void(const ForallNode*,Matcher*)>([&forall, &matched](
                const ForallNode* node, Matcher* ctx) {
          if (!matched) forall = node;
          matched = true;
        })
  );

  if (!matched) return stmt;
  string reason;

  if (should_use_CUDA_codegen()) {
    IndexVar i1, i2;
    IndexStmt parallelized256 = stmt.split(forall.getIndexVar(), i1, i2, 256);
    parallelized256 = Parallelize(i1, ParallelUnit::GPUBlock, OutputRaceStrategy::NoRaces).apply(parallelized256, &reason);
    if (parallelized256 == IndexStmt()) {
      return stmt;
    }
    parallelized256 = Parallelize(i2, ParallelUnit::GPUThread, OutputRaceStrategy::NoRaces).apply(parallelized256, &reason);
    if (parallelized256 == IndexStmt()) {
      return stmt;
    }
    return parallelized256;
  }
  else {
    IndexStmt parallelized = Parallelize(forall.getIndexVar(), ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces).apply(stmt, &reason);
    if (parallelized == IndexStmt()) {
      // can't parallelize
      return stmt;
    }
    return parallelized;
  }
}

// Takes in a set of pairs of IndexVar and level for a given tensor and orders
// the IndexVars by tensor level
static vector<pair<IndexVar, bool>> 
varOrderFromTensorLevels(set<pair<IndexVar, pair<int, bool>>> tensorLevelVars) {
  vector<pair<IndexVar, pair<int, bool>>> sortedPairs(tensorLevelVars.begin(), 
                                                      tensorLevelVars.end());
  auto comparator = [](const pair<IndexVar, pair<int, bool>> &left, 
                       const pair<IndexVar, pair<int, bool>> &right) {
    return left.second.first < right.second.first;
  };
  std::sort(sortedPairs.begin(), sortedPairs.end(), comparator);

  vector<pair<IndexVar, bool>> varOrder;
  std::transform(sortedPairs.begin(),
                sortedPairs.end(),
                std::back_inserter(varOrder),
                [](const std::pair<IndexVar, pair<int, bool>>& p) {
                  return pair<IndexVar, bool>(p.first, p.second.second);
                });
  return varOrder;
}


// Takes in varOrders from many tensors and creates a map of dependencies between IndexVars
static map<IndexVar, set<IndexVar>>
depsFromVarOrders(map<string, vector<pair<IndexVar,bool>>> varOrders) {
  map<IndexVar, set<IndexVar>> deps;
  for (const auto& varOrderPair : varOrders) {
    const auto& varOrder = varOrderPair.second;
    for (auto firstit = varOrder.begin(); firstit != varOrder.end(); ++firstit) {
      for (auto secondit = firstit + 1; secondit != varOrder.end(); ++secondit) {
        if (firstit->second || secondit->second) { // one of the dimensions must enforce constraints
          if (deps.count(secondit->first)) {
            deps[secondit->first].insert(firstit->first);
          } else {
            deps[secondit->first] = {firstit->first};
          }
        }
      }
    }
  }
  return deps;
}


static vector<IndexVar>
topologicallySort(map<IndexVar,set<IndexVar>> hardDeps,
                  map<IndexVar,multiset<IndexVar>> softDeps,
                  vector<IndexVar> originalOrder) {
  vector<IndexVar> sortedVars;
  unsigned long countVars = originalOrder.size();
  while (sortedVars.size() < countVars) {
    // Scan for variable with no dependencies
    IndexVar freeVar;
    size_t freeVarPos = std::numeric_limits<size_t>::max();
    size_t minSoftDepsRemaining = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < originalOrder.size(); ++i) {
      IndexVar var = originalOrder[i];
      if (!hardDeps.count(var) || hardDeps[var].empty()) {  
        const size_t softDepsRemaining = softDeps.count(var) ? 
                                         softDeps[var].size() : 0;
        if (softDepsRemaining < minSoftDepsRemaining) {
          freeVar = var;
          freeVarPos = i;
          minSoftDepsRemaining = softDepsRemaining;
        }
      }
    }

    // No free var found there is a cycle
    taco_iassert(freeVarPos != std::numeric_limits<size_t>::max())
        << "Cycles in iteration graphs must be resolved, through transpose, "
        << "before the expression is passed to the topological sorting "
        << "routine.";

    sortedVars.push_back(freeVar);

    // remove dependencies on variable
    for (auto& varTensorDepsPair : hardDeps) {
      auto& varTensorDeps = varTensorDepsPair.second;
      if (varTensorDeps.count(freeVar)) {
        varTensorDeps.erase(freeVar);
      }
    }
    for (auto& varTensorDepsPair : softDeps) {
      auto& varTensorDeps = varTensorDepsPair.second;
      if (varTensorDeps.count(freeVar)) {
        varTensorDeps.erase(freeVar);
      }
    }
    originalOrder.erase(originalOrder.begin() + freeVarPos);
  }
  return sortedVars;
}


IndexStmt reorderLoopsTopologically(IndexStmt stmt) {
  // Collect tensorLevelVars which stores the pairs of IndexVar and tensor
  // level that each tensor is accessed at
  struct DAGBuilder : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    // int is level, bool is if level enforces constraints (ie not dense)
    map<string, set<pair<IndexVar, pair<int, bool>>>> tensorLevelVars;
    IndexStmt innerBody;
    map <IndexVar, ParallelUnit> forallParallelUnit;
    map <IndexVar, OutputRaceStrategy> forallOutputRaceStrategy;
    vector<IndexVar> indexVarOriginalOrder;
    Iterators iterators;

    DAGBuilder(Iterators iterators) : iterators(iterators) {};

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      MergeLattice lattice = MergeLattice::make(foralli, iterators, ProvenanceGraph(), {}); // TODO
      indexVarOriginalOrder.push_back(i);
      forallParallelUnit[i] = foralli.getParallelUnit();
      forallOutputRaceStrategy[i] = foralli.getOutputRaceStrategy();

      // Iterator and if Iterator enforces constraints
      vector<pair<Iterator, bool>> depIterators;
      for (Iterator iterator : lattice.points()[0].iterators()) {
        if (!iterator.isDimensionIterator()) {
          depIterators.push_back({iterator, true});
        }
      }

      for (Iterator iterator : lattice.points()[0].locators()) {
        depIterators.push_back({iterator, false});
      }

      // add result iterators that append
      for (Iterator iterator : lattice.results()) {
        depIterators.push_back({iterator, !iterator.hasInsert()});
      }

      for (const auto& iteratorPair : depIterators) {
        int level = iteratorPair.first.getMode().getLevel();
        string tensor = to<ir::Var>(iteratorPair.first.getTensor())->name;
        if (tensorLevelVars.count(tensor)) {
          tensorLevelVars[tensor].insert({{i, {level, iteratorPair.second}}});
        }
        else {
          tensorLevelVars[tensor] = {{{i, {level, iteratorPair.second}}}};
        }
      }

      if (!isa<Forall>(foralli.getStmt())) {
        innerBody = foralli.getStmt();
        return; // Only reorder first contiguous section of ForAlls
      }
      IndexNotationVisitor::visit(node);
    }
  };

  Iterators iterators(stmt);
  DAGBuilder dagBuilder(iterators);
  stmt.accept(&dagBuilder);

  // Construct tensor dependencies (sorted list of IndexVars) from tensorLevelVars
  map<string, vector<pair<IndexVar, bool>>> tensorVarOrders;
  for (const auto& tensorLevelVar : dagBuilder.tensorLevelVars) {
    tensorVarOrders[tensorLevelVar.first] = 
        varOrderFromTensorLevels(tensorLevelVar.second);
  }
  const auto hardDeps = depsFromVarOrders(tensorVarOrders);

  struct CollectSoftDependencies : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    map<IndexVar, multiset<IndexVar>> softDeps;

    void visit(const AssignmentNode* op) {
      op->lhs.accept(this);
      op->rhs.accept(this);
    }

    void visit(const AccessNode* node) {
      const auto& modeOrdering = node->tensorVar.getFormat().getModeOrdering();
      for (size_t i = 1; i < (size_t)node->tensorVar.getOrder(); ++i) {
        const auto srcVar = node->indexVars[modeOrdering[i - 1]];
        const auto dstVar = node->indexVars[modeOrdering[i]];
        softDeps[dstVar].insert(srcVar);
      }
    }
  };
  CollectSoftDependencies collectSoftDeps;
  stmt.accept(&collectSoftDeps);

  const auto sortedVars = topologicallySort(hardDeps, collectSoftDeps.softDeps, 
                                            dagBuilder.indexVarOriginalOrder);

  // Reorder Foralls use a rewriter in case new nodes introduced outside of Forall
  struct TopoReorderRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const vector<IndexVar>& sortedVars;
    IndexStmt innerBody;
    const map <IndexVar, ParallelUnit> forallParallelUnit;
    const map <IndexVar, OutputRaceStrategy> forallOutputRaceStrategy;

    TopoReorderRewriter(const vector<IndexVar>& sortedVars, IndexStmt innerBody,
                        const map <IndexVar, ParallelUnit> forallParallelUnit,
                        const map <IndexVar, OutputRaceStrategy> forallOutputRaceStrategy)
        : sortedVars(sortedVars), innerBody(innerBody),
        forallParallelUnit(forallParallelUnit), forallOutputRaceStrategy(forallOutputRaceStrategy)  {
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      // first forall must be in collected variables
      taco_iassert(util::contains(sortedVars, i));
      stmt = innerBody;
      for (auto it = sortedVars.rbegin(); it != sortedVars.rend(); ++it) {
        stmt = forall(*it, stmt, forallParallelUnit.at(*it), forallOutputRaceStrategy.at(*it), node->transfers, node->computingOn, foralli.getUnrollFactor());
      }
      return;
    }

  };
  TopoReorderRewriter rewriter(sortedVars, dagBuilder.innerBody, 
                               dagBuilder.forallParallelUnit, dagBuilder.forallOutputRaceStrategy);
  return rewriter.rewrite(stmt);
}

IndexStmt scalarPromote(IndexStmt stmt, ProvenanceGraph provGraph, 
                        bool isWholeStmt, bool promoteScalar) {
  std::map<Access,const ForallNode*> hoistLevel;
  std::map<Access,IndexExpr> reduceOp;
  struct FindHoistLevel : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    std::map<Access,const ForallNode*>& hoistLevel;
    std::map<Access,IndexExpr>& reduceOp;
    std::map<Access,std::set<IndexVar>> hoistIndices;
    std::set<IndexVar> derivedIndices;
    std::set<IndexVar> indices;
    const ProvenanceGraph& provGraph;
    const bool isWholeStmt;
    const bool promoteScalar;
    
    FindHoistLevel(std::map<Access,const ForallNode*>& hoistLevel,
                   std::map<Access,IndexExpr>& reduceOp,
                   const ProvenanceGraph& provGraph,
                   bool isWholeStmt, bool promoteScalar) : 
        hoistLevel(hoistLevel), reduceOp(reduceOp), provGraph(provGraph),
        isWholeStmt(isWholeStmt), promoteScalar(promoteScalar) {}

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      // Don't allow hoisting out of forall's for GPU warp and block reduction
      if (foralli.getParallelUnit() == ParallelUnit::GPUWarpReduction || 
          foralli.getParallelUnit() == ParallelUnit::GPUBlockReduction) {
        FindHoistLevel findHoistLevel(hoistLevel, reduceOp, provGraph, false, 
                                      promoteScalar);
        foralli.getStmt().accept(&findHoistLevel);
        return;
      }

      std::vector<Access> resultAccesses;
      std::tie(resultAccesses, std::ignore) = getResultAccesses(foralli);
      for (const auto& resultAccess : resultAccesses) {
        if (!promoteScalar && resultAccess.getIndexVars().empty()) {
          continue;
        }

        std::set<IndexVar> resultIndices(resultAccess.getIndexVars().begin(),
                                         resultAccess.getIndexVars().end());
        if (std::includes(indices.begin(), indices.end(), 
                          resultIndices.begin(), resultIndices.end()) &&
            !util::contains(hoistLevel, resultAccess)) {
          hoistLevel[resultAccess] = node;
          hoistIndices[resultAccess] = indices;
          if (!isWholeStmt || resultIndices != derivedIndices) {
            reduceOp[resultAccess] = IndexExpr();
          }
        }
      }

      auto newIndices = provGraph.newlyRecoverableParents(i, derivedIndices);
      newIndices.push_back(i);
      derivedIndices.insert(newIndices.begin(), newIndices.end());

      const auto underivedIndices = getIndexVars(foralli);
      for (const auto& newIndex : newIndices) {
        if (util::contains(underivedIndices, newIndex)) {
          indices.insert(newIndex);
        }
      }

      IndexNotationVisitor::visit(node);

      for (const auto& newIndex : newIndices) {
        indices.erase(newIndex);
        derivedIndices.erase(newIndex);
      }
    }

    void visit(const AssignmentNode* op) {
      if (util::contains(hoistLevel, op->lhs) && 
          hoistIndices[op->lhs] == indices) {
        hoistLevel.erase(op->lhs);
      }
      if (util::contains(reduceOp, op->lhs)) {
        reduceOp[op->lhs] = op->op;
      }
    }
  };
  FindHoistLevel findHoistLevel(hoistLevel, reduceOp, provGraph, isWholeStmt, 
                                promoteScalar);
  stmt.accept(&findHoistLevel);
  
  struct HoistWrites : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const std::map<Access,const ForallNode*>& hoistLevel;
    const std::map<Access,IndexExpr>& reduceOp;

    HoistWrites(const std::map<Access,const ForallNode*>& hoistLevel,
                const std::map<Access,IndexExpr>& reduceOp) : 
        hoistLevel(hoistLevel), reduceOp(reduceOp) {}

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();
      IndexStmt body = rewrite(foralli.getStmt());

      std::vector<IndexStmt> consumers;
      for (const auto& resultAccess : hoistLevel) {
        if (resultAccess.second == node) {
          // This assumes the index expression yields at most one result tensor; 
          // will not work correctly if there are multiple results.
          TensorVar resultVar = resultAccess.first.getTensorVar();
          TensorVar val("t" + i.getName() + resultVar.getName(), 
                        Type(resultVar.getType().getDataType(), {}));
          body = ReplaceReductionExpr(
              map<Access,Access>({{resultAccess.first, val()}})).rewrite(body);

          IndexExpr op = util::contains(reduceOp, resultAccess.first) 
                       ? reduceOp.at(resultAccess.first) : IndexExpr();
          IndexStmt consumer = Assignment(Access(resultAccess.first), val(), op);
          consumers.push_back(consumer);
        }
      }

      if (body == foralli.getStmt()) {
        taco_iassert(consumers.empty());
        stmt = node;
        return;
      }

      stmt = forall(i, body, foralli.getParallelUnit(),
                    foralli.getOutputRaceStrategy(), node->transfers, node->computingOn, foralli.getUnrollFactor());
      for (const auto& consumer : consumers) {
        stmt = where(consumer, stmt);
      }
    }
  };
  HoistWrites hoistWrites(hoistLevel, reduceOp);
  return hoistWrites.rewrite(stmt);
}

IndexStmt scalarPromote(IndexStmt stmt) {
  return scalarPromote(stmt, ProvenanceGraph(stmt), true, false);
}

// TODO Temporary function to insert workspaces into SpMM kernels
static IndexStmt optimizeSpMM(IndexStmt stmt) {
  if (!isa<Forall>(stmt)) {
    return stmt;
  }
  Forall foralli = to<Forall>(stmt);
  IndexVar i = foralli.getIndexVar();

  if (!isa<Forall>(foralli.getStmt())) {
    return stmt;
  }
  Forall forallk = to<Forall>(foralli.getStmt());
  IndexVar k = forallk.getIndexVar();

  if (!isa<Forall>(forallk.getStmt())) {
    return stmt;
  }
  Forall forallj = to<Forall>(forallk.getStmt());
  IndexVar j = forallj.getIndexVar();

  if (!isa<Assignment>(forallj.getStmt())) {
    return stmt;
  }
  Assignment assignment = to<Assignment>(forallj.getStmt());

  if (!isa<Mul>(assignment.getRhs())) {
    return stmt;
  }
  Mul mul = to<Mul>(assignment.getRhs());

  taco_iassert(isa<Access>(assignment.getLhs()));
  if (!isa<Access>(mul.getA())) {
    return stmt;
  }
  if (!isa<Access>(mul.getB())) {
    return stmt;
  }

  Access Aaccess = to<Access>(assignment.getLhs());
  Access Baccess = to<Access>(mul.getA());
  Access Caccess = to<Access>(mul.getB());

  if (Aaccess.getIndexVars().size() != 2 ||
      Baccess.getIndexVars().size() != 2 ||
      Caccess.getIndexVars().size() != 2) {
    return stmt;
  }

  if (!compare(Aaccess.getIndexVars(), {i,j}) ||
      !compare(Baccess.getIndexVars(), {i,k}) ||
      !compare(Caccess.getIndexVars(), {k,j})) {
    return stmt;
  }

  TensorVar A = Aaccess.getTensorVar();
  if (A.getFormat().getModeFormats()[0].getName() != "dense" ||
      A.getFormat().getModeFormats()[1].getName() != "compressed" ||
      A.getFormat().getModeOrdering()[0] != 0 ||
      A.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  // I think we can to linear combination of rows as long as there are no permutations in the format and the
  // level formats are ordered. The i -> k -> j loops should iterate over the data structures without issue.
  TensorVar B = Baccess.getTensorVar();
  if (!B.getFormat().getModeFormats()[0].isOrdered() ||
      !B.getFormat().getModeFormats()[1].isOrdered() ||
      B.getFormat().getModeOrdering()[0] != 0 ||
      B.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  TensorVar C = Caccess.getTensorVar();
  if (!C.getFormat().getModeFormats()[0].isOrdered() ||
      !C.getFormat().getModeFormats()[1].isOrdered() ||
      C.getFormat().getModeOrdering()[0] != 0 ||
      C.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  // It's an SpMM statement so return an optimized SpMM statement
  TensorVar w("w",
              Type(A.getType().getDataType(), 
              {A.getType().getShape().getDimension(1)}),
              taco::dense);
  return forall(i,
                where(forall(j,
                             A(i,j) = w(j)),
                      forall(k,
                             forall(j,
                                    w(j) += B(i,k) * C(k,j)))));
}

IndexStmt insertTemporaries(IndexStmt stmt)
{
  IndexStmt spmm = optimizeSpMM(stmt);
  if (spmm != stmt) {
    return spmm;
  }

  // TODO Implement general workspacing when scattering into sparse result modes

  // Result dimensions that are indexed by free variables dominated by a
  // reduction variable are scattered into.  If any of these are compressed
  // then we introduce a dense workspace to scatter into instead.  The where
  // statement must push the reduction loop into the producer side, leaving
  // only the free variable loops on the consumer side.

  //vector<IndexVar> reductionVars = getReductionVars(stmt);
  //...

  return stmt;
}

IndexStmt hoistSuchThats(IndexStmt stmt) {
  struct SuchThatHoister : public IndexNotationRewriter {
    IndexStmt apply(IndexStmt stmt) {
      stmt = IndexNotationRewriter::rewrite(stmt);
      if (!(this->predicates.empty() && this->calls.empty())) {
        stmt = SuchThat(stmt, this->predicates, this->calls);
      }
      return stmt;
    }

    void visit(const SuchThatNode* node) {
      this->stmt = node->stmt;
      util::append(this->predicates, node->predicate);
      for (auto it : node->calls) {
        this->calls.insert({it.first, it.second});
      }
    }

    std::vector<IndexVarRel> predicates;
    std::map<IndexVar, std::shared_ptr<LeafCallInterface>> calls;
  };
  return SuchThatHoister().apply(stmt);
}

}
