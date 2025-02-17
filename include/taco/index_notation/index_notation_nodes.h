#ifndef TACO_INDEX_NOTATION_NODES_H
#define TACO_INDEX_NOTATION_NODES_H

#include <vector>
#include <memory>
#include <functional>

#include "taco/type.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes_abstract.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/util/strings.h"

namespace taco {

struct AccessWindow;
struct IndexSet;

// IndexVarIterationModifier is a marker interface for describing iteration
// transformations onto a particular index variable. Currently, the type is
// inhabited only by
// * AccessWindow
// * IndexSet
struct IndexVarIterationModifier {
  virtual ~IndexVarIterationModifier() = default;

  // match performs dynamic dispatch on the subclass that implements IndexVarIterationModifier.
  static void match(std::shared_ptr<IndexVarIterationModifier> ptr,
                    std::function<void(std::shared_ptr<AccessWindow>)> windowFunc,
                    std::function<void(std::shared_ptr<IndexSet>)> indexSetFunc) {
    auto windowPtr = std::dynamic_pointer_cast<AccessWindow>(ptr);
    auto indexSetPtr = std::dynamic_pointer_cast<IndexSet>(ptr);
    if (windowPtr != nullptr) {
      windowFunc(windowPtr);
    } else if (indexSetPtr != nullptr) {
      indexSetFunc(indexSetPtr);
    } else {
      taco_iassert("IndexVarIterationModifier was not AccessWindow or IndexVarIterationModifier");
    }
  }
};

// An AccessNode carries the windowing information for an IndexVar + TensorVar
// combination. An AccessWindow contains the lower and upper bounds of each
// windowed mode (0-indexed). AccessWindow is extracted from AccessNode so that
// it can be referenced externally.
struct AccessWindow : IndexVarIterationModifier {
  ~AccessWindow() = default;

  int lo;
  int hi;
  int stride;
  friend bool operator==(const AccessWindow& a, const AccessWindow& b) {
    return a.lo == b.lo && a.hi == b.hi && a.stride == b.stride;
  }
};

// An AccessNode also carries the information about an index set for an IndexVar +
// TensorVar combination. An IndexSet contains the set of dimensions projected
// out from a tensor via an index set.
struct IndexSet : IndexVarIterationModifier {
  ~IndexSet() = default;

  std::shared_ptr<std::vector<int>> set;
  TensorBase tensor;
  friend bool operator==(const IndexSet& a, const IndexSet& b) {
    return *a.set == *b.set && a.tensor == b.tensor;
  }
};

struct AccessNode : public IndexExprNode {
  AccessNode(TensorVar tensorVar, const std::vector<IndexVar>& indices, 
             const std::map<int, std::shared_ptr<IndexVarIterationModifier>> &modifiers,
             bool isAccessingStructure)
      : IndexExprNode(isAccessingStructure ? Bool : tensorVar.getType().getDataType()), 
        tensorVar(tensorVar), indexVars(indices), 
        isAccessingStructure(isAccessingStructure) {
    // Unpack the input modifiers into the appropriate maps for each mode.
    for (auto &it : modifiers) {
      IndexVarIterationModifier::match(it.second, [&](std::shared_ptr<AccessWindow> w) {
        this->windowedModes[it.first] = *w;
      }, [&](std::shared_ptr<IndexSet> i) {
        this->indexSetModes[it.first] = *i;
      });
    }
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  virtual void setAssignment(const Assignment& assignment) {}

  // packageModifiers collects all IndexVarIterationModifiers applied to this
  // AccessNode into a map.
  std::map<int, std::shared_ptr<IndexVarIterationModifier>> packageModifiers() const {
    std::map<int, std::shared_ptr<IndexVarIterationModifier>> ret;
    for (auto& it : this->windowedModes) {
      ret[it.first] = std::make_shared<AccessWindow>(it.second);
    }
    for (auto& it : this->indexSetModes) {
      ret[it.first] = std::make_shared<IndexSet>(it.second);
    }
    return ret;
  }

  TensorVar tensorVar;
  std::vector<IndexVar> indexVars;
  std::map<int, AccessWindow> windowedModes;
  std::map<int, IndexSet> indexSetModes;
  bool isAccessingStructure;

protected:
  /// Initialize an AccessNode with just a TensorVar. If this constructor is used,
  /// then indexVars must be set afterwards.
  explicit AccessNode(TensorVar tensorVar) : 
      IndexExprNode(tensorVar.getType().getDataType()), 
      tensorVar(tensorVar), isAccessingStructure(false) {}
};

struct LiteralNode : public IndexExprNode {
  template <typename T> LiteralNode(T val) : IndexExprNode(type<T>()) {
    this->val = malloc(sizeof(T));
    *static_cast<T*>(this->val) = val;
  }

  ~LiteralNode() {
    free(val);
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  template <typename T> T getVal() const {
    taco_iassert(getDataType() == type<T>())
        << "Attempting to get data of wrong type";
    return *static_cast<T*>(val);
  }

  void* val;
};


struct UnaryExprNode : public IndexExprNode {
  IndexExpr a;

protected:
  UnaryExprNode(IndexExpr a) : IndexExprNode(a.getDataType()), a(a) {}
};


struct NegNode : public UnaryExprNode {
  NegNode(IndexExpr operand) : UnaryExprNode(operand) {}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};


struct BinaryExprNode : public IndexExprNode {
  virtual std::string getOperatorString() const = 0;

  IndexExpr a;
  IndexExpr b;

protected:
  BinaryExprNode() : IndexExprNode() {}
  BinaryExprNode(IndexExpr a, IndexExpr b)
      : IndexExprNode(max_type(a.getDataType(), b.getDataType())), a(a), b(b) {}
};


struct AddNode : public BinaryExprNode {
  AddNode() : BinaryExprNode() {}
  AddNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "+";
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};


struct SubNode : public BinaryExprNode {
  SubNode() : BinaryExprNode() {}
  SubNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "-";
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};


struct MulNode : public BinaryExprNode {
  MulNode() : BinaryExprNode() {}
  MulNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "*";
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};


struct DivNode : public BinaryExprNode {
  DivNode() : BinaryExprNode() {}
  DivNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "/";
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};


struct SqrtNode : public UnaryExprNode {
  SqrtNode(IndexExpr operand) : UnaryExprNode(operand) {}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

};


struct CastNode : public IndexExprNode {
  CastNode(IndexExpr operand, Datatype newType);

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  IndexExpr a;
};


struct CallIntrinsicNode : public IndexExprNode {
  CallIntrinsicNode(const std::shared_ptr<Intrinsic>& func,
                    const std::vector<IndexExpr>& args); 

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  std::shared_ptr<Intrinsic> func;
  std::vector<IndexExpr> args;
};


struct ReductionNode : public IndexExprNode {
  ReductionNode(IndexExpr op, IndexVar var, IndexExpr a);

  void accept(IndexExprVisitorStrict* v) const {
     v->visit(this);
  }

  IndexExpr op;  // The binary reduction operator, which is a `BinaryExprNode`
                 // with undefined operands)
  IndexVar var;
  IndexExpr a;
};


// Index Statements
struct AssignmentNode : public IndexStmtNode {
  AssignmentNode(const Access& lhs, const IndexExpr& rhs, const IndexExpr& op)
      : lhs(lhs), rhs(rhs), op(op) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  Access    lhs;
  IndexExpr rhs;
  IndexExpr op;
};

struct YieldNode : public IndexStmtNode {
  YieldNode(const std::vector<IndexVar>& indexVars, IndexExpr expr)
      : indexVars(indexVars), expr(expr) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  std::vector<IndexVar> indexVars;
  IndexExpr expr;
};

struct ForallNode : public IndexStmtNode {
  ForallNode(IndexVar indexVar, IndexStmt stmt, ParallelUnit parallel_unit, OutputRaceStrategy  output_race_strategy, std::vector<Transfer> transfers, std::vector<TensorVar> computingOn, size_t unrollFactor = 0)
      : indexVar(indexVar), stmt(stmt), parallel_unit(parallel_unit), output_race_strategy(output_race_strategy), unrollFactor(unrollFactor), transfers(transfers),  computingOn(computingOn) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexVar indexVar;
  IndexStmt stmt;
  ParallelUnit parallel_unit;
  OutputRaceStrategy  output_race_strategy;
  size_t unrollFactor = 0;

  // All of the transfer objects queued up at this level.
  std::vector<Transfer> transfers;
  std::vector<TensorVar> computingOn;
};

struct WhereNode : public IndexStmtNode {
  WhereNode(IndexStmt consumer, IndexStmt producer)
      : consumer(consumer), producer(producer) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexStmt consumer;
  IndexStmt producer;
};

struct MultiNode : public IndexStmtNode {
  MultiNode(IndexStmt stmt1, IndexStmt stmt2) : stmt1(stmt1), stmt2(stmt2) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexStmt stmt1;
  IndexStmt stmt2;
};

struct SuchThatNode : public IndexStmtNode {
  SuchThatNode(IndexStmt stmt, std::vector<IndexVarRel> predicate, std::map<IndexVar, std::shared_ptr<LeafCallInterface>> calls) : stmt(stmt), predicate(predicate), calls(calls) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexStmt stmt;
  std::vector<IndexVarRel> predicate;
  std::map<IndexVar, std::shared_ptr<LeafCallInterface>> calls;
};

struct SequenceNode : public IndexStmtNode {
  SequenceNode(IndexStmt definition, IndexStmt mutation)
      : definition(definition), mutation(mutation) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexStmt definition;
  IndexStmt mutation;
};

struct AssembleNode : public IndexStmtNode {
  AssembleNode(IndexStmt queries, IndexStmt compute, 
               Assemble::AttrQueryResults results)
      : queries(queries), compute(compute), results(results) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexStmt queries;
  IndexStmt compute;
  Assemble::AttrQueryResults results;
};

struct PlaceNode : public IndexStmtNode {
  PlaceNode(IndexExpr e, std::vector<std::pair<Grid, GridPlacement>> placements) : expr(e), placements(placements) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexExpr expr;
  std::vector<std::pair<Grid, GridPlacement>> placements;
};

struct PartitionNode : public IndexStmtNode {
  PartitionNode(IndexExpr expr) : expr(expr) {}

  void accept(IndexStmtVisitorStrict* v) const {
    v->visit(this);
  }

  IndexExpr expr;
};


/// Returns true if expression e is of type E.
template <typename E>
inline bool isa(const IndexExprNode* e) {
  return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}

/// Casts the expression e to type E.
template <typename E>
inline const E* to(const IndexExprNode* e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e);
}

/// Returns true if statement e is of type S.
template <typename S>
inline bool isa(const IndexStmtNode* s) {
  return s != nullptr && dynamic_cast<const S*>(s) != nullptr;
}

/// Casts the index statement node s to subtype S.
template <typename SubType>
inline const SubType* to(const IndexStmtNode* s) {
  taco_iassert(isa<SubType>(s)) <<
      "Cannot convert " << typeid(s).name() << " to " << typeid(SubType).name();
  return static_cast<const SubType*>(s);
}

template <typename I>
inline const typename I::Node* getNode(const I& stmt) {
  taco_iassert(isa<typename I::Node>(stmt.ptr));
  return static_cast<const typename I::Node*>(stmt.ptr);
}

}
#endif
