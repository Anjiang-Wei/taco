#ifndef TACO_LOWERER_IMPL_H
#define TACO_LOWERER_IMPL_H

#include <utility>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <taco/index_notation/index_notation.h>

#include "taco/lower/lower.h"
#include "taco/lower/iterator.h"
#include "taco/util/scopedset.h"
#include "taco/util/uncopyable.h"
#include "taco/ir_tags.h"
#include "taco/ir/ir_rewriter.h"

namespace taco {

class TensorVar;
class IndexVar;

class IndexStmt;
class Assignment;
class Yield;
class Forall;
class Where;
class Multi;
class SuchThat;
class Sequence;

class IndexExpr;
class Access;
class Literal;
class Neg;
class Add;
class Sub;
class Mul;
class Div;
class Sqrt;
class Cast;
class CallIntrinsic;

class MergeLattice;
class MergePoint;
class ModeAccess;

namespace ir {
class Stmt;
class Expr;
}

class LowererImpl : public util::Uncopyable {
public:
  LowererImpl();
  virtual ~LowererImpl() = default;

  /// Lower an index statement to an IR function.
  ir::Stmt lower(IndexStmt stmt, std::string name, LowerOptions options);

protected:

  /// Lower an assignment statement.
  virtual ir::Stmt lowerAssignment(Assignment assignment);

  /// Lower a yield statement.
  virtual ir::Stmt lowerYield(Yield yield);


  /// Lower a forall statement.
  virtual ir::Stmt lowerForall(Forall forall);

  /// Lower a forall that needs to be cloned so that one copy does not have guards
  /// used for vectorized and unrolled loops
  virtual ir::Stmt lowerForallCloned(Forall forall);

  /// Lower a forall that iterates over all the coordinates in the forall index
  /// var's dimension, and locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallDimension(Forall forall,
                                        std::vector<Iterator> locaters,
                                        std::vector<Iterator> inserters,
                                        std::vector<Iterator> appenders,
                                        std::set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt);

  /// Lower a forall that iterates over all the coordinates in the forall index
  /// var's dimension, and locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallDenseAcceleration(Forall forall,
                                                std::vector<Iterator> locaters,
                                                std::vector<Iterator> inserters,
                                                std::vector<Iterator> appenders,
                                                std::set<Access> reducedAccesses,
                                                ir::Stmt recoveryStmt);


  /// Lower a forall that iterates over the coordinates in the iterator, and
  /// locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallCoordinate(Forall forall, Iterator iterator,
                                         std::vector<Iterator> locaters,
                                         std::vector<Iterator> inserters,
                                         std::vector<Iterator> appenders,
                                         std::set<Access> reducedAccesses,
                                         ir::Stmt recoveryStmt);

  /// Lower a forall that iterates over the positions in the iterator, accesses
  /// the iterators coordinate, and locates tensor positions from the locate
  /// iterators.
  virtual ir::Stmt lowerForallPosition(Forall forall, Iterator iterator,
                                       std::vector<Iterator> locaters,
                                       std::vector<Iterator> inserters,
                                       std::vector<Iterator> appenders,
                                       std::set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt);

  virtual ir::Stmt lowerForallFusedPosition(Forall forall, Iterator iterator,
                                       std::vector<Iterator> locaters,
                                       std::vector<Iterator> inserters,
                                       std::vector<Iterator> appenders,
                                       std::set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt);

  /// Used in lowerForallFusedPosition to generate code to
  /// search for the start of the iteration of the loop (a separate kernel on GPUs)
  virtual ir::Stmt searchForFusedPositionStart(Forall forall, Iterator posIterator);

    /**
     * Lower the merge lattice to code that iterates over the sparse iteration
     * space of coordinates and computes the concrete index notation statement.
     * The merge lattice dictates the code to iterate over the coordinates, by
     * successively iterating to the exhaustion of each relevant sparse iteration
     * space region (i.e., the regions in a venn diagram).  The statement is then
     * computed and/or indices assembled at each point in its sparse iteration
     * space.
     *
     * \param lattice
     *      A merge lattice that describes the sparse iteration space of the
     *      concrete index notation statement.
     * \param coordinate
     *      An IR expression that resolves to the variable containing the current
     *      coordinate the merge lattice is at.
     * \param statement
     *      A concrete index notation statement to compute at the points in the
     *      sparse iteration space described by the merge lattice.
     *
     * \return
     *       IR code to compute the forall loop.
     */
  virtual ir::Stmt lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
                                     IndexStmt statement, 
                                     const std::set<Access>& reducedAccesses);

  virtual ir::Stmt resolveCoordinate(std::vector<Iterator> mergers, ir::Expr coordinate, bool emitVarDecl);

    /**
     * Lower the merge point at the top of the given lattice to code that iterates
     * until one region of the sparse iteration space of coordinates and computes
     * the concrete index notation statement.
     *
     * \param pointLattice
     *      A merge lattice whose top point describes a region of the sparse
     *      iteration space of the concrete index notation statement.
     * \param coordinate
     *      An IR expression that resolves to the variable containing the current
     *      coordinate the merge point is at.
     *      A concrete index notation statement to compute at the points in the
     *      sparse iteration space region described by the merge point.
     */
  virtual ir::Stmt lowerMergePoint(MergeLattice pointLattice,
                                   ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                   const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared);

  /// Lower a merge lattice to cases.
  virtual ir::Stmt lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                   MergeLattice lattice,
                                   const std::set<Access>& reducedAccesses);

  /// Lower a forall loop body.
  virtual ir::Stmt lowerForallBody(ir::Expr coordinate, IndexStmt stmt,
                                   std::vector<Iterator> locaters,
                                   std::vector<Iterator> inserters,
                                   std::vector<Iterator> appenders,
                                   const std::set<Access>& reducedAccesses);


  /// Lower a where statement.
  virtual ir::Stmt lowerWhere(Where where);

  /// Lower a sequence statement.
  virtual ir::Stmt lowerSequence(Sequence sequence);

  /// Lower an assemble statement.
  virtual ir::Stmt lowerAssemble(Assemble assemble);

  /// Lower a multi statement.
  virtual ir::Stmt lowerMulti(Multi multi);

  /// Lower a suchthat statement.
  virtual ir::Stmt lowerSuchThat(SuchThat suchThat);

  /// Lower an access expression.
  virtual ir::Expr lowerAccess(Access access);

  /// Lower a literal expression.
  virtual ir::Expr lowerLiteral(Literal literal);

  /// Lower a negate expression.
  virtual ir::Expr lowerNeg(Neg neg);
	
  /// Lower an addition expression.
  virtual ir::Expr lowerAdd(Add add);

  /// Lower a subtraction expression.
  virtual ir::Expr lowerSub(Sub sub);

  /// Lower a multiplication expression.
  virtual ir::Expr lowerMul(Mul mul);

  /// Lower a division expression.
  virtual ir::Expr lowerDiv(Div div);

  /// Lower a square root expression.
  virtual ir::Expr lowerSqrt(Sqrt sqrt);

  /// Lower a cast expression.
  virtual ir::Expr lowerCast(Cast cast);

  /// Lower an intrinsic function call expression.
  virtual ir::Expr lowerCallIntrinsic(CallIntrinsic call);


  /// Lower a concrete index variable statement.
  ir::Stmt lower(IndexStmt stmt);

  /// Lower a concrete index variable expression.
  ir::Expr lower(IndexExpr expr);


  /// Check whether the lowerer should generate code to assemble result indices.
  bool generateAssembleCode() const;

  /// Check whether the lowerer should generate code to compute result values.
  bool generateComputeCode() const;


  /// Retrieve a tensor IR variable.
  ir::Expr getTensorVar(TensorVar) const;

  /// Retrieves a result values array capacity variable.
  ir::Expr getCapacityVar(ir::Expr) const;

  /// Retrieve the values array of the tensor var.
  ir::Expr getValuesArray(TensorVar, bool exclusive = true) const;

  /// Retrieve the dimension of an index variable (the values it iterates over),
  /// which is encoded as the interval [0, result).
  ir::Expr getDimension(IndexVar indexVar) const;

  /// Retrieve the chain of iterators that iterate over the access expression.
  std::vector<Iterator> getIterators(Access) const;

  /// Retrieve the access expressions that have been exhausted.
  std::set<Access> getExhaustedAccesses(MergePoint, MergeLattice) const;

  /// Retrieve the reduced tensor component value corresponding to an access.
  ir::Expr getReducedValueVar(Access) const;

  /// Retrieve the coordinate IR variable corresponding to an index variable.
  ir::Expr getCoordinateVar(IndexVar) const;

  /// Retrieve the coordinate IR variable corresponding to an iterator.
  ir::Expr getCoordinateVar(Iterator) const;


  /**
   * Retrieve the resolved coordinate variables of an iterator and it's parent
   * iterators, which are the coordinates after per-iterator coordinates have
   * been merged with the min function.
   *
   * \param iterator
   *      A defined iterator (that take part in a chain of parent iterators).
   *
   * \return
   *       IR expressions that resolve to resolved coordinates for the
   *       iterators.  The first entry is the resolved coordinate of this
   *       iterator followed by its parent's, its grandparent's, etc.
   */
  std::vector<ir::Expr> coordinates(Iterator iterator) const;

  /**
   * Retrieve the resolved coordinate variables of the iterators, which are the
   * coordinates after per-iterator coordinates have been merged with the min
   * function.
   *
   * \param iterators
   *      A set of defined iterators.
   *
   * \return
   *      IR expressions that resolve to resolved coordinates for the iterators,
   *      in the same order they were given.
   */
  std::vector<ir::Expr> coordinates(std::vector<Iterator> iterators);

  /// Generate code to initialize result indices.
  ir::Stmt initResultArrays(std::vector<Access> writes, 
                            std::vector<Access> reads,
                            std::set<Access> reducedAccesses);

  /// Generate code to finalize result indices.
  ir::Stmt finalizeResultArrays(std::vector<Access> writes);

  /**
   * Replace scalar tensor pointers with stack scalar for lowering.
   */
  ir::Stmt defineScalarVariable(TensorVar var, bool zero);

  ir::Stmt initResultArrays(IndexVar var, std::vector<Access> writes,
                            std::vector<Access> reads,
                            std::set<Access> reducedAccesses);

  ir::Stmt resizeAndInitValues(const std::vector<Iterator>& appenders,
                               const std::set<Access>& reducedAccesses);
  /**
   * Generate code to zero-initialize values array in range
   * [begin * size, (begin + 1) * size).
   */
  ir::Stmt zeroInitValues(ir::Expr tensor, ir::Expr begin, ir::Expr size);
  // lgZeroInitValues is similar to zeroInitValues but for legion tensors.
  // It is used to zero out the values allocation during assembly, but understands
  // the multi-dimensional structure of the values region.
  ir::Stmt lgZeroInitValues(const Access& acc);

  /// Declare position variables and initialize them with a locate.
  ir::Stmt declLocatePosVars(std::vector<Iterator> iterators);

  /// Emit loops to reduce duplicate coordinates.
  ir::Stmt reduceDuplicateCoordinates(ir::Expr coordinate, 
                                      std::vector<Iterator> iterators, 
                                      bool alwaysReduce);

  /**
   * Create code to declare and initialize while loop iteration variables,
   * including both pos variables (of e.g. compressed modes) and crd variables
   * (e.g. dense modes).
   *
   * \param iterators
   *      Iterators whose iteration variables will be declared and initialized.
   *
   * \return
   *      A IR statement that declares and initializes each iterator's iterators
   *      variable
   */
  ir::Stmt codeToInitializeIteratorVars(std::vector<Iterator> iterators, std::vector<Iterator> rangers, std::vector<Iterator> mergers, ir::Expr coord, IndexVar coordinateVar);
  ir::Stmt codeToInitializeIteratorVar(Iterator iterator, std::vector<Iterator> iterators, std::vector<Iterator> rangers, std::vector<Iterator> mergers, ir::Expr coordinate, IndexVar coordinateVar);

  /// Returns true iff the temporary used in the where statement is dense and sparse iteration over that
  /// temporary can be automaticallty supported by the compiler.
  std::pair<bool,bool> canAccelerateDenseTemp(Where where);

  /// Initializes a temporary workspace
  std::vector<ir::Stmt> codeToInitializeTemporary(Where where);

  /// Gets the size of a temporary tensorVar in the where statement
  ir::Expr getTemporarySize(Where where);

  /// Initializes helper arrays to give dense workspaces sparse acceleration
  std::vector<ir::Stmt> codeToInitializeDenseAcceleratorArrays(Where where);

  /// Recovers a derived indexvar from an underived variable.
  ir::Stmt codeToRecoverDerivedIndexVar(IndexVar underived, IndexVar indexVar, bool emitVarDecl);

  /// Conditionally increment iterator position variables.
  ir::Stmt codeToIncIteratorVars(ir::Expr coordinate, IndexVar coordinateVar,
          std::vector<Iterator> iterators, std::vector<Iterator> mergers);

  ir::Stmt codeToLoadCoordinatesFromPosIterators(std::vector<Iterator> iterators, bool declVars, bool boundsChecks = false);

  /// Create statements to append coordinate to result modes.
  ir::Stmt appendCoordinate(std::vector<Iterator> appenders, ir::Expr coord);

  /// Create statements to append positions to result modes.
  ir::Stmt generateAppendPositions(std::vector<Iterator> appenders);

  /// Create an expression to index into a tensor value array.
  ir::Expr generateValueLocExpr(Access access) const;

  /// Expression that evaluates to true if none of the iterators are exhausted
  ir::Expr checkThatNoneAreExhausted(std::vector<Iterator> iterators);

  /// Create an expression that can be used to filter out (some) zeros in the
  /// result
  ir::Expr generateAssembleGuard(IndexExpr expr);

  /// Check whether the result tensor should be assembled by ungrouped insertion
  bool isAssembledByUngroupedInsertion(TensorVar result);
  bool isAssembledByUngroupedInsertion(ir::Expr result);

  /// Check whether the statement writes to a result tensor
  bool hasStores(ir::Stmt stmt);

  std::pair<std::vector<Iterator>,std::vector<Iterator>>
  splitAppenderAndInserters(const std::vector<Iterator>& results);

  /// Expression that returns the beginning of a window to iterate over
  /// in a compressed iterator. It is used when operating over windows of
  /// tensors, instead of the full tensor.
  ir::Expr searchForStartOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end);

  /// Expression that returns the end of a window to iterate over
  /// in a compressed iterator. It is used when operating over windows of
  /// tensors, instead of the full tensor.
  ir::Expr searchForEndOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end);

  /// Statement that guards against going out of bounds of the window that
  /// the input iterator was configured with.
  ir::Stmt upperBoundGuardForWindowPosition(Iterator iterator, ir::Expr access);

  /// Expression that recovers a canonical index variable from a position in
  /// a windowed position iterator. A windowed position iterator iterates over
  /// values in the range [lo, hi). This expression projects values in that
  /// range back into the canonical range of [0, n).
  ir::Expr projectWindowedPositionToCanonicalSpace(Iterator iterator, ir::Expr expr);

  // projectCanonicalSpaceToWindowedPosition is the opposite of
  // projectWindowedPositionToCanonicalSpace. It takes an expression ranging
  // through the canonical space of [0, n) and projects it up to the windowed
  // range of [lo, hi).
  ir::Expr projectCanonicalSpaceToWindowedPosition(Iterator iterator, ir::Expr expr);

  /// strideBoundsGuard inserts a guard against accessing values from an
  /// iterator that don't fit in the stride that the iterator is configured
  /// with. It takes a boolean incrementPosVars to control whether the outer
  /// loop iterator variable should be incremented when the guard is fired.
  ir::Stmt strideBoundsGuard(Iterator iterator, ir::Expr access, bool incrementPosVar);

  bool anyParentInSet(IndexVar var, std::set<IndexVar>& s);

  // Helper methods for lowering Legion code.

  // declareLaunchDomain declares the index space domain for a distributed task launch.
  // It returns the statements to assign a result to domain, and also returns the color
  // space itself.
  ModeFunction declareLaunchDomain(ir::Expr domain, Forall forall, const std::vector<IndexVar>& distVars);

  // statementAccessesRegion returns true if an ir statement reads or writes
  // a region component of a tensor. The input expression target must be a
  // ir::GetProperty.
  bool statementAccessesRegion(ir::Stmt stmt, ir::Expr target);

  // declarePartitionBoundsVars declares partition bounds for a particular value of
  // an iterator through the partition's domain.
  std::vector<ir::Stmt> declarePartitionBoundsVars(ir::Expr domainIter, TensorVar tensor);

  // createDomainPointColorings generates code that colors a point in the launch domain
  // based on the current value of domainIter. It modifies fullyReplicatedTensors with
  // tensors that are fully replicated across the domain. It also takes in a parameter
  // tvMask, which if defined, restricts the generated operations to be specific to
  // tvMask only.
  std::vector<ir::Stmt> createDomainPointColorings(
      Forall forall,
      ir::Expr domainIter,
      std::map<TensorVar, ir::Expr> domains,
      std::set<TensorVar>& fullyReplicatedTensors,
      std::vector<ir::Expr> colorings,
      TensorVar tvMask = TensorVar()
  );

  // createIndexPartitions creates IndexPartitions from constructed point colorings
  // of each tensor being transferred at the current level. It also takes in a parameter
  // tvMask, which if defined, restricts the generated operations to be specific to
  // tvMask only.
  std::vector<ir::Stmt> createIndexPartitions(
      Forall forall,
      ir::Expr domain,
      ir::Expr colorSpace,
      std::map<TensorVar, std::map<int, std::vector<ir::Expr>>>& tensorLogicalPartitions,
      std::map<TensorVar, std::map<int, ir::Expr>>& tensorDenseRunPartitions,
      std::set<TensorVar> fullyReplicatedTensors,
      std::vector<ir::Expr> colorings,
      const std::vector<IndexVar>& distIvars,
      TensorVar tvMask = TensorVar()
  );

  // getPrivilegeForTensor returns the Legion privilege and coherence mode to access
  // the input tensor with.
  std::pair<ir::Expr, ir::Expr> getPrivilegeForTensor(Forall forall, const TensorVar& tv);

  // lowerIndexLaunch lowers a loop into a parallel index launch.
  std::vector<ir::Stmt> lowerIndexLaunch(
      Forall forall,
      ir::Expr domain,
      std::map<TensorVar, std::map<int, std::vector<ir::Expr>>>& tensorLogicalPartitions,
      std::set<ir::GetProperty::Hashable> tensorsAccessed,
      int taskID,
      ir::Stmt& unpackTensorData
  );

  // lowerSerialTaskLoop lowers a loop of serial task launches.
  std::vector<ir::Stmt> lowerSerialTaskLoop(
      Forall forall,
      ir::Expr domain,
      ir::Expr domainIter,
      Datatype pointT,
      std::map<TensorVar, std::map<int, std::vector<ir::Expr>>>& tensorLogicalPartitions,
      std::set<ir::GetProperty::Hashable> tensorsAccessed,
      int taskID,
      ir::Stmt& unpackTensorData
  );

  // loweringToGPU denotes whether this Lowerer is configured to lower to GPUs. It uses
  // a combination of the static should_use_CUDA_codegen() function and containsGPULoops.
  bool loweringToGPU();
  // containsGPULoops is true if the target IndexStmt contains GPU parallelized loops.
  bool containsGPULoops = false;

private:
  bool assemble;
  bool compute;
  bool legion = false;
  std::string funcName;

  std::set<TensorVar> needCompute;

  int markAssignsAtomicDepth = 0;
  ParallelUnit atomicParallelUnit;

  std::set<TensorVar> assembledByUngroupedInsert;

  /// Map used to hoist temporary workspace initialization
  std::map<Forall, Where> temporaryInitialization;

  /// Map from tensor variables in index notation to variables in the IR
  std::map<TensorVar, ir::Expr> tensorVars;
  /// A backwards map from tensor IR variables to TensorVars.
  std::map<ir::Expr, TensorVar> exprToTensorVar;
  std::map<TensorVar, ir::Expr> scalars;

  // Set of tensors that will be written to.
  std::set<TensorVar> resultTensors;
  // A set of tensors that are distributed. This set contains the input
  // and output tensors, but does not contain any temporary workspaces
  // that are local to a single memory.
  std::set<TensorVar> legionTensors;

  struct TemporaryArrays {
    ir::Expr values;
  };
  std::map<TensorVar, TemporaryArrays> temporaryArrays;

  // This set of fields is used by distributed assembly.
  // assembleQueryResults is a set of TensorVars that contain the result
  // of attribute queries for assemble.
  std::set<TensorVar> assembleQueryResults;
  // loweringAssembleQueries is set to true if we are currently lowering
  // an assemble query.
  bool loweringAssembleQueries = false;
  // loweringAssembleCompute is set to true if we are currently lowering
  // the compute phase of an assemble operation.
  bool loweringAssembleCompute = false;
  // assembleQueryIndexSpace is an expression of an index space used by an
  // attribute query to launch tasks over. It is intended to be reused by
  // some assembly functions to distribute over the same space as the compute.
  ir::Expr assembleQueryIndexSpace;
  // assembleIsomorphicQueryAndCompute marks whether or not the structure of the
  // assemble attribute queries and compute statements are the same.
  bool assembleIsomorphicQueryAndCompute = false;
  // assembleComputeIVarToQueryIVar is a mapping of IndexVars in the compute
  // statement to IndexVars in the attribute queries. It is only set if
  // assembleIsomorphicQueryAndCompute is true.
  std::map<IndexVar, IndexVar> assembleComputeIVarToQueryIVar;
  // assembleQueryPartitions is a map of expressions used to partition
  // tensors in the query phase. These partitions can then be reused by
  // the compute phase.
  // TODO (rohany): I don't think that this will work when there are nested
  //  distributions. In that case, we'll need to mark the partitions with
  //  colors, and then look up the right colors in the subtasks.
  std::map<IndexVar, std::map<TensorVar, std::map<int, std::vector<ir::Expr>>>> assembleQueryPartitions;

  /// Map form temporary to indexList var if accelerating dense workspace
  std::map<TensorVar, ir::Expr> tempToIndexList;

  /// Map form temporary to indexListSize if accelerating dense workspace
  std::map<TensorVar, ir::Expr> tempToIndexListSize;

  /// Map form temporary to bitGuard var if accelerating dense workspace
  std::map<TensorVar, ir::Expr> tempToBitGuard;

  std::set<TensorVar> guardedTemps;

  /// Map from result tensors to variables tracking values array capacity.
  std::map<ir::Expr, ir::Expr> capacityVars;

  /// Map from index variables to their dimensions, currently [0, expr).
  std::map<IndexVar, ir::Expr> dimensions;

  /// Map from index variables to their bounds, currently also [0, expr) but allows adding minimum in future too
  std::map<IndexVar, std::vector<ir::Expr>> underivedBounds;

  /// Map from indexvars to their variable names
  std::map<IndexVar, ir::Expr> indexVarToExprMap;
  /// Maintain a backwards map from variables to the corresponding IndexVar's.
  std::map<ir::Expr, IndexVar> exprToIndexVarMap;

  /// Tensor and mode iterators to iterate over in the lowered code
  Iterators iterators;

  /// Keep track of relations between IndexVars
  ProvenanceGraph provGraph;

  bool ignoreVectorize = false; // already being taken into account

  std::vector<ir::Stmt> whereConsumers;
  std::vector<TensorVar> whereTemps;
  std::map<TensorVar, const AccessNode *> whereTempsToResult;

  bool captureNextLocatePos = false;
  ir::Stmt capturedLocatePos; // used for whereConsumer when want to replicate same locating

  bool emitUnderivedGuards = true;

  int inParallelLoopDepth = 0;

  std::map<ParallelUnit, ir::Expr> parallelUnitSizes;
  std::map<ParallelUnit, IndexVar> parallelUnitIndexVars;

  /// Keep track of what IndexVars have already been defined
  std::set<IndexVar> definedIndexVars;
  std::vector<IndexVar> definedIndexVarsOrdered;

  /// Map from tensor accesses to variables storing reduced values.
  std::map<Access, ir::Expr> reducedValueVars;

  /// Set of locate-capable iterators that can be legally accessed.
  util::ScopedSet<Iterator> accessibleIterators;

  /// Visitor methods can add code to emit it to the function header.
  std::vector<ir::Stmt> header;
  // taskHeader is similar to header, but is emitted at the header of
  // the current task.
  std::vector<ir::Stmt> taskHeader;

  /// Visitor methods can add code to emit it to the function footer.
  std::vector<ir::Stmt> footer;

  int taskCounter = 1;

  class Visitor;
  friend class Visitor;
  std::shared_ptr<Visitor> visitor;

  // presentIvars is a set of all index variables that are present in a CIN expression.
  std::set<IndexVar> presentIvars;
  std::map<IndexVar, std::map<TensorVar, std::vector<std::vector<ir::Expr>>>> derivedBounds;
  IndexVar curDistVar;
  std::map<IndexVar, std::set<IndexVar>> varsInScope;
  int distLoopDepth = 0;

  // We support distributing onto multiple partitions at the same time, as long
  // as they are on the same grids and index variables. This just allows us to
  // re-use as many partitions as possible, if necessary.
  std::map<TensorVar, ir::Expr> computingOnPartition;
  std::vector<TensorVar> computingOnTensorVar;

  bool performingLegionReduction = false;
  bool performingScalarReduction = false;
  ir::Expr scalarReductionResult;

  std::vector<TensorVar> tensorVarOrdering;

  bool isPlacementCode = false;
  // Manages transferring information between the distribution syntax and
  // the lowerer. This is set by the old distribution syntax.
  std::vector<std::pair<Grid, GridPlacement>> placements;
  // Information about the Tensor Distribution Notation statement being lowered.
  std::vector<TensorDistributionNotation> tensorDistributionNotation;
  // getDataDistributionNestingDepth is a layer of indirection between the old
  // and new notations for data distribution, allowing for the lowerer to understand
  // which level of nested data distribution is being considered.
  size_t getDataDistributionNestingDepth() {
    taco_iassert(this->isPlacementCode);
    taco_iassert(!this->placements.empty() || !this->tensorDistributionNotation.empty());
    return std::max(this->placements.size(), this->tensorDistributionNotation.size());
  }

  bool isPartitionCode = false;

  std::map<IndexVar, int> indexVarFaces;
  std::map<IndexVar, std::shared_ptr<LeafCallInterface>> calls;

  bool waitOnFutureMap;
  bool partitionPackAsPointer;

  // rcmfCrdDomains contains variables that hold the domains of RectCompressedModeFormat's
  // that are needed to sometimes override the position bounds computed by modes.
  std::map<TensorVar, std::map<int, ir::Expr>> rcmfCrdDomains;
  // getOverridenPosBounds returns bounds on the position variable
  // in cases where the real position bounds may be tighter than the
  // bounds returned by the posIterator. This can happen when a merger
  // loop is strip mined and distributed. If the pos bounds are not
  // overridden, then an empty vector is returned.
  std::vector<ir::Expr> getOveriddenPosBounds(Iterator it);

  // setPlacementPrivilege controls whether or not the generated placement code
  // has a parameter to control the privilege to launch placement tasks with.
  bool setPlacementPrivilege;
  // placementPrivilegeVar is the ir::Var to represent the privilege.
  ir::Expr placementPrivilegeVar;

  // LegionLoweringKind controls how the lowerer should generate code
  // for the target statement.
  enum LegionLoweringKind {
    PARTITION_AND_COMPUTE,
    PARTITION_ONLY,
    COMPUTE_ONLY,
  };
  LegionLoweringKind legionLoweringKind;
  // computeOnlyPartitions holds onto a partition argument for each tensor
  // when the LegionLoweringKind is COMPUTE_ONLY. These partition arguments
  // are the top level partition for each tensor. All sub-partitions are
  // indexed through the iteration space point identifier scheme.
  std::map<TensorVar, ir::Expr> computeOnlyPartitions;
  // topLevelPartitionPack is a struct containing fields of partitions for each
  // tensor that is partitioned at the top level of the computation. It is used
  // when the LegionLoweringKind is COMPUTE_ONLY, as this argument should be
  // passed in as a parameter.
  ir::Expr topLevelPartitionPack;
  ir::Expr getTopLevelTensorPartition(TensorVar& t) {
    // TODO (rohany): I should probably extract the naming of the field into a helper
    //  method so that the logic is centralized.
    return ir::FieldAccess::make(topLevelPartitionPack, t.getName() + "Partition", true /* isDeref */, Auto);
  }
  Datatype getTopLevelTensorPartitionPackType() {
    // Copy the name.
    auto name = this->funcName;
    // TODO (rohany): This is just a hack to get things to work out.
    // Strip off partitionFor from the front of the name, if it exists.
    auto pos = name.find("partitionFor");
    if (pos != std::string::npos) {
      name.erase(pos, std::string("partitionFor").size());
    }
    return Datatype("partitionPackFor" + name);
  }

  // Maintain a set of all of the defined index variables. Note that we cannot use
  // the existing definedIndexVarsOrdered because it does not expand the distributed fused
  // index variables out into their component variables.
  std::vector<IndexVar> definedIndexVarsExpanded;
  std::vector<ir::Expr> iterationSpacePointIdentifiers;
  void pushIterationSpacePointIdentifier() {
    auto len = definedIndexVarsExpanded.size();
    std::stringstream ss;
    ss << "pointID" << len;
    auto var = ir::Var::make(ss.str(), Int64);
    this->iterationSpacePointIdentifiers.push_back(var);
  }
  ir::Expr getIterationSpacePointIdentifier() {
    return this->iterationSpacePointIdentifiers[this->definedIndexVarsExpanded.size() - 1];
  }

  // DenseFormatRuns constructs metadata about all of the runs of dense
  // formats in a tensor.
  struct DenseFormatRuns {
    DenseFormatRuns(const Access& a, const Iterators& iterators);
    struct DenseRun {
      std::vector<int> modes;
      std::vector<int> levels;
    };
    std::vector<DenseRun> runs;
  };

  // ValuesAnalyzer maintains information about the dimensionality of the
  // values array of each tensor, as well as information about how to access it.
  struct ValuesAnalyzer {
    void addAccess(const Access& access, const Iterators& iterators, const std::map<IndexVar, ir::Expr>& indexVarToExprMap);
    int getValuesDim(const TensorVar& tv) const;
    ir::Expr getAccessPoint(const Access& access) const;

    // Member variables.
    std::map<TensorVar, int> valuesDims;
    // valuesAccess maintains a map between accesses and the variables needed
    // to index into the values region.
    std::map<Access, std::vector<ir::Expr>> valuesAccess;
    // valuesAccessWidths maintains a map between accesses and the size of
    // each dimension for each index variable used to index into the region.
    std::map<Access, std::vector<ir::Expr>> valuesAccessWidths;
  } valuesAnalyzer;

  // getAllNeededParentPositions get all of the position variables needed for multi-dimensional
  // access into an iterator.
  std::vector<ir::Expr> getAllNeededParentPositions(Iterator& iter);

  // construct an AffineProjection between the bottom-up partition of the top dense
  // level pack of `from` to the fully-dense level pack of `to`. It returns an empty
  // Expr (i.e. !ret.defined()) when `to` is not partitioned by `from`.
  ir::Expr constructAffineProjection(Access& from, Access& to);

  // construct a SparseGatherProjection between the partition of a
  // sparse level of `from` to the dense level pack of `to`. It returns
  // an empty Expr (i.e. !ret.defined()) when `to` cannot be partitioned
  // with a SparseGatherProjection.
  ir::Expr constructSparseGatherProjection(Access& from, Access& to, int posMode);

  // tensorVarToAccess contains a mapping between TensorVar's to the access in
  // which they appear in the statement. We require that each tensor is used once
  // in a statement right now, so we only populate this map when this->legion = true.
  std::map<TensorVar, Access> tensorVarToAccess;

  // getAllAccessors returns a deduplicated list of all accessors present in a
  // statement. The returnd list of ir::Expr are gauranteed to be ir::GetProperty's.
  std::vector<ir::Expr> getAllAccessors(ir::Stmt stmt);

  // stmtHasTasks returns whether or not the input statement contains any
  // for loops that are tasks.
  bool stmtHasTasks(ir::Stmt stmt);

  // stmtHasAssemble returns whether or not the input IndexNotation statement contains
  // any AssembleNodes.
  bool stmtHasAssemble(IndexStmt stmt);

  // BoundsInferenceExprRewriter rewrites an expression by replacing Var's corresponding
  // to IndexVariable's with their upper and lower bounds, if those variables are not
  // current defined.
  struct BoundsInferenceExprRewriter : public ir::IRRewriter {
    BoundsInferenceExprRewriter(ProvenanceGraph &pg, Iterators &iterators,
                                std::map<IndexVar, std::vector<ir::Expr>> &underivedBounds,
                                std::map<IndexVar, ir::Expr> &indexVarToExprMap,
                                std::set<IndexVar> &inScopeVars,
                                std::map<ir::Expr, IndexVar>& exprToIndexVarMap,
                                std::vector<IndexVar>& definedIndexVars,
                                bool lower,
                                std::set<IndexVar> presentIvars);
    void visit(const ir::Var* var);
    void visit(const ir::GetProperty* gp);

    // Fields needed for the rewriter.
    ProvenanceGraph& pg;
    Iterators& iterators;
    std::map<IndexVar, std::vector<ir::Expr>>& underivedBounds;
    std::map<IndexVar, ir::Expr>& indexVarToExprMap;
    std::vector<IndexVar>& definedIndexVars;
    std::map<ir::Expr, IndexVar>& exprToIndexVarMap;
    std::set<IndexVar>& inScopeVars;
    bool lower;
    std::set<IndexVar> presentIvars;
    bool changed = false;
  };

  // These two fields maintain information about if the optimization
  // to write into sparse outputs with the same non-zero structure
  // as the input tensor is enabled.
  bool preservesNonZeros = false;
  NonZeroAnalyzerResult nonZeroAnalyzerResult;

  // Some common Legion expressions and types. Symbols that are needed outside of
  // the lowerer are defined in ir.{h, cpp}.
  static inline ir::Expr exclusive = ir::Symbol::make("EXCLUSIVE");
  static inline ir::Expr simultaneous = ir::Symbol::make("LEGION_SIMULTANEOUS");
  static inline ir::Expr task = ir::Symbol::make("task");
  static inline ir::Expr virtualMap = ir::Symbol::make("Mapping::DefaultMapper::VIRTUAL_MAP");
  static inline ir::Expr placementMap = ir::Symbol::make("TACOMapper::PLACEMENT");
  static inline ir::Expr placementShard = ir::Symbol::make("TACOMapper::PLACEMENT_SHARD");
  static inline ir::Expr untrackValidRegions = ir::Symbol::make("TACOMapper::UNTRACK_VALID_REGIONS");
  static inline ir::Expr sameAddressSpace = ir::Symbol::make("Mapping::DefaultMapper::SAME_ADDRESS_SPACE");
  static inline ir::Expr backpressureTask = ir::Symbol::make("TACOMapper::BACKPRESSURE_TASK");
};

}
#endif
