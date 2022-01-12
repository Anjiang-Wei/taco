#ifndef TACO_MODE_FORMAT_IMPL_H
#define TACO_MODE_FORMAT_IMPL_H

#include <vector>
#include <initializer_list>
#include <memory>
#include <string>
#include <map>
#include <tuple>

#include "taco/format.h"
#include "taco/ir/ir.h"
#include "taco/lower/mode.h"
#include "taco/index_notation/index_notation.h"

namespace taco {

class ModeFormatImpl;
class ModeFormatPack;
class ModePack;

class AttrQuery {
public:
  enum Aggregation { IDENTITY, COUNT, MIN, MAX };
  struct Attr {
    Attr(std::tuple<std::string,Aggregation,std::vector<IndexVar>> attr); 

    std::string label;
    Aggregation aggr;
    std::vector<IndexVar> params;
  };

  AttrQuery();
  AttrQuery(const std::vector<IndexVar>& groupBy, const Attr& attr);
  AttrQuery(const std::vector<IndexVar>& groupBy, 
            const std::vector<Attr>& attrs);

  const std::vector<IndexVar>& getGroupBy() const;
  const std::vector<Attr>& getAttrs() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const AttrQuery::Attr&);
std::ostream& operator<<(std::ostream&, const AttrQuery&);

class AttrQueryResult {
public:
  AttrQueryResult() = default;
  AttrQueryResult(ir::Expr resultVar, ir::Expr resultValuesArray, ir::Expr resultValuesAccessor);

  ir::Expr getResult(const std::vector<ir::Expr>& indices, 
                     const std::string& attr) const;

  friend std::ostream& operator<<(std::ostream&, const AttrQueryResult&);

private:
  ir::Expr resultVar;
  ir::Expr resultValuesArray;
  ir::Expr resultValuesAccessor;
};

// ModeRegion is information about a region used by a mode.
struct ModeRegion {
  ir::Expr region;
  ir::Expr regionParent;
  ir::Expr field;
  ir::Expr accessorRO;
  ir::Expr accessorRW;
};

std::ostream& operator<<(std::ostream&, const AttrQueryResult&);

/// Mode functions implement parts of mode capabilities, such as position
/// iteration and locate.  The lower machinery requests mode functions from
/// mode type implementations (`ModeTypeImpl`) and use these to generate code
/// to iterate over and assemble tensors.
class ModeFunction {
public:
  /// Construct an undefined mode function.
  ModeFunction();

  /// Construct a mode function.
  ModeFunction(ir::Stmt body, const std::vector<ir::Expr>& results);

  /// Retrieve the mode function's body where arguments are inlined.  The body
  /// may be undefined (when the result expression compute the mode function).
  ir::Stmt compute() const;

  /// Retrieve the ith mode function result.
  ir::Expr operator[](size_t i) const;

  /// The number of results
  size_t numResults() const;

  /// Retrieve the mode function's result expressions.
  const std::vector<ir::Expr>& getResults() const;

  /// True if the mode function is defined.
  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const ModeFunction&);


/// The abstract class to inherit from to add a new mode format to the system.
/// The mode type implementation can then be passed to the `ModeType`
/// constructor.
class ModeFormatImpl {
public:
  ModeFormatImpl(std::string name, bool isFull, bool isOrdered, bool isUnique, 
                 bool isBranchless, bool isCompact, bool isZeroless, 
                 bool hasCoordValIter, bool hasCoordPosIter, bool hasLocate, 
                 bool hasInsert, bool hasAppend, bool hasSeqInsertEdge, 
                 bool hasInsertCoord, bool isYieldPosPure);

  virtual ~ModeFormatImpl();

  /// Create a copy of the mode type with different properties.
  virtual ModeFormat copy(
      std::vector<ModeFormat::Property> properties) const = 0;


  virtual std::vector<AttrQuery> attrQueries(
      std::vector<IndexVar> parentCoords, 
      std::vector<IndexVar> childCoords) const;


  /// The coordinate iteration capability's iterator function computes a range
  /// [result[0], result[1]) of coordinates to iterate over.
  /// `coord_iter_bounds(i_{1}, ..., i_{k−1}) -> begin_{k}, end_{k}`
  virtual ModeFunction coordIterBounds(std::vector<ir::Expr> parentCoords,
                                       Mode mode) const;

  /// The coordinate iteration capability's access function maps a coordinate
  /// iterator variable to a position (result[0]) and reports if a position
  /// could not be found (result[1]).
  /// `coord_iter_access(p_{k−1}, i_{1}, ..., i_{k}) -> p_{k}, found`
  virtual ModeFunction coordIterAccess(ir::Expr parentPos,
                                       std::vector<ir::Expr> coords,
                                       Mode mode) const;

  virtual ModeFunction coordBounds(ir::Expr parentPos, Mode mode) const;


  /// The position iteration capability's iterator function computes a range
  /// [result[0], result[1]) of positions to iterate over.
  /// `pos_iter_bounds(p_{k−1}) -> begin_{k}, end_{k}`
  /// The interface provides a list of parentPositions in case several parent positions
  /// are needed to access the bounds of the position iteration.
  virtual ModeFunction posIterBounds(std::vector<ir::Expr> parentPositions, Mode mode) const;

  /// The position iteration capability's access function maps a position
  /// iterator variable to a coordinate (result[0]) and reports if a coordinate
  /// could not be found (result[1]).
  /// `pos_iter_access(p_{k}, i_{1}, ..., i_{k−1}) -> i_{k}, found`
  virtual ModeFunction posIterAccess(ir::Expr pos, 
                                     std::vector<ir::Expr> coords,
                                     Mode mode) const;


  /// The locate capability locates the position of a coordinate (result[0])
  /// and reports if the coordinate could not be found (result[1]).
  /// `locate(p_{k−1}, i_{1}, ..., i_{k}) -> p_{k}, found`
  virtual ModeFunction locate(ir::Expr parentPos,
                              std::vector<ir::Expr> coords,
                              Mode mode) const;


  /// Level functions that implement grouped insert capabilitiy.
  /// @{
  virtual ir::Stmt
  getInsertCoord(ir::Expr p, const std::vector<ir::Expr>& i, Mode mode) const;

  virtual ir::Expr getWidth(Mode mode) const;

  virtual ir::Stmt
  getInsertInitCoords(ir::Expr pBegin, ir::Expr pEnd, Mode mode) const;

  virtual ir::Stmt
  getInsertInitLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;

  virtual ir::Stmt
  getInsertFinalizeLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;
  /// @}

  
  /// Level functions that implement append capabilitiy.
  /// @{
  virtual ir::Stmt
  getAppendCoord(ir::Expr p, ir::Expr i, Mode mode) const;

  // Similarly to posIterBounds, this function also takes a vector of parent
  // positions in case the level format needs them.
  virtual ir::Stmt
  getAppendEdges(std::vector<ir::Expr> parentPositions, ir::Expr pBegin, ir::Expr pEnd,
                 Mode mode) const;

  virtual ir::Expr getSize(ir::Expr parentSize, Mode mode) const;

  // parentPos is the position variables of the nearest sparse ancestor of a mode.
  virtual ir::Stmt
  getAppendInitEdges(ir::Expr parentPos, ir::Expr pPrevBegin, ir::Expr pPrevEnd, Mode mode) const;

  virtual ir::Stmt
  getAppendInitLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;

  // parentPos is the position variable of the nearest sparse ancestor of a mode.
  // It can be undefined if the mode does not have a sparse ancestor.
  virtual ir::Stmt
  getAppendFinalizeLevel(ir::Expr parentPos, ir::Expr szPrev, ir::Expr sz, Mode mode) const;
  /// @}

  /// Level functions that implement ungrouped insert capabilitiy.
  /// @{
  virtual ir::Expr getAssembledSize(ir::Expr prevSize, Mode mode) const;

  virtual ir::Stmt
  getSeqInitEdges(ir::Expr prevSize, std::vector<ir::Expr> parentDims,
                  std::vector<AttrQueryResult> queries, Mode mode) const;
  
  virtual ir::Stmt
  getSeqInsertEdges(ir::Expr parentPos, std::vector<ir::Expr> parentDims, ir::Expr colorSpace,
                    std::vector<ir::Expr> coords, std::vector<AttrQueryResult> queries, Mode mode) const;

  virtual ir::Stmt
  getInitCoords(ir::Expr prevSize, std::vector<AttrQueryResult> queries, 
                Mode mode) const;

  virtual ir::Stmt
  getInitYieldPos(ir::Expr prevSize, Mode mode) const;
  
  virtual ModeFunction
  getYieldPos(ir::Expr parentPos, std::vector<ir::Expr> coords, 
              Mode mode) const;

  virtual ir::Stmt
  getInsertCoord(ir::Expr parentPos, ir::Expr pos, std::vector<ir::Expr> coords, 
                 Mode mode) const;

  virtual ir::Stmt
  getFinalizeYieldPos(ir::Expr prevSize, Mode mode) const;
  /// @}

  /// One confusion here was about dense tensors, as they needed to somehow
  /// pass through information. The idea here is to associate an index space
  /// with each consecutive run of dense levels in a tensor. Then, there is a
  /// separate _initial_ partitioning process for each run of dense levels,
  /// and that partition can be then passed down to the child levels of the
  /// tensor to use to create dependent partitions. This can be implemented in
  /// maybe a similar manner as the way that multiple coordinates are stored
  /// in array of structs format by having the different modes point into the
  /// same array via the modepack.
  ///
  /// Another revision to this --- we can't actually do this runs of dense
  /// levels _below_ the first sparse level. The problem is that we'd need an
  /// index space for _each entry_ in the level above. But, lo-and-behold, I
  /// think that we could do it by just having a 1-higher dimension so that
  /// we have a separate dimension of the index space for each entry, so then
  /// indexing is simple. This is also great, because now we can express how
  /// to partition a dense dimension run given a partition of the parent --
  /// simply extend the partition along the first dimension. It seems like you
  /// would want something like this if you had a computation on a tensor with
  /// format {Sparse, Dense, Dense}, and you wanted to run a serial loop over
  /// the sparse tensor that launched distributed jobs on the {Dense, Dense}
  /// sub-levels. This is also a reasonable thing to do if you have control
  /// replication as those child jobs are going to be cheaper to launch.
  ///
  /// Level functions related to partitioning. Very much a WIP.

  // This group of functions contains capabilities to perform a partitioning
  // of the position space of a tensor mode.

  // Initialize data structures needed to create a coloring of the position space.
  virtual ir::Stmt getInitializePosColoring(Mode mode) const;
  // Create an entry in a position space coloring given the domain point, and the
  // upper and lower bounds of the position space accessed.
  virtual ir::Stmt getCreatePosColoringEntry(Mode mode, ir::Expr domainPoint, ir::Expr lowerBound, ir::Expr upperBound) const;
  // Finalize any data structures needed to create a coloring of the position space.
  virtual ir::Stmt getFinalizePosColoring(Mode mode) const;
  // Create an initial partition of this format level using the created coloring
  // from above. This function returns a ModeFunction where the first n elements
  // are a partition for each region in the mode (i.e. n == this->getRegions().size()).
  // The last two elements are an initial partition to use as an upwards partition
  // and a downwards partition.
  virtual ModeFunction getCreatePartitionWithPosColoring(Mode mode, ir::Expr domain, ir::Expr partitionColor) const;

  // Analogously to the position space coloring methods, there should be a corresponding
  // set of methods to color and partition a tensor level via a coordinate space. However,
  // this would ideally be implemented by Dense levels that are allowed to span multiple
  // dimensions and correspond exactly to the DenseFormatRuns in the implementation. So
  // for now, the Lowerer handles these cases explicitly, but in a different implementation
  // this indirection could be used.

  // The idea here is that given an IndexPartition of an object, the resulting
  // ModeFunction computes a partition of each array in the mode, and returns
  // the partition to use to partition the lower levels of the tree.
  virtual ModeFunction getPartitionFromParent(ir::Expr parentPartition, Mode mode, ir::Expr partitionColor) const;
  virtual ModeFunction getPartitionFromChild(ir::Expr childPartition, Mode mode, ir::Expr partitionColor) const;

  // TODO (rohany): It seems like we'll also want a "getAccessors" method that complements
  //  the getArrays method to maintain accessor variables for each of the arrays in the format.
  // TODO (rohany): In addition / in complement to the "getAccessors" method, we'll also want
  //  a method that just tells (from the format) how to set up the accessors for each tensor,
  //  and we can then insert one of these at the head of each task so that we can correctly unpack
  //  each of the packed regions from the input regions.

  /// @}

  /// Returns arrays associated with a tensor mode
  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, int level) const = 0;

  // getRegions is an analogous function to getArrays that returns all of the Regions used
  // by a particular mode, as well as the parent of each region. Next, the regions returned
  // from getRegions must be returned in the same order as the partitions in the ModeFunctions
  // returned from getPartitionFrom{Parent,Child}.
  virtual std::vector<ModeRegion> getRegions(ir::Expr tensor, int level) const;

  friend bool operator==(const ModeFormatImpl&, const ModeFormatImpl&);
  friend bool operator!=(const ModeFormatImpl&, const ModeFormatImpl&);

  const std::string name;

  const bool isFull;
  const bool isOrdered;
  const bool isUnique;
  const bool isBranchless;
  const bool isCompact;
  const bool isZeroless;
  
  const bool hasCoordValIter;
  const bool hasCoordPosIter;
  const bool hasLocate;
  const bool hasInsert;
  const bool hasAppend;
  const bool hasSeqInsertEdge;
  const bool hasInsertCoord;
  const bool isYieldPosPure;

protected:
  /// Check if other mode format is identical. Can assume that this method will 
  /// always be called with an argument that is of the same class.
  virtual bool equals(const ModeFormatImpl& other) const;
  // Utility function to store variables inside the Mode.
  ir::Expr getModeVar(Mode mode, const std::string varName, Datatype type) const;
};

// If we are building in Debug configuration, then by default start with an
// allocation of only size 1. This allows codepaths around reallocation to
// get triggered and potentially expose bugs.
#ifndef NDEBUG
static const int DEFAULT_ALLOC_SIZE = 1;
#else
static const int DEFAULT_ALLOC_SIZE = 1 << 20;
#endif

}
#endif

