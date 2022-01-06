#ifndef TACO_MODE_FORMAT_RECT_COMPRESSED_H
#define TACO_MODE_FORMAT_RECT_COMPRESSED_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

// If we are building in Debug configuration, then by default start with an
// allocation of only size 1. This allows codepaths around reallocation to
// get triggered and potentially expose bugs.
#ifndef NDEBUG
static const int LG_DEFAULT_ALLOC_SIZE = 1;
#else
// TODO (rohany): See what a decent default is here.
static const int LG_DEFAULT_ALLOC_SIZE = 1 << 20;
#endif

// TODO (rohany): I want a Legion ModePack that I can sub in here, or adjust
//  the ModePack default structure to use the LegionTensor when possible.

class RectCompressedModeFormat : public ModeFormatImpl {
public:
  using ModeFormatImpl::getInsertCoord;

  RectCompressedModeFormat();
  RectCompressedModeFormat(int posDim);
  RectCompressedModeFormat(bool isFull, bool isOrdered,
                           bool isUnique, bool isZeroless, int posDim, long long allocSize = LG_DEFAULT_ALLOC_SIZE);

  ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;

  // Implementing attrQueries allows us to use the Assemble infrastructure.
  std::vector<AttrQuery>
  attrQueries(std::vector<IndexVar> parentCoords,
              std::vector<IndexVar> childCoords) const override;
  ir::Expr getAssembledSize(ir::Expr prevSize, Mode mode) const override;
  ir::Stmt getInitYieldPos(ir::Expr prevSize, Mode mode) const override;
  ModeFunction getYieldPos(ir::Expr parentPos, std::vector<ir::Expr> coords,
                           Mode mode) const override;
  ir::Stmt getSeqInitEdges(ir::Expr prevSize,
                           std::vector<ir::Expr> parentDims,
                           std::vector<AttrQueryResult> queries,
                           Mode mode) const override;
  ir::Stmt getSeqInsertEdges(ir::Expr parentPos,
                             std::vector<ir::Expr> parentDims,
                             std::vector<ir::Expr> coords,
                             std::vector<AttrQueryResult> queries,
                             Mode mode) const override;
  ir::Stmt getInitCoords(ir::Expr prevSize,
                         std::vector<AttrQueryResult> queries,
                         Mode mode) const override;
  ir::Stmt getInsertCoord(ir::Expr parentPos, ir::Expr pos,
                          std::vector<ir::Expr> coords,
                          Mode mode) const override;
  ir::Stmt getFinalizeYieldPos(ir::Expr prevSize, Mode mode) const override;
  // End functions related to Assemble infrastructure.

  // Similarly to the compressed level, this format supports position iterate.
  ModeFunction posIterBounds(std::vector<ir::Expr> parentPos, Mode mode) const override;
  ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords, Mode mode) const override;

  // Definitions for insertion into a tensor level.
  ir::Stmt getAppendCoord(ir::Expr pos, ir::Expr coord, Mode mode) const override;
  ir::Stmt getAppendEdges(std::vector<ir::Expr> parentPositions, ir::Expr posBegin, ir::Expr posEnd, Mode mode) const override;
  ir::Expr getSize(ir::Expr szPrev, Mode mode) const override;
  ir::Stmt getAppendInitEdges(ir::Expr parentPos, ir::Expr parentPosBegin, ir::Expr parentPosEnd, Mode mode) const override;
  ir::Stmt getAppendInitLevel(ir::Expr parentSize, ir::Expr size, Mode mode) const override;
  ir::Stmt getAppendFinalizeLevel(ir::Expr parentPos, ir::Expr parentSize, ir::Expr size, Mode mode) const override;

  // Partitioning capabilities.
  ir::Stmt getInitializePosColoring(Mode mode) const override;
  ir::Stmt getFinalizePosColoring(Mode mode) const override;
  ir::Stmt getCreatePosColoringEntry(Mode mode, ir::Expr domainPoint, ir::Expr lowerBound, ir::Expr upperBound) const override;
  ModeFunction getCreatePartitionWithPosColoring(Mode mode, ir::Expr domain, ir::Expr partitionColor) const override;
  ModeFunction getPartitionFromParent(ir::Expr parentPartition, Mode mode, ir::Expr partitionColor) const override;
  ModeFunction getPartitionFromChild(ir::Expr childPartition, Mode mode, ir::Expr partitionColor) const override;

  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, int level) const override;
  std::vector<ModeRegion> getRegions(ir::Expr tensor, int level) const override;

  ir::Stmt declareModeVariables(Mode& mode) const;

  // Public methods to get information about regions.
  enum RECT_COMPRESSED_REGIONS {
    POS = 0,
    CRD,
    POS_PARENT,
    CRD_PARENT,
  };
  ir::Expr getRegion(ModePack pack, RECT_COMPRESSED_REGIONS reg) const;
  ir::Expr getAccessor(ModePack pack, RECT_COMPRESSED_REGIONS reg, ir::RegionPrivilege priv = ir::RO) const;
protected:

  ir::Expr getPosCapacity(Mode mode) const;
  ir::Expr getCoordCapacity(Mode mode) const;
  ir::Expr packToPoint(const std::vector<ir::Expr>& args) const;
  ir::Expr getPosBounds(Mode mode) const;
  ir::Expr getCrdBounds(Mode mode) const;
  ir::Stmt initPosBounds(Mode mode) const;
  ir::Stmt initCrdBounds(Mode mode) const;
  ir::Expr getCrdColoring(Mode mode) const;

  ModeFunction partitionPosFromCrd(Mode mode, ir::Expr crdIndexPartition,
                                   std::function<std::vector<ir::Expr>(std::vector<ir::Expr>)> maybeAddColor) const;

  // hasSparseAncestor returns true if there is a sparse mode
  // above this mode, i.e. it returns false if all parent levels
  // are dense.
  bool hasSparseAncestor(Mode mode) const;

  std::string getModeSizeVarName(Mode& mode) const;

  static inline ir::Expr fidRect1 = ir::Symbol::make("FID_RECT_1");
  static inline ir::Expr fidCoord = ir::Symbol::make("FID_COORD");
  const long long allocSize;
  int posDim = -1;
};

}

#endif //TACO_MODE_FORMAT_RECT_COMPRESSED_H
