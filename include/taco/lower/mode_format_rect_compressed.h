#ifndef TACO_MODE_FORMAT_RECT_COMPRESSED_H
#define TACO_MODE_FORMAT_RECT_COMPRESSED_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

// TODO (rohany): See what a decent default is here.
static const int LG_DEFAULT_ALLOC_SIZE = 1 << 20;

// TODO (rohany): I want a Legion ModePack that I can sub in here, or adjust
//  the ModePack default structure to use the LegionTensor when possible.

class RectCompressedModeFormat : public ModeFormatImpl {
public:
  using ModeFormatImpl::getInsertCoord;

  RectCompressedModeFormat();
  RectCompressedModeFormat(bool isFull, bool isOrdered,
                           bool isUnique, bool isZeroless, long long allocSize = LG_DEFAULT_ALLOC_SIZE);

  ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;

  // Similarly to the compressed level, this format supports position iterate.
  ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const override;
  ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords, Mode mode) const override;

  // Definitions for insertion into a tensor level.
  ir::Stmt getAppendCoord(ir::Expr pos, ir::Expr coord, Mode mode) const override;
  ir::Stmt getAppendEdges(ir::Expr parentPos, ir::Expr posBegin, ir::Expr posEnd, Mode mode) const override;
  ir::Expr getSize(ir::Expr szPrev, Mode mode) const override;
  ir::Stmt getAppendInitEdges(ir::Expr parentPosBegin, ir::Expr parentPosEnd, Mode mode) const override;
  ir::Stmt getAppendInitLevel(ir::Expr parentSize, ir::Expr size, Mode mode) const override;
  ir::Stmt getAppendFinalizeLevel(ir::Expr parentSize, ir::Expr size, Mode mode) const override;

  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, int level) const override;
protected:
  ir::Expr getPosRegion(ModePack pack) const;
  ir::Expr getCoordRegion(ModePack pack) const;

  ir::Expr getPosCapacity(Mode mode) const;
  ir::Expr getCoordCapacity(Mode mode) const;

  const long long allocSize;
};

}

#endif //TACO_MODE_FORMAT_RECT_COMPRESSED_H
