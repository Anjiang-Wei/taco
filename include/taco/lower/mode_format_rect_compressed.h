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

//  // TODO (rohany): Do I want this right now?
//  std::vector<AttrQuery>
//  attrQueries(std::vector<IndexVar> parentCoords, std::vector<IndexVar> childCoords) const override;

  // TODO (rohany): I definitely want these -- instead of the i, i+1, they will do i.lo and i.hi.
  ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const override;
  ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords, Mode mode) const override;


  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, int level) const override;

protected:
  ir::Expr getPosRegion(ModePack pack) const;
  ir::Expr getCoordRegion(ModePack pack) const;

  const long long allocSize;
};

}

#endif //TACO_MODE_FORMAT_RECT_COMPRESSED_H
