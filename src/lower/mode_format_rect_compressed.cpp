#include "taco/lower/mode_format_rect_compressed.h"

namespace taco {

RectCompressedModeFormat::RectCompressedModeFormat() :
  RectCompressedModeFormat(false, true, true, false) {}

RectCompressedModeFormat::RectCompressedModeFormat(bool isFull, bool isOrdered, bool isUnique, bool isZeroless,
                                                   long long allocSize) :
    ModeFormatImpl("lgRectCompressed", isFull, isOrdered, isUnique, false, true, isZeroless, false, true, false,
                   false, true, true, true, false), allocSize(allocSize) {}


ModeFormat RectCompressedModeFormat::copy(std::vector<ModeFormat::Property> properties) const {
  bool isFull = this->isFull;
  bool isOrdered = this->isOrdered;
  bool isUnique = this->isUnique;
  bool isZeroless = this->isZeroless;
  for (const auto property : properties) {
    switch (property) {
      case ModeFormat::FULL:
        isFull = true;
        break;
      case ModeFormat::NOT_FULL:
        isFull = false;
        break;
      case ModeFormat::ORDERED:
        isOrdered = true;
        break;
      case ModeFormat::NOT_ORDERED:
        isOrdered = false;
        break;
      case ModeFormat::UNIQUE:
        isUnique = true;
        break;
      case ModeFormat::NOT_UNIQUE:
        isUnique = false;
        break;
      case ModeFormat::ZEROLESS:
        isZeroless = true;
        break;
      case ModeFormat::NOT_ZEROLESS:
        isZeroless = false;
        break;
      default:
        break;
    }
  }
  const auto compressedVariant =
      std::make_shared<RectCompressedModeFormat>(isFull, isOrdered, isUnique, isZeroless);
  return ModeFormat(compressedVariant);
}

ModeFunction RectCompressedModeFormat::posIterBounds(ir::Expr parentPos, Mode mode) const {
  auto pack = mode.getModePack();
  ir::Expr pBegin = ir::FieldAccess::make(ir::Load::make(this->getPosRegion(pack), parentPos), "lo", false, Int64);
  ir::Expr pEnd = ir::Add::make(ir::FieldAccess::make(ir::Load::make(this->getPosRegion(pack), parentPos), "hi", false, Int64), 1);
  return ModeFunction(ir::Stmt(), {pBegin, pEnd});
}

ModeFunction RectCompressedModeFormat::posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords, Mode mode) const {
  // TODO (rohany): Not sure what this assert does for us.
  taco_iassert(mode.getPackLocation() == 0);

  ir::Expr idxArray = this->getCoordRegion(mode.getModePack());
  // TODO (rohany): I don't understand what's happening here. Ask someone
  //  about this before landing it.
  ir::Expr stride = (int)mode.getModePack().getNumModes();
  ir::Expr idx = ir::Load::make(idxArray, ir::Mul::make(pos, stride));
  return ModeFunction(ir::Stmt(), {idx, true});
}


std::vector<ir::Expr> RectCompressedModeFormat::getArrays(ir::Expr tensor, int mode, int level) const {
  std::string arraysName = util::toString(tensor) + std::to_string(level);
  // TODO (rohany): These get properties might have to change for Legion?
  return {ir::GetProperty::make(tensor, ir::TensorProperty::Indices,
                            level - 1, 0, arraysName + "_pos"),
          ir::GetProperty::make(tensor, ir::TensorProperty::Indices,
                            level - 1, 1, arraysName + "_crd")};
}

ir::Expr RectCompressedModeFormat::getPosRegion(ModePack pack) const {
  return pack.getArray(0);
}

ir::Expr RectCompressedModeFormat::getCoordRegion(ModePack pack) const {
  return pack.getArray(1);
}

}