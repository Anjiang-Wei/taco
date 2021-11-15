#include "taco/lower/mode_format_lg_singleton.h"

#include "taco/util/strings.h"

using namespace taco::ir;

namespace taco {

LgSingletonModeFormat::LgSingletonModeFormat() : LgSingletonModeFormat(false, true, true, false) {}


LgSingletonModeFormat::LgSingletonModeFormat(bool isFull, bool isOrdered,
                                         bool isUnique, bool isZeroless,
                                         long long allocSize) :
    ModeFormatImpl("lgsingleton", isFull, isOrdered, isUnique, true, true,
                   isZeroless, false, true, false, false, true, false, true,
                   true), allocSize(allocSize) {}

ModeFormat LgSingletonModeFormat::copy(
    std::vector<ModeFormat::Property> properties) const {
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
  const auto singletonVariant =
      std::make_shared<LgSingletonModeFormat>(isFull, isOrdered, isUnique,
                                            isZeroless);
  return ModeFormat(singletonVariant);
}

ModeFunction LgSingletonModeFormat::posIterBounds(Expr parentPos,
                                                Mode mode) const {
  return ModeFunction(Stmt(), {parentPos, ir::Add::make(parentPos, 1)});
}

ModeFunction LgSingletonModeFormat::posIterAccess(ir::Expr pos,
                                                std::vector<ir::Expr> coords,
                                                Mode mode) const {
  auto pack = mode.getModePack();
  Expr coordAcc = this->getAccessor(pack, CRD);
  Expr stride = (int)mode.getModePack().getNumModes();
  Expr offset = (int)mode.getPackLocation();
  Expr loc = ir::Add::make(ir::Mul::make(pos, stride), offset);
  Expr idx = Load::make(coordAcc, loc);
  return ModeFunction(Stmt(), {idx, true});
}

std::vector<ir::Expr> LgSingletonModeFormat::getArrays(ir::Expr tensor, int mode, int level) const {
  std::string arraysName = util::toString(tensor) + std::to_string(level);
  std::vector<ir::Expr> arrays(2);
  arrays[CRD] = ir::GetProperty::make(tensor, ir::TensorProperty::Indices, level - 1, 0, arraysName + "_crd");
  arrays[CRD_PARENT] = ir::GetProperty::make(tensor, ir::TensorProperty::IndicesParents,level - 1, 0, arraysName + "_crd_parent");
  return arrays;
}

std::vector<ModeRegion> LgSingletonModeFormat::getRegions(ir::Expr tensor, int level) const {
  // TODO (rohany): getArrays does not the mode argument, so we omit it here.
  auto arrays = this->getArrays(tensor, 0, level);
  auto crdReg = arrays[CRD].as<ir::GetProperty>();

  // TODO (rohany): Do the crd arrays need to be int64's?
  auto makeCrdAcc = [&](ir::RegionPrivilege priv) {
    // The CRD array will always have dimensionality = 1.
    return ir::GetProperty::makeIndicesAccessor(tensor, crdReg->name, crdReg->mode, crdReg->index, ir::GetProperty::AccessorArgs {
        .dim = 1,
        .elemType = Int32,
        .field = fidCoord,
        .regionAccessing = crdReg,
        .priv = priv,
    });
  };

  return {
      ModeRegion{.region = arrays[CRD], .regionParent = arrays[CRD_PARENT], .field = fidCoord, .accessorRO = makeCrdAcc(
          ir::RO), .accessorRW = makeCrdAcc(ir::RW)},
  };
}

ir::Expr LgSingletonModeFormat::getRegion(ModePack pack, RECT_COMPRESSED_REGIONS reg) const {
  return pack.getArray(int(reg));
}

ir::Expr LgSingletonModeFormat::getAccessor(ModePack pack, RECT_COMPRESSED_REGIONS reg, ir::RegionPrivilege priv) const {
  auto tensor = pack.getTensor();
  auto region = this->getRegions(tensor, pack.getLevel())[reg];
  switch (priv) {
    case ir::RO:
      return region.accessorRO;
    case ir::RW:
      return region.accessorRW;
    default:
      taco_iassert(false);
      return ir::Expr();
  }
}

}