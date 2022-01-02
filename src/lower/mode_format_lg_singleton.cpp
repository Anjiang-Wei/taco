#include "taco/lower/mode_format_lg_singleton.h"
#include "taco/util/strings.h"
#include "ir/ir_generators.h"

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

ModeFunction LgSingletonModeFormat::posIterBounds(std::vector<Expr> parentPositions,
                                                Mode mode) const {
  taco_iassert(parentPositions.size() == 1);
  auto parentPos = parentPositions[0];
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

Stmt LgSingletonModeFormat::getAppendCoord(ir::Expr pos, ir::Expr coord, Mode mode) const {
  taco_iassert(mode.getPackLocation() == 0);
  auto pack = mode.getModePack();

  auto idxArray = this->getRegion(pack, CRD);
  auto idxArrayParent = this->getRegion(pack, CRD_PARENT);
  auto idxArrayAcc = this->getAccessor(pack, CRD, ir::RW);
  ir::Expr stride = (int)mode.getModePack().getNumModes();
  ir::Stmt storeIdx = ir::Store::make(idxArrayAcc, ir::Mul::make(pos, stride), coord);

  if (pack.getNumModes() > 1) {
    return storeIdx;
  }

  auto crdAcc = this->getAccessor(pack, CRD, ir::RW);
  auto newCrdAcc = ir::makeCreateAccessor(crdAcc, idxArray, fidCoord);
  auto maybeResizeIdx = lgDoubleSizeIfFull(idxArray, getCoordCapacity(mode), pos, idxArrayParent, idxArray, fidCoord, ir::Assign::make(crdAcc, newCrdAcc), ir::readWrite);
  return ir::Block::make({maybeResizeIdx, storeIdx});
}

Stmt LgSingletonModeFormat::getAppendInitLevel(ir::Expr parentSize, ir::Expr size, Mode mode) const {
  auto pack = mode.getModePack();
  if (mode.getPackLocation() != (pack.getNumModes() - 1)) {
    return Stmt();
  }

  auto defaultCapacity = ir::Literal::make(allocSize, Datatype::Int32);
  auto crdCapacity = getCoordCapacity(mode);
  auto crdArray = this->getRegion(pack, CRD);
  auto crdParent = this->getRegion(pack, CRD_PARENT);
  auto crdAcc = this->getAccessor(pack, CRD, RW);
  auto initCrdCapacity = VarDecl::make(crdCapacity, defaultCapacity);
  auto allocCrd = ir::makeLegionMalloc(crdArray, crdCapacity, crdParent, fidCoord, ir::readWrite);
  auto newCrdAcc = ir::makeCreateAccessor(crdAcc, crdArray, fidCoord);
  auto updateAcc = ir::Assign::make(crdAcc, newCrdAcc);
  return Block::make(initCrdCapacity, allocCrd, updateAcc);
}

Stmt LgSingletonModeFormat::getAppendFinalizeLevel(ir::Expr, ir::Expr, ir::Expr size, Mode mode) const {
  // We just need to emit a block that casts the region down to the final size.
  auto pack = mode.getModePack();
  auto crdReg = this->getRegion(pack, CRD).as<ir::GetProperty>();
  auto field = ir::FieldAccess::make(mode.getTensorExpr(), "indices", true /* isDeref*/, Auto);
  auto levelLoad = ir::Load::make(field, crdReg->mode);
  auto idxLoad = ir::Load::make(levelLoad, crdReg->index);
  auto subreg = ir::Call::make("getSubRegion", {ir::ctx, ir::runtime, this->getRegion(pack, CRD_PARENT),
                                                ir::makeConstructor(Rect(1), {0, ir::Sub::make(size, 1)})}, Auto);
  return ir::Assign::make(idxLoad, subreg);
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

ir::Expr LgSingletonModeFormat::getCoordCapacity(Mode mode) const {
  const std::string varName = mode.getName() + "_crd_size";
  return this->getModeVar(mode, varName, Int());
}

}