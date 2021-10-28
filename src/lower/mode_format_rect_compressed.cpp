#include "taco/lower/mode_format_rect_compressed.h"
#include "ir/ir_generators.h"

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
  taco_iassert(mode.getPackLocation() == 0);

  ir::Expr idxArray = this->getCoordRegion(mode.getModePack());
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

ir::Stmt RectCompressedModeFormat::getAppendCoord(ir::Expr pos, ir::Expr coord, Mode mode) const {
  taco_iassert(mode.getPackLocation() == 0);

  ir::Expr idxArray = this->getCoordRegion(mode.getModePack());
  ir::Expr stride = (int)mode.getModePack().getNumModes();
  ir::Stmt storeIdx = ir::Store::make(idxArray, ir::Mul::make(pos, stride), coord);

  if (mode.getModePack().getNumModes() > 1) {
    return storeIdx;
  }

  // TODO (rohany): Remove this hard coded field.
  // TODO (rohany): Add management around physical regions.
  auto maybeResizeIdx = lgDoubleSizeIfFull(idxArray, getCoordCapacity(mode), pos, idxArray, ir::Symbol::make("FID_COORD"));
  return ir::Block::make({maybeResizeIdx, storeIdx});
}

ir::Stmt RectCompressedModeFormat::getAppendEdges(ir::Expr parentPos, ir::Expr posBegin, ir::Expr posEnd,
                                                  Mode mode) const {
  ir::Expr posArray = this->getPosRegion(mode.getModePack());
  ModeFormat parentModeType = mode.getParentModeType();
  auto lo = ir::FieldAccess::make(ir::Load::make(posArray, parentPos), "lo", false /* deref */, Int64);
  auto hi = ir::FieldAccess::make(ir::Load::make(posArray, parentPos), "hi", false /* deref */, Int64);
  ir::Stmt setLo, setHi;

  // If we can append to our parent mode, then we don't need to set things up to
  // do a scan to finalize the level. So, we can just set the lo and hi values directly.
  if (!parentModeType.defined() || parentModeType.hasAppend()) {
    setLo = ir::Assign::make(lo, posBegin);
    // End bounds are inclusive.
    setHi = ir::Assign::make(hi, ir::Sub::make(posEnd, 1));
  } else {
    // If we are instead inserting into the mode, set things up to finalize
    // the level with a scan. In this case, we don't set a value for lo.
    setLo = ir::Stmt();
    setHi = ir::Assign::make(hi, ir::Sub::make(ir::Sub::make(posEnd, posBegin), 1));
  }
  return ir::Block::make({setLo, setHi});
}

ir::Expr RectCompressedModeFormat::getSize(ir::Expr szPrev, Mode mode) const {
  // TODO (rohany): I'm not sure that this is correct. It seems like it should
  //  either get the current size from the region itself or look at high.
  return ir::Load::make(this->getPosRegion(mode.getModePack()), szPrev);
}

ir::Stmt RectCompressedModeFormat::getAppendInitEdges(ir::Expr pPrevBegin, ir::Expr pPrevEnd,
                                                      Mode mode) const {
  if (isa<ir::Literal>(pPrevBegin)) {
    taco_iassert(to<ir::Literal>(pPrevBegin)->equalsScalar(0));
    return ir::Stmt();
  }

  ir::Expr posArray = this->getPosRegion(mode.getModePack());
  ir::Expr posCapacity = this->getPosCapacity(mode);
  ModeFormat parentModeType = mode.getParentModeType();
  if (!parentModeType.defined() || parentModeType.hasAppend()) {
    // TODO (rohany): Don't make a symbol here.
    // TODO (rohany): Management around physical/logical regions.
    return lgDoubleSizeIfFull(posArray, posCapacity, pPrevEnd, posArray, ir::Symbol::make("FID_RECT_1"));
  }

  // Initialize all of the spots in the pos array.
  ir::Expr pVar = ir::Var::make("p" + mode.getName(), Int64);
  ir::Expr lb = pPrevBegin;
  ir::Expr ub = ir::Add::make(pPrevEnd, 1);
  // Start off each component in the position array as <0, 0>.
  auto store = ir::Store::make(posArray, pVar, ir::makeConstructor(Rect(1), {0, 0}));
  auto initPos = ir::For::make(pVar, lb, ub, 1, store);
  // TODO (rohany): Don't make a symbol here.
  // TODO (rohany): Management around physical/logical regions.
  ir::Stmt maybeResizePos = lgAtLeastDoubleSizeIfFull(posArray, posCapacity, pPrevEnd, posArray, ir::Symbol::make("FID_RECT_1"));
  return ir::Block::make({maybeResizePos, initPos});
}

ir::Stmt RectCompressedModeFormat::getAppendInitLevel(ir::Expr szPrev, ir::Expr size, Mode mode) const {
  const bool szPrevIsZero = isa<ir::Literal>(szPrev) &&
                            to<ir::Literal>(szPrev)->equalsScalar(0);

  ir::Expr defaultCapacity = ir::Literal::make(allocSize, Datatype::Int32);
  ir::Expr posArray = this->getPosRegion(mode.getModePack());
  ir::Expr initCapacity = szPrevIsZero ? defaultCapacity : szPrev;
  ir::Expr posCapacity = initCapacity;

  std::vector<ir::Stmt> initStmts;
  if (szPrevIsZero) {
    posCapacity = getPosCapacity(mode);
    initStmts.push_back(ir::VarDecl::make(posCapacity, initCapacity));
  }
  // TODO (rohany): I need to have separate management of the physical and logical regions.
  // TODO (rohany): Make it part of the mode constructor to choose what the field ID is -- FID_VAL isn't right here.
  initStmts.push_back(ir::makeLegionMalloc(posArray, posCapacity, posArray, ir::Symbol::make("FID_VAL")));
  // Start off each component in the position array as <0, 0>.
  initStmts.push_back(ir::Store::make(posArray, 0, ir::makeConstructor(Rect(1), {0, 0})));

  if (mode.getParentModeType().defined() &&
      !mode.getParentModeType().hasAppend() && !szPrevIsZero) {
    ir::Expr pVar = ir::Var::make("p" + mode.getName(), Int());
    // Start off each component in the position array as <0, 0>.
    ir::Stmt storePos = ir::Store::make(posArray, pVar, ir::makeConstructor(Rect(1), {0, 0}));
    initStmts.push_back(ir::For::make(pVar, 1, initCapacity, 1, storePos));
  }

  if (mode.getPackLocation() == (mode.getModePack().getNumModes() - 1)) {
    ir::Expr crdCapacity = this->getCoordCapacity(mode);
    ir::Expr crdArray = this->getCoordRegion(mode.getModePack());
    initStmts.push_back(ir::VarDecl::make(crdCapacity, defaultCapacity));
    // TODO (rohany): I need to have separate management of the physical and logical regions.
    // TODO (rohany): Make it part of the mode constructor to choose what the field ID is -- FID_VAL isn't right here.
    initStmts.push_back(ir::makeLegionMalloc(crdArray, crdCapacity, crdArray, ir::Symbol::make("FID_VAL")));
  }

  return ir::Block::make(initStmts);
}

ir::Stmt RectCompressedModeFormat::getAppendFinalizeLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const {
  ModeFormat parentModeType = mode.getParentModeType();
  if ((isa<ir::Literal>(szPrev) && to<ir::Literal>(szPrev)->equalsScalar(1)) ||
      !parentModeType.defined() || parentModeType.hasAppend()) {
    return ir::Stmt();
  }

  ir::Expr csVar = ir::Var::make("cs" + mode.getName(), Int64);
  ir::Stmt initCs = ir::VarDecl::make(csVar, 0);

  auto pos = this->getPosRegion(mode.getModePack());
  ir::Expr pVar = ir::Var::make("p" + mode.getName(), Int64);

  auto lo = ir::FieldAccess::make(ir::Load::make(pos, pVar), "lo", false /* deref */, Int64);
  auto hi = ir::FieldAccess::make(ir::Load::make(pos, pVar), "hi", false /* deref */, Int64);

  ir::Expr numElems = ir::Var::make("numElems" + mode.getName(), Int64);
  ir::Stmt getNumElems = ir::VarDecl::make(numElems, hi);
  // Increment lo and hi by the accumulator.
  auto setLo = ir::Assign::make(lo, ir::Add::make(lo, csVar));
  auto setHi = ir::Assign::make(hi, ir::Add::make(hi, csVar));
  auto incCs = ir::Assign::make(csVar, ir::Add::make(numElems, 1));
  auto body = ir::Block::make({getNumElems, setLo, setHi, incCs});
  auto finalize = ir::For::make(pVar, 0, szPrev, 1, body);
  return ir::Block::make({initCs, finalize});
}

ModeFunction RectCompressedModeFormat::getPartitionFromParent(ir::Expr parentPartition, Mode mode) const {
  // Partition the pos region in the same way that the parent is partitioned,
  // as there is a pos entry for each of the entries in the parent.
  auto posPart = ir::Var::make("posPart" + mode.getName(), LogicalPartition);
  auto createPosPart = ir::VarDecl::make(posPart, ir::Call::make("copyParentPartition", {}, Auto));
  // Then, using the partition of pos, create a dependent partition of the crd array.
  auto crdPart = ir::Var::make("crdPart" + mode.getName(), LogicalPartition);
  auto createCrdPart = ir::VarDecl::make(crdPart, ir::Call::make("create_partition_by_image_range", {posPart}, Auto));
  // The partition to pass down to children is the coloring of the crd array.
  return ModeFunction(ir::Block::make({createPosPart, createCrdPart}), {posPart, crdPart, crdPart});
}

ModeFunction RectCompressedModeFormat::getPartitionFromChild(ir::Expr childPartition, Mode mode) const {
  // Here, we have a partition of the level below us. To go back up
  // from a child partition, we first partition the crd array, and then
  // use that to create a dependent partition of the pos array.
  auto crdPart = ir::Var::make("crdPart" + mode.getName(), LogicalPartition);
  auto createCrdPart = ir::VarDecl::make(crdPart, ir::Call::make("copyChildPartition", {childPartition}, Auto));
  auto posPart = ir::Var::make("posPart" + mode.getName(), LogicalPartition);
  auto createPosPart = ir::VarDecl::make(posPart, ir::Call::make("create_partition_by_preimage_range", {crdPart}, Auto));
  // The resulting partition is a partition of the pos array.
  return ModeFunction(ir::Block::make({createCrdPart, createPosPart}), {posPart, crdPart, posPart});
}

ir::Expr RectCompressedModeFormat::getPosRegion(ModePack pack) const {
  return pack.getArray(0);
}

ir::Expr RectCompressedModeFormat::getCoordRegion(ModePack pack) const {
  return pack.getArray(1);
}

// TODO (rohany): This probably has to be changed for the same reasons as below.
ir::Expr RectCompressedModeFormat::getPosCapacity(Mode mode) const {
  const std::string varName = mode.getName() + "_pos_size";

  if (!mode.hasVar(varName)) {
    ir::Expr posCapacity = ir::Var::make(varName, Int());
    mode.addVar(varName, posCapacity);
    return posCapacity;
  }

  return mode.getVar(varName);
}

// TODO (rohany): This probably needs to be changed (at least how we
//  access the capacity needs to be cached rather than making runtime calls).
ir::Expr RectCompressedModeFormat::getCoordCapacity(Mode mode) const {
  const std::string varName = mode.getName() + "_crd_size";

  if (!mode.hasVar(varName)) {
    ir::Expr idxCapacity = ir::Var::make(varName, Int());
    mode.addVar(varName, idxCapacity);
    return idxCapacity;
  }

  return mode.getVar(varName);
}

}