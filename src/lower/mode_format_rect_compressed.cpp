#include "taco/lower/mode_format_rect_compressed.h"
#include "ir/ir_generators.h"
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
  // TODO (rohany): It seems like we might need to avoid just accessing the final element
  //  in the region if that isn't part of our partition.
  auto posRegAcc = this->getAccessor(pack, POS);
  ir::Expr pBegin = ir::FieldAccess::make(ir::Load::make(posRegAcc, parentPos), "lo", false, Int64);
  ir::Expr pEnd = ir::Add::make(ir::FieldAccess::make(ir::Load::make(posRegAcc, parentPos), "hi", false, Int64), 1);
  return ModeFunction(ir::Stmt(), {pBegin, pEnd});
}

ModeFunction RectCompressedModeFormat::posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords, Mode mode) const {
  taco_iassert(mode.getPackLocation() == 0);

  auto pack = mode.getModePack();
  auto idxArray = this->getAccessor(pack, CRD);
  taco_iassert(pack.getNumModes() == 1);
  ir::Expr stride = (int)pack.getNumModes();
  ir::Expr idx = ir::Load::make(idxArray, ir::Mul::make(pos, stride));
  return ModeFunction(ir::Stmt(), {idx, true});
}


std::vector<ir::Expr> RectCompressedModeFormat::getArrays(ir::Expr tensor, int mode, int level) const {
  std::string arraysName = util::toString(tensor) + std::to_string(level);
  std::vector<ir::Expr> arrays(4);
  arrays[POS] = ir::GetProperty::make(tensor, ir::TensorProperty::Indices, level - 1, 0, arraysName + "_pos");
  arrays[POS_PARENT] = ir::GetProperty::make(tensor, ir::TensorProperty::IndicesParents, level - 1, 0, arraysName + "_pos_parent");
  arrays[CRD] = ir::GetProperty::make(tensor, ir::TensorProperty::Indices,level - 1, 1, arraysName + "_crd");
  arrays[CRD_PARENT] = ir::GetProperty::make(tensor, ir::TensorProperty::IndicesParents,level - 1, 1, arraysName + "_crd_parent");
  return arrays;
}

std::vector<ModeRegion> RectCompressedModeFormat::getRegions(ir::Expr tensor, int level) const {
  // TODO (rohany): getArrays does not the mode argument, so we omit it here.
  auto arrays = this->getArrays(tensor, 0, level);
  auto posReg = arrays[POS].as<ir::GetProperty>();
  auto crdReg = arrays[CRD].as<ir::GetProperty>();

  // Set up our accessors too.
  auto makePosAcc = [&](ir::RegionPrivilege priv) {
    // TODO (rohany): Do the entries in the pos and crd arrays need to be int64's?
    // TODO (rohany): The dimensionality of the pos region will need to be changed here.
    return ir::GetProperty::makeIndicesAccessor(tensor, posReg->name, posReg->mode, posReg->index, ir::GetProperty::AccessorArgs {
        .dim = 1,
        .elemType = Rect(1),
        .field = fidRect1,
        .regionAccessing = posReg,
        .priv = priv,
    });
  };
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
      ModeRegion{.region = arrays[POS], .regionParent = arrays[POS_PARENT], .field = fidRect1, .accessorRO = makePosAcc(
          ir::RO), .accessorRW = makePosAcc(ir::RW)},
      ModeRegion{.region = arrays[CRD], .regionParent = arrays[CRD_PARENT], .field = fidCoord, .accessorRO = makeCrdAcc(
          ir::RO), .accessorRW = makeCrdAcc(ir::RW)},
  };
}

ir::Stmt RectCompressedModeFormat::getAppendCoord(ir::Expr pos, ir::Expr coord, Mode mode) const {
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
  auto maybeResizeIdx = lgDoubleSizeIfFull(idxArray, getCoordCapacity(mode), pos, idxArrayParent, idxArray, fidCoord, ir::Assign::make(crdAcc, newCrdAcc));
  return ir::Block::make({maybeResizeIdx, storeIdx});
}

ir::Stmt RectCompressedModeFormat::getAppendEdges(ir::Expr parentPos, ir::Expr posBegin, ir::Expr posEnd,
                                                  Mode mode) const {
  auto posArray = this->getAccessor(mode.getModePack(), POS, ir::RW);
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
  return ir::Load::make(this->getRegion(mode.getModePack(), POS), szPrev);
}

ir::Stmt RectCompressedModeFormat::getAppendInitEdges(ir::Expr pPrevBegin, ir::Expr pPrevEnd,
                                                      Mode mode) const {
  auto pack = mode.getModePack();
  if (isa<ir::Literal>(pPrevBegin)) {
    taco_iassert(to<ir::Literal>(pPrevBegin)->equalsScalar(0));
    return ir::Stmt();
  }

  auto posArray = this->getRegion(pack, POS);
  auto posArrayParent = this->getRegion(pack, POS_PARENT);
  ir::Expr posCapacity = this->getPosCapacity(mode);
  ModeFormat parentModeType = mode.getParentModeType();
  if (!parentModeType.defined() || parentModeType.hasAppend()) {
    // TODO (rohany): Don't make a symbol here.
    // TODO (rohany): Management around physical/logical regions.
    auto posAcc = this->getAccessor(pack, POS, ir::RW);
    auto newPosAcc = ir::makeCreateAccessor(posAcc, posArray, fidRect1);
    return lgDoubleSizeIfFull(posArray, posCapacity, pPrevEnd, posArrayParent, posArray, ir::Symbol::make("FID_RECT_1"), ir::Assign::make(posAcc, newPosAcc));
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
  auto posAcc = this->getAccessor(pack, POS, ir::RW);
  auto newPosAcc = ir::makeCreateAccessor(posAcc, posArray, fidRect1);
  ir::Stmt maybeResizePos = lgAtLeastDoubleSizeIfFull(posArray, posCapacity, pPrevEnd, posArrayParent, posArray, ir::Symbol::make("FID_RECT_1"), ir::Assign::make(posAcc, newPosAcc));
  return ir::Block::make({maybeResizePos, initPos});
}

ir::Stmt RectCompressedModeFormat::getAppendInitLevel(ir::Expr szPrev, ir::Expr size, Mode mode) const {
  const bool szPrevIsZero = isa<ir::Literal>(szPrev) &&
                            to<ir::Literal>(szPrev)->equalsScalar(0);

  auto pack = mode.getModePack();
  ir::Expr defaultCapacity = ir::Literal::make(allocSize, Datatype::Int32);
  auto posArray = this->getRegion(pack, POS);
  auto posParent = this->getRegion(pack, POS_PARENT);
  ir::Expr initCapacity = szPrevIsZero ? defaultCapacity : szPrev;
  ir::Expr posCapacity = initCapacity;

  std::vector<ir::Stmt> initStmts;
  if (szPrevIsZero) {
    posCapacity = getPosCapacity(mode);
    initStmts.push_back(ir::VarDecl::make(posCapacity, initCapacity));
  }

  // TODO (rohany): We don't need to malloc a particular size here, as we know this region
  //  is already built to be the correct size. However, having this be more general will
  //  help us out a bit in the longer run where this might not be true.
  initStmts.push_back(ir::makeLegionMalloc(posArray, posCapacity, posParent, fidRect1));
  // Reinitialize the accessor for the new physical region. We need a RW accessor here.
  auto posAcc = this->getAccessor(pack, POS, ir::RW);
  auto newPosAcc = ir::makeCreateAccessor(posAcc, posArray, fidRect1);
  initStmts.push_back(ir::Assign::make(posAcc, newPosAcc));
  // Start off each component in the position array as <0, 0>.
  initStmts.push_back(ir::Store::make(posAcc, 0, ir::makeConstructor(Rect(1), {0, 0})));

  if (mode.getParentModeType().defined() &&
      !mode.getParentModeType().hasAppend() && !szPrevIsZero) {
    ir::Expr pVar = ir::Var::make("p" + mode.getName(), Int());
    // Start off each component in the position array as <0, 0>.
    ir::Stmt storePos = ir::Store::make(posAcc, pVar, ir::makeConstructor(Rect(1), {0, 0}));
    initStmts.push_back(ir::For::make(pVar, 1, initCapacity, 1, storePos));
  }

  if (mode.getPackLocation() == (mode.getModePack().getNumModes() - 1)) {
    ir::Expr crdCapacity = this->getCoordCapacity(mode);
    auto crdArray = this->getRegion(pack, CRD);
    auto crdParent = this->getRegion(pack, CRD_PARENT);
    initStmts.push_back(ir::VarDecl::make(crdCapacity, defaultCapacity));
    // TODO (rohany): I need to have separate management of the physical and logical regions.
    initStmts.push_back(ir::makeLegionMalloc(crdArray, crdCapacity, crdParent, fidCoord));
    // Reinitialize the CRD RW accessor.
    auto crdAcc = this->getAccessor(pack, CRD, ir::RW);
    auto newCrdAcc = ir::makeCreateAccessor(crdAcc, crdArray, fidCoord);
    initStmts.push_back(ir::Assign::make(crdAcc, newCrdAcc));
  }

  return ir::Block::make(initStmts);
}

ir::Stmt RectCompressedModeFormat::getAppendFinalizeLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const {
  auto pack = mode.getModePack();
  ModeFormat parentModeType = mode.getParentModeType();
  if ((isa<ir::Literal>(szPrev) && to<ir::Literal>(szPrev)->equalsScalar(1)) ||
      !parentModeType.defined() || parentModeType.hasAppend()) {
    // TODO (rohany): I think that we also need to do subregion casts here?
    return ir::Stmt();
  }

  ir::Expr csVar = ir::Var::make("cs" + mode.getName(), Int64);
  ir::Stmt initCs = ir::VarDecl::make(csVar, 0);

  auto pos = this->getAccessor(pack, POS, ir::RW);
  ir::Expr pVar = ir::Var::make("p" + mode.getName(), Int64);

  auto lo = ir::FieldAccess::make(ir::Load::make(pos, pVar), "lo", false /* deref */, Int64);
  auto hi = ir::FieldAccess::make(ir::Load::make(pos, pVar), "hi", false /* deref */, Int64);

  ir::Expr numElems = ir::Var::make("numElems" + mode.getName(), Int64);
  ir::Stmt getNumElems = ir::VarDecl::make(numElems, hi);
  // Increment lo and hi by the accumulator. Note that we have to set this order
  // appropriately so that the lowering infrastructure doesn't turn this into
  // a compound add as the Realm type system doesn't allow Point<N> += operations.
  auto setLo = ir::Assign::make(lo, ir::Add::make(csVar, lo));
  auto setHi = ir::Assign::make(hi, ir::Add::make(csVar, hi));
  auto incCs = ir::Assign::make(csVar, ir::Add::make(csVar, ir::Add::make(numElems, 1)));
  auto body = ir::Block::make({getNumElems, setLo, setHi, incCs});
  auto finalize = ir::For::make(pVar, 0, szPrev, 1, body);

  // Now, we need to cast the regions back to subregions of the tightest possible size.
  // TODO (rohany): We don't need to do this for pos right now, as pos regions should all
  //  have the right size.
  // TODO (rohany): It's definitely weird here to have to directly access the LegionTensor.

  auto crdReg = this->getRegion(pack, CRD).as<ir::GetProperty>();
  auto field = ir::FieldAccess::make(mode.getTensorExpr(), "indices", true /* isDeref*/, Auto);
  auto levelLoad = ir::Load::make(field, crdReg->mode);
  auto idxLoad = ir::Load::make(levelLoad, crdReg->index);
  auto subreg = ir::Call::make("getSubRegion", {ir::ctx, ir::runtime, this->getRegion(pack, CRD_PARENT),
                                                ir::makeConstructor(Rect(1), {0, ir::Sub::make(sz, 1)})}, Auto);
  auto setSubReg = ir::Assign::make(idxLoad, subreg);
  return ir::Block::make({initCs, finalize, setSubReg});
}

ModeFunction RectCompressedModeFormat::getPartitionFromParent(ir::Expr parentPartition, Mode mode, ir::Expr partitionColor) const {
  auto maybeAddColor = [&](std::vector<ir::Expr> args) {
    if (!partitionColor.defined()) {
      return args;
    }
    args.push_back(partitionColor);
    return args;
  };
  auto pack = mode.getModePack();
  // Partition the pos region in the same way that the parent is partitioned,
  // as there is a pos entry for each of the entries in the parent.
  // TODO (rohany): Add these variables to the mode.
  auto posPart = ir::Var::make("posPart" + mode.getName(), LogicalPartition);
  auto createPosPart = ir::VarDecl::make(posPart, ir::Call::make("copyPartition", maybeAddColor(
      {ir::ctx, ir::runtime, parentPartition, this->getRegion(pack, POS)}), Auto));
  // Then, using the partition of pos, create a dependent partition of the crd array.
  auto crdPart = ir::Var::make("crdPart" + mode.getName(), LogicalPartition);
  auto posIndexPart = ir::MethodCall::make(posPart, "get_index_partition", {}, false /* deref */, Auto);
  auto createCrdPartCall = ir::Call::make(
    "runtime->create_partition_by_image_range",
    maybeAddColor({
      ir::ctx,
      ir::MethodCall::make(this->getRegion(pack, CRD), "get_index_space", {}, false /* deref */, Auto),
      posPart,
      this->getRegion(pack, POS_PARENT),
      fidRect1,
      ir::Call::make("runtime->get_index_partition_color_space_name", {ir::ctx, posIndexPart}, Auto),
    }),
    Auto
  );
  auto getCrdLogicalPartCall = ir::Call::make(
    "runtime->get_logical_partition",
    {
      ir::ctx,
      this->getRegion(pack, CRD_PARENT),
      createCrdPartCall,
    },
    Auto
  );
  auto createCrdPart = ir::VarDecl::make(crdPart, getCrdLogicalPartCall);
  // The partition to pass down to children is the coloring of the crd array.
  return ModeFunction(ir::Block::make({createPosPart, createCrdPart}), {posPart, crdPart, crdPart});
}

ModeFunction RectCompressedModeFormat::getPartitionFromChild(ir::Expr childPartition, Mode mode, ir::Expr) const {
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

ir::Expr RectCompressedModeFormat::getRegion(ModePack pack, RECT_COMPRESSED_REGIONS reg) const {
  return pack.getArray(int(reg));
}

ir::Expr RectCompressedModeFormat::getAccessor(ModePack pack, RECT_COMPRESSED_REGIONS reg, ir::RegionPrivilege priv) const {
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
//  Maybe not, since the mode makes the variable once, it probably assigns to it
//  and tracks it.
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