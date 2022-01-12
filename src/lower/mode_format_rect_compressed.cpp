#include "taco/lower/mode_format_rect_compressed.h"
#include "taco/lower/mode_format_dense.h"
#include "ir/ir_generators.h"

namespace taco {

RectCompressedModeFormat::RectCompressedModeFormat() :
  RectCompressedModeFormat(false, true, true, false, -1 /* posDim */) {}

RectCompressedModeFormat::RectCompressedModeFormat(int posDim) :
  RectCompressedModeFormat(false, true, true, false, posDim) {}

RectCompressedModeFormat::RectCompressedModeFormat(bool isFull, bool isOrdered, bool isUnique, bool isZeroless, int posDim, long long allocSize) :
  ModeFormatImpl("lgRectCompressed", isFull, isOrdered, isUnique, false, true, isZeroless, false, true, false,
                 false, true, true, true, false), allocSize(allocSize), posDim(posDim) {}


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
      std::make_shared<RectCompressedModeFormat>(isFull, isOrdered, isUnique, isZeroless, posDim, allocSize);
  return ModeFormat(compressedVariant);
}

ModeFunction RectCompressedModeFormat::posIterBounds(std::vector<ir::Expr> parentPositions, Mode mode) const {
  taco_iassert(parentPositions.size() == size_t(this->posDim));
  auto pack = mode.getModePack();
  auto accessPoint = this->packToPoint(parentPositions);
  // TODO (rohany): It seems like we might need to avoid just accessing the final element
  //  in the region if that isn't part of our partition.
  auto posRegAcc = this->getAccessor(pack, POS);
  ir::Expr pBegin = ir::FieldAccess::make(ir::Load::make(posRegAcc, accessPoint), "lo", false, Int64);
  ir::Expr pEnd = ir::Add::make(ir::FieldAccess::make(ir::Load::make(posRegAcc, accessPoint), "hi", false, Int64), 1);
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

std::vector<AttrQuery> RectCompressedModeFormat::attrQueries(std::vector<IndexVar> parentCoords,
                                                             std::vector<IndexVar> childCoords) const {
  std::vector<IndexVar> groupBy(parentCoords.begin(), parentCoords.end() - 1);
  std::vector<IndexVar> aggregatedCoords = {parentCoords.back()};
  if (!isUnique) {
    aggregatedCoords.insert(aggregatedCoords.end(), childCoords.begin(),
                            childCoords.end());
  }
  return {AttrQuery(groupBy, {AttrQuery::Attr(std::make_tuple("nnz", AttrQuery::COUNT, aggregatedCoords))})};
}

ir::Expr RectCompressedModeFormat::getAssembledSize(ir::Expr prevSize, Mode mode) const {
  // The result of the scan over the pos array is the total number
  // of nonzeros in the level, so that is the resulting size of the level.
  return ir::FieldAccess::make(this->getSeqInsertEdgesResultVar(mode), "scanResult", false /* deref */, Int());
}

ir::Stmt RectCompressedModeFormat::getInitYieldPos(ir::Expr prevSize, Mode mode) const {
  return ir::Stmt();
}

ModeFunction RectCompressedModeFormat::getYieldPos(ir::Expr parentPos, std::vector<ir::Expr> coords, Mode mode) const {
  taco_iassert(!this->hasSparseAncestor(mode));
  // This way of constructing the access point doesn't work if we have
  // a sparse ancestor. This can be fixed by extending the interface
  // of getYieldPos to take in all parent positions.
  std::vector<ir::Expr> pointArgs;
  taco_iassert(int(coords.size()) >= (this->posDim + 1));
  // The final coordinate in coords is the coordinate of _this_ level, so
  // we don't want to include it. We start at coords.size() - (this->posDim + 1)
  // to start at the coordinates that make up this multi-dimensional pos array,
  // while excluding the final coordinate.
  for (size_t i = coords.size() - (this->posDim + 1); i < (coords.size() - 1); i++) {
    pointArgs.push_back(coords[i]);
  }
  auto posAcc = this->getAccessor(mode.getModePack(), POS, ir::RW);
  auto point = this->packToPoint(pointArgs);
  auto lo = ir::FieldAccess::make(ir::Load::make(posAcc, point), "lo", false /* deref */, Int64);
  auto pVar = ir::Var::make("p" + mode.getName(), Int());
  auto getPtr = ir::VarDecl::make(pVar, lo);
  auto incPtr = ir::Assign::make(lo, ir::Add::make(lo, 1));
  return ModeFunction(ir::Block::make(getPtr, incPtr), {pVar});
}

ir::Stmt RectCompressedModeFormat::getFinalizeYieldPos(ir::Expr, Mode mode) const {
  // We the RectCompressedFinalizeYieldPositions task to perform this
  // computation, we all we do is launch it using the partition used to
  // perform the scan.
  std::vector<ir::Stmt> results;
  auto makeLauncher = ir::makeConstructor(
    RectCompressedFinalizeYieldPositions,
    {
      ir::ctx,
      ir::runtime,
      this->getRegion(mode.getModePack(), POS),
      ir::FieldAccess::make(this->getSeqInsertEdgesResultVar(mode), "partition", false /* deref */, Auto),
      this->fidRect1
    }
  );
  auto launcher = ir::Var::make(mode.getName() + "_finalize_yield_pos_launcher", RectCompressedFinalizeYieldPositions);
  results.push_back(ir::VarDecl::make(launcher, makeLauncher));
  results.push_back(ir::SideEffect::make(ir::Call::make("runtime->execute_index_space", {ir::ctx, launcher}, Auto)));
  return ir::Block::make(results);
}

ir::Stmt RectCompressedModeFormat::getSeqInitEdges(ir::Expr prevSize, std::vector<ir::Expr> parentDims,
                                                   std::vector<AttrQueryResult> queries, Mode mode) const {
  // The seqInsertEdges call does all of the setup that we need, so there is nothing
  // more to be done here. This will have to do more work once we support compressed
  // levels with sparse ancestors.
  taco_iassert(!this->hasSparseAncestor(mode));
  return {};
}

ir::Stmt RectCompressedModeFormat::getSeqInsertEdges(ir::Expr parentPos, std::vector<ir::Expr> parentDims, ir::Expr colorSpace,
                                                     std::vector<ir::Expr> coords, std::vector<AttrQueryResult> queries, Mode mode) const {

  // As with getFinalizeYieldPositions, we have a task to perform this computation in
  // a distributed manner, so we just launch it here.
  taco_iassert(coords.size() == parentDims.size());
  // If the input color space is undefined, then create a dummy size 1 color space.
  if (!colorSpace.defined()) {
    colorSpace = ir::Call::make("runtime->create_index_space", {ir::ctx, ir::makeConstructor(Rect(1), {0, 0})}, Auto);
  }
  auto call = ir::Call::make(
    "RectCompressedGetSeqInsertEdges::compute",
    {
      ir::ctx,
      ir::runtime,
      colorSpace,
      this->getRegion(mode.getModePack(), POS),
      this->fidRect1,
      queries[0].getResult({}, "nnz"),
      ir::fidVal
    },
    Auto
  );
  auto resultVar = this->getSeqInsertEdgesResultVar(mode);
  return ir::VarDecl::make(resultVar, call);
}

ir::Stmt RectCompressedModeFormat::getInitCoords(ir::Expr prevSize, std::vector<AttrQueryResult> queries,
                                                 Mode mode) const {
  // The assemble infrastructure will manage actually physically mapping these
  // regions if necessary, we all we need to do is cast the crd region to
  // its correct size.
  auto pack = mode.getModePack();
  auto size = this->getAssembledSize(prevSize, mode);
  auto crdArray = this->getRegion(pack, CRD);
  auto crdArrayGP = crdArray.as<ir::GetProperty>();
  auto field = ir::FieldAccess::make(mode.getTensorExpr(), "indices", true /* isDeref*/, Auto);
  auto levelLoad = ir::Load::make(field, crdArrayGP->mode);
  auto idxLoad = ir::Load::make(levelLoad, crdArrayGP->index);
  auto subreg = ir::Call::make("getSubRegion", {ir::ctx, ir::runtime, this->getRegion(pack, CRD_PARENT),
                                                ir::makeConstructor(Rect(1), {0, ir::Sub::make(size, 1)})}, Auto);
  // Note that we need to set the field in the LegionTensor as well
  // as the variable for the crd region.
  auto setSubReg = ir::Assign::make(crdArray, subreg);
  auto setLegionTensor = ir::Assign::make(idxLoad, crdArray);
  return ir::Block::make(setSubReg, setLegionTensor);
}

ir::Stmt RectCompressedModeFormat::getInsertCoord(ir::Expr parentPos, ir::Expr pos, std::vector<ir::Expr> coords,
                                                  Mode mode) const {
  taco_iassert(mode.getPackLocation() == 0);
  auto pack = mode.getModePack();
  auto crdAcc = this->getAccessor(pack, CRD, ir::RW);
  auto stride = (int)pack.getNumModes();
  return ir::Store::make(crdAcc, ir::Mul::make(pos, stride), coords.back());
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
  taco_iassert(this->posDim != -1);
  // TODO (rohany): getArrays does not the mode argument, so we omit it here.
  auto arrays = this->getArrays(tensor, 0, level);
  auto posReg = arrays[POS].as<ir::GetProperty>();
  auto crdReg = arrays[CRD].as<ir::GetProperty>();

  // Set up our accessors too.
  auto makePosAcc = [&](ir::RegionPrivilege priv) {
    // TODO (rohany): Do the entries in the pos and crd arrays need to be int64's?
    return ir::GetProperty::makeIndicesAccessor(tensor, posReg->name, posReg->mode, posReg->index, ir::GetProperty::AccessorArgs(
        posDim,
        Rect(1),
        fidRect1,
        posReg,
        arrays[POS_PARENT],
        priv
    ));
  };
  auto makeCrdAcc = [&](ir::RegionPrivilege priv) {
    // The CRD array will always have dimensionality = 1.
    return ir::GetProperty::makeIndicesAccessor(tensor, crdReg->name, crdReg->mode, crdReg->index, ir::GetProperty::AccessorArgs(
        1,
        Int32,
        fidCoord,
        crdReg,
        arrays[CRD_PARENT],
        priv
    ));
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
  auto maybeResizeIdx = lgDoubleSizeIfFull(idxArray, getCoordCapacity(mode), pos, idxArrayParent, idxArray, fidCoord, ir::Assign::make(crdAcc, newCrdAcc), ir::readWrite);
  return ir::Block::make({maybeResizeIdx, storeIdx});
}

ir::Stmt RectCompressedModeFormat::getAppendEdges(std::vector<ir::Expr> parentPositions, ir::Expr posBegin, ir::Expr posEnd,
                                                  Mode mode) const {
  taco_iassert(parentPositions.size() == size_t(this->posDim));
  auto parentPos = this->packToPoint(parentPositions);
  auto posArray = this->getAccessor(mode.getModePack(), POS, ir::RW);
  ModeFormat parentModeType = mode.getParentModeType();
  auto lo = ir::FieldAccess::make(ir::Load::make(posArray, parentPos), "lo", false /* deref */, Int64);
  auto hi = ir::FieldAccess::make(ir::Load::make(posArray, parentPos), "hi", false /* deref */, Int64);

  // We need to keep around the code that performs a scan over empty cells for
  // position splits -- without the intermediate rectangles filled, we cannot
  // perform binary searches through the pos array.
  ir::Stmt setLo, setHi;
  if (!parentModeType.defined() || parentModeType.hasAppend()) {
    setLo = ir::Assign::make(lo, posBegin);
    setHi = ir::Assign::make(hi, ir::Sub::make(posEnd, 1));
  } else {
    setLo = ir::Stmt();
    setHi = ir::Assign::make(hi, ir::Sub::make(ir::Sub::make(posEnd, posBegin), 1));
  }
  return ir::Block::make({setLo, setHi});
}

ir::Expr RectCompressedModeFormat::getSize(ir::Expr szPrev, Mode mode) const {
  auto name = this->getModeSizeVarName(mode);
  taco_iassert(mode.hasVar(name));
  return mode.getVar(name);
}

std::string RectCompressedModeFormat::getModeSizeVarName(Mode& mode) const {
  return mode.getName() + "Size";
}

ir::Stmt RectCompressedModeFormat::declareModeVariables(Mode& mode) const {
  std::vector<ir::Stmt> results;

  auto sizeVarName = this->getModeSizeVarName(mode);
  if (!mode.hasVar(sizeVarName)) {
    auto var = ir::Var::make(sizeVarName, Int64);
    mode.addVar(sizeVarName, var);
    auto call = ir::Call::make("runtime->get_index_space_domain", {ir::ctx, ir::getIndexSpace(this->getRegion(mode.getModePack(), CRD))}, Auto);
    auto hi = ir::MethodCall::make(call, "hi", {}, false /* deref */, Int64);
    auto size = ir::Add::make(ir::Load::make(hi, 0), 1);
    results.push_back(ir::VarDecl::make(var, size));
  }

  return ir::Block::make(results);
}

ir::Stmt RectCompressedModeFormat::getAppendInitEdges(ir::Expr parentPos, ir::Expr pPrevBegin, ir::Expr pPrevEnd, Mode mode) const {
  auto pack = mode.getModePack();
  if (isa<ir::Literal>(pPrevBegin)) {
    taco_iassert(to<ir::Literal>(pPrevBegin)->equalsScalar(0));
    return ir::Stmt();
  }

  // If the parent mode has append (is sparse), all we need to do is
  // potentially double the size of the pos array.
  auto posArray = this->getRegion(pack, POS);
  auto posArrayParent = this->getRegion(pack, POS_PARENT);
  ir::Expr posCapacity = this->getPosCapacity(mode);
  ModeFormat parentModeType = mode.getParentModeType();
  auto posAcc = this->getAccessor(pack, POS, ir::RW);
  auto newPosAcc = ir::makeCreateAccessor(posAcc, posArray, fidRect1);
  if (!parentModeType.defined() || parentModeType.hasAppend()) {
    return lgDoubleSizeIfFull(posArray, posCapacity, parentPos, posArrayParent, posArray, fidRect1, ir::Assign::make(posAcc, newPosAcc), ir::readWrite);
  }

  // At this point, we are a sparse level with at least one dense level above
  // us, which means that the first dimension of our pos array will be indexed
  // by parentPos. Here, we need to initialize all of the entries in the pos
  // array for the current parentPos. We emit loops that iterate over the
  // remaining dimensions of the pos array, as the first one is fixed to parentPos.
  std::vector<ir::Expr> pVars(this->posDim);
  pVars[0] = parentPos;
  for (int i = 1; i < this->posDim; i++) {
    pVars[i] = ir::Var::make("p" + mode.getName() + util::toString(i), Int());
  }
  auto initPos = ir::Store::make(posAcc, this->packToPoint(pVars), ir::makeConstructor(Rect(1), {0, -1}));
  for (int i = this->posDim - 1; i >= 1; i--) {
    auto rect = ir::FieldAccess::make(this->getPosBounds(mode), "bounds", false /* isDeref */, Auto);
    auto domHi = ir::FieldAccess::make(rect, "hi", false /* isDeref */, Int());
    auto hi = ir::Add::make(ir::Load::make(domHi, i), 1);
    initPos = ir::For::make(pVars[i], 0, hi, 1, initPos);
  }
  ir::Stmt maybeResizePos = lgDoubleSizeIfFull(posArray, posCapacity, parentPos, posArrayParent, posArray, fidRect1, ir::Assign::make(posAcc, newPosAcc), ir::readWrite);
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

  std::vector<ir::Stmt> initStmts = {this->initPosBounds(mode)};
  if (szPrevIsZero) {
    posCapacity = getPosCapacity(mode);
    initStmts.push_back(ir::VarDecl::make(posCapacity, initCapacity));
  }

  // If we have a sparse ancestor, then we'll need to perform an initial
  // allocation of the pos region. Otherwise, the dimensions of the pos
  // region are statically known and we can directly allocate the region.
  if (this->hasSparseAncestor(mode) || !mode.getParentMode().defined()) {
    initStmts.push_back(ir::makeLegionMalloc(posArray, posCapacity, posParent, fidRect1, ir::readWrite));
  } else {
    auto mallocCall = ir::Call::make("legionMalloc", {ir::ctx, ir::runtime, posArray, posParent, fidRect1, ir::readWrite}, Auto);
    initStmts.push_back(ir::Assign::make(posArray, mallocCall));
  }

  // Reinitialize the accessor for the new physical region. We need a RW accessor here.
  auto posAcc = this->getAccessor(pack, POS, ir::RW);
  auto newPosAcc = ir::makeCreateAccessor(posAcc, posArray, fidRect1);
  initStmts.push_back(ir::Assign::make(posAcc, newPosAcc));

  // Start off each component in the position array as <0, -1>, an empty rectangle
  auto zeros = std::vector<ir::Expr>(this->posDim, 0);
  initStmts.push_back(ir::Store::make(posAcc, this->packToPoint(zeros), ir::makeConstructor(Rect(1), {0, -1})));

  // This initialization doesn't happen when the pos region has a sparse ancestor.
  if (mode.getParentModeType().defined() &&
      !mode.getParentModeType().hasAppend() && !szPrevIsZero) {
    taco_iassert(!this->hasSparseAncestor(mode));

    // We need to emit a loop for each dimension of the pos region.
    std::vector<ir::Expr> pVars(this->posDim);
    for (int i = 0; i < this->posDim; i++) {
      pVars[i] = ir::Var::make("p" + mode.getName() + util::toString(i), Int());
    }
    // Start off each component in the position array as <0, -1>, an empty rectangle
    auto loop = ir::Store::make(posAcc, this->packToPoint(pVars), ir::makeConstructor(Rect(1), {0, -1}));
    for (int i = this->posDim - 1; i >= 0; i--) {
      auto rect = ir::FieldAccess::make(this->getPosBounds(mode), "bounds", false /* isDeref */, Auto);
      auto domHi = ir::FieldAccess::make(rect, "hi", false /* isDeref */, Int());
      auto hi = ir::Add::make(ir::Load::make(domHi, i), 1);
      loop = ir::For::make(pVars[i], 0, hi, 1, loop);
    }
    initStmts.push_back(loop);
  }

  if (mode.getPackLocation() == (mode.getModePack().getNumModes() - 1)) {
    ir::Expr crdCapacity = this->getCoordCapacity(mode);
    auto crdArray = this->getRegion(pack, CRD);
    auto crdParent = this->getRegion(pack, CRD_PARENT);
    initStmts.push_back(ir::VarDecl::make(crdCapacity, defaultCapacity));
    initStmts.push_back(ir::makeLegionMalloc(crdArray, crdCapacity, crdParent, fidCoord, ir::readWrite));
    // Reinitialize the CRD RW accessor.
    auto crdAcc = this->getAccessor(pack, CRD, ir::RW);
    auto newCrdAcc = ir::makeCreateAccessor(crdAcc, crdArray, fidCoord);
    initStmts.push_back(ir::Assign::make(crdAcc, newCrdAcc));
  }

  return ir::Block::make(initStmts);
}

ir::Stmt RectCompressedModeFormat::getAppendFinalizeLevel(ir::Expr parentPos, ir::Expr szPrev, ir::Expr sz, Mode mode) const {
  auto pack = mode.getModePack();
  ModeFormat parentModeType = mode.getParentModeType();

  auto getSubregionCasts = [&]() {
    std::vector<ir::Stmt> result;
    // If our mode has a sparse parent, or is the first mode in the tensor,
    // we'll need to cast down the region to the actual size.
    if (this->hasSparseAncestor(mode) || !mode.getParentMode().defined()) {
      // We also need to restrict the pos array to a subregion if our parent
      // region is not dense (i.e. we have a variable size).
      auto posReg = this->getRegion(pack, POS).as<ir::GetProperty>();
      auto field = ir::FieldAccess::make(mode.getTensorExpr(), "indices", true /* isDeref*/, Auto);
      auto levelLoad = ir::Load::make(field, posReg->mode);
      auto idxLoad = ir::Load::make(levelLoad, posReg->index);
      auto size = parentPos.defined() ? parentPos : szPrev;
      auto subreg = ir::Call::make("getSubRegion", {ir::ctx, ir::runtime, this->getRegion(pack, POS_PARENT),
                                                    ir::makeConstructor(Rect(1), {0, ir::Sub::make(size, 1)})}, Auto);
      auto setSubReg = ir::Assign::make(idxLoad, subreg);
      result.push_back(setSubReg);
    }

    auto crdReg = this->getRegion(pack, CRD).as<ir::GetProperty>();
    auto field = ir::FieldAccess::make(mode.getTensorExpr(), "indices", true /* isDeref*/, Auto);
    auto levelLoad = ir::Load::make(field, crdReg->mode);
    auto idxLoad = ir::Load::make(levelLoad, crdReg->index);
    auto subreg = ir::Call::make("getSubRegion", {ir::ctx, ir::runtime, this->getRegion(pack, CRD_PARENT),
                                                  ir::makeConstructor(Rect(1), {0, ir::Sub::make(sz, 1)})}, Auto);
    auto setSubReg = ir::Assign::make(idxLoad, subreg);
    result.push_back(setSubReg);
    return ir::Block::make(result);
  };

  // If our parent is guaranteed to write into every entry of the pos array, then we are done,
  // and just need to do subregion cast operations.
  if ((isa<ir::Literal>(szPrev) && to<ir::Literal>(szPrev)->equalsScalar(1)) ||
      !parentModeType.defined() || parentModeType.hasAppend()) {
    return getSubregionCasts();
  }

  // Otherwise, we need to perform a prefix sum to fill in the values of the pos array. We need
  // this even though we have rectangles so that we can binary search over the pos arrays.
  auto pos = this->getAccessor(pack, POS, ir::RW);

  // Before we do the initialization loop, we first need to malloc the whole pos region
  // itself. This is important because the optimization performed by realloc drops access
  // to the old portion of the region. So, we'll just re-allocate it here. However, we
  // only need to do this if the pos region has a sparse ancestor. If the region does not
  // have a sparse ancestor then it will have been malloc'd up front and will never have
  // been realloced.
  ir::Stmt reallocFullRegion;
  if (this->hasSparseAncestor(mode)) {
    std::vector<ir::Stmt> stmts;
    // Unmap the existing pos region.
    auto posReg = this->getRegion(pack, POS);
    auto posParent = this->getRegion(pack, POS_PARENT);
    stmts.push_back(ir::SideEffect::make(ir::Call::make("runtime->unmap_region", {ir::ctx, posReg}, Auto)));
    // Now malloc the region. Its first dimension will be of parentPos size.
    stmts.push_back(ir::makeLegionMalloc(posReg, parentPos, posParent, fidRect1, ir::readWrite));
    // Now update the accessor.
    stmts.push_back(ir::Assign::make(pos, ir::makeCreateAccessor(pos, posReg, fidRect1)));
    reallocFullRegion = ir::Block::make(stmts);
  }

  // Get the index space domain of the pos array.
  auto posDom = this->getPosBounds(mode);

  // We now need to iterate over the full multi-dimensional pos region
  // to perform the prefix sum over the rectangles. We handle doing the
  // prefix sum over multi-dimensional pos regions by simply linearizing
  // the pos region in a row-major layout. This loop is not emitted when
  // the pos region has a sparse direct parent, as then every location
  // in the pos region will be written to.
  ir::Expr csVar = ir::Var::make("cs" + mode.getName(), Int64);
  ir::Stmt initCs = ir::VarDecl::make(csVar, 0);

  // We need to emit a loop for each dimension of the pos region.
  std::vector<ir::Expr> pVars(this->posDim);
  for (int i = 0; i < this->posDim; i++) {
    pVars[i] = ir::Var::make("p" + mode.getName() + util::toString(i), Int());
  }

  // Increment lo and hi by the accumulator. Note that we have to set this order
  // appropriately so that the lowering infrastructure doesn't turn this into
  // a compound add as the Realm type system doesn't allow Point<N> += operations.
  auto lo = ir::FieldAccess::make(ir::Load::make(pos, this->packToPoint(pVars)), "lo", false, Int64);
  auto hi = ir::FieldAccess::make(ir::Load::make(pos, this->packToPoint(pVars)), "hi", false, Int64);
  ir::Expr numElems = ir::Var::make("numElems" + mode.getName(), Int64);
  ir::Stmt getNumElems = ir::VarDecl::make(numElems, hi);
  auto setLo = ir::Assign::make(lo, ir::Add::make(csVar, lo));
  auto setHi = ir::Assign::make(hi, ir::Add::make(csVar, hi));
  auto incCs = ir::Assign::make(csVar, ir::Add::make(csVar, ir::Add::make(numElems, 1)));
  auto loop = ir::Block::make({getNumElems, setLo, setHi, incCs});
  for (int i = this->posDim - 1; i >= 0; i--) {
    auto rect = ir::FieldAccess::make(posDom, "bounds", false, Auto);
    auto domHi = ir::FieldAccess::make(rect, "hi", false, Int());
    // If parentPos is defined, then this mode has a sparse ancestor which
    // indexes the first dimension of the pos region. So, we'll use that variable
    // instead of the runtime bounds (since the first dimension is given an infinite size).
    ir::Expr loopHi;
    if (parentPos.defined() && i == 0) {
      loopHi = parentPos;
    } else {
      loopHi = ir::Add::make(ir::Load::make(domHi, i), 1);
    }
    loop = ir::For::make(pVars[i], 0, loopHi, 1, loop);
  }
  auto setSubReg = getSubregionCasts();
  return ir::Block::make({reallocFullRegion, initCs, loop, setSubReg});
}

ir::Stmt RectCompressedModeFormat::getInitializePosColoring(Mode mode) const {
  // Get the bounds of the CRD region.
  auto declareColoring = ir::VarDecl::make(this->getCrdColoring(mode), ir::makeConstructor(DomainPointColoring, {}));
  auto declareBounds = this->initCrdBounds(mode);
  // Create a domainPointColoring for the CRD region.
  return ir::Block::make(declareBounds, declareColoring);
}

ir::Stmt RectCompressedModeFormat::getFinalizePosColoring(Mode mode) const {
  return ir::Stmt();
}

ir::Stmt RectCompressedModeFormat::getCreatePosColoringEntry(Mode mode, ir::Expr domainPoint, ir::Expr lowerBound, ir::Expr upperBound) const {
  std::vector<ir::Stmt> stmts;
  auto pointT = Point(1);
  auto rectT = Rect(1);

  auto start = ir::Var::make(mode.getName() + "CrdStart", pointT);
  auto end = ir::Var::make(mode.getName() + "CrdEnd", pointT);
  auto domainMax = ir::Load::make(ir::FieldAccess::make(this->getCrdBounds(mode), "bounds.hi", false /* deref */, Int64));
  stmts.push_back(ir::VarDecl::make(start, ir::makeConstructor(pointT, {lowerBound})));
  stmts.push_back(ir::VarDecl::make(end, ir::makeConstructor(pointT, {ir::Min::make(upperBound, domainMax)})));
  auto rect = ir::Var::make(mode.getName() + "CrdRect", rectT);
  stmts.push_back(ir::VarDecl::make(rect, ir::makeConstructor(rectT, {start, end})));

  // It's possible that this partitioning makes a rectangle that goes out of bounds
  // of crd's index space. If so, replace the rectangle with an empty Rect.
  auto crdBounds = this->getCrdBounds(mode);
  auto lb = ir::MethodCall::make(crdBounds, "contains", {ir::FieldAccess::make(rect, "lo", false, Auto)}, false, Bool);
  auto hb = ir::MethodCall::make(crdBounds, "contains", {ir::FieldAccess::make(rect, "hi", false, Auto)}, false, Bool);
  auto guard = ir::Or::make(ir::Neg::make(lb), ir::Neg::make(hb));
  stmts.push_back(ir::IfThenElse::make(guard, ir::Block::make(
      ir::Assign::make(rect, ir::MethodCall::make(rect, "make_empty", {}, false, Auto)))));
  stmts.push_back(ir::Assign::make(ir::Load::make(this->getCrdColoring(mode), domainPoint), rect));

  return ir::Block::make(stmts);
}

ModeFunction RectCompressedModeFormat::getCreatePartitionWithPosColoring(Mode mode, ir::Expr domain, ir::Expr partitionColor) const {
  // Here, we use the coloring to create a partition of the crd
  // region using create_partition_by_domain. Then, we use that
  // partition to create a partition of the pos region with
  // create_partition_by_preimage_range. This logic is similar
  // to getPartitionFromChild.
  auto maybeAddColor = [&](std::vector<ir::Expr> args) {
    if (!partitionColor.defined()) {
      return args;
    }
    args.push_back(partitionColor);
    return args;
  };
  auto pack = mode.getModePack();

  std::vector<ir::Stmt> stmts;

  auto crd = this->getRegion(pack, CRD);
  auto crdIndexSpace = ir::MethodCall::make(crd, "get_index_space", {}, false /* deref */, Auto);
  auto crdIndexPart = ir::Var::make(mode.getName() + "_crd_index_part", IndexPartition);
  auto createCrdIndexPart = ir::Call::make("runtime->create_index_partition", maybeAddColor({ir::ctx, crdIndexSpace, domain, this->getCrdColoring(mode), ir::computePart}), Auto);
  stmts.push_back(ir::VarDecl::make(crdIndexPart, createCrdIndexPart));
  auto crdPart = ir::Var::make(mode.getName() + "_crd_part", LogicalPartition);
  stmts.push_back(ir::VarDecl::make(crdPart, ir::Call::make("runtime->get_logical_partition", {ir::ctx, crd, crdIndexPart}, Auto)));
  auto posPartition = this->partitionPosFromCrd(mode, crdIndexPart, maybeAddColor);
  auto posPart = posPartition[0];
  stmts.push_back(posPartition.compute());
  return ModeFunction(ir::Block::make(stmts), {posPart, crdPart, posPart, crdPart});
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
      this->getRegion(pack, CRD),
      createCrdPartCall,
    },
    Auto
  );
  auto createCrdPart = ir::VarDecl::make(crdPart, getCrdLogicalPartCall);
  // The partition to pass down to children is the coloring of the crd array.
  return ModeFunction(ir::Block::make({createPosPart, createCrdPart}), {posPart, crdPart, crdPart});
}

ModeFunction RectCompressedModeFormat::getPartitionFromChild(ir::Expr childPartition, Mode mode, ir::Expr partitionColor) const {
  auto maybeAddColor = [&](std::vector<ir::Expr> args) {
    if (!partitionColor.defined()) {
      return args;
    }
    args.push_back(partitionColor);
    return args;
  };
  auto pack = mode.getModePack();
  // Here, we have a partition of the level below us. There is an entry in
  // the level below us for each entry in the crd array, so we can copy
  // that partition to make the partition of crd.
  auto crdPart = ir::Var::make("crdPart" + mode.getName(), LogicalPartition);
  auto crdPartIndexPartition = ir::MethodCall::make(crdPart, "get_index_partition", {}, false /* deref */, IndexPartition);
  auto createCrdPart = ir::VarDecl::make(crdPart, ir::Call::make("copyPartition", maybeAddColor(
      {ir::ctx, ir::runtime, childPartition, this->getRegion(pack, CRD)}), Auto));
  // Now, using this partition of crd, create a dependent partition of the pos array.
  auto posPartition = this->partitionPosFromCrd(mode, crdPartIndexPartition, maybeAddColor);
  // The resulting partition is a partition of the pos array.
  auto posPart = posPartition[0];
  return ModeFunction(ir::Block::make(createCrdPart, posPartition.compute()), {posPart, crdPart, posPart});
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

ir::Expr RectCompressedModeFormat::getPosCapacity(Mode mode) const {
  const std::string varName = mode.getName() + "_pos_size";
  return this->getModeVar(mode, varName, Int());
}

ir::Expr RectCompressedModeFormat::getCoordCapacity(Mode mode) const {
  const std::string varName = mode.getName() + "_crd_size";
  return this->getModeVar(mode, varName, Int());
}

ir::Expr RectCompressedModeFormat::packToPoint(const std::vector<ir::Expr>& args) const {
  auto pointTy = Point(args.size());
  return ir::makeConstructor(pointTy, args);
}

ir::Expr RectCompressedModeFormat::getPosBounds(Mode mode) const {
  const std::string varName = mode.getName() + "_pos_domain";
  return this->getModeVar(mode, varName, Domain(this->posDim));
}

ir::Expr RectCompressedModeFormat::getCrdBounds(Mode mode) const {
  const std::string varName = mode.getName() + "_crd_domain";
  return this->getModeVar(mode, varName, Domain(1));
}

ir::Expr RectCompressedModeFormat::getCrdColoring(Mode mode) const {
  const std::string varName = mode.getName() + "_crd_coloring";
  return this->getModeVar(mode, varName, DomainPointColoring);
}

ir::Expr RectCompressedModeFormat::getSeqInsertEdgesResultVar(Mode mode) const {
  const std::string varName = mode.getName() + "_seq_insert_edges_result";
  return this->getModeVar(mode, varName, Auto);
}

ir::Stmt RectCompressedModeFormat::initPosBounds(Mode mode) const {
  auto pack = mode.getModePack();
  auto posArray = this->getRegion(pack, POS);
  auto getIspace = ir::MethodCall::make(posArray, "get_index_space", {}, false, Auto);
  auto getDomain = ir::Call::make("runtime->get_index_space_domain", {ir::ctx, getIspace}, Auto);
  return ir::VarDecl::make(this->getPosBounds(mode), getDomain);
}

ir::Stmt RectCompressedModeFormat::initCrdBounds(Mode mode) const {
  auto pack = mode.getModePack();
  auto crdArray = this->getRegion(pack, CRD);
  auto getIspace = ir::MethodCall::make(crdArray, "get_index_space", {}, false, Auto);
  auto getDomain = ir::Call::make("runtime->get_index_space_domain", {ir::ctx, getIspace}, Auto);
  return ir::VarDecl::make(this->getCrdBounds(mode), getDomain);
}

bool RectCompressedModeFormat::hasSparseAncestor(Mode mode) const {
  bool hasSparseParent = false;
  auto parentMode = mode.getParentMode();
  while (parentMode.defined()) {
    if (!parentMode.getModeFormat().is<DenseModeFormat>()) {
      hasSparseParent = true;
      break;
    }
    parentMode = parentMode.getParentMode();
  }
  return hasSparseParent;
}

ModeFunction RectCompressedModeFormat::partitionPosFromCrd(Mode mode, ir::Expr crdIndexPartition,
                                                           std::function<std::vector<ir::Expr>(
                                                               std::vector<ir::Expr>)> maybeAddColor) const {
  auto pack = mode.getModePack();
  // There are some interesting things going on to create the backwards partition of
  // the pos array from the crd array. First, we use create_partition_by_image_range
  // to find the backpointers from each crd entry into the containing pos entry.
  // However, this isn't enough. Only the pos entries that are non-empty will be contained
  // in one of these dependent partitions, meaning that we won't be able to scan past the
  // empty pos entries that denote empty rectangles. To remedy this, we exploit the fact
  // that the crd array is sorted by increasing coordinate position. Therefore, if we know
  // that two pos entries i and j are in a partition, then all entries between i and j must
  // also be in the partition (this intuition extends to multiple dimensions). So, we can
  // "densify" the sparse partition computed via create_partition_by_image_range to get
  // the final partition of the pos region.
  // TODO (rohany): Handle formats like {Sparse, Dense}.
  auto posSparsePart = ir::Var::make("posSparsePart" + mode.getName(), IndexPartition);
  auto createPosSparsePart = ir::Call::make(
      "runtime->create_partition_by_preimage_range",
      {
          ir::ctx,
          crdIndexPartition,
          this->getRegion(pack, POS),
          this->getRegion(pack, POS_PARENT),
          fidRect1,
          ir::Call::make("runtime->get_index_partition_color_space_name", {ir::ctx, crdIndexPartition}, Auto)
      },
      Auto
  );
  auto definedPosSparsePart = ir::VarDecl::make(posSparsePart, createPosSparsePart);
  auto posIndexPart = ir::Var::make("posIndexPart" + mode.getName(), IndexPartition);
  auto createPosIndexPart = ir::VarDecl::make(posIndexPart, ir::Call::make("densifyPartition", maybeAddColor(
      {ir::ctx, ir::runtime, ir::getIndexSpace(this->getRegion(pack, POS)), posSparsePart}), Auto));
  auto posPart = ir::Var::make("posPart" + mode.getName(), LogicalPartition);
  auto createPosPart = ir::VarDecl::make(posPart, ir::Call::make("runtime->get_logical_partition",
                                                                 {ir::ctx, this->getRegion(pack, POS), posIndexPart}, Auto));
  return ModeFunction(ir::Block::make({definedPosSparsePart, createPosIndexPart, createPosPart}), {posPart});
}

}
