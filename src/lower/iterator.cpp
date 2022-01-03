#include "taco/lower/iterator.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "mode_access.h"
#include "taco/ir/ir.h"
#include "taco/storage/storage.h"
#include "taco/storage/array.h"
#include "taco/util/strings.h"
#include "taco/lower/mode_format_impl.h"
#include "taco/lower/mode_format_rect_compressed.h"

using namespace std;
using namespace taco::ir;

namespace taco {

// class Iterator
struct Iterator::Content {
  IndexVar indexVar;
  Mode     mode;

  Iterator               parent;  // Pointer to parent iterator
  std::weak_ptr<Content> child;   // (Non-reference counted) pointer to child iterator

  ir::Expr tensor;
  ir::Expr posVar;
  ir::Expr coordVar;
  ir::Expr endVar;
  ir::Expr segendVar;
  ir::Expr validVar;
  ir::Expr beginVar;

  // AccessWindow represents a window (or slice) into a tensor mode, given by
  // the expressions representing an upper and lower bound. An iterator
  // is windowed if window is not NULL.
  struct Window {
    // windowVar is a Var specific to this iterator. It is intended to
    // be used as temporary storage to avoid duplicate memory loads of
    // expressions that are used in window/stride related bounds checking.
    ir::Expr windowVar;
    ir::Expr lo;
    ir::Expr hi;
    ir::Expr stride;
    Window(ir::Expr _lo, ir::Expr _hi, ir::Expr _stride, ir::Expr _windowVar) :
      windowVar(_windowVar), lo(_lo), hi(_hi), stride(_stride) {};
  };
  std::shared_ptr<Window> window;
  Iterator indexSetIterator;

  // The iterator that this iterator is tracking, if any.
  Iterator trackingIterator;
};

Iterator::Iterator() : content(nullptr) {
}

Iterator::Iterator(std::shared_ptr<Content> content) : content(content) {
}

Iterator::Iterator(IndexVar indexVar, bool isFull) : content(new Content) {
  content->indexVar = indexVar;
  content->coordVar = Var::make(indexVar.getName(), Int());
  content->posVar = Var::make(indexVar.getName() + "_pos", Int());

  if (!isFull) {
    content->beginVar = Var::make(indexVar.getName() + "_begin", Int());
    content->endVar = Var::make(indexVar.getName() + "_end", Int());
  }
}

Iterator::Iterator(ir::Expr tensor) : content(new Content) {
  content->tensor = tensor;
  content->posVar = 0;
  content->coordVar = 0;
  content->endVar = 1;
}

Iterator::Iterator(IndexVar indexVar, Expr tensor, Mode mode, Iterator parent,
                   string name, bool useNameForPos) : content(new Content) {
  content->indexVar = indexVar;

  content->mode = mode;
  content->parent = parent;
  content->parent.setChild(*this);

  string modeName = mode.getName();
  content->tensor = tensor;

  string posNamePrefix = "p" + modeName;
  if (useNameForPos) {
    posNamePrefix = name;
  }
  content->posVar   = Var::make(name,            Int());
  content->endVar   = Var::make("p" + modeName + "_end",   Int());
  content->beginVar = Var::make("p" + modeName + "_begin", Int());

  content->coordVar = Var::make(name, Int());
  content->segendVar = Var::make(modeName + "_segend", Int());
  content->validVar = Var::make("v" + modeName, Bool);
}

bool Iterator::isRoot() const {
  return !getParent().defined();
}

bool Iterator::isLeaf() const {
  return !getChild().defined();
}

const Iterator& Iterator::getParent() const {
  taco_iassert(defined());
  return content->parent;
}

const Iterator Iterator::getChild() const {
  taco_iassert(defined());
  return Iterator(content->child.lock());
}

void Iterator::setChild(const Iterator& iterator) const {
  taco_iassert(defined());
  content->child = iterator.content; 
}

IndexVar Iterator::getIndexVar() const {
  return content->indexVar;
}

Expr Iterator::getTensor() const {
  taco_iassert(defined());
  return content->tensor;
}

const Mode& Iterator::getMode() const {
  taco_iassert(defined());
  return content->mode;
}

Expr Iterator::getPosVar() const {
  taco_iassert(defined());
  return content->posVar;
}

Expr Iterator::getCoordVar() const {
  taco_iassert(defined());
  return content->coordVar;
}

Expr Iterator::getIteratorVar() const {
  return hasPosIter() ? getPosVar() : getCoordVar();
}

Expr Iterator::getDerivedVar() const {
  return hasPosIter() ? getCoordVar() : getPosVar();
}

Expr Iterator::getEndVar() const {
  taco_iassert(defined());
  return content->endVar;
}

Expr Iterator::getSegendVar() const {
  taco_iassert(defined());
  return content->segendVar;
}

Expr Iterator::getValidVar() const {
  taco_iassert(defined());
  return content->validVar;
}

Expr Iterator::getBeginVar() const {
  taco_iassert(defined());
  return content->beginVar;
}

bool Iterator::isDimensionIterator() const {
  return !content->mode.defined() && !content->tensor.defined();
}

bool Iterator::isModeIterator() const {
  return content->mode.defined();
}

bool Iterator::isDense() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeFormat() == ModeFormat::Dense;
}

bool Iterator::isLgSparse() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeFormat().is<RectCompressedModeFormat>();
}

bool Iterator::isFull() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return !content->beginVar.defined() && !content->endVar.defined();
  return getMode().defined() && getMode().getModeFormat().isFull();
}

bool Iterator::isOrdered() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isOrdered();
}

bool Iterator::isUnique() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isUnique();
}

bool Iterator::isBranchless() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isBranchless();
}

bool Iterator::isCompact() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isCompact();
}

bool Iterator::isZeroless() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().isZeroless();
}

bool Iterator::hasCoordIter() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasCoordValIter();
}

bool Iterator::hasPosIter() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasCoordPosIter();
}

bool Iterator::hasLocate() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasLocate();
}

bool Iterator::hasInsert() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasInsert();
}

bool Iterator::hasAppend() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasAppend();
}

bool Iterator::hasSeqInsertEdge() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasSeqInsertEdge();
}

bool Iterator::hasInsertCoord() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasInsertCoord();
}

bool Iterator::isYieldPosPure() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().isYieldPosPure();
}

ModeFunction Iterator::coordBounds(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->coordIterBounds(coords, getMode());
}

ModeFunction Iterator::coordBounds(const ir::Expr& parentPos) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->coordBounds(parentPos, getMode());
}

ModeFunction Iterator::coordAccess(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->coordIterAccess(getParent().getPosVar(),
                                                   coords, getMode());
}

ModeFunction Iterator::posBounds(const ir::Expr& parentPos) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->posIterBounds({parentPos}, getMode());
}

ModeFunction Iterator::posBounds(const std::vector<ir::Expr>& parentPositions) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->posIterBounds(parentPositions, getMode());
}

ModeFunction Iterator::posAccess(const ir::Expr& pos, 
                                 const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->posIterAccess(pos, coords, getMode());
}

ModeFunction Iterator::locate(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->locate(getParent().getPosVar(),
                                              coords, getMode());
}

Stmt Iterator::getInsertCoord(const Expr& p, const std::vector<Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertCoord(p, coords, getMode());
}

Expr Iterator::getWidth() const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getWidth(getMode());
}

Stmt Iterator::getInsertInitCoords(const Expr& pBegin, const Expr& pEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertInitCoords(pBegin, pEnd,
                                                           getMode());
}

Stmt Iterator::getInsertInitLevel(const Expr& szPrev, const Expr& sz) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertInitLevel(szPrev, sz,
                                                          getMode());
}

Stmt Iterator::getInsertFinalizeLevel(const Expr& szPrev, const Expr& sz) const{
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertFinalizeLevel(szPrev, sz,
                                                              getMode());
}

Stmt Iterator::getAppendCoord(const Expr& p, const Expr& i) const {
  taco_iassert(defined() && content->mode.defined());
  return content->mode.getModeFormat().impl->getAppendCoord(p, i, content->mode);
}

Stmt Iterator::getAppendEdges(const std::vector<Expr>& parentPositions, const Expr& pBegin,
                              const Expr& pEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getAppendEdges(parentPositions, pBegin, pEnd, getMode());
}

Expr Iterator::getSize(const ir::Expr& szPrev) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getSize(szPrev, getMode());
}

Stmt Iterator::getAppendInitEdges(const Expr& parentPos,
                                  const Expr& pPrevBegin,
                                  const Expr& pPrevEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return content->mode.getModeFormat().impl->getAppendInitEdges(parentPos,
                                                                pPrevBegin,
                                                                pPrevEnd,
                                                                content->mode);
}

Stmt Iterator::getAppendInitLevel(const Expr& szPrev, const Expr& sz) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getAppendInitLevel(szPrev, sz,
                                                          getMode());
}

Stmt Iterator::getAppendFinalizeLevel(const Expr& parentPos, const Expr& szPrev, const Expr& sz) const{
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getAppendFinalizeLevel(parentPos, szPrev, sz, getMode());
}

Expr Iterator::getAssembledSize(const Expr& prevSize) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getAssembledSize(prevSize, getMode());
}

Stmt Iterator::getSeqInitEdges(const Expr& prevSize, 
    const std::vector<AttrQueryResult>& queries) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getSeqInitEdges(prevSize, queries, 
                                                         getMode());
}

Stmt Iterator::getSeqInsertEdge(const Expr& parentPos, 
    const std::vector<Expr>& coords, 
    const std::vector<AttrQueryResult>& queries) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getSeqInsertEdge(parentPos, coords, 
                                                          queries, getMode());
}

Stmt Iterator::getInitCoords(const Expr& prevSize, 
    const std::vector<AttrQueryResult>& queries) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInitCoords(prevSize, queries, 
                                                       getMode());
}

Stmt Iterator::getInitYieldPos(const Expr& prevSize) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInitYieldPos(prevSize, getMode());
}

ModeFunction Iterator::getYieldPos(const Expr& parentPos, 
    const std::vector<Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getYieldPos(parentPos, coords, 
                                                     getMode());
}

Stmt Iterator::getInsertCoord(const Expr& parentPos, const Expr& pos, 
    const std::vector<Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertCoord(parentPos, pos, coords, 
                                                        getMode());
}

Stmt Iterator::getFinalizeYieldPos(const Expr& prevSize) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getFinalizeYieldPos(prevSize, 
                                                             getMode());
}

bool Iterator::defined() const {
  return content != nullptr;
}

bool Iterator::isWindowed() const {
  return this->content->window != nullptr;
}

ir::Expr Iterator::getWindowLowerBound() const {
  taco_iassert(this->isWindowed());
  return this->content->window->lo;
}

ir::Expr Iterator::getWindowUpperBound() const {
  taco_iassert(this->isWindowed());
  return this->content->window->hi;
}

ir::Expr Iterator::getStride() const {
  taco_iassert(this->isWindowed());
  return this->content->window->stride;
}

ir::Expr Iterator::getWindowVar() const {
  taco_iassert(this->isWindowed());
  return this->content->window->windowVar;
}

bool Iterator::isStrided() const {
  // It's not necessary but makes things simpler to require a window in order
  // to have a stride.
  taco_iassert(this->isWindowed());
  auto strideLiteral = this->content->window->stride.as<ir::Literal>();
  return !(strideLiteral != nullptr && strideLiteral->getIntValue() == 1);
}

void Iterator::setWindowBounds(ir::Expr lo, ir::Expr hi, ir::Expr stride) {
  auto windowVarName = this->getIndexVar().getName() + this->getMode().getName() + "_window";
  auto wvar = ir::Var::make(windowVarName, Int());
  this->content->window = std::make_shared<Content::Window>(Content::Window(lo, hi, stride, wvar));
}

bool Iterator::hasIndexSet() const {
  return this->content->indexSetIterator.defined();
}
Iterator Iterator::getIndexSetIterator() const {
  taco_iassert(this->hasIndexSet());
  return this->content->indexSetIterator;
}

void Iterator::setIndexSetIterator(Iterator iter) {
  this->content->indexSetIterator = iter;
}

ir::Stmt Iterator::getInitializePosColoring() const {
  taco_iassert(this->defined());
  return getMode().getModeFormat().impl->getInitializePosColoring(getMode());
}

ir::Stmt Iterator::getCreatePosColoringEntry(ir::Expr domainPoint, ir::Expr lowerBound, ir::Expr upperBound) const {
  taco_iassert(this->defined());
  return getMode().getModeFormat().impl->getCreatePosColoringEntry(getMode(), domainPoint, lowerBound, upperBound);
}

ir::Stmt Iterator::getFinalizePosColoring() const {
  taco_iassert(this->defined());
  return getMode().getModeFormat().impl->getFinalizePosColoring(getMode());
}

ModeFunction Iterator::getCreatePartitionWithPosColoring(ir::Expr domain, ir::Expr partitionColor) const {
  taco_iassert(this->defined());
  return getMode().getModeFormat().impl->getCreatePartitionWithPosColoring(getMode(), domain, partitionColor);
}

ModeFunction Iterator::getPartitionFromParent(ir::Expr parentPartition, ir::Expr partitionColor) const {
  return getMode().getModeFormat().impl->getPartitionFromParent(parentPartition, getMode(), partitionColor);
}

ModeFunction Iterator::getPartitionFromChild(ir::Expr childPartition, ir::Expr partitionColor) const {
  return getMode().getModeFormat().impl->getPartitionFromChild(childPartition, getMode(), partitionColor);
}

std::vector<ModeRegion> Iterator::getRegions() {
  auto mode = getMode();
  return mode.getModeFormat().impl->getRegions(getTensor(), mode.getLevel());
}

ir::Stmt Iterator::declareModeVariables() const {
  if (this->content->mode.getModeFormat().is<RectCompressedModeFormat>()) {
    auto rcmf = this->content->mode.getModeFormat().as<RectCompressedModeFormat>();
    return rcmf->declareModeVariables(this->content->mode);
  }
  return ir::Stmt();
}

Iterator Iterator::getTrackingIterator() const {
  return this->content->trackingIterator;
}

bool operator==(const Iterator& a, const Iterator& b) {
  if (a.isDimensionIterator() && b.isDimensionIterator()) {
    return a.getIndexVar() == b.getIndexVar();
  }
  if (a.content == b.content) {
    return true;
  }

  return (a.getIndexVar() == b.getIndexVar() && a.getTensor() == b.getTensor()
      && a.getParent() == b.getParent());
}

bool operator<(const Iterator& a, const Iterator& b) {
  if (a == b) return false;
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const Iterator& iterator) {
  // Undefined iterator
  if (!iterator.defined()) {
    return os << "Iterator()";
  }
  // Dimension iterator
  if (iterator.isDimensionIterator()) {
    return os << "\u0394" << iterator.getIndexVar().getName();
  }
  return os << iterator.getTensor();
}


// class Iterators
struct Iterators::Content {
  map<ModeAccess,Iterator> levelIterators;
  map<Iterator,ModeAccess> modeAccesses;
  map<IndexVar,Iterator>   modeIterators;
};


Iterators::Iterators()
  : content(new Content)
{
}


Iterators::Iterators(IndexStmt stmt) : Iterators(stmt, createIRTensorVars(stmt))
{
}


Iterators::Iterators(IndexStmt stmt, const map<TensorVar, Expr>& tensorVars)
: Iterators()
{
  ProvenanceGraph provGraph = ProvenanceGraph(stmt);
  set<IndexVar> underivedAdded;

  // Figure out whether we are able to get away with keeping the non-zero
  // structure of the input for the output tensor.
  NonZeroAnalyzerResult nonZeroResult;
  auto preservesNonZeros = preservesNonZeroStructure(stmt, nonZeroResult);

  // Create dimension iterators
  match(stmt,
    function<void(const ForallNode*, Matcher*)>([&](auto n, auto m) {
      content->modeIterators.insert({n->indexVar, Iterator(n->indexVar, !provGraph.hasCoordBounds(n->indexVar) && provGraph.isCoordVariable(n->indexVar))});
      for (const IndexVar& underived : provGraph.getUnderivedAncestors(n->indexVar)) {
        if (!underivedAdded.count(underived)) {
          content->modeIterators.insert({underived, underived});
          underivedAdded.insert(underived);
        }
      }
      m->match(n->stmt);
    })
  );

  // Create access iterators
  match(stmt,
    function<void(const AccessNode*)>([&](auto n) {
      taco_iassert(util::contains(tensorVars, n->tensorVar));
      Expr tensorIR = tensorVars.at(n->tensorVar);
      Format format = n->tensorVar.getFormat();
      this->createAccessIterators(Access(n), format, tensorIR, provGraph, tensorVars);
    }),
    function<void(const AssignmentNode*, Matcher*)>([&](auto n, auto m) {
      m->match(n->rhs);
      m->match(n->lhs);
    })
  );

  if (preservesNonZeros) {
    // If our output preserves the non-zero structure of the input
    // tensor, then we'll modify the result iterator to track variables
    // of the input tensor that control iteration. However, information
    // about the mode etc will be retained.
    auto resultTV = nonZeroResult.resultAccess->getTensorVar();
    auto inputTV = nonZeroResult.inputAccess->getTensorVar();

    for (int i = 0; i < resultTV.getOrder(); i++) {
      auto resultIter = this->content->levelIterators[{*nonZeroResult.resultAccess, i+1}];
      auto inputIter = this->content->levelIterators[{*nonZeroResult.inputAccess, i+1}];
      taco_iassert(resultIter.defined());
      taco_iassert(inputIter.defined());
      resultIter.content->trackingIterator = inputIter;
    }
  }

  // Reverse the levelIterators map for fast modeAccess lookup
  for (auto& iterator : content->levelIterators) {
    content->modeAccesses.insert({iterator.second, iterator.first});
  }
}


void Iterators::createAccessIterators(Access access, Format format, Expr tensorIR,
                                      ProvenanceGraph provGraph,
                                      const map<TensorVar, Expr> &tensorVars) {
  TensorVar tensorConcrete = access.getTensorVar();
  taco_iassert(tensorConcrete.getOrder() == format.getOrder())
      << tensorConcrete << ", Format" << format;
  Shape shape = tensorConcrete.getType().getShape();

  Iterator parent(tensorIR);
  content->levelIterators.insert({{access,0}, parent});

  int level = 1;
  Mode parentMode;

  // This logic is for setting up iterators for pos splits. In a fused position split,
  // the original logic assigned all level iterators to have the index variable of the
  // fused pos split. This logic seems incorrect. Instead, having only the deepest level
  // that the pos split occurred use that index variable seemed like the right thing to
  // satisfy how the code was used in lowerForallFusedPosition. This block attempts to
  // find the deepest match for the position split in the coordinate tree of the tensor.
  int deepestPosMatch = -1;
  {
    int counter = 0;
    for (auto modeTypePack : format.getModeFormatPacks()) {
      taco_iassert(modeTypePack.getModeFormats().size() == 1);
      for (auto& modeType : modeTypePack.getModeFormats()) {
        int modeNumber = format.getModeOrdering()[counter];
        IndexVar indexVar = access.getIndexVars()[modeNumber];
        IndexVar iteratorIndexVar;
        if (provGraph.getPosIteratorDescendant(indexVar, &iteratorIndexVar) &&
            provGraph.isPosOfAccess(iteratorIndexVar, access)) {
          deepestPosMatch = std::max(deepestPosMatch, modeNumber);
        }
      }
      counter++;
    }
  }

  for (ModeFormatPack modeTypePack : format.getModeFormatPacks()) {
    vector<Expr> arrays;
    taco_iassert(modeTypePack.getModeFormats().size() > 0);

    int modeNumber = format.getModeOrdering()[level-1];
    ModePack modePack(modeTypePack.getModeFormats().size(),
                      modeTypePack.getModeFormats()[0], tensorIR,
                      modeNumber, level);

    int pos = 0;
    for (auto& modeType : modeTypePack.getModeFormats()) {
      int modeNumber = format.getModeOrdering()[level-1];
      Dimension dim = shape.getDimension(modeNumber);
      IndexVar indexVar = access.getIndexVars()[modeNumber];
      IndexVar posFuseVar;
      IndexVar iteratorIndexVar = indexVar;
      // As described above, only set the deepest position split match to be the fused iterator.
      if (provGraph.getPosIteratorDescendant(indexVar, &posFuseVar) &&
          provGraph.isPosOfAccess(posFuseVar, access) && modeNumber == deepestPosMatch) {
        iteratorIndexVar = posFuseVar;
      }
      Mode mode(tensorIR, dim, level, modeType, modePack, pos, parentMode);

      string name = iteratorIndexVar.getName() + tensorConcrete.getName();
      Iterator iterator(iteratorIndexVar, tensorIR, mode, parent, name, true);

      // If the access that this iterator corresponds to has a window, then
      // adjust the iterator appropriately.
      if (access.isModeWindowed(modeNumber)) {
        auto lo = ir::Literal::make(access.getWindowLowerBound(modeNumber));
        auto hi = ir::Literal::make(access.getWindowUpperBound(modeNumber));
        auto stride = ir::Literal::make(access.getStride(modeNumber));
        iterator.setWindowBounds(lo, hi, stride);
      }
      // If the access that corresponds to this iterator has an index set,
      // then we need to construct an iterator for the index set.
      if (access.isModeIndexSet(modeNumber)) {
        auto tv = access.getModeIndexSetTensor(modeNumber);
        auto tvVar = tensorVars.at(tv);
        auto tvFormat = tv.getFormat();
        auto tvShape = tv.getType().getShape();
        auto accessIvar = access.getIndexVars()[modeNumber];
        ModePack tvModePack(1, tvFormat.getModeFormats()[0], tvVar, 0, 1);
        Mode tvMode(tvVar, tvShape.getDimension(0), 1, tvFormat.getModeFormats()[0], tvModePack, 0, Mode());
        // Finally, construct the iterator and register it as an indexSetIterator.
        auto iter = Iterator(accessIvar, tvVar, tvMode, {tvVar}, accessIvar.getName() + tv.getName() + "_filter");
        iterator.setIndexSetIterator(iter);
        // Also add the iterator to the modeAccesses map.
        content->modeAccesses.insert({iter, {access, modeNumber + 1}});
      }

      content->levelIterators.insert({{access,modeNumber+1}, iterator});
      if (iteratorIndexVar != indexVar) {
        // add to allowing lowering to find correct iterator for this pos variable
        content->modeIterators[iteratorIndexVar] = iterator;
      }

      parent = iterator;
      parentMode = mode;
      pos++;
      level++;
    }
  }
}

Iterator Iterators::getLevelIteratorByModeAccess(ModeAccess modeAccess) const
{
  taco_iassert(content != nullptr);
  taco_iassert(util::contains(content->levelIterators, modeAccess))
      << "Cannot find " << modeAccess << " in "
      << util::join(content->levelIterators);
  return content->levelIterators.at(modeAccess);
}

Iterator Iterators::getLevelIteratorByMode(const TensorVar& tv, int mode) const {
  taco_iassert(content != nullptr);
  Iterator iter;
  for (auto levelIter : this->levelIterators()) {
    // Don't consider phantom iterators generated by Assemble.
    if (levelIter.first.getAccess().isAccessingStructure()) continue;
    if (levelIter.first.getAccess().getTensorVar() == tv && levelIter.first.getModePos() == size_t(mode + 1)) {
      taco_iassert(!iter.defined());
      iter = levelIter.second;
    }
  }
  return iter;
}

Iterator Iterators::getLevelIteratorByLevel(const TensorVar& tv, int level) const {
  auto mode = tv.getFormat().getModeOrdering()[level];
  return this->getLevelIteratorByMode(tv, mode);
}

std::map<ModeAccess,Iterator> Iterators::levelIterators() const
{
  return content->levelIterators;
}


ModeAccess Iterators::modeAccess(Iterator iterator) const
{
  taco_iassert(content != nullptr);
  taco_iassert(util::contains(content->modeAccesses, iterator));
  return content->modeAccesses.at(iterator);
}


Iterator Iterators::modeIterator(IndexVar indexVar) const
{
  taco_iassert(content != nullptr);
  taco_iassert(util::contains(content->modeIterators, indexVar));
  return content->modeIterators.at(indexVar);
}

std::map<IndexVar, Iterator> Iterators::modeIterators() const {
  return content->modeIterators;
}


// Free functions
std::vector<Iterator> getAppenders(const std::vector<Iterator>& iterators) {
  vector<Iterator> appendIterators;
  for (auto& iterator : iterators) {
    if (iterator.hasAppend()) {
      appendIterators.push_back(iterator);
    }
  }
  return appendIterators;
}


std::vector<Iterator> getInserters(const std::vector<Iterator>& iterators) {
 vector<Iterator> result;
  for (auto& iterator : iterators) {
    if (iterator.hasInsert()) {
      taco_iassert(iterator.hasLocate())
          << "Iterators with insert must also have locate";
      result.push_back(iterator);
    }
  }
  return result;
}

}
