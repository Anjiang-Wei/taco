#include "taco/index_notation/distribution.h"
#include "taco/index_notation/index_notation.h"

namespace taco {

struct Transfer::Content {
  Content(Access a) : access(a) {}
  Access access;
};

Transfer::Transfer(taco::Access a) : content(new Content(a)) {}

Access Transfer::getAccess() const {
  return this->content->access;
}

bool operator==(Transfer& a, Transfer& b) {
  return a.content->access.getTensorVar() == b.content->access.getTensorVar();
}

std::ostream& operator<<(std::ostream& o, const Transfer& t) {
  o << "transfer(" << t.getAccess() << ")";
  return o;
}

GridPlacement::AxisMatch Face(int face) {
  return GridPlacement::AxisMatch::makeFace(face);
}

GridPlacement::AxisMatch Replicate() {
  return GridPlacement::AxisMatch::makeReplicated();
}


PlacementGrid::Axis operator|(ir::Expr e, GridPlacement::AxisMatch axis) {
  return PlacementGrid::Axis(e, axis);
}

struct DistVar::Content {
  std::string name;
};

DistVar::DistVar() : content(new Content) {
  this->content->name = "random_name";
}

DistVar::DistVar(const std::string &name) : content(new Content) {
  this->content->name = name;
}

const std::string& DistVar::getName() const {
  return this->content->name;
}

bool operator==(const DistVar& a, const DistVar& b) {
  return a.content == b.content;
}

bool operator<(const DistVar& a, const DistVar& b) {
  return a.content < b.content;
}

struct MachineDimensionName::Content {
  Kind kind;
  // Used when kind == Var.
  DistVar var;
  // Used when kind == Restriction.
  int restriction;
};

MachineDimensionName::MachineDimensionName(int n) : content(new Content) {
  this->content->kind = Restriction;
  this->content->restriction = n;
}

MachineDimensionName::MachineDimensionName(DistVar v) : content(new Content) {
  this->content->kind = Var;
  this->content->var = v;
}

MachineDimensionName::MachineDimensionName() : content(nullptr) {}

MachineDimensionName MachineDimensionName::broadcast() {
  MachineDimensionName name;
  name.content = std::make_shared<MachineDimensionName::Content>();
  name.content->kind = Broadcast;
  return name;
}

MachineDimensionName::Kind MachineDimensionName::getKind() const {
  return this->content->kind;
}

const DistVar MachineDimensionName::getDistVar() const {
  taco_iassert(this->getKind() == Var);
  return this->content->var;
}

int MachineDimensionName::getRestriction() const {
  taco_iassert(this->getKind() == Restriction);
  return this->content->restriction;
}

struct TensorDistributionNotation::Content {
  std::vector<DistVar> lhs;
  std::vector<MachineDimensionName> rhs;
  Grid machine;
  ParallelUnit pu;
};

TensorDistributionNotation::TensorDistributionNotation() : content(nullptr) {}

bool TensorDistributionNotation::isValid() const {
  return this->content != nullptr;
}

TensorDistributionNotation::TensorDistributionNotation(std::vector<DistVar> lhs, Grid machine,
                                                       std::vector<MachineDimensionName> rhs,
                                                       ParallelUnit pu) : content(new Content) {
  this->content->lhs = lhs;
  this->content->rhs = rhs;
  this->content->machine = machine;
  this->content->pu = pu;
}

const std::vector<MachineDimensionName>& TensorDistributionNotation::getRHS() const {
  taco_iassert(this->isValid());
  return this->content->rhs;
}

const std::vector<DistVar>& TensorDistributionNotation::getLHS() const {
  taco_iassert(this->isValid());
  return this->content->lhs;
}

const Grid& TensorDistributionNotation::getMachine() const {
  taco_iassert(this->isValid());
  return this->content->machine;
}

ParallelUnit TensorDistributionNotation::getParallelUnit() const {
  return this->content->pu;
}

}