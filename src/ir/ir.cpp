#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir_printer.h"

#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/type.h"

namespace taco {
namespace ir {

// class Expr
Expr::Expr(bool n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(int8_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(int16_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(int32_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(int64_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(uint8_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(uint16_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(uint32_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(uint64_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(float n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(double n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(std::complex<float> n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(std::complex<double> n) : IRHandle(Literal::make(n)) {
}

Expr Literal::zero(Datatype datatype) {
  Expr zero;
  switch (datatype.getKind()) {
    case Datatype::Bool:
      zero = Literal::make(false);
      break;
    case Datatype::UInt8:
      zero = Literal::make((uint8_t)0);
      break;
    case Datatype::UInt16:
      zero = Literal::make((uint16_t)0);
      break;
    case Datatype::UInt32:
      zero = Literal::make((uint32_t)0);
      break;
    case Datatype::UInt64:
      zero = Literal::make((uint64_t)0);
      break;
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      zero = Literal::make((int8_t)0);
      break;
    case Datatype::Int16:
      zero = Literal::make((int16_t)0);
      break;
    case Datatype::Int32:
      zero = Literal::make((int32_t)0);
      break;
    case Datatype::Int64:
      zero = Literal::make((int64_t)0);
      break;
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      zero = Literal::make((float)0.0);
      break;
    case Datatype::Float64:
      zero = Literal::make((double)0.0);
      break;
    case Datatype::Complex64:
      zero = Literal::make(std::complex<float>());
      break;
    case Datatype::Complex128:
      zero = Literal::make(std::complex<double>());
      break;
    case Datatype::CppType:
      taco_ierror;
      break;
    case Datatype::Undefined:
      taco_ierror;
      break;
  }
    taco_iassert(zero.defined());
    return zero;
}

Literal::~Literal() {
  free(value.get());
}

bool Literal::getBoolValue() const {
  taco_iassert(type.isBool()) << "Type must be boolean";
  return getValue<bool>();
}

int64_t Literal::getIntValue() const {
  taco_iassert(type.isInt()) << "Type must be integer";
  switch (type.getKind()) {
    case Datatype::Int8:
      return getValue<int8_t>();
    case Datatype::Int16:
      return getValue<int16_t>();
    case Datatype::Int32:
      return getValue<int32_t>();
    case Datatype::Int64:
      return getValue<int64_t>();
    case Datatype::Int128:
      taco_not_supported_yet;
    default:
      break;
  }
  taco_ierror << "not an integer type";
  return 0ll;
}

uint64_t Literal::getUIntValue() const {
  taco_iassert(type.isUInt()) << "Type must be unsigned integer";
  switch (type.getKind()) {
    case Datatype::UInt8:
      return getValue<uint8_t>();
    case Datatype::UInt16:
      return getValue<uint16_t>();
    case Datatype::UInt32:
      return getValue<uint32_t>();
    case Datatype::UInt64:
      return getValue<uint64_t>();
    case Datatype::UInt128:
      taco_not_supported_yet;
    default:
      break;
  }
  taco_ierror << "not an unsigned integer type";
  return 0ull;
}

double Literal::getFloatValue() const {
  taco_iassert(type.isFloat()) << "Type must be floating point";
  switch (type.getKind()) {
    case Datatype::Float32:
      static_assert(sizeof(float) == 4, "Float not 32 bits");
      return getValue<float>();
    case Datatype::Float64:
      return getValue<double>();
    default:
      break;
  }
  taco_ierror << "not a floating point type";
  return 0.0;
}

std::complex<double> Literal::getComplexValue() const {
  taco_iassert(type.isComplex()) << "Type must be a complex number";
  switch (type.getKind()) {
    case Datatype::Complex64:
      return getValue<std::complex<float>>();
    case Datatype::Complex128:
      return getValue<std::complex<double>>();
    default:
      break;
  }
  taco_ierror << "not a floating point type";
  return 0.0;
}

template <typename T> bool compare(const Literal* literal, double val) {
      return literal->getValue<T>() == static_cast<T>(val);
}

bool Literal::equalsScalar(double scalar) const {
  switch (type.getKind()) {
    case Datatype::Bool:
      return compare<bool>(this, scalar);
    break;
    case Datatype::UInt8:
      return compare<uint8_t>(this, scalar);
    break;
    case Datatype::UInt16:
      return compare<uint16_t>(this, scalar);
    break;
    case Datatype::UInt32:
      return compare<uint32_t>(this, scalar);
    break;
    case Datatype::UInt64:
      return compare<uint64_t>(this, scalar);
    break;
    case Datatype::UInt128:
      taco_not_supported_yet;
    break;
    case Datatype::Int8:
      return compare<int8_t>(this, scalar);
    break;
    case Datatype::Int16:
      return compare<int16_t>(this, scalar);
    break;
    case Datatype::Int32:
      return compare<int32_t>(this, scalar);
    break;
    case Datatype::Int64:
      return compare<int64_t>(this, scalar);
    break;
    case Datatype::Int128:
      taco_not_supported_yet;
    break;
    case Datatype::Float32:
      return compare<float>(this, scalar);
    break;
    case Datatype::Float64:
      return compare<double>(this, scalar);
    break;
    case Datatype::Complex64:
      return compare<std::complex<float>>(this, scalar);
    break;
    case Datatype::Complex128:
      return compare<std::complex<double>>(this, scalar);
    break;
    case Datatype::CppType:
      taco_not_supported_yet;
      break;
    case Datatype::Undefined:
      taco_not_supported_yet;
    break;
  }
  return false;
}

Expr Var::make(std::string name, Datatype type, 
               bool is_ptr, bool is_tensor, bool is_parameter) {
  Var *var = new Var;
  var->type = type;
  var->name = name;

  // TODO: is_ptr and is_tensor should be part of type
  var->is_ptr = is_ptr;
  var->is_tensor = is_tensor;
  var->is_parameter = is_parameter;

  return var;
}

Expr Symbol::make(std::string name) {
  Symbol* s = new Symbol;
  s->type = Datatype(Datatype::CppType);
  s->name = name;
  return s;
}

Expr Neg::make(Expr a) {
  Neg *neg = new Neg;
  neg->a = a;
  neg->type = a.type();
  return neg;
}

Expr Sqrt::make(Expr a) {
  Sqrt *sqrt = new Sqrt;
  sqrt->a = a;
  sqrt->type = a.type();
  return sqrt;
}

// Binary Expressions
// helper
Datatype max_expr_type(Expr a, Expr b);
Datatype max_expr_type(Expr a, Expr b) {
  return max_type(a.type(), b.type());
}

Expr Add::make(Expr a, Expr b) {
  return Add::make(a, b, max_expr_type(a, b));
}

Expr Add::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  Add *add = new Add;
  add->type = type;
  add->a = a;
  add->b = b;
  return add;
}

Expr Sub::make(Expr a, Expr b) {
  return Sub::make(a, b, max_expr_type(a, b));
}

Expr Sub::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";
  
  Sub *sub = new Sub;
  sub->type = type;
  sub->a = a;
  sub->b = b;
  return sub;
}

Expr Mul::make(Expr a, Expr b) {
  return Mul::make(a, b, max_expr_type(a, b));
}

Expr Mul::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";
  
  Mul *mul = new Mul;
  mul->type = type;
  mul->a = a;
  mul->b = b;
  return mul;
}

Expr Div::make(Expr a, Expr b) {
  return Div::make(a, b, max_expr_type(a, b));
}

Expr Div::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";
  
  Div *div = new Div;
  div->type = type;
  div->a = a;
  div->b = b;
  return div;
}

Expr Rem::make(Expr a, Expr b) {
  return Rem::make(a, b, max_expr_type(a, b));
}

Expr Rem::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";
  
  Rem *rem = new Rem;
  rem->type = type;
  rem->a = a;
  rem->b = b;
  return rem;
}

Expr Min::make(Expr a, Expr b) {
  return Min::make({a, b}, max_expr_type(a, b));
}

Expr Min::make(Expr a, Expr b, Datatype type) {
  return Min::make({a, b}, type);
}

Expr Min::make(std::vector<Expr> operands) {
  taco_iassert(operands.size() > 0);
  return Min::make(operands, operands[0].type());
}

Expr Min::make(std::vector<Expr> operands, Datatype type) {
  Min* min = new Min;
  min->operands = operands;
  min->type = type;
  return min;
}

Expr Max::make(Expr a, Expr b) {
  return Max::make(a, b, max_expr_type(a, b));
}

Expr Max::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  return Max::make({a, b}, type);
}

Expr Max::make(std::vector<Expr> operands) {
  taco_iassert(operands.size() > 0);
  return Max::make(operands, operands[0].type());
}

Expr Max::make(std::vector<Expr> operands, Datatype type) {
  Max* max = new Max;
  max->operands = operands;
  max->type = type;
  return max;
}

Expr BitAnd::make(Expr a, Expr b) {
  BitAnd *bitAnd = new BitAnd;
  bitAnd->type = UInt();
  bitAnd->a = a;
  bitAnd->b = b;
  return bitAnd;
}

Expr BitOr::make(Expr a, Expr b) {
  BitOr *bitOr = new BitOr;
  bitOr->type = UInt();
  bitOr->a = a;
  bitOr->b = b;
  return bitOr;
}

// Boolean binary ops
Expr Eq::make(Expr a, Expr b) {
  Eq *eq = new Eq;
  eq->type = Bool;
  eq->a = a;
  eq->b = b;
  return eq;
}

Expr Neq::make(Expr a, Expr b) {
  Neq *neq = new Neq;
  neq->type = Bool;
  neq->a = a;
  neq->b = b;
  return neq;
}

Expr Gt::make(Expr a, Expr b) {
  Gt *gt = new Gt;
  gt->type = Bool;
  gt->a = a;
  gt->b = b;
  return gt;
}

Expr Lt::make(Expr a, Expr b) {
  Lt *lt = new Lt;
  lt->type = Bool;
  lt->a = a;
  lt->b = b;
  return lt;
}

Expr Gte::make(Expr a, Expr b) {
  Gte *gte = new Gte;
  gte->type = Bool;
  gte->a = a;
  gte->b = b;
  return gte;
}

Expr Lte::make(Expr a, Expr b) {
  Lte *lte = new Lte;
  lte->type = Bool;
  lte->a = a;
  lte->b = b;
  return lte;
}

Expr Or::make(Expr a, Expr b) {
  Or *ornode = new Or;
  ornode->type = Bool;
  ornode->a = a;
  ornode->b = b;
  return ornode;
}

Expr And::make(Expr a, Expr b) {
  And *andnode = new And;
  andnode->type = Bool;
  andnode->a = a;
  andnode->b = b;
  return andnode;
}

Expr Cast::make(Expr a, Datatype newType) {
  Cast *cast = new Cast;
  cast->type = newType;
  cast->a = a;
  return cast;
}

Expr Call::make(const std::string& func, const std::vector<Expr>& args, 
                Datatype type) {
  Call *call = new Call;
  call->type = type;
  call->func = func;
  call->args = args;
  return call;
}

Expr makeConstructor(Datatype type, std::vector<Expr> args) {
  return ir::Call::make(type.getName(), args, type);
}

Expr getLogicalRegion(Expr e) {
  return ir::Call::make("get_logical_region", {e}, Auto);
};

Expr getIndexSpace(Expr e) {
  return ir::Call::make("get_index_space", {e}, Auto);
}

ir::Expr regionPrivilegeToExpr(RegionPrivilege rp) {
  switch (rp) {
    case RO:
      return readOnly;
    case RW:
      return readWrite;
    default:
      taco_iassert(false);
  }
  return Expr();
}

std::ostream& operator<<(std::ostream& os, const RegionPrivilege& rp) {
  switch (rp) {
    case RO:
      os << "RO"; break;
    case RW:
      os << "RW"; break;
    default:
      taco_iassert(false);
  }
  return os;
}

std::string accessorTypeString(ir::Expr expr) {
  auto op = expr.as<GetProperty>();
  taco_iassert(op != nullptr);

  auto sanitize = [](Datatype typ) {
    auto raw = util::toString(typ);
    std::stringstream result;
    for (auto c : raw) {
      switch (c) {
        case '<':
        case '>':
          result << '_';
          break;
        default:
          result << c;
          break;
      }
    }
    return result.str();
  };

  switch (op->property) {
    case TensorProperty::ValuesReadAccessor: {
      std::stringstream ss;
      ss << "AccessorRO" << op->type << op->mode;
      return ss.str();
    }
    case TensorProperty::ValuesWriteAccessor: {
      std::stringstream ss;
      ss << "AccessorRW" << op->type << op->mode;
      return ss.str();
    }
    case TensorProperty::ValuesReductionAccessor: {
      std::stringstream ss;
      ss << "AccessorReduce" << op->type << op->mode;
      return ss.str();
    }
    case TensorProperty::ValuesReductionNonExclusiveAccessor: {
      std::stringstream ss;
      ss << "AccessorReduceNonExcl" << op->type << op->mode;
      return ss.str();
    }
    case TensorProperty::IndicesAccessor: {
      std::stringstream ss;
      ss << "Accessor" << op->accessorArgs.priv << sanitize(op->accessorArgs.elemType) << op->accessorArgs.dim;
      return ss.str();
    }
    default:
      taco_iassert(false);
      return "";
  }
}

Expr makeCreateAccessor(ir::Expr acc, ir::Expr physReg, ir::Expr field) {
  return ir::Call::make("createAccessor<" + accessorTypeString(acc) + ">", {physReg, field}, Auto);
}


Expr MethodCall::make(Expr var, const std::string &func, const std::vector<Expr> &args, bool deref, Datatype type) {
  MethodCall *call = new MethodCall;
  call->type = type;
  call->var = var;
  call->func = func;
  call->args = args;
  call->deref = deref;
  return call;
}

Expr Deref::make(Expr var, Datatype type) {
  Deref *deref = new Deref;
  deref->type = type;
  deref->var = var;
  return deref;
}

Expr FieldAccess::make(Expr var, std::string field, bool isDeref, Datatype type) {
  FieldAccess* fa = new FieldAccess;
  fa->type = type;
  fa->var = var;
  fa->field = field;
  fa->isDeref = isDeref;
  return fa;
}

// Load
Expr Load::make(Expr arr) {
  return Load::make(arr, Literal::make((int64_t)0));
}

Expr Load::make(Expr arr, Expr loc) {
  taco_iassert(loc.type().isInt() || loc.type().isUInt() || loc.type() == Datatype::CppType)
      << "Can't load from a non-integer offset";
  Load *load = new Load;
  load->type = arr.type();
  load->arr = arr;
  load->loc = loc;
  return load;
}

// Malloc
Expr Malloc::make(Expr size) {
  taco_iassert(size.defined());
  Malloc *malloc = new Malloc;
  malloc->size = size;
  return malloc;
}

// Sizeof
Expr Sizeof::make(Type type) {
  Sizeof *szeof = new Sizeof;
  szeof->type = UInt64;
  szeof->sizeofType = type;
  return szeof;
}

// Block
Stmt Block::make() {
  return Block::make({});
}

static bool nop(const Stmt& stmt) {
  if (!stmt.defined()) return true;
  if (isa<Block>(stmt) && to<Block>(stmt)->contents.size() == 0) return true;
  return false;
}

Stmt Block::make(std::vector<Stmt> stmts) {
  Block *block = new Block;
  for (auto& stmt : stmts) {
    if (nop(stmt)) continue;
    block->contents.push_back(stmt);
  }
  return block;
}

Stmt Block::blanks(std::vector<Stmt> stmts) {
  Block *block = new Block;

  // Add first defined statement to result.
  size_t i = 0;
  for (; i < stmts.size(); i++) {
    Stmt stmt = stmts[i];
    if (nop(stmt)) continue;
    block->contents.push_back(stmt);
    break;
  }
  i++;

  // Add additional defined statements to result prefixed with a blank line.
  for (; i < stmts.size(); i++) {
    Stmt stmt = stmts[i];
    if (nop(stmt)) continue;
    block->contents.push_back(BlankLine::make());
    block->contents.push_back(stmt);
  }

  return block;
}

// Scope
Stmt Scope::make(Stmt scopedStmt) {
  taco_iassert(scopedStmt.defined());

  if (isa<Scope>(scopedStmt)) {
    return scopedStmt;
  }

  Scope *scope = new Scope;
  scope->scopedStmt = scopedStmt;
  return scope;
}

// Store to an array
Stmt Store::make(Expr arr, Expr loc, Expr data, bool use_atomics, ParallelUnit atomic_parallel_unit) {
  Store *store = new Store;
  store->arr = arr;
  store->loc = loc;
  store->data = data;
  store->use_atomics = use_atomics;
  store->atomic_parallel_unit = atomic_parallel_unit;
  return store;
}

// Conditional
Stmt IfThenElse::make(Expr cond, Stmt then) {
  return IfThenElse::make(cond, then, Stmt());
}

Stmt IfThenElse::make(Expr cond, Stmt then, Stmt otherwise) {
  taco_iassert(then.defined());
  taco_iassert(cond.defined());
  taco_iassert(cond.type().isBool()) << "Can only branch on boolean";

  IfThenElse* ite = new IfThenElse;
  ite->cond = cond;
  ite->then = then;
  ite->otherwise = otherwise;
  ite->then = Scope::make(then);
  ite->otherwise = otherwise.defined() ? Scope::make(otherwise) : otherwise;
  return ite;
}

Stmt Case::make(std::vector<std::pair<Expr,Stmt>> clauses, bool alwaysMatch) {
  for (auto clause : clauses) {
    taco_iassert(clause.first.type().isBool()) << "Can only branch on boolean";
  }

  std::vector<std::pair<Expr,Stmt>> scopedClauses;
  for (auto& clause : clauses) {
    scopedClauses.push_back({clause.first, Scope::make(clause.second)});
  }
  
  Case* cs = new Case;
  cs->clauses = scopedClauses;
  cs->alwaysMatch = alwaysMatch;
  return cs;
}

Stmt Switch::make(std::vector<std::pair<Expr,Stmt>> cases, Expr controlExpr) {
  for (auto switchCase : cases) {
    taco_iassert(switchCase.first.type().isUInt()) << "Can only switch on uint";
  }

  std::vector<std::pair<Expr,Stmt>> scopedCases;
  for (auto& switchCase : cases) {
    scopedCases.push_back({switchCase.first, Scope::make(switchCase.second)});
  }
  
  Switch* sw = new Switch;
  sw->cases = scopedCases;
  sw->controlExpr = controlExpr;
  return sw;
}

// For loop
Stmt For::make(Expr var, Expr start, Expr end, Expr increment, Stmt body,
  LoopKind kind, ParallelUnit parallel_unit, size_t unrollFactor, int vec_width,
  bool isTask, int taskID) {
  For *loop = new For;
  loop->var = var;
  loop->start = start;
  loop->end = end;
  loop->increment = increment;
  loop->contents = Scope::make(body);
  loop->kind = kind;
  loop->unrollFactor = unrollFactor;
  loop->vec_width = vec_width;
  loop->parallel_unit = parallel_unit;
  loop->isTask = isTask;
  loop->taskID = taskID;
  return loop;
}

// While loop
Stmt While::make(Expr cond, Stmt contents, LoopKind kind,
  int vec_width) {
  While *loop = new While;
  loop->cond = cond;
  loop->contents = Scope::make(contents);
  loop->kind = kind;
  loop->vec_width = vec_width;
  return loop;
}

// Function
Stmt Function::make(std::string name,
                    std::vector<Expr> outputs, std::vector<Expr> inputs,
                    Stmt body, Datatype returnType) {
  Function *func = new Function;
  func->name = name;
  func->body = Scope::make(body);
  func->inputs = inputs;
  func->outputs = outputs;
  func->returnType = returnType;
  return func;
}

std::pair<std::vector<Datatype>,Datatype> Function::getReturnType() const {
  struct InferReturnType : IRVisitor {
    std::pair<std::vector<Datatype>,Datatype> returnType;

    using IRVisitor::visit;

    void visit(const Yield* stmt) {
      if (returnType.second != Datatype()) {
        taco_iassert(returnType.second == stmt->val.type());
        taco_iassert(returnType.first.size() == stmt->coords.size());
        taco_iassert([&]() {
            for (size_t i = 0; i < stmt->coords.size(); ++i) {
              if (returnType.first[i] != stmt->coords[i].type()) {
                return false;
              }
            }
            return true;
          }()); 
        return;
      }
      for (auto& coord : stmt->coords) {
        returnType.first.push_back(coord.type());
      }
      returnType.second = stmt->val.type();
    }

    std::pair<std::vector<Datatype>,Datatype> inferType(Stmt stmt) {
      returnType = {{}, Datatype()};
      stmt.accept(this);
      return returnType;
    }
  };
  return InferReturnType().inferType(this);
}

// VarDecl
Stmt VarDecl::make(Expr var, Expr rhs) {
  taco_iassert(var.as<Var>())
    << "Can only declare a Var";
  VarDecl* decl = new VarDecl;
  decl->var = var;
  decl->rhs = rhs;
  return decl;
}

// VarAssign
Stmt Assign::make(Expr lhs, Expr rhs, bool use_atomics, ParallelUnit atomic_parallel_unit) {
  taco_iassert(lhs.as<Var>() || lhs.as<GetProperty>() || lhs.as<Load>() || lhs.as<FieldAccess>())
    << "Can only assign to a Var or GetProperty";
  Assign *assign = new Assign;
  assign->lhs = lhs;
  assign->rhs = rhs;
  assign->use_atomics = use_atomics;
  assign->atomic_parallel_unit = atomic_parallel_unit;
  return assign;
}

// Yield
Stmt Yield::make(std::vector<Expr> coords, Expr val) {
  for (auto coord : coords) {
    taco_iassert(coord.as<Var>()) << "Coordinates must be instances of Var";
  }
  Yield *yield = new Yield;
  yield->coords = coords;
  yield->val = val;
  return yield;
}

// Allocate
Stmt Allocate::make(Expr var, Expr num_elements, bool is_realloc, Expr old_elements, bool clear, LegionAllocPack pack) {
  taco_iassert(var.as<GetProperty>() ||
               (var.as<Var>() && var.as<Var>()->is_ptr)) <<
      "Can only allocate memory for a pointer-typed Var";
  taco_iassert(num_elements.type().isInt() || num_elements.type().isUInt()) <<
      "Can only allocate an integer-valued number of elements";
  Allocate* alloc = new Allocate;
  alloc->var = var;
  alloc->num_elements = num_elements;
  alloc->is_realloc = is_realloc;
  taco_iassert(!is_realloc || old_elements.ptr != NULL);
  alloc->old_elements = old_elements;
  alloc->clear = clear;
  alloc->pack = pack;
  return alloc;
}

Stmt makeLegionMalloc(Expr var, Expr numElements, Expr logicalRegion, Expr fieldID, Expr priv) {
  return Allocate::make(var,
    numElements,
    false /* is_realloc */,
    Expr() /* old_elements */,
    false /* clear */,
    Allocate::LegionAllocPack(logicalRegion, fieldID, priv)
  );
}

Stmt makeLegionRealloc(Expr var, Expr numElements, Expr logicalRegion, Expr oldPhysicalRegion, Expr fieldID, Expr priv) {
  return Allocate::make(var,
    numElements,
    true /* is_realloc */,
    oldPhysicalRegion,
    false /* clear */,
    Allocate::LegionAllocPack(logicalRegion, fieldID, priv)
  );
}

// Free
Stmt Free::make(Expr var) {
  taco_iassert(var.as<GetProperty>() ||
               (var.as<Var>() && var.as<Var>()->is_ptr)) <<
      "Can only allocate memory for a pointer-typed Var";
  Free* free = new Free;
  free->var = var;
  return free;
}

// Comment
Stmt Comment::make(std::string text) {
  Comment* comment = new Comment;
  comment->text = text;
  return comment;
}

// BlankLine
Stmt BlankLine::make() {
  return new BlankLine;
}

// Continue
Stmt Continue::make() {
  return new Continue;
}

// Break
Stmt Break::make() {
  return new Break;
}

Stmt Return::make(Expr val) {
  Return* ret = new Return;
  ret->ret = val;
  return ret;
}

// Print
Stmt Print::make(std::string fmt, std::vector<Expr> params) {
  Print* pr = new Print;
  pr->fmt = fmt;
  pr->params = params;
  return pr;
}

Stmt SideEffect::make(Expr e) {
  SideEffect* se = new SideEffect;
  se->e = e;
  return se;
}
  
Expr GetProperty::make(Expr tensor, TensorProperty property, int mode,
                       int index, std::string name) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = property;
  gp->mode = mode;
  gp->name = name;
  gp->index = index;
  
  //TODO: deal with the fact that some of these are pointers
  if (property == TensorProperty::Values)
    gp->type = tensor.type();
  else
    gp->type = Int();
  
  return gp;
}

Expr GetProperty::makeDenseLevelRun(Expr tensor, int index) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = TensorProperty::DenseLevelRun;
  gp->mode = -1;
  gp->index = index;
  const Var* tensorVar = tensor.as<Var>();
  gp->name = tensorVar->name + "_dense_run_" + util::toString(index);
  gp->type = IndexSpace;
  return gp;
}

Expr GetProperty::makeIndicesAccessor(Expr tensor, std::string nameBase, int mode, int index, AccessorArgs args) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = TensorProperty::IndicesAccessor;
  gp->mode = mode;
  gp->index = index;
  gp->name = nameBase + "_accessor";
  // TODO (rohany): I'm making accessors not an Auto type anymore, and instead making them
  //  the type of their element. I don't know if this is going to cause some complications
  //  in the end, but I'm keeping this here so that we know what has changed. This is important
  //  to do so that arithmetic operations on the results of accessors are typed correctly.
  // gp->type = Auto;
  gp->type = args.elemType;
  gp->accessorArgs = args;
  return gp;
}

GetProperty::Hashable GetProperty::toHashable() const {
  return Hashable {
    .tensor = this->tensor,
    .property = this->property,
    .mode = this->mode,
    .index = this->index,
  };
}

bool GetProperty::isAccessor() const {
  switch (this->property) {
    case TensorProperty::ValuesWriteAccessor:
    case TensorProperty::ValuesReadAccessor:
    case TensorProperty::ValuesReductionAccessor:
    case TensorProperty::ValuesReductionNonExclusiveAccessor:
    case TensorProperty::IndicesAccessor:
      return true;
    default:
      return false;
  }
}

bool GetProperty::isReductionAccessor() const {
  switch (this->property) {
    case TensorProperty::ValuesReductionAccessor:
    case TensorProperty::ValuesReductionNonExclusiveAccessor:
      return true;
    default:
      return false;
  }
}

bool operator<(const GetProperty::Hashable& a, const GetProperty::Hashable& b) {
  if (a.tensor != b.tensor) return a.tensor < b.tensor;
  if (a.property != b.property) return a.property < b.property;
  if (a.mode != b.mode) return a.mode < b.mode;
  return a.index < b.index;
}

bool operator==(const GetProperty::Hashable& a, const GetProperty::Hashable& b) {
  if (a.tensor != b.tensor) return false;
  if (a.property != b.property) return false;
  if (a.mode != b.mode) return false;
  if (a.index != b.index) return false;
  return true;
}

bool operator!=(const GetProperty::Hashable& a, const GetProperty::Hashable& b) {
  return !(a == b);
}

// Sort
Stmt Sort::make(std::vector<Expr> args) {
  Sort* sort = new Sort;
  sort->args = args;
  return sort;
}

Stmt PackTaskArgs::make(Expr var, int forTaskID, std::vector<Expr> prefixVars, std::vector<Expr> prefixExprs) {
  PackTaskArgs* pa = new PackTaskArgs;
  pa->var = var;
  pa->forTaskID = forTaskID;
  pa->prefixVars = prefixVars;
  pa->prefixExprs = prefixExprs;
  taco_iassert(prefixVars.size() == prefixExprs.size());
  return pa;
}

Stmt UnpackTensorData::make(std::vector<ir::Expr> regions, std::vector<ir::Expr> regionParents) {
  UnpackTensorData* utd = new UnpackTensorData;
  // Assert that all of the regions are GetProperty objects.
  for (const auto& it : regions) {
    taco_iassert(it.as<GetProperty>());
  }
  for (const auto& it : regionParents) {
    taco_iassert(it.as<GetProperty>());
  }
  utd->regions = regions;
  utd->regionParents = regionParents;
  taco_iassert(regions.size() == regionParents.size());
  return utd;
}

Stmt DeclareStruct::make(std::string name, std::vector<std::string> fields, std::vector<Datatype> fieldTypes) {
  DeclareStruct* ds = new DeclareStruct;
  ds->name = name;
  ds->fields = fields;
  ds->fieldTypes = fieldTypes;
  return ds;
}

// GetProperty
Expr GetProperty::make(Expr tensor, TensorProperty property, int mode) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = property;
  gp->mode = mode;
  
  //TODO: deal with the fact that these are pointers.
  if (property == TensorProperty::Values)
    gp->type = tensor.type();
  else
    gp->type = Int();
  
  const Var* tensorVar = tensor.as<Var>();
  switch (property) {
    case TensorProperty::Order:
      gp->name = tensorVar->name + "_order";
      break;
    case TensorProperty::Dimension:
      gp->name = tensorVar->name + util::toString(mode + 1) + "_dimension";
      break;
    case TensorProperty::ComponentSize:
      gp->name = tensorVar->name + "_csize";
      break;
    case TensorProperty::ModeOrdering:
      gp->name = tensorVar->name  + util::toString(mode + 1) + "_mode_ordering";
      break;
    case TensorProperty::ModeTypes:
      gp->name = tensorVar->name  + util::toString(mode + 1) + "_mode_type";
      break;
    case TensorProperty::Indices:
      taco_ierror << "Must provide both mode and index for the Indices property";
      break;
    case TensorProperty::IndicesParents:
      taco_ierror << "Must provide both mode and index for the Indices property";
      break;
    case TensorProperty::DenseLevelRun:
      taco_ierror << "Must provide index for the DenseLevelRun property";
      break;
    case TensorProperty::IndicesAccessor:
      taco_ierror << "Use specialized method for creating indices accessors";
      break;
    case TensorProperty::Values:
      gp->name = tensorVar->name + "_vals";
      break;
    case TensorProperty::ValuesSize:
      gp->name = tensorVar->name + "_vals_size";
      break;
    case TensorProperty::ValuesParent:
      gp->name = tensorVar->name + "_vals_parent";
      break;
    case TensorProperty::IndexSpace:
      gp->name = tensorVar->name + "_index_space";
      // This has an index space type (TODO: can I write it here without the dimensions?).
      gp->type = Datatype(Datatype::CppType);
      break;
    case TensorProperty::ValuesReadAccessor:
      gp->type = tensor.type();
      gp->name = tensorVar->name + "_vals_ro_accessor";
      break;
    case TensorProperty::ValuesWriteAccessor:
      gp->type = tensor.type();
      gp->name = tensorVar->name + "_vals_rw_accessor";
      break;
    case TensorProperty::ValuesReductionAccessor:
      gp->type = tensor.type();
      gp->name = tensorVar->name + "_vals_red_accessor";
      break;
    case TensorProperty::ValuesReductionNonExclusiveAccessor:
      gp->type = tensor.type();
      gp->name = tensorVar->name + "_vals_red_accessor_non_excl";
      break;
  }
  
  return gp;
}
  
// visitor methods
template<> void ExprNode<Literal>::accept(IRVisitorStrict *v)
    const { v->visit((const Literal*)this); }
template<> void ExprNode<Var>::accept(IRVisitorStrict *v)
    const { v->visit((const Var*)this); }
template<> void ExprNode<Symbol>::accept(IRVisitorStrict *v)
    const { v->visit((const Symbol*)this); }
template<> void ExprNode<Neg>::accept(IRVisitorStrict *v)
    const { v->visit((const Neg*)this); }
template<> void ExprNode<Sqrt>::accept(IRVisitorStrict *v)
    const { v->visit((const Sqrt*)this); }
template<> void ExprNode<Add>::accept(IRVisitorStrict *v)
    const { v->visit((const Add*)this); }
template<> void ExprNode<Sub>::accept(IRVisitorStrict *v)
    const { v->visit((const Sub*)this); }
template<> void ExprNode<Mul>::accept(IRVisitorStrict *v)
    const { v->visit((const Mul*)this); }
template<> void ExprNode<Div>::accept(IRVisitorStrict *v)
    const { v->visit((const Div*)this); }
template<> void ExprNode<Rem>::accept(IRVisitorStrict *v)
    const { v->visit((const Rem*)this); }
template<> void ExprNode<Min>::accept(IRVisitorStrict *v)
    const { v->visit((const Min*)this); }
template<> void ExprNode<Max>::accept(IRVisitorStrict *v)
    const { v->visit((const Max*)this); }
template<> void ExprNode<BitAnd>::accept(IRVisitorStrict *v)
    const { v->visit((const BitAnd*)this); }
template<> void ExprNode<BitOr>::accept(IRVisitorStrict *v)
    const { v->visit((const BitOr*)this); }
template<> void ExprNode<Eq>::accept(IRVisitorStrict *v)
    const { v->visit((const Eq*)this); }
template<> void ExprNode<Neq>::accept(IRVisitorStrict *v)
    const { v->visit((const Neq*)this); }
template<> void ExprNode<Gt>::accept(IRVisitorStrict *v)
    const { v->visit((const Gt*)this); }
template<> void ExprNode<Lt>::accept(IRVisitorStrict *v)
    const { v->visit((const Lt*)this); }
template<> void ExprNode<Gte>::accept(IRVisitorStrict *v)
    const { v->visit((const Gte*)this); }
template<> void ExprNode<Lte>::accept(IRVisitorStrict *v)
    const { v->visit((const Lte*)this); }
template<> void ExprNode<And>::accept(IRVisitorStrict *v)
    const { v->visit((const And*)this); }
template<> void ExprNode<Or>::accept(IRVisitorStrict *v)
    const { v->visit((const Or*)this); }
template<> void ExprNode<Cast>::accept(IRVisitorStrict *v)
    const { v->visit((const Cast*)this); }
template<> void ExprNode<Call>::accept(IRVisitorStrict *v)
    const { v->visit((const Call*)this); }
template<> void ExprNode<MethodCall>::accept(IRVisitorStrict *v)
const { v->visit((const MethodCall*)this); }
template<> void StmtNode<IfThenElse>::accept(IRVisitorStrict *v)
    const { v->visit((const IfThenElse*)this); }
template<> void StmtNode<Case>::accept(IRVisitorStrict *v)
    const { v->visit((const Case*)this); }
template<> void StmtNode<Switch>::accept(IRVisitorStrict *v)
    const { v->visit((const Switch*)this); }
template<> void ExprNode<Load>::accept(IRVisitorStrict *v)
    const { v->visit((const Load*)this); }
template<> void ExprNode<Malloc>::accept(IRVisitorStrict *v)
    const { v->visit((const Malloc*)this); }
template<> void ExprNode<Sizeof>::accept(IRVisitorStrict *v)
    const { v->visit((const Sizeof*)this); }
template<> void StmtNode<Store>::accept(IRVisitorStrict *v)
    const { v->visit((const Store*)this); }
template<> void StmtNode<For>::accept(IRVisitorStrict *v)
    const { v->visit((const For*)this); }
template<> void StmtNode<While>::accept(IRVisitorStrict *v)
    const { v->visit((const While*)this); }
template<> void StmtNode<Block>::accept(IRVisitorStrict *v)
    const { v->visit((const Block*)this); }
template<> void StmtNode<Scope>::accept(IRVisitorStrict *v)
    const { v->visit((const Scope*)this); }
template<> void StmtNode<Function>::accept(IRVisitorStrict *v)
    const { v->visit((const Function*)this); }
template<> void StmtNode<VarDecl>::accept(IRVisitorStrict *v)
    const { v->visit((const VarDecl*)this); }
template<> void StmtNode<Assign>::accept(IRVisitorStrict *v)
    const { v->visit((const Assign*)this); }
template<> void StmtNode<Yield>::accept(IRVisitorStrict *v)
    const { v->visit((const Yield*)this); }
template<> void StmtNode<Allocate>::accept(IRVisitorStrict *v)
    const { v->visit((const Allocate*)this); }
template<> void StmtNode<Free>::accept(IRVisitorStrict *v)
    const { v->visit((const Free*)this); }
template<> void StmtNode<Comment>::accept(IRVisitorStrict *v)
    const { v->visit((const Comment*)this); }
template<> void StmtNode<BlankLine>::accept(IRVisitorStrict *v)
    const { v->visit((const BlankLine*)this); }
template<> void StmtNode<Continue>::accept(IRVisitorStrict *v)
  const { v->visit((const Continue*)this); }
template<> void StmtNode<Print>::accept(IRVisitorStrict *v)
    const { v->visit((const Print*)this); }
template<> void ExprNode<GetProperty>::accept(IRVisitorStrict *v)
    const { v->visit((const GetProperty*)this); }
template<> void StmtNode<Sort>::accept(IRVisitorStrict *v)
  const { v->visit((const Sort*)this); }
template<> void StmtNode<Break>::accept(IRVisitorStrict *v)
  const { v->visit((const Break*)this); }
template<> void ExprNode<Deref>::accept(IRVisitorStrict *v)
const { v->visit((const Deref*)this); }
template<> void ExprNode<FieldAccess>::accept(IRVisitorStrict *v)
const { v->visit((const FieldAccess*)this); }
template<> void StmtNode<SideEffect>::accept(IRVisitorStrict *v)
const { v->visit((const SideEffect*)this); }
template<> void StmtNode<PackTaskArgs>::accept(IRVisitorStrict *v)
const { v->visit((const PackTaskArgs*)this); }
template<> void StmtNode<Return>::accept(IRVisitorStrict *v)
const { v->visit((const Return*)this); }
template<> void StmtNode<UnpackTensorData>::accept(IRVisitorStrict *v)
const { v->visit((const UnpackTensorData*)this); }
template<> void StmtNode<DeclareStruct>::accept(IRVisitorStrict *v)
const { v->visit((const DeclareStruct*)this); }

// printing methods
std::ostream& operator<<(std::ostream& os, const Stmt& stmt) {
  if (!stmt.defined()) return os << "Stmt()" << std::endl;
  IRPrinter printer(os);
  stmt.accept(&printer);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Expr& expr) {
  if (!expr.defined()) return os << "Expr()";
  IRPrinter printer(os);
  expr.accept(&printer);
  return os;
}

// Definitions of common legion expressions.
ir::Expr ctx = ir::Symbol::make("ctx");
ir::Expr runtime = ir::Symbol::make("runtime");
ir::Expr AffineProjectionBot = ir::Symbol::make("AffineProjection::BOT");
ir::Expr GPUFBMem = ir::Symbol::make("Legion::Memory::Kind::GPU_FB_MEM");
ir::Expr disjointPart = ir::Symbol::make("LEGION_DISJOINT_COMPLETE_KIND");
ir::Expr aliasedPart = ir::Symbol::make("LEGION_ALIASED_COMPLETE_KIND");
ir::Expr computePart = ir::Symbol::make("LEGION_COMPUTE_KIND");
ir::Expr readOnly = ir::Symbol::make("READ_ONLY");
ir::Expr readWrite = ir::Symbol::make("READ_WRITE");
ir::Expr fidVal = ir::Symbol::make("FID_VAL");

} // namespace ir
} // namespace taco
