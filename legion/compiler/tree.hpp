#ifndef __TREE
#define __TREE
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <string.h>
#include <assert.h>

using namespace Legion;
using namespace Legion::Mapping;

template <typename T1, typename T2>
struct PairHash
{
  using VAL = std::pair<T1, T2>;
  std::size_t operator()(VAL const& pair) const noexcept
  {
    return std::hash<T1>{}(pair.first) << 1 ^ std::hash<T2>{}(pair.second);
  }
};

enum ProcessorEnum
{
	CPU,
	GPU,
	IO,
	PY,
	PROC,
	OMP,
	ALLPROC,
};

const char* ProcessorEnumName[] =
{
	"CPU",
	"GPU",
	"IO",
	"PY",
	"PROC",
	"OMP",
	"ALLPROC",
};

enum MemoryEnum
{
	SYSMEM,
	FBMEM,
	RDMEM,
	ZCMEM,
	SOCKMEM,
	ALLMEM,
};

const char* MemoryEnumName[] =
{
	"SYSMEM",
	"FBMEM",
	"RDMEM",
	"ZCMEM",
	"SOCKMEM",
	"ALLMEM",
};

enum ArgTypeEnum
{
	INT,
	BOOL,
	IPOINT,
	ISPACE,
	MSPACE,
};

const char* ArgTypeName[] =
{
	"INT",
	"BOOL",
	"IPOINT",
	"ISPACE",
	"MSPACE",
};

enum BinOpEnum
{
	PLUS,
	MINUS,
	MULTIPLY,
	DIVIDE,
	MOD,
	BIGGER,
	SMALLER,
	GE,
	LE,
	EQ,
	NEQ,
	OR,
	AND,
};

const char* BinOpName[] = 
{
	"+",
	"-",
	"*",
	"/",
	"%",
	">",
	"<",
	">=",
	"<=",
	"==",
	"!=",
	"||",
	"&&",
};


enum APIEnum
{
	SIZE,
	SPLIT,
	MERGE,
	SWAP,
	SLICE,
	REVERSE,
	BALANCE_SPLIT,
	VOLUME,
	HAS,
};

const char* APIName[] =
{
	"SIZE",
	"SPLIT",
	"MERGE",
	"SWAP",
	"SLICE",
	"REVERSE",
	"BALANCE_SPLIT",
	"VOLUME",
	"HAS",
};

enum NodeType
{
	ProgramType,
	StmtType,
	ProcLstType,
	ProcType,
	MemLstType,
	MemType,
	ProcCustomType,
	RegionCustomType,
	ArgTypeType,
	AssignType,
	IndexTaskMapType,
	ArgType,
	ArgLstType,
	FuncDefType,
	BinaryExprType,
	IdentifierExprType,
	IdentifierLstType,
	IntValType,
	BoolValType,
	TupleExprType,
	SliceExprType,
	FuncInvokeType,
	IndexExprType,
	NegativeExprType,
	ExclamationType,
	APIType,
	TenaryExprType,
	UnpackExprType,
	PrintType,
	PrintArgsType,
	ReturnType,
	FuncStmtsType,
	ObjectInvokeType,
	MSpaceType,
	TupleIntType,
};

const char* NodeTypeName[] =
{
  "ProgramType",
  "StmtType",
  "ProcLstType",
  "ProcType",
  "MemLstType",
  "MemType",
  "ProcCustomType",
  "RegionCustomType",
  "ArgTypeType",
  "AssignType",
  "IndexTaskMapType",
  "ArgType",
  "ArgLstType",
  "FuncDefType",
  "BinaryExprType",
  "IdentifierExprType",
  "IdentifierLstType",
  "IntValType",
  "BoolValType",
  "TupleExprType",
  "SliceExprType",
  "FuncInvokeType",
  "IndexExprType",
  "NegativeExprType",
  "ExclamationType",
  "APIType",
  "TenaryExprType",
  "UnpackExprType",
  "PrintType",
  "PrintArgsType",
  "ReturnType",
  "FuncStmtsType",
  "ObjectInvokeType",
  "MSpaceType",
  "TupleIntType",
};

class Node
{
public:
	NodeType type;
	virtual ~Node() {}
	virtual void print() {};
	virtual Node* run() 
	{
		std::cout << "Run method TBD:" << NodeTypeName[this->type] << std::endl;
		return NULL;
	};
};

class ExprNode : public Node
{
public:
	
	ExprNode() {}
	virtual ~ExprNode() {};
	virtual void print() = 0;
	virtual Node* run()
	{
		std::cout << "Run method TBD" << NodeTypeName[this->type] << std::endl;
		return NULL;
	}
};

class StmtNode: public Node
{
public:
	StmtNode() { type = StmtType; }
};

class ProgramNode : public Node
{
// Program: Stmt+  
public:
	std::vector<StmtNode*> stmt_list;

	ProgramNode() { type = ProgramType; }
	void print();
	Node* run()
	{
		for (size_t i = 0; i < stmt_list.size(); i++)
		{
			stmt_list[i]->run();
		}
		return NULL;
	}
};

ProgramNode* root;
std::unordered_map<std::string, Node*> global_symbol;
std::stack<std::unordered_map<std::string, Node*>> local_symbol;


class FuncStmtsNode : public Node
{
public:
	std::vector<StmtNode*> stmtlst;
	FuncStmtsNode()
	{
		type = FuncStmtsType;
	}
	void print()
	{
		printf("FuncStmtsNode\n");
		for (size_t i = 0; i < stmtlst.size(); i++)
		{
			stmtlst[i]->print();
		}
	}
};

class APINode : public Node
{
public:
	APIEnum api;
	APINode(APIEnum x)
	{
		type = APIType;
		api = x;
	}
	void print() { printf("APINode %s\n", APIName[api]); }
};

class ProcNode : public ExprNode
{
public:
	ProcessorEnum proc_type;

	ProcNode(ProcessorEnum x) { type = ProcType; proc_type = x;}
	void print() { printf("ProcNode: %s\n", ProcessorEnumName[proc_type]); }
	Node* run() { return this; }
};

class MemNode : public ExprNode
{
public:
	MemoryEnum mem_type;

	MemNode(MemoryEnum x) { type = MemType; mem_type = x; }
	void print() { printf("MemNode: %s\n", MemoryEnumName[mem_type]); }
	Node* run() { return this; }
};

class ProcLstNode : public Node
{
public:
	std::vector<ProcessorEnum> proc_type_lst;

	ProcLstNode() { type = ProcLstType; }
	void print();
};

class MemLstNode : public Node
{
public:
	std::vector<MemoryEnum> mem_type_lst;

	MemLstNode() { type = MemLstType; }
	void print();
};

class ProcCustomNode : public StmtNode
{
public:
	std::string taskname;
	std::vector<ProcessorEnum> proc_types;

	ProcCustomNode(const char* x, const ProcLstNode* y)
	{
		type = ProcCustomType;
		taskname = std::string(x);
		proc_types = y->proc_type_lst;
	}
	void print() { printf("ProcCustomNode %s %s\n", taskname.c_str(), ProcessorEnumName[proc_types[0]]); }
	Node* run();
};

class RegionCustomNode : public StmtNode
{
public:
	std::string taskname;
	std::string region_name;
	ProcessorEnum processor_type;
	std::vector<MemoryEnum> mem_types;

	RegionCustomNode(const char* x, const char* y, ProcNode* z, const MemLstNode* a)
	{
		type = RegionCustomType;
		taskname = std::string(x);
		region_name = std::string(y);
		processor_type = z->proc_type;
		mem_types = a->mem_type_lst;
	}
	void print()
	{
		printf("RegionCustomNode %s %s %s: %s as first\n", 
			taskname.c_str(), region_name.c_str(), ProcessorEnumName[processor_type],
			MemoryEnumName[mem_types[0]]);
	}
	Node* run();
};


class ArgTypeNode : public Node
{
public:
	ArgTypeEnum argtype;

	ArgTypeNode(ArgTypeEnum x) { type = ArgTypeType; argtype = x; }
	void print() { printf("ArgTypeNode %s\n", ArgTypeName[argtype]); }
};

class AssignNode : public StmtNode
{
public:
	std::string var_name;
	Node* right_node;

	AssignNode(const char* x, Node* y) { type = AssignType; var_name = std::string(x); right_node = y; }
	void print()
	{
		printf("AssignNode: %s\n", var_name.c_str());
		right_node->print();
	}
	Node* run()
	{
		Node* simplified = right_node->run();
		if (!(simplified->type == IntValType || simplified->type == BoolValType ||\
			simplified->type == MSpaceType || simplified->type == TupleExprType))
		{
			std::cout << "Cannot Assign " << NodeTypeName[simplified->type] << std::endl;
			assert(false);
		}
		if (local_symbol.size() >= 1)
		{
			local_symbol.top().insert({var_name, simplified});
		}
		else
		{
			global_symbol.insert({var_name, simplified});
		}
		// printf("AssignNode: %s inserted\n", var_name.c_str());
		// simplified->print();
		return NULL;
	}
};


class BinaryExprNode : public ExprNode
{
public:
	ExprNode* left;
	ExprNode* right;
	BinOpEnum op;
	BinaryExprNode(ExprNode* x, BinOpEnum a, ExprNode* y)
	{
		type = BinaryExprType;
		left = x; right = y; op = a;
	}
	void print()
	{
		printf("BinaryExprNode, %s\n", BinOpName[op]);
		printf("left\n"); left->print();
		printf("right\n"); right->print();
		printf("---------\n");
	}
	Node* run();
};

class IdentifierLstNode : public Node
{
public:
	std::vector<std::string> idlst;
	IdentifierLstNode(const char* x, const char* y)
	{
		type = IdentifierLstType;
		idlst.push_back(std::string(x));
		idlst.push_back(std::string(y));
	}
	void append(const char* x)
	{
		idlst.push_back(std::string(x));
	}
};

class IdentifierExprNode : public ExprNode
{
public:
	std::string name;

	IdentifierExprNode(const char* x)
	{
		type = IdentifierExprType;
		name = std::string(x);
	}
	void print()
	{
		printf("IdentifierExprNode %s\n", name.c_str());
	}
	Node* run()
	{
		if (local_symbol.size() > 0)
		{
			if (local_symbol.top().count(name) > 0)
			{
				return local_symbol.top().at(name);
			}
		}
		if (global_symbol.count(name) == 0)
		{
			std::cout << name << " not found" << std::endl;
			assert(false);
		}
		return global_symbol.at(name);
	}
};

class IntValNode : public ExprNode
{
public:
	int intval;

	IntValNode(int x) { type = IntValType; intval = x; }
	void print()
	{
		printf("%d", intval);
	}
	IntValNode* run()
	{
		return this;
	}
	IntValNode* negate()
	{
		return new IntValNode(-intval);
	}
	Node* binary_op(IntValNode* rt, BinOpEnum op);
};

class TupleIntNode : public ExprNode
{
public:
	std::vector<int> tupleint;
	TupleIntNode(std::vector<int> x) { type = TupleIntType; tupleint = x;}
	void print() 
	{
		for (size_t i = 0; i < tupleint.size(); i++) 
		{
			printf("%d", tupleint[i]);
			if (i != tupleint.size() - 1)
			{
				printf(",");
			}
		}
	}
	TupleIntNode* run() { return this; }
	TupleIntNode* negate();
	TupleIntNode* slice(int a, int b);
	TupleIntNode* binary_op(TupleIntNode* rt, BinOpEnum op);
	IntValNode* at(int x);
	IntValNode* at(IntValNode* x);
};

class BoolValNode : public ExprNode
{
public:
	bool boolval;

	BoolValNode(bool x) { type = BoolValType; boolval = x; }
	void print()
	{
		printf("BoolValNode: %s\n", boolval ? "true" : "false");
	}
	Node* run() { return this; }
	BoolValNode* binary_op(BoolValNode* rt, BinOpEnum op);
};

class IndexTaskMapNode : public StmtNode
{
	std::vector<std::string> task_name;
	std::string func_name;

public:
	IndexTaskMapNode(const char* x, const char* y)
	{
		type = IndexTaskMapType;
		task_name.push_back(std::string(x));
		func_name = std::string(y);
	}
	IndexTaskMapNode(IdentifierLstNode* x, const char* y)
	{
		type = IndexTaskMapType;
		task_name = x->idlst;
		func_name= std::string(y);
	}
	void print() { printf("IndexTaskMapNode (%s,...), %s\n", task_name[0].c_str(), func_name.c_str()); }
	Node* run();
};

class ArgNode : public Node
{
public:
	ArgTypeEnum argtype;
	std::string argname;
	ArgNode(ArgTypeNode* x, const char* y)
	{
		type = ArgType;
		argtype = x->argtype;
		argname = std::string(y);
	}
};

class ArgLstNode : public Node
{
public:
	std::vector<ArgNode*> arg_lst;
	ArgLstNode()
	{
		type = ArgLstType;
	}
	void print()
	{
		printf("ArgLstNode: %ld\n", arg_lst.size());
		for (size_t i = 0; i < arg_lst.size(); i++)
		{
			printf("Arg %ld: %s %s\n", i, ArgTypeName[arg_lst[i]->argtype], arg_lst[i]->argname.c_str());
		}
	}
};

class FuncDefNode: public StmtNode
{
public:
	std::string func_name;
	ArgLstNode* func_args;
	FuncStmtsNode* func_stmts;
	FuncDefNode(const char* x, ArgLstNode* y, FuncStmtsNode* z)
	{
		type = FuncDefType;
		func_name = std::string(x);
		func_args = y;
		func_stmts = z;
	}
	void print()
	{
		printf("FuncDefNode %s\n", func_name.c_str());
		func_args->print();
		func_stmts->print();
	}
	Node* run()
	{
		global_symbol.insert({func_name, this});
		return this;
	}
	Node* invoked();
};


class TupleExprNode : public ExprNode
{
public:
	std::vector<Node*> exprlst;
	TupleExprNode() { type = TupleExprType; }
	TupleExprNode(Node* a)
	{
		type = TupleExprType;
		exprlst.push_back(a);
	}
	TupleExprNode(std::vector<Node*> v)
	{
		type = TupleExprType;
		exprlst = v;
	}
	void print()
	{
		for (size_t i = 0; i < exprlst.size(); i++)
		{
			exprlst[i]->print();
			if (i != exprlst.size() - 1)
			{
				printf(",");
			}
		}
	}
	Node* run();
	ExprNode* Convert2TupleInt(); // if all nodes in exprlst are IntValNode(IntValType), then can be converted
	TupleExprNode* negate();
	TupleExprNode* slice(int a, int b);
	TupleExprNode* binary_op(TupleExprNode* right_op, BinOpEnum op);
	Node* at(int x);
	Node* at(IntValNode* x);
	IntValNode* one_int_only();
};

class SliceExprNode : public ExprNode
{
public:
	ExprNode* left;
	ExprNode* right;
	SliceExprNode(ExprNode* l, ExprNode* r)
	{
		type = SliceExprType;
		left = l;
		right = r;
	}
	void print()
	{
		printf("SliceExprNode\n");
		left->print();
		right->print();
	}
	Node* run()
	{
		return this;
	}
};

class ForTupleExprNode : public ExprNode
{
public:
	ExprNode* expr;
	std::string identifier;
	ExprNode* range;
	ForTupleExprNode(ExprNode* x, const char* y, ExprNode* z)
	{
		expr = x;
		identifier = std::string(y);
		if ((z->type != TupleExprType && z->type != TupleIntType))
		{
			std::cout << "For-comprehension must itererate over TupleExpr or TupleInt" << std::endl;
			assert(false);
		}
		range = z;
	}
	void print()
	{
		printf("ForTupleExprNode\n");
		expr->print();
		std::cout << identifier << std::endl;
		range->print();
	}
	Node* run();
};

class FuncInvokeNode : public ExprNode
{
public:
	ExprNode* func_node;
	TupleExprNode* args_node;
	FuncInvokeNode(ExprNode* x, TupleExprNode* y)
	{
		type = FuncInvokeType;
		func_node = x;
		args_node = y;
	}
	void print()
	{
		printf("FuncInvokeNode\n");
		func_node->print();
		args_node->print();
	}
	// Node* broadcast(); // deal with !()
	Node* run();
};

class ObjectInvokeNode : public ExprNode
{
public:
	ExprNode* obj;
	APIEnum api;
	ObjectInvokeNode(ExprNode* x, APINode* y)
	{
		type = ObjectInvokeType;
		obj = x;
		api = y->api;
	}
	void print()
	{
		printf("ObjectInvokeNode\n");
		obj->print();
		printf("%s\n", APIName[api]);
	}
	Node* run();
};

class IndexExprNode : public ExprNode
{
public:
	ExprNode* tuple;
	ExprNode* index;
	IndexExprNode(ExprNode* x, ExprNode* y)
	{
		type = IndexExprType;
		tuple = x;
		index = y;
	}
	IndexExprNode(ExprNode* x, TupleExprNode* y)
	{
		type = IndexExprType;
		tuple = x; // must be a machine model (identifier)
		index = y;
	}
	void print()
	{
		printf("IndexExprNode\n");
		tuple->print();
		index->print();
	}
	Node* run();
};

class NegativeExprNode : public ExprNode
{
public:
	ExprNode* neg;
	NegativeExprNode(ExprNode* x)
	{
		type = NegativeExprType;
		neg = x;
	}
	void print()
	{
		printf("NegativeExprNode\n");
		neg->print();
	}
	Node* run();
};

/*
class ExclamationNode : public ExprNode
{
public:
	ExprNode* expr;
	ExclamationNode(ExprNode* x)
	{
		type = ExclamationType;
		expr = x;
	}
	void print()
	{
		printf("ExclamationNode\n");
		expr->print();
	}
};
*/

class TenaryExprNode : public ExprNode
{
public:
	ExprNode* bool_exp;
	ExprNode* true_exp;
	ExprNode* false_expr;
	TenaryExprNode(ExprNode* x, ExprNode* y, ExprNode* z)
	{
		type = TenaryExprType;
		bool_exp = x; true_exp = y; false_expr = z;
	}
	void print()
	{
		printf("TenaryExprNode\n");
		bool_exp->print();
		true_exp->print();
		false_expr->print();
	}
	Node* run()
	{
		Node* simplified_node = bool_exp->run();
		assert(simplified_node->type == BoolValType);
		BoolValNode* bool_node = (BoolValNode*) simplified_node;
		if (bool_node->boolval)
		{
			return true_exp->run();
		}
		else
		{
			return false_expr->run();
		}
	}
};

class UnpackExprNode : public ExprNode
{
public:
	ExprNode* expr;

	UnpackExprNode(ExprNode* x) { type = UnpackExprType; expr = x; }
	void print()
	{
		printf("UnpackExprNode\n");
		expr->print();
	}
	Node* run()
	{
		Node* res = expr->run();
		assert(res->type == TupleExprType);
		return new UnpackExprNode((ExprNode*) res);
	}
};


class PrintArgsNode : public Node
{
public:
	std::vector<ExprNode*> printargs;

	PrintArgsNode() { type = PrintArgsType; }
};

class PrintNode : public StmtNode
{
public:
	std::string format_string;
	std::vector<ExprNode*> printargs;
	PrintNode(const char* x, PrintArgsNode* y)
	{
		type = PrintType;
		format_string = std::string(x);
		printargs = y->printargs;
	}
	Node* run();
	void print()
	{
		printf("PrintNode\n");
	}
};

class ReturnNode : public StmtNode
{
public:
	ExprNode* ret_expr;
	ReturnNode(ExprNode* x)
	{
		type = ReturnType;
		ret_expr = x;
	}
	void print() { printf("ReturnNode\n"); ret_expr->print(); }
	Node* run()
	{
		Node* result = ret_expr->run();
		assert(result->type == TupleExprType || result->type == TupleIntType || \
			result->type == IntValType || result->type == BoolValType);
		// std::cout << "return Node executed" << std::endl;
		// result->print();
		return result;
	}
};


class ConstraintsNode : public Node
{
public:
	bool reverse;
	bool aos;
	bool compact;
	bool align;
	BinOpEnum align_op;
	int align_int;
	ConstraintsNode()
	{
		reverse = false;
		aos = true;
		compact = false;
		align = false;
		align_op = PLUS;
		align_int = 0;
	}
	void update(const char* x)
	{
		if (!strcmp(x, "reverse"))
		{
			reverse = true;
		}
		else if (!strcmp(x, "positive"))
		{
			reverse = false;
		}
		else if (!strcmp(x, "aos"))
		{
			aos = true;
		}
		else if (!strcmp(x, "soa"))
		{
			aos = false;
		}
		else if (!strcmp(x, "compact"))
		{
			compact = true;
		}
		else
		{
			std::cout << "unsupported update in ConstraintsNode" << std::endl;
			assert(false);
		}
	}
	void print() { std::cout << "ConstraintsNode" << std::endl; }
	void update(BinOpEnum x, int y)
	{
		align = true;
		align_op = x;
		align_int = y;
	}
	Node* run() { return NULL; }
};

class LayoutCustomNode : public StmtNode
{
	//Layout *taskname *region_name *memory_type
public:
	std::string task_name;
	std::string region_name;
	MemoryEnum mem_type;
	ConstraintsNode* constraint;
	LayoutCustomNode(const char* x0, const char* x1, MemNode* x2, ConstraintsNode* x3)
	{
		task_name = std::string(x0);
		region_name = std::string(x1);
		mem_type = x2->mem_type;
		constraint = x3;
	}
	void print() { printf("LayoutCustomNode\n"); }
	Node* run();
};

class MemoryCollectNode : public StmtNode
{
	// CollectMemory $task_name $region_name
public:
	std::string task_name;
	std::string region_name;
	MemoryCollectNode(const char* x0, const char* x1)
	{
		task_name = std::string(x0);
		region_name = std::string(x1);
	}
	void print() { printf("MemoryCollectNode %s %s", task_name.c_str(), region_name.c_str()); }
	Node* run();
};

class InstanceLimitNode : public StmtNode
{
	// InstanceLimit Foo CPU 5
public:
	std::string task_name;
	ProcessorEnum proc_type;
	int num;
	InstanceLimitNode(const char* task_name_, ProcNode* proc, int num_)
	{
		task_name = std::string(task_name_);
		proc_type = proc->proc_type;
		num = num_;
	}
	void print() { printf("InstanceLimitNode\n"); }
	Node* run();
};

class MSpace;

class Tree2Legion
{
public:
	Tree2Legion() {}
	Tree2Legion(std::string filename);
	static std::unordered_map<std::string, std::vector<Processor::Kind>> task_policies;

	using HashFn1 = PairHash<std::string, std::string>;
  	static std::unordered_map<std::pair<std::string, std::string>, 
		std::unordered_map<Processor::Kind, std::vector<Memory::Kind>>, HashFn1> region_policies;
	
	static std::unordered_map<std::pair<std::string, std::string>,
		std::unordered_map<Memory::Kind, ConstraintsNode*>, HashFn1> 
		layout_constraints;
	
	static std::unordered_set<std::pair<std::string, std::string>, HashFn1> memory_collect;
	
	static std::unordered_map<std::string, std::unordered_map<Processor::Kind, int>> task2limit;
	
	// static std::unordered_map<std::string, MSpace*> task2mspace;
	static std::unordered_map<std::string, FuncDefNode*> task2func;

	bool should_fall_back(std::string task_name)
	{
		if (task2func.count(task_name) > 0)
		{
			return false;
		}
		if (task2func.count("*") > 0)
		{
			return false;
		}
		// std::cout << task_name << " will fallback for sharding/slicing, warning!" << std::endl;
		return true;
	}

	bool should_collect_memory(std::string task_name, std::vector<std::string> region_name)
	{
		if (memory_collect.size() == 0)
		{
			return false;
		}
		if (memory_collect.count({"*", "*"}) > 0)
		{
			return true;
		}
		if (memory_collect.count({task_name, "*"}) > 0)
		{
			return true;
		}
		for (auto &str: region_name)
		{
			if (memory_collect.count({task_name, str}) > 0)
				return true;
			if (memory_collect.count({"*", str}) > 0)
				return true;
		}
		return false;
	}
	  
	void print()
	{
		std::cout << "I am invoked!" << std::endl;
	}

	bool prerun_validate(std::string task);
	std::vector<int> run(std::string task, std::vector<int> x, std::vector<int> point_space);
	std::vector<Memory::Kind> query_memory_policy(std::string task_name, std::string region_name, Processor::Kind proc_kind)
	{
		std::pair<std::string, std::string> key = {task_name, region_name};
		if (region_policies.count(key) > 0)
		{
			std::unordered_map<Processor::Kind, std::vector<Memory::Kind> > value = region_policies.at(key);
			if (value.count(proc_kind) > 0)
			{
				return value.at(proc_kind);
			}
			if (value.count(Processor::NO_KIND) > 0)
			{
				return value.at(Processor::NO_KIND);
			}
		}
		return {};
	}
	std::vector<Memory::Kind> query_memory_list(std::string task_name, std::vector<std::string> region_names, Processor::Kind proc_kind)
	{
		// region_names: no "*" included; will need to consider "*"
		std::vector<Memory::Kind> res;
		// exact match first
		for (auto &region_name : region_names)
		{
			std::vector<Memory::Kind> to_append = query_memory_policy(task_name, region_name, proc_kind);
			res.insert(res.end(), to_append.begin(), to_append.end());
		}
		// task_name *
		std::vector<Memory::Kind> to_append2 = query_memory_policy(task_name, "*", proc_kind);
		res.insert(res.end(), to_append2.begin(), to_append2.end());
		// * region_name
		for (auto &region_name : region_names)
		{
			std::vector<Memory::Kind> to_append3 = query_memory_policy("*", region_name, proc_kind);
			res.insert(res.end(), to_append3.begin(), to_append3.end());
		}
		// * *
		std::vector<Memory::Kind> to_append4 = query_memory_policy("*", "*", proc_kind);
		res.insert(res.end(), to_append4.begin(), to_append4.end());
		return res;
	}
	ConstraintsNode* query_constraint_one_region(const std::string &task_name, const std::string &region_name,
												 const Memory::Kind &mem_kind)
	{
		std::pair<std::string, std::string> key = {task_name, region_name};
		if (layout_constraints.count(key) > 0)
		{
			std::unordered_map<Memory::Kind, ConstraintsNode*> value = layout_constraints.at(key);
			if (value.count(mem_kind) > 0)
			{
				return value.at(mem_kind);
			}
			if (value.count(Memory::NO_MEMKIND) > 0)
			{
				return value.at(Memory::NO_MEMKIND);
			}
		}
		return NULL;
	}
	ConstraintsNode* query_constraint(const std::string &task_name, const std::vector<std::string> &region_names,
									  const Memory::Kind &mem_kind)
	{
		// exact match first
		for (auto &region_name : region_names)
		{
			ConstraintsNode* res1 = query_constraint_one_region(task_name, region_name, mem_kind);
			if (res1 != NULL)
			{
				return res1;
			}
		}
		ConstraintsNode* res2 = query_constraint_one_region(task_name, "*", mem_kind);
		if (res2 != NULL)
		{
			return res2;
		}
		for (auto &region_name : region_names)
		{
			ConstraintsNode* res3 = query_constraint_one_region("*", region_name, mem_kind);
			if (res3 != NULL)
			{
				return res3;
			}
		}
		ConstraintsNode* res4 = query_constraint_one_region("*", "*", mem_kind);
		if (res4 != NULL)
		{
			return res4;
		}
		return NULL;
	}
	int query_max_instance(std::string task_name, Processor::Kind proc_kind)
	{
		if (task2limit.count(task_name) > 0)
		{
			std::unordered_map<Processor::Kind, int> kind_int = task2limit.at(task_name);
			if (kind_int.count(proc_kind) > 0)
			{
				int res = kind_int.at(proc_kind);
				assert(res > 0);
				return res;
			}
			return 0;
		}
		return 0;
	}
};

std::unordered_map<std::string, std::vector<Processor::Kind>> Tree2Legion::task_policies;
using HashFn1 = PairHash<std::string, std::string>;
std::unordered_map<std::pair<std::string, std::string>, 
	std::unordered_map<Processor::Kind, std::vector<Memory::Kind> >, HashFn1>
	Tree2Legion::region_policies;

std::unordered_map<std::pair<std::string, std::string>,
	std::unordered_map<Memory::Kind, ConstraintsNode*>, HashFn1>
	Tree2Legion::layout_constraints;
std::unordered_map<std::string, std::unordered_map<Processor::Kind, int>> Tree2Legion::task2limit;
// std::unordered_map<std::string, MSpace*> Tree2Legion::task2mspace;
std::unordered_map<std::string, FuncDefNode*> Tree2Legion::task2func;
std::unordered_set<std::pair<std::string, std::string>, HashFn1> Tree2Legion::memory_collect;
#endif
