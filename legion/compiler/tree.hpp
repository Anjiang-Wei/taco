#ifndef __TREE
#define __TREE
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <unordered_map>
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
};

const char* ProcessorEnumName[] =
{
	"CPU",
	"GPU",
	"IO",
	"PY",
	"PROC",
	"OMP",
};

enum MemoryEnum
{
	SYSMEM,
	FBMEM,
	RDMEM,
	ZCMEM,
	SOCKMEM,
};

const char* MemoryEnumName[] =
{
	"SYSMEM",
	"FBMEM",
	"RDMEM",
	"ZCMEM",
	"SOCKMEM",
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
	TaskDefaultType,
	ProcLstType,
	ProcType,
	RegionDefaultType,
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
	IntValType,
	BoolValType,
	TupleExprType,
	FuncInvokeType,
	IndexExprType,
	NegativeExprType,
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
  "TaskDefaultType",
  "ProcLstType",
  "ProcType",
  "RegionDefaultType",
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
  "IntValType",
  "BoolValType",
  "TupleExprType",
  "FuncInvokeType",
  "IndexExprType",
  "NegativeExprType",
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

class TaskDefaultNode : public StmtNode
{
public:
	std::vector<ProcessorEnum> proc_type_lst;

	TaskDefaultNode(const ProcLstNode* x) { type = TaskDefaultType; proc_type_lst = x->proc_type_lst; }
	void print();
	Node* run();
};

class MemLstNode : public Node
{
public:
	std::vector<MemoryEnum> mem_type_lst;

	MemLstNode() { type = MemLstType; }
	void print();
};

class RegionDefaultNode : public StmtNode
{
public:
	ProcessorEnum proc_type;
	std::vector<MemoryEnum> mem_type_lst;

	RegionDefaultNode(const ProcNode* x, const MemLstNode* y)
	{
		type = RegionDefaultType;
		proc_type = x->proc_type;
		mem_type_lst = y->mem_type_lst;
	}
	void print();
	Node* run();
};

class ProcCustomNode : public StmtNode
{
public:
	std::string taskname;
	ProcessorEnum proc_type;

	ProcCustomNode(const char* x, const ProcNode* y)
	{
		type = ProcCustomType;
		taskname = std::string(x);
		proc_type = y->proc_type;
	}
	void print() { printf("ProcCustomNode %s %s\n", taskname.c_str(), ProcessorEnumName[proc_type]); }
	Node* run();
};

class RegionCustomNode : public StmtNode
{
public:
	std::string taskname;
	std::string region_name;
	MemoryEnum mem_type;

	RegionCustomNode(const char* x, const char* y, const MemNode* z)
	{
		type = RegionCustomType;
		taskname = std::string(x);
		region_name = std::string(y);
		mem_type = z->mem_type;
	}
	void print() { printf("RegionCustomNode %s %s %s\n", taskname.c_str(), region_name.c_str(), MemoryEnumName[mem_type]); }
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
	Node* run()
	{
		return this;
	}
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
			printf("%d, ", tupleint[i]);
		}
	}
	Node* run() { return this; }
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
};

class IndexTaskMapNode : public StmtNode
{
	std::string task_name;
	std::string machine_name;
	std::string func_name;

public:
	IndexTaskMapNode(const char* x, const char* y, const char* z)
	{
		type = IndexTaskMapType;
		task_name = std::string(x);
		machine_name = std::string(y);
		func_name = std::string(z);
	}
	void print() { printf("IndexTaskMapNode %s %s %s\n", task_name.c_str(), machine_name.c_str(), func_name.c_str()); }
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
		printf("TupleExprNode: %ld\n", exprlst.size());
		for (size_t i = 0; i < exprlst.size(); i++)
		{
			exprlst[i]->print();
		}
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
};

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
	std::vector<IdentifierExprNode*> printargs;

	PrintArgsNode() { type = PrintArgsType; }
};

class PrintNode : public StmtNode
{
public:
	std::string format_string;
	std::vector<IdentifierExprNode*> printargs;
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

class MSpace;

class Tree2Legion
{
public:
	Tree2Legion() { launch_space = NULL; }
	Tree2Legion(std::string filename);
	// ~Tree2Legion() { if (launch_space != NULL) delete launch_space; } // this will result in SEGV!
	static std::vector<Processor::Kind> default_task_policy;
	static std::unordered_map<std::string, Processor::Kind> task_policies;

	static std::unordered_map<Processor::Kind, std::vector<Memory::Kind>> default_region_policy;
	using HashFn1 = PairHash<std::string, std::string>;
  	static std::unordered_map<std::pair<std::string, std::string>, Memory::Kind, HashFn1> region_policies;
	
	static std::unordered_map<std::string, MSpace*> task2mspace;
	static std::unordered_map<std::string, FuncDefNode*> task2func;

	bool should_fall_back(std::string task_name)
	{
		if (task2mspace.count(task_name) > 0)
		{
			return false;
		}
		if (task2mspace.count("IndexTaskMapDefault") > 0)
		{
			return false;
		}
		return true;
	}
	
	// TupleIntNode* machine_space;
	TupleIntNode* launch_space;
	  
	void print() { std::cout << "I am invoked!" << std::endl; }

	void set_launch_space(std::vector<int> x)
	{
		launch_space = new TupleIntNode(x);
	}
	std::vector<int> run(std::string task, std::vector<int> x);
};

std::vector<Processor::Kind> Tree2Legion::default_task_policy;
std::unordered_map<std::string, Processor::Kind> Tree2Legion::task_policies;
std::unordered_map<Processor::Kind, std::vector<Memory::Kind>> Tree2Legion::default_region_policy;
using HashFn1 = PairHash<std::string, std::string>;
std::unordered_map<std::pair<std::string, std::string>, Memory::Kind, HashFn1> Tree2Legion::region_policies;

std::unordered_map<std::string, MSpace*> Tree2Legion::task2mspace;
std::unordered_map<std::string, FuncDefNode*> Tree2Legion::task2func;
#endif
