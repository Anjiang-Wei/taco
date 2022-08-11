#include "tree.hpp"
#include "MSpace.cpp"
#include <iostream>
#include <algorithm>

// #define DEBUG_TREE true

using namespace Legion;
using namespace Legion::Mapping;

class NSMapper;
Processor::Kind MyProc2LegionProc(ProcessorEnum);
Memory::Kind MyMem2LegionMem(MemoryEnum);

void handleError(int errorCode, const char *errorMessage)
{
  printf("Error #%d: %s\n", errorCode, errorMessage);
}


void ProgramNode::print()
{
  printf("ProgramNode's Stmt: %ld\n", stmt_list.size());
  for (size_t i = 0; i < stmt_list.size(); i++)
  {
    printf("Stmt %ld\n", i);
    stmt_list[i]->print();
    printf("------------\n");
  }
}

void ProcLstNode::print()
{
  printf("ProcLstNode: %ld\n", proc_type_lst.size());
  for (size_t i = 0; i < proc_type_lst.size(); i++)
  {
    printf("%s ", ProcessorEnumName[proc_type_lst[i]]);
  }
  printf("\n");
}

void MemLstNode::print()
{
  printf("MemLstNode: %ld\n", mem_type_lst.size());
  for (size_t i = 0; i < mem_type_lst.size(); i++)
  {
    printf("%s ", MemoryEnumName[mem_type_lst[i]]);
  }
  printf("\n");
}

Processor::Kind MyProc2LegionProc(ProcessorEnum myproc)
{
  switch (myproc)
  {
    case CPU:
      return Processor::LOC_PROC;
    case GPU:
      return Processor::TOC_PROC;
    case IO:
      return Processor::IO_PROC;
    case PY:
      return Processor::PY_PROC;
    case PROC:
      return Processor::PROC_SET;
    case OMP:
      return Processor::OMP_PROC;
    case ALLPROC:
      return Processor::NO_KIND;
    default:
      assert(false);
      break;
  }
  assert(false);
  return Processor::LOC_PROC;
}

std::vector<Processor::Kind> MyProc2LegionProcList(std::vector<ProcessorEnum> myprocs)
{
  std::vector<Processor::Kind> res;
  res.resize(myprocs.size());
  std::transform(myprocs.begin(), myprocs.end(), res.begin(), MyProc2LegionProc);
  return res;
}

Memory::Kind MyMem2LegionMem(MemoryEnum mymem)
{
  switch(mymem)
  {
    case SYSMEM:
      return Memory::SYSTEM_MEM;
    case FBMEM:
      return Memory::GPU_FB_MEM; 
    case RDMEM:
      return Memory::REGDMA_MEM;
    case ZCMEM:
      return Memory::Z_COPY_MEM;
    case SOCKMEM:
      return Memory::SOCKET_MEM;
    case ALLMEM:
      return Memory::NO_MEMKIND;
    default:
      assert(false);
      break;
  }
  assert(false);
  return Memory::Z_COPY_MEM;
}

std::vector<Memory::Kind> MyMem2LegionMemList(std::vector<MemoryEnum> myprocs)
{
  std::vector<Memory::Kind> res;
  res.resize(myprocs.size());
  std::transform(myprocs.begin(), myprocs.end(), res.begin(), MyMem2LegionMem);
  return res;
}

Node* ProcCustomNode::run()
{
  std::vector<Processor::Kind> res = MyProc2LegionProcList(this->proc_types);
  Tree2Legion::task_policies.insert({taskname, res});
  return NULL;
}

Node* RegionCustomNode::run()
{
  std::pair<std::string, std::string> key = std::make_pair(taskname, region_name);
  Processor::Kind proc = MyProc2LegionProc(this->processor_type);
  std::vector<Memory::Kind> mem = MyMem2LegionMemList(this->mem_types);
  if (Tree2Legion::region_policies.count(key) > 0)
  {
    Tree2Legion::region_policies.at(key).insert({proc, mem});
  }
  else
  {
    std::unordered_map<Processor::Kind, std::vector<Memory::Kind>> value = {{proc, mem}};
    Tree2Legion::region_policies.insert({key, value});
  }
  return NULL;
}

Node* LayoutCustomNode::run()
{
  std::pair<std::string, std::string> key = std::make_pair(task_name, region_name);
  Memory::Kind mem = MyMem2LegionMem(mem_type);
  if (Tree2Legion::layout_constraints.count(key) > 0)
  {
    Tree2Legion::layout_constraints.at(key).insert({mem, this->constraint});
  }
  else
  {
    std::unordered_map<Memory::Kind, ConstraintsNode*> value = {{mem, this->constraint}};
    Tree2Legion::layout_constraints.insert({key, value});
  }
  return NULL;
}

Node* InstanceLimitNode::run()
{
  Processor::Kind proc_kind = MyProc2LegionProc(proc_type);
  if (Tree2Legion::task2limit.count(task_name) > 0)
  {
    Tree2Legion::task2limit.at(task_name).insert({proc_kind, num});
  }
  else
  {
    std::unordered_map<Processor::Kind, int> kind_int;
    kind_int.insert({proc_kind, num});
    Tree2Legion::task2limit.insert({task_name, kind_int});
  }
  return NULL;
}

Node* MemoryCollectNode::run()
{
  Tree2Legion::memory_collect.insert({task_name, region_name});
  return NULL;
}

bool Tree2Legion::prerun_validate(std::string task, Processor::Kind proc_kind)
{
  MSpace* mspace_node;
  if (task2mspace.count(task) > 0)
  {
    mspace_node = task2mspace.at(task);
  }
  else if (task2mspace.count("*") > 0)
  {
    mspace_node = task2mspace.at("*");
  }
  // run_validate will be invoked in slicing, so it must be sharded by user
  assert(mspace_node != NULL);
  if (proc_kind == MyProc2LegionProc(mspace_node->proc_type))
  {
    return true;
  }
  return false;
}

std::vector<int> Tree2Legion::run(std::string task, std::vector<int> x, std::vector<int> point_space)
{
  #ifdef DEBUG_TREE
      std::cout << "in Tree2Legion::run " << vec2str(x) << std::endl;
  #endif
  MSpace* mspace_node;
  FuncDefNode* func_node;
  TupleIntNode* launch_space = new TupleIntNode(point_space);// task2launch_space.at(task);
  if (task2mspace.count(task) > 0)
  {
    mspace_node = task2mspace.at(task);
    func_node = task2func.at(task);
  }
  else if (task2mspace.count("*") > 0)
  {
    mspace_node = task2mspace.at("*");
    func_node = task2func.at("*");
  }
  else 
  {
    std::cout << "Fail in Tree2Legion::run when searching task names" << std::endl;
    assert(false);
  }

  std::unordered_map<std::string, Node*> func_symbols;
  
  TupleIntNode* ipoint_input = new TupleIntNode(x);
  func_symbols.insert({func_node->func_args->arg_lst[0]->argname, ipoint_input});
  func_symbols.insert({func_node->func_args->arg_lst[1]->argname, launch_space});
  func_symbols.insert({func_node->func_args->arg_lst[2]->argname, mspace_node});

  assert(local_symbol.size() == 0);
  local_symbol.push(func_symbols);
  Node* res = func_node->invoked();
  local_symbol.pop();

  delete ipoint_input;

  assert(res->type == TupleIntType || res->type == TupleExprType || res->type == IntValType);
  // printf("Index Launch Computation Finished\n");
  if (res->type == TupleIntType)
  {
    TupleIntNode* res2 = (TupleIntNode*) res;
    if (res2->tupleint.size() != mspace_node->get_size().size())
    {
      std::cout << "Not compatible with the machine model's dimension!" << std::endl;
      assert(false);
    }
    return mspace_node->get_node_proc(res2->tupleint);
  }
  else if (res->type == TupleExprType)
  {
    TupleExprNode* res2 = (TupleExprNode*) res;
    std::vector<int> res3;
    for (size_t i = 0; i < res2->exprlst.size(); i++)
    {
      assert(res2->exprlst[i]->type == IntValType);
      res3.push_back(((IntValNode*) res2->exprlst[i])->intval);
    }
    if (res3.size() != mspace_node->get_size().size())
    {
      std::cout << "Not compatible with the machine model's dimension!" << std::endl;
      assert(false);
    }
    return mspace_node->get_node_proc(res3);
  }
  else
  {
    IntValNode* res2 = (IntValNode*) res;
     return mspace_node->get_node_proc(std::vector<int>{res2->intval});
  }
  assert(false);
  return x;
}

Node* IndexTaskMapNode::run()
{
  if (global_symbol.count(func_name) == 0 || global_symbol.count(machine_name) == 0)
  {
    std::cout << "IndexTaskMap's function or machine model undefined" << std::endl;
    assert(false);
  }
  Node* fun_node = global_symbol.at(func_name);
  assert(fun_node->type == FuncDefType);
  FuncDefNode* func_node_c = (FuncDefNode*) fun_node;

  std::vector<ArgNode*> params = func_node_c->func_args->arg_lst;

  Node* machine_node = global_symbol.at(machine_name);
  assert(machine_node->type == MSpaceType);
  MSpace* mspace_node = (MSpace*) machine_node;

  Tree2Legion::task2mspace.insert({task_name, mspace_node});
  Tree2Legion::task2func.insert({task_name, func_node_c});

  // IPoint x, ISpace y, MSpace z
  if (!(params.size() == 3 && params[0]->argtype == IPOINT && \
      params[1]->argtype == ISPACE && params[2]->argtype == MSPACE))
  {
    std::cout << "Entry function input must be (IPoint, ISpace, MSpace)" << std::endl;
    assert(false);
  }
  return NULL;
}

Node* FuncDefNode::invoked()
{
  // iterate over all statements in func_stmts
  for (size_t i = 0; i < func_stmts->stmtlst.size(); i++)
  {
    if (func_stmts->stmtlst[i]->type != ReturnType)
    {
      func_stmts->stmtlst[i]->run();
    }
    else
    {
      // std::cout << "return detected" << std::endl;
      return func_stmts->stmtlst[i]->run();
    }
  }
  std::cout << "Error: function without return" << std::endl;
  assert(false);
}

Node* FuncInvokeNode::run()
{
  Node* args = args_node->run();
  assert(args->type == TupleExprType);
  TupleExprNode* args_c = (TupleExprNode*) args;
  if (func_node->type == MSpaceType)
  {
    MSpace* func_c = (MSpace*) func_node;
    assert(args_c->exprlst.size() == 1);
    assert(args_c->exprlst[0]->type == ProcType);
    ProcNode* proc_c = (ProcNode*) (args_c->exprlst[0]);
    func_c->set_proc_type(proc_c->proc_type);
    return func_c;
  }
  else if (func_node->type == ObjectInvokeType)
  {
    ObjectInvokeNode* func_c = (ObjectInvokeNode*) func_node;
    if (func_c->api == HAS)
    {
      assert(func_c->obj->type == IndexExprType);
      IndexExprNode* index_node = (IndexExprNode*) func_c->obj;
      Node* mspace_node = index_node->tuple->run(); // get the MSpace node from the Identifier node's run()
      assert(mspace_node->type == MSpaceType);
      MSpace* machine_node = (MSpace*) mspace_node;

      Node* int_node = index_node->index->run();
      assert(int_node->type == IntValType);
      int int_val = ((IntValNode*) int_node)->intval;

      assert(args_c->exprlst.size() == 1);
      Node* mem_node = args_c->exprlst[0]->run();
      assert(mem_node->type == MemType);
      MemNode* memnode = (MemNode*) mem_node;
      MemoryEnum mem = memnode->mem_type;
      
      return new BoolValNode(machine_node->has_mem(int_val, mem));
    }
    else if (func_c->api == REVERSE)
    {
      Node* mspace_node_ = func_c->obj->run();
      assert(mspace_node_->type == MSpaceType);
      MSpace* mspace_node = (MSpace*) mspace_node_;
      
      assert(args_c->exprlst.size() == 1);
      Node* intnode_1 = args_c->exprlst[0]->run();
      assert(intnode_1->type == IntValType);
      IntValNode* int_node_1 = (IntValNode*) intnode_1;

      return new MSpace(mspace_node, func_c->api, int_node_1->intval);
    }
    else if (func_c->api == SPLIT || func_c->api == SWAP || \
            func_c->api == MERGE || func_c->api == BALANCE_SPLIT)
    {
      Node* mspace_node_ = func_c->obj->run();
      assert(mspace_node_->type == MSpaceType);
      MSpace* mspace_node = (MSpace*) mspace_node_;
      
      assert(args_c->exprlst.size() == 2);
      Node* intnode_1 = args_c->exprlst[0]->run();
      Node* intnode_2 = args_c->exprlst[1]->run();
      assert(intnode_1->type == IntValType);
      assert(intnode_2->type == IntValType);
      IntValNode* int_node_1 = (IntValNode*) intnode_1;
      IntValNode* int_node_2 = (IntValNode*) intnode_2;

      return new MSpace(mspace_node, func_c->api, int_node_1->intval, int_node_2->intval);
    }
    else if (func_c->api == SLICE)
    {
      Node* mspace_node_ = func_c->obj->run();
      assert(mspace_node_->type == MSpaceType);
      MSpace* mspace_node = (MSpace*) mspace_node_;
      
      assert(args_c->exprlst.size() == 3);
      Node* intnode_1 = args_c->exprlst[0]->run();
      Node* intnode_2 = args_c->exprlst[1]->run();
      Node* intnode_3 = args_c->exprlst[2]->run();
      assert(intnode_1->type == IntValType);
      assert(intnode_2->type == IntValType);
      assert(intnode_3->type == IntValType);
      IntValNode* int_node_1 = (IntValNode*) intnode_1;
      IntValNode* int_node_2 = (IntValNode*) intnode_2;
      IntValNode* int_node_3 = (IntValNode*) intnode_3;

      return new MSpace(mspace_node, func_c->api, int_node_1->intval, int_node_2->intval, int_node_3->intval);
    }
    else
    {
      std::cout << "unsupported func_c->api" << std::endl;
    }
  }
  else if (func_node->type == IdentifierExprType)
  {
    Node* func_node_ = func_node->run();
    assert(func_node_->type == FuncDefType);
    FuncDefNode* func_def = (FuncDefNode*) func_node_;
    std::vector<ArgNode*> params = func_def->func_args->arg_lst;
    // insert arguments and type checking into local_symbol;
    std::unordered_map<std::string, Node*> func_symbols;
    if (params.size() != args_c->exprlst.size())
    {
      std::cout << "argument number mismatch!" << std::endl;
      assert(false);
    }
    for (size_t i = 0; i < args_c->exprlst.size(); i++)
    {
      Node* feed_in = args_c->exprlst[i];
      switch (params[i]->argtype)
      {
        case INT:
          assert(feed_in->type == IntValType);
          break;
        case BOOL:
          assert(feed_in->type == BoolValType);
          break;
        case IPOINT:
          assert(feed_in->type == TupleIntType);
          break;
        case ISPACE:
          assert(feed_in->type == TupleIntType);
          break;
        case MSPACE:
          assert(feed_in->type == MSpaceType);
          break;
        default:
          assert(false);
      }
      func_symbols.insert({params[i]->argname, feed_in});
    }
    local_symbol.push(func_symbols);
    Node* res = func_def->invoked();
    local_symbol.pop();
    return res;
  }
  else
  {
    std::cout << "unsupported in FuncInvokeNode" << std::endl;
    assert(false);
  }
  assert(false);
  return NULL;
}

Node* BinaryExprNode::run()
{
  Node* simplified_left = left->run();
  Node* simplified_right = right->run();
  if (simplified_left->type != simplified_right->type)
  {
    std::cout << "type mismatch!!" << std::endl;
    std::cout << NodeTypeName[simplified_left->type] << std::endl;
    std::cout << NodeTypeName[simplified_right->type] << std::endl;
    assert(false);
  }
  if (simplified_left->type == BoolValType)
  {
    BoolValNode* lt = (BoolValNode*) simplified_left;
    BoolValNode* rt = (BoolValNode*) simplified_right;
    switch(op)
    {
      case EQ:
        return new BoolValNode(lt->boolval == rt->boolval);
      case NEQ:
        return new BoolValNode(lt->boolval != rt->boolval);
      case OR:
        std::cout << "OR detected: " <<  (lt->boolval || rt->boolval) << std::endl;
        return new BoolValNode(lt->boolval || rt->boolval);
      case AND:
        return new BoolValNode(lt->boolval && rt->boolval);
      default:
        assert(false);
    }
  }
  else if (simplified_left->type == IntValType)
  {
    IntValNode* lt = (IntValNode*) simplified_left;
    IntValNode* rt = (IntValNode*) simplified_right;
    switch (op)
    {
      case PLUS:
        // std::cout << "PLUS: " << lt->intval + rt->intval << std::endl;
        return new IntValNode(lt->intval + rt->intval);
      case MINUS:
        return new IntValNode(lt->intval - rt->intval);
      case MULTIPLY:
        // std::cout << "MUL: " << lt->intval * rt->intval << std::endl;
        return new IntValNode(lt->intval * rt->intval);
      case DIVIDE:
        return new IntValNode(lt->intval / rt->intval);
      case MOD:
        return new IntValNode(lt->intval % rt->intval);
      case BIGGER:
        return new BoolValNode(lt->intval > rt->intval);
      case SMALLER:
        return new BoolValNode(lt->intval < rt->intval);
      case GE:
        return new BoolValNode(lt->intval >= rt->intval);
      case LE:
        return new BoolValNode(lt->intval <= rt->intval);
      case EQ:
        return new BoolValNode(lt->intval == rt->intval);
      case NEQ:
        return new BoolValNode(lt->intval != rt->intval);
      default:
        assert(false);
    }
  }
  assert(false);
  return NULL;
}

Node* ObjectInvokeNode::run()
{
  assert(obj->type == IdentifierExprType);
  Node* obj_tbd = obj->run();
  if (obj_tbd->type != MSpaceType && obj_tbd->type != TupleIntType)
  {
    std::cout << NodeTypeName[obj_tbd->type] <<  " does not support volume/size" << std::endl;
    assert(false);
  }
  if (obj_tbd->type == TupleIntType)
  {
    return obj_tbd;
  }
  MSpace* machine_ = (MSpace*) obj_tbd;
  if (api == VOLUME)
  {
    return new IntValNode(machine_->get_volume());
  }
  else if (api == SIZE)
  {
    return new TupleIntNode(machine_->get_size());
  }
  else
  {
    std::cout << "unsupported ObjectInvokeNode api" << std::endl;
    assert(false);
  }
  assert(false);
  return NULL;
}

Node* IndexExprNode::run()
{
	Node* index_ = index->run();
  assert(index_->type == IntValType);
  IntValNode* int_node = (IntValNode*) index_;
  int int_index = int_node->intval;

  Node* tuple_ = tuple->run();
  if (tuple_->type == TupleIntType)
  {
    TupleIntNode* tuple_int = (TupleIntNode*) tuple_;
    if (int_index >= (int) tuple_int->tupleint.size())
    {
      std::cout << "Index Out Of Bound!" << std::endl;
      assert(false);
    }
    return new IntValNode(tuple_int->tupleint[int_index]);
  }
  else
  {
    std::cout << "unsupported IndexExprNode tuple_->type" << std::endl;
    assert(false);
  }
  assert(false);
  return NULL;
}


Node* TupleExprNode::run()
{
  TupleExprNode* res = new TupleExprNode();
  for (size_t i = 0; i < exprlst.size(); i++)
  {
    Node* simplified = exprlst[i]->run();
    if (simplified->type == UnpackExprType)
    {
      UnpackExprNode* unpack_node = (UnpackExprNode*) simplified;
      assert(unpack_node->expr->type == TupleExprType);
      TupleExprNode* tuple_node = (TupleExprNode*) unpack_node->expr;
      for (size_t i = 0; i < tuple_node->exprlst.size(); i++)
      {
        res->exprlst.push_back(tuple_node->exprlst[i]->run());
      }
    }
    else
    {
      res->exprlst.push_back(simplified);
    }
  }
  return res;
}

Node* PrintNode::run()
{
  std::string s = format_string;
  std::string delimiter = "{}";
  size_t pos = 0;
  std::string token;
  size_t i = 0;
  while ((pos = s.find(delimiter)) != std::string::npos)
  {
    token = s.substr(0, pos);
    std::cout << token;
    if (i >= printargs.size()) 
    {
      std::cout << "Not enough arguments for print()" << std::endl; 
      assert(false);
    }
    printargs[i]->run()->print();
    s.erase(0, pos + delimiter.length());
    i++;
  }
  std::cout << s << std::endl;
  return NULL;
}

Tree2Legion::Tree2Legion(std::string filename)
{
  extern FILE* yyin;
  extern int yyparse();
  yyin = fopen(filename.c_str(), "r");
  if (yyin == NULL)
  {
    std::cout << "Mapping policy file does not exist" << std::endl;
    assert(false);
  }
  yyparse();
  // std::cout << root->stmt_list.size() << std::endl;
  // root->print();
  root->run();
}
