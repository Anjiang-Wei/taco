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
      std::cout << "Unsupported Processor Type" << std::endl;
      assert(false);
      break;
  }
  std::cout << "Reach undesired region in MyProc2LegionProc" << std::endl;
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
      std::cout << "Reach undesired region in MyMem2LegionMem" << std::endl;
      assert(false);
      break;
  }
  std::cout << "Reach undesired region in MyMem2LegionMem" << std::endl;
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

TupleIntNode* TaskNode::get_point()
{
  if (index_launch)
    return ipoint;
  else
    return new TupleIntNode(std::vector<int>());
}

TupleIntNode* TaskNode::get_space()
{
  if (index_launch)
    return ispace;
  else
    return new TupleIntNode(std::vector<int>());
}

TaskNode* TaskNode::get_parent()
{
  if (parent != NULL)
    return parent;
  else
  {
    // todo: query the Legion runtime, and record the result in parent
  }
}

std::vector<int> TaskNode::get_proc_coordinate_from_Legion()
{
  if (coordinate.size() != 0)
  {
    return coordinate;
  }
  else
  {
    // todo: query the Legion runtime, and record the result in coordinate
  }
}

std::vector<std::vector<int>> Tree2Legion::runsingle(std::string task)
{
  // todo: design this interface
  return {};
}

std::vector<std::vector<int>> Tree2Legion::run(std::string task, std::vector<int> x,
            std::vector<int> point_space, Processor::Kind proc_kind = Processor::NO_KIND)
{
  // todo: redesign this interface, we may want to pass the whole Task object in here
  #ifdef DEBUG_TREE
      std::cout << "in Tree2Legion::run " << vec2str(x) << std::endl;
  #endif
  FuncDefNode* func_node;
  if (task2func.count(task) > 0)
  {
    func_node = task2func.at(task);
  }
  else if (task2func.count("*") > 0)
  {
    func_node = task2func.at("*");
  }
  else 
  {
    std::cout << "Fail in Tree2Legion::run when searching task names" << std::endl;
    assert(false);
  }

  std::unordered_map<std::string, Node*> func_symbols;
  TaskNode* task_node = new TaskNode();
  task_node->init(task, x, point_space);
  func_symbols.insert({func_node->func_args->arg_lst[0]->argname, task_node});

  if (local_symbol.size() != 0)
  {
    std::cout << "local_symbol.size() != 0 inside Tree2Legion::run" << std::endl;
    assert(false);
  }
  local_symbol.push(func_symbols);
  Node* res = func_node->invoked();
  local_symbol.pop();

  delete task_node;

  if (res->type == TupleIntType)
  {
    TupleIntNode* res2 = (TupleIntNode*) res;
    if (proc_kind != Processor::NO_KIND)
    {
      if (MyProc2LegionProc(res2->final_proc) != proc_kind)
      {
        printf("%s is actually mapped to %s, but machine model is for %s", 
                task.c_str(),
                processor_kind_to_string(proc_kind).c_str(),
                ProcessorEnumName[res2->final_proc]);
        assert(false);
      }
    }
    return std::vector<std::vector<int>>({res2->tupleint});
  }
  else if (res->type == SetTupleIntType)
  {
    SetTupleIntNode* res2 = (SetTupleIntNode*) res;
    if (proc_kind != Processor::NO_KIND)
    {
      if (MyProc2LegionProc(res2->final_proc) != proc_kind)
      {
        printf("%s is actually mapped to %s, but machine model is for %s", 
                task.c_str(),
                processor_kind_to_string(proc_kind).c_str(),
                ProcessorEnumName[res2->final_proc]);
        assert(false);
      }
    }
    return res2->tupletupleint;
  }
  else
  {
    printf("Must return TupleIntType or SetTupleIntType after invoking mapping function\n");
    assert(false);
  }
  return {};
}

Node* SingleTaskMapNode::run()
{
  if (global_symbol.count(func_name) == 0)
  {
    printf("SingleTaskMap's function undefined\n");
    assert(false);
  }
  Node* fun_node = global_symbol.at(func_name);
  if (fun_node->type != FuncDefType)
  {
    printf("SingleTaskMap's mapping function is undefined\n");
    assert(false);
  }
  FuncDefNode* func_node_c = (FuncDefNode*) fun_node;

  std::vector<ArgNode*> params = func_node_c->func_args->arg_lst;

  for (int i = 0; i < task_name.size(); i++)
  {
    Tree2Legion::task2func.insert({task_name[i], func_node_c});
  }
  if (!(params.size() == 1 && (params[0]->argtype == TASK)))
  {
    std::cout << "Entry mapping function's must be like func(Task t)" << std::endl;
    assert(false);
  }
  return NULL;
}


Node* IndexTaskMapNode::run()
{
  if (global_symbol.count(func_name) == 0)
  {
    printf("IndexTaskMap's function undefined\n");
    assert(false);
  }
  Node* fun_node = global_symbol.at(func_name);
  if (fun_node->type != FuncDefType)
  {
    printf("IndexTaskMap's mapping function is undefined\n");
    assert(false);
  }
  FuncDefNode* func_node_c = (FuncDefNode*) fun_node;

  std::vector<ArgNode*> params = func_node_c->func_args->arg_lst;

  for (int i = 0; i < task_name.size(); i++)
  {
    Tree2Legion::task2func.insert({task_name[i], func_node_c});
  }
  // IPoint x, ISpace y
  if (!(params.size() == 2 && params[0]->argtype == IPOINT && params[1]->argtype == ISPACE))
  {
    std::cout << "Entry mapping function's input must be (IPoint, ISpace)" << std::endl;
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


// Node* FuncInvokeNode::broadcast()
// {
//   /*
//   ExprNode* func_node;
// 	TupleExprNode* args_node; // std::vector<Node*> exprlst;
//   */
//   bool exclamation = false;
//   int num_element = 0;
//   // first, we check whether there is broadcast needed
//   for (int i = 0; i < exprlst.size(); i++)
//   {
//     if (exprlst[i]->type == ExclamationType)
//     {
//       ExclamationNode* exclam_node = (ExclamationNode*) exprlst[i];
//       int num_element_;
//       // exclam_node has: ExprNode* expr;
//       // expr can be either TupleIntNode, std::vector<int> tupleint;
//       // or TupleExprNode, std::vector<Node*> exprlst; (recursive case)
//       if (exclam_node->expr->type == TupleIntType)
//       {
//         TupleIntNode* tuple_int_node = (TupleIntNode*) exclam_node->expr;
//         num_element_ = tuple_int_node->tupleint.size();
//       }
//       else if (exclam_node->expr->type == TupleExprType)
//       {
//         TupleExprNode* tuple_expr_node = (TupleExprNode*) exclam_node->expr;
//         num_element_ = tuple_expr_node->exprlst.size();
//       }
//       else
//       {
//         std::cout << "Unsupported type after '!'" << std::endl;
//         assert(false);
//       }
//       if (exclamation)
//       {
//         if (num_element != num_element_)
//         {
//           std::cout << "Number of elements of '!' should be the same" << std::endl;
//           assert(false);
//         }
//       }
//       else
//       {
//         exclamation = true;
//         num_element = num_element_;
//       }
//     }
//   }
//   if (exclamation == false)
//   {
//     return this;
//   }
//   // Turn func_(a, !(1, 2, 3), !(x, y, z)) 
//   // into (func_(a, 1, x), func_(a, 2, y), func_(a, 3, z))
//   std::vector<FuncInvokeNode*> tuple_res;
//   for (int k = 0; k < num_element; k++)
//   {
//     std::vector<Node*> func_args;
//     for (int i = 0; i < exprlst.size(); i++)
//     {
//       if (exprlst[i]->type == ExclamationType)
//       {
//         ExclamationNode* exclam_node = (ExclamationNode*) exprlst[i];
//         if (exclam_node->expr->type == TupleIntType)
//         {
//           TupleIntNode* tuple_int_node = (TupleIntNode*) exclam_node->expr;
//           // convert it to IntValNode
//           num_element_ = tuple_int_node->tupleint.size();
//         }
//         else if (exclam_node->expr->type == TupleExprType)
//         {
//           TupleExprNode* tuple_expr_node = (TupleExprNode*) exclam_node->expr;
//           num_element_ = tuple_expr_node->exprlst.size();
//         }
//         else
//         {
//           std::cout << "Unsupported type after '!'" << std::endl;
//           assert(false);
//         }
//       }
//     }
//     TupleExprNode* func_args_node = new TupleExprNode(func_args);
//     FuncInvokeNode* func_invoke_node = new FuncInvokeNode(this->func_node, func_args_node);
//     tuple_res.push_back(func_invoke_node);
//   }
//   return new TupleExprNode(tuple_res);
// }


Node* FuncInvokeNode::run()
{
  Node* args = args_node->run();
  assert(args->type == TupleExprType);
  TupleExprNode* args_c = (TupleExprNode*) args;
  // Machine model initialization
  if (func_node->type == MSpaceType)
  {
    MSpace* func_c = (MSpace*) func_node;
    assert(args_c->exprlst.size() == 1);
    assert(args_c->exprlst[0]->type == ProcType);
    ProcNode* proc_c = (ProcNode*) (args_c->exprlst[0]);
    func_c->set_proc_type(proc_c->proc_type);
    return func_c;
  }
  // functions for built-in objects (Machine Model)
  else if (func_node->type == ObjectInvokeType)
  {
    ObjectInvokeNode* func_c = (ObjectInvokeNode*) func_node;
    if (func_c->api == HAS)
    {
      // m[0].has(GPU)
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
    else if (func_c->api == AUTO_SPLIT || func_c->api == GREEDY_SPLIT)
    {
      Node* mspace_node_ = func_c->obj->run();
      assert(mspace_node_->type == MSpaceType);
      MSpace* mspace_node = (MSpace*) mspace_node_;

      assert(args_c->exprlst.size() == 2);
      Node* intnode_1 = args_c->exprlst[0]->run();
      Node* tuple_int_node_2 = args_c->exprlst[1]->run();
      if (tuple_int_node_2->type == TupleExprType)
      {
        tuple_int_node_2 = ((TupleExprNode*) tuple_int_node_2)->Convert2TupleInt();
      }
      assert(intnode_1->type == IntValType);
      if (tuple_int_node_2->type != TupleIntType)
      {
        printf("Autosplit's second argument must be tuple of integers, while getting %s\n",
              NodeTypeName[tuple_int_node_2->type]);
        assert(false);
      }
      IntValNode* int_node_1 = (IntValNode*) intnode_1;
      TupleIntNode* node_2 = (TupleIntNode*) tuple_int_node_2;

      return new MSpace(mspace_node, func_c->api, int_node_1->intval, node_2->tupleint);
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
  // functions for built-in objects (Task Object)
  else if (func_node->type == TaskNodeType)
  {
    // todo: support task.processor(machine_model), return a point in machine model space
  }
  // user-defined function
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
  if (simplified_left->type == TupleExprType)
  {
    simplified_left = ((TupleExprNode*) simplified_left)->Convert2TupleInt();
  }
  if (simplified_right->type == TupleExprType)
  {
    simplified_right = ((TupleExprNode*) simplified_right)->Convert2TupleInt();
  }
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
    return lt->binary_op(rt, op);
  }
  else if (simplified_left->type == IntValType)
  {
    IntValNode* lt = (IntValNode*) simplified_left;
    IntValNode* rt = (IntValNode*) simplified_right;
    return lt->binary_op(rt, op);
  }
  else if (simplified_left->type == TupleExprType)
  {
    TupleExprNode* lt = (TupleExprNode*) simplified_left;
    TupleExprNode* rt = (TupleExprNode*) simplified_right;
    return lt->binary_op(rt, op);
  }
  else if (simplified_left->type == TupleIntType)
  {
    TupleIntNode* lt = (TupleIntNode*) simplified_left;
    TupleIntNode* rt = (TupleIntNode*) simplified_right;
    return lt->binary_op(rt, op);
  }
  printf("Unsupported operator type in BinaryExprNode\n");
  assert(false);
  return NULL;
}


IntValNode* TupleIntNode::len()
{
  return new IntValNode(tupleint.size());
}

IntValNode* TupleIntNode::volume()
{
  int res = 1;
  for (size_t i = 0; i < tupleint.size(); i++)
  {
    res *= tupleint[i];
  }
  return new IntValNode(res);
}

Node* ObjectInvokeNode::run()
{
  Node* obj_tbd = obj->run();
  if (obj_tbd->type == TupleExprType)
  {
    obj_tbd = ((TupleExprNode*) obj_tbd)->Convert2TupleInt();
  }
  if (obj_tbd->type != MSpaceType && obj_tbd->type != TupleIntType)
  {
    std::cout << NodeTypeName[obj_tbd->type] <<  " does not support volume/size" << std::endl;
    assert(false);
  }
  if (obj_tbd->type == TupleIntType)
  {
    TupleIntNode* tuple_int = (TupleIntNode*) obj_tbd;
    if (api == VOLUME)
    {
      return tuple_int->volume();
    }
    else if (api == SIZE)
    {
      return tuple_int;
    }
    else if (api == LEN)
    {
      return tuple_int->len();
    }
    printf("TupleInt only support volume/size/len\n");
    assert(false);
    return NULL;
  }
  else if (obj_tbd->type == MSpaceType)
  {
    MSpace* machine_ = (MSpace*) obj_tbd;
    if (api == VOLUME)
    {
      return new IntValNode(machine_->get_volume());
    }
    else if (api == SIZE)
    {
      return new TupleIntNode(machine_->get_size());
    }
    else if (api == LEN)
    {
      return new IntValNode(machine_->get_size().size());
    }
    printf("MSpace only support Volume/Size/LEN in ObjectInvokeNode\n");
    assert(false);
    return NULL;
  }
  else if (obj_tbd->type == TaskNodeType)
  {
    // todo: should also support TaskNode: ipoint (get_point), ispace (get_space), parent (get_parent);
  }
  printf("ObjectInvokeNode only supports TupleInt / MSpace / TaskNode\n");
  assert(false);
  return NULL;
}

Node* IntValNode::binary_op(IntValNode* rt, BinOpEnum op)
{
  switch (op)
    {
    case PLUS:
      return new IntValNode(this->intval + rt->intval);
    case MINUS:
      return new IntValNode(this->intval - rt->intval);
    case MULTIPLY:
      return new IntValNode(this->intval * rt->intval);
    case DIVIDE:
      return new IntValNode(this->intval / rt->intval);
    case MOD:
      return new IntValNode(this->intval % rt->intval);
    case BIGGER:
      return new BoolValNode(this->intval > rt->intval);
    case SMALLER:
      return new BoolValNode(this->intval < rt->intval);
    case GE:
      return new BoolValNode(this->intval >= rt->intval);
    case LE:
      return new BoolValNode(this->intval <= rt->intval);
    case EQ:
      return new BoolValNode(this->intval == rt->intval);
    case NEQ:
      return new BoolValNode(this->intval != rt->intval);
    default:
      printf("Unsupported binary operator for Integer\n");
      assert(false);
  }
  return NULL;
}

TupleExprNode* TupleExprNode::binary_op(TupleExprNode* right_op, BinOpEnum op)
{
  // self as left_op
  if (exprlst.size() != right_op->exprlst.size())
  {
    printf("Dimension mismatch in TupleExprNode's binary operator\n");
    assert(false);
  }
  std::vector<Node*> res;
  for (int i = 0; i < exprlst.size(); i++)
  {
    if (!(exprlst[i]->type == IntValType && right_op->exprlst[i]->type == IntValType))
    {
      printf("Only IntValType inside TupleExprNode is allowed in binary operation\n");
      assert(false);
    }
    IntValNode* int_node_left = (IntValNode*) exprlst[i];
    IntValNode* int_node_right = (IntValNode*) (right_op->exprlst[i]);
    res.push_back(int_node_left->binary_op(int_node_right, op));
  }
  return new TupleExprNode(res);
}

TupleIntNode* TupleIntNode::binary_op(TupleIntNode* rt, BinOpEnum op)
{
  // self as left_op
  std::vector<int> res;
  if (tupleint.size() != rt->tupleint.size())
  {
    printf("Dimension mismatch in TupleIntNode's binary operation\n");
    assert(false);
  }
  for (int i = 0; i < tupleint.size(); i++)
  {
    int new_res;
    switch (op)
    {
      case PLUS:
        new_res = this->tupleint[i] + rt->tupleint[i]; break;
      case MINUS:
        new_res = this->tupleint[i] - rt->tupleint[i]; break;
      case MULTIPLY:
        new_res = this->tupleint[i] * rt->tupleint[i]; break;
      case DIVIDE:
        new_res = this->tupleint[i] / rt->tupleint[i]; break;
      case MOD:
        new_res = this->tupleint[i] % rt->tupleint[i]; break;
      default:
        printf("Unsupported binary operator for TupleIntNode\n");
        assert(false);
    }
    res.push_back(new_res);
  }
  return new TupleIntNode(res);
}

BoolValNode* BoolValNode::binary_op(BoolValNode* rt, BinOpEnum op)
{
  switch(op)
  {
  case EQ:
    return new BoolValNode(this->boolval == rt->boolval);
  case NEQ:
    return new BoolValNode(this->boolval != rt->boolval);
  case OR:
    return new BoolValNode(this->boolval || rt->boolval);
  case AND:
    return new BoolValNode(this->boolval && rt->boolval);
  default:
    printf("Unsupported binary operator for Boolean variable\n");
    assert(false);
  }
  return NULL;
}

TupleIntNode* TupleIntNode::slice(int a, int b)
{
  if (b >= tupleint.size() && b >= 0)
  {
    printf("slice's right index is out of bound!\n");
    assert(false);
  }
  a = (a < 0 ? tupleint.size() + a : a);
  b = (b <= 0 ? tupleint.size() + b : b); // b == 0 means not slicing the right side
  std::vector<int> res;
  for (int i = a; i < b; i++)
  {
    res.push_back(tupleint[i]);
  }
  return new TupleIntNode(res);
}

TupleExprNode* TupleExprNode::slice(int a, int b)
{
  if (b >= exprlst.size() && b >= 0)
  {
    printf("slice's right index is out of bound!\n");
    assert(false);
  }
  a = (a < 0 ? exprlst.size() + a : a);
  b = (b <= 0 ? exprlst.size() + b : b); // b == 0 means not slicing the right side
  std::vector<Node*> res;
  for (int i = a; i < b; i++)
  {
    res.push_back(exprlst[i]);
  }
  return new TupleExprNode(res);
}

TupleExprNode* TupleExprNode::negate()
{
  std::vector<Node*> res;
  for (int i = 0; i < exprlst.size(); i++)
  {
    if (exprlst[i]->type != IntValType)
    {
      printf("We can only negate IntValType inside TupleExprNode\n");
      assert(false);
    }
    IntValNode* int_node = (IntValNode*) exprlst[i];
    res.push_back(int_node->negate());
  }
  return new TupleExprNode(res);
}


Node* NegativeExprNode::run()
{
  if (neg->type == IntValType)
  {
    return ((IntValNode*)neg)->negate();
  }
  else if (neg->type == TupleExprType)
  {
    return ((TupleExprNode*)neg)->negate();
  }
  else if (neg->type == TupleIntType)
  {
    return ((TupleIntNode*)neg)->negate();
  }
  else
  {
    printf("Negating node must be applied to Int/TupleExpr/TupleInt\n");
  }
  return NULL;
}

TupleIntNode* TupleIntNode::negate()
{
  std::vector<int> res;
  for (int i = 0; i < tupleint.size(); i++)
  {
    res.push_back(-tupleint[i]);
  }
  return new TupleIntNode(res);
}

IntValNode* TupleIntNode::at(int x)
{
  if (x < tupleint.size())
  {
    return new IntValNode(this->tupleint[x >= 0 ? x : tupleint.size() + x]);
  }
  printf("Index out of bound for TupleIntNode\n");
  assert(false);
  return NULL;
}

IntValNode* TupleIntNode::at(IntValNode* x)
{
  return this->at(x->intval);
}

Node* TupleExprNode::at(int x)
{
  if (x < exprlst.size())
  {
    return this->exprlst[x >= 0 ? x : exprlst.size() + x];
  }
  printf("Index out of bound for TupleIntNode\n");
  assert(false);
  return NULL;
}

Node* TupleExprNode::at(IntValNode* x)
{
  return this->at(x->intval);
}

Node* IndexExprNode::run()
{
	Node* index_ = index->run();
  Node* tuple_ = tuple->run();
  if (index_->type == TupleExprType)
  {
    TupleExprNode* index_node = (TupleExprNode*) index_;
    IntValNode* int_node = index_node->one_int_only();
    if (int_node != NULL) // integer indexing
    {
      if (tuple_->type == TupleIntType)
      {
        TupleIntNode* tuple_int = (TupleIntNode*) tuple_;
        return tuple_int->at(int_node);
      }
      else if (tuple_->type == TupleExprType)
      {
        TupleExprNode* tuple_expr = (TupleExprNode*) tuple_;
        return tuple_expr->at(int_node);
      }
      else if (tuple_->type == MSpaceType) // dynamic machine model
      {
        MSpace* mspace = (MSpace*) tuple_;
        return new TupleIntNode(
          mspace->get_node_proc(std::vector<int>{int_node->intval}),
          mspace->proc_type);
      }
      else
      {
        printf("Unsupported IndexExprNode tuple's type: %s using integer index\n", NodeTypeName[tuple_->type]);
        index_->print();
        tuple_->print();
        assert(false);
      }
    }
    else if (tuple_->type == MSpaceType)
    // index_node is a tuple of expression, dynamic machine model
    {
      MSpace* machine_space = (MSpace*) tuple_;
      Node* converted = index_node->Convert2TupleInt(true);
      if (converted->type == TupleIntType)
      {
        TupleIntNode* tuple_int_node = (TupleIntNode*) converted;
        // get_node_proc returns a bool, indicating SetTupleInt or TupleInt
        // pass vector<int> and vector<vector<int>> into get_node_proc as arguments
        std::vector<int> result1;
        std::vector<std::vector<int>> result2;
        bool is_result1 = machine_space->get_node_proc(tuple_int_node->tupleint, result1, result2);
        if (is_result1)
        {
          return new TupleIntNode(result1, machine_space->proc_type);
        }
        else
        {
          return new SetTupleIntNode(result2, machine_space->proc_type);
        }
      }
      else
      {
        printf("Must only use tuple of integers to index a machine model\n");
        assert(false);
      }
    }
    else
    {
      printf("Please use tuple-indexing only for a machine model\n");
      assert(false);
    }
  }
  else if (index_->type == SliceExprType)
  {
    SliceExprNode* slice_node = (SliceExprNode*) index_;
    Node* left = slice_node->left == NULL ? NULL : slice_node->left->run();
    Node* right = slice_node->right == NULL ? NULL : slice_node->right->run();
    if ( (!(left == NULL || left->type == IntValType)) 
         || (!(right == NULL || right->type == IntValType)) )
    {
      printf("Left/Right side of SliceExprNode must be NULL or integer\n");
      assert(false);
    }
    int left_int = (left == NULL ? 0 : ((IntValNode*)left)->intval);
    int right_int = (right == NULL ? 0 : ((IntValNode*)right)->intval);
    if (tuple_->type == TupleIntType)
    {
      TupleIntNode* tuple_int = (TupleIntNode*) tuple_;
      return tuple_int->slice(left_int, right_int);
    }
    else if (tuple_->type == TupleExprType)
    {
      TupleExprNode* tuple_expr = (TupleExprNode*) tuple_;
      return tuple_expr->slice(left_int, right_int);
    }
    else if (tuple_->type == MSpaceType) // slicing a machine will return TupleInt 
    {
      MSpace* machine = (MSpace*) tuple_;
      return (new TupleIntNode(machine->get_size()))->slice(left_int, right_int);
    }
    else
    {
      printf("unsupported IndexExprNode's type for slicing\n");
      assert(false);
    }
  }
  printf("Unsupported index type in IndexExprNode\n");
  assert(false);
  return NULL;
}

IntValNode* TupleExprNode::one_int_only()
{
  if (exprlst.size() == 1 && exprlst[0]->type == IntValType)
  {
    return (IntValNode*) exprlst[0];
  }
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
      // support both TupleExprType and TupleIntType
      if (unpack_node->expr->type == TupleExprType)
      {
        TupleExprNode* tuple_node = (TupleExprNode*) unpack_node->expr;
        for (size_t i = 0; i < tuple_node->exprlst.size(); i++)
        {
          res->exprlst.push_back(tuple_node->exprlst[i]->run());
        }
      }
      else if (unpack_node->expr->type == TupleIntType)
      {
        TupleIntNode* tuple_int_node = (TupleIntNode*) unpack_node->expr;
        for (size_t i = 0; i < tuple_int_node->tupleint.size(); i++)
        {
          res->exprlst.push_back(new IntValNode(tuple_int_node->tupleint[i]));
        }
      }
      else
      {
        printf("Unsupported node type after UnpackingNode*\n");
        assert(false);
      }
    }
    else
    {
      res->exprlst.push_back(simplified);
    }
  }
  return res;
}

ExprNode* TupleExprNode::Convert2TupleInt(bool allow_star)
{
  // if all nodes in std::vector<Node*> exprlst; are IntValNode(IntValType), then can be converted to TupleIntNode
  std::vector<int> tuple_int;
  for (int i = 0; i < this->exprlst.size(); i++)
  {
    if (this->exprlst[i]->type == IntValType)
    {
      IntValNode* int_node = (IntValNode*) this->exprlst[i];
      tuple_int.push_back(int_node->intval);
    }
    else if (this->exprlst[i]->type == StarExprType && allow_star==true)
    {
      tuple_int.push_back(-1);
    }
    else
    {
      return this;
    }
  }
  return new TupleIntNode(tuple_int);
}

void push_local_symbol_with_top_merge(std::unordered_map<std::string, Node*> x)
{
	if (local_symbol.size() == 0)
	{
		local_symbol.push(x);
	}
	else
	{
		std::unordered_map<std::string, Node*> current_top = local_symbol.top();
		x.insert(current_top.begin(), current_top.end());
		local_symbol.push(x);
	}
}

Node* ForTupleExprNode::run()
{
  std::vector<Node*> result;
  std::unordered_map<std::string, Node*> variable_binding;
  // range can only be TupleExprType or TupleIntType
  if (range->type == TupleExprType)
  {
    TupleExprNode* range_ = (TupleExprNode*) range;
    for (size_t i = 0; i < range_->exprlst.size(); i++)
    {
      Node* feed_in = range_->exprlst[i];
      variable_binding[identifier] = feed_in; // insert or overwrite
      push_local_symbol_with_top_merge(variable_binding);
      Node* res = expr->run();
      local_symbol.pop();
      result.push_back(res);
    }
    return new TupleExprNode(result);
  }
  else // TupleIntType
  {
    TupleIntNode* range_ = (TupleIntNode*) range;
    for (size_t i = 0; i < range_->tupleint.size(); i++)
    {
      Node* feed_in = new IntValNode(range_->tupleint[i]);
      variable_binding[identifier] = feed_in; // insert or overwrite
      push_local_symbol_with_top_merge(variable_binding);
      Node* res = expr->run();
      local_symbol.pop();
      result.push_back(res);
    }
    return new TupleExprNode(result);
  }
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
