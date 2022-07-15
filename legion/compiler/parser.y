%{
#define YYDEBUG 1
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include "tree.hpp"
#include "tree.cpp"
#include "MSpace.cpp"
int yylex();
void yyerror(const char*);
%}

%define parse.error verbose

%token T_Size T_Split T_Merge T_Swap T_Slice T_Reverse T_Balance_split T_Volume T_Has
%token T_Reverse_Dimension T_Positive_Dimension T_AOS T_SOA T_Compact T_Align
%token T_CPU T_GPU T_IO T_PY T_PROC T_OMP
%token T_SYSMEM T_FBMEM T_RDMEM T_ZCMEM T_SOCKMEM
%token T_Int T_Bool T_IPoint T_ISpace T_MSpace T_Def T_Return T_True T_False
%token T_Task T_Region T_Layout T_IndexTaskMap T_Print
%token T_Le T_Ge T_Eq T_Ne
%token T_And T_Or

%union
{
    char* string;
    int intVal;

    class ProgramNode* program;
    class StmtNode* stmt;
    class ProcLstNode* proclst;
    class ProcNode* proc;
    class MemLstNode* memlst;
    class MemNode* mem;
    class ProcCustomNode* proccustom;
    class RegionCustomNode* regioncustom;
    class LayoutCustomNode* layoutcustom;
    class ConstraintsNode* constraints;
    class ArgTypeNode* argtype;
    class AssignNode* assign;
    class ExprNode* expr;
    class IndexTaskMapNode* indextaskmap;
    class FuncDefNode* funcdef;
    class ArgLstNode* args;
    class TupleExprNode* exprn;
    class APINode* prop;
    class PrintArgsNode* printargs;
    class PrintNode* printstmt;
    class ReturnNode* returnstmt;
    class FuncStmtsNode* funcstmt;
    class ObjectInvokeNode* objinvoke;
}

// Semantic value
%token <string> T_Identifier
%token <string> T_StringConstant
%token <intVal> T_IntConstant
%type <program> Program
%type <stmt> Stmt
%type <proclst> ProcLst
%type <proc> Proc
%type <memlst> MemLst
%type <mem> Mem
%type <proccustom> ProcCustom
%type <regioncustom> RegionCustom
%type <layoutcustom> LayoutCustom
%type <constraints> Constraints
%type <argtype> TYPE
%type <assign> Assign_Stmt
%type <expr> Expr
%type <indextaskmap> IndexTaskMap
%type <funcdef> FuncDef
%type <args> ArgLst
%type <args> ArgLst_
%type <exprn> ExprN
%type <exprn> ExprN_1
%type <prop> Prop
%type <printargs> Print_Args
%type <printstmt> Print_Stmt
%type <returnstmt> Return_Stmt
%type <funcstmt> Func_Stmts
%type <stmt> Func_Stmt

%left '='
%left ','
%left '?' ':'
%left T_Or
%left T_And
%left T_Eq T_Ne
%left '<' '>' T_Le T_Ge
%left '+' '-'
%left '*' '/' '%'
%left '!'
%left '[' ']' '(' ')' '.'

%%

Program:
    Stmt                { root = new ProgramNode(); root->stmt_list.push_back($1); $$ = root; }
|   Program Stmt        { $1->stmt_list.push_back($2); $$ = $1; }
;

Stmt:
    ProcCustom        { $$ = $1; }
|   RegionCustom      { $$ = $1; }
|   LayoutCustom      { $$ = $1; }
|   FuncDef           { $$ = $1; }
|   IndexTaskMap      { $$ = $1; }
|   Assign_Stmt       { $$ = $1; }
|   Print_Stmt        { $$ = $1; }
;

ProcCustom:
    T_Task T_Identifier ProcLst  ';'  { $$ = new ProcCustomNode($2, $3); }
;

RegionCustom:
    T_Region T_Identifier T_Identifier T_Identifier MemLst ';' { $$ = new RegionCustomNode($2, $3, $4, $5); }
;

LayoutCustom:
    T_Layout T_Identifier T_Identifier T_Identifier Constraints ';'   { $$ = new LayoutCustomNode($2, $3, $4, $5); }
;

Constraints:
    /* empty */                             { $$ = new ConstraintsNode(); }
|   Constraints T_Reverse_Dimension         { $1->update("row"); $$ = $1; }
|   Constraints T_Positive_Dimension        { $1->update("column"); $$ = $1; }
|   Constraints T_AOS                       { $1->update("aos"); $$ = $1; }
|   Constraints T_SOA                       { $1->update("soa"); $$ = $1; }
|   Constraints T_Compact                   { $1->update("compact"); $$ = $1; }
|   Constraints T_Align '<' T_IntConstant   { $1->update(SMALLER, $4); $$ =f $1; }
|   Constraints T_Align T_Le T_IntConstant  { $1->update(LE, $4); $$ = $1; }
|   Constraints T_Align '>' T_IntConstant   { $1->update(BIGGER, $4); $$ = $1; }
|   Constraints T_Align T_Ge T_IntConstant  { $1->update(GE, $4); $$ = $1; }
|   Constraints T_Align T_Eq T_IntConstant  { $1->update(EQ, $4); $$ = $1; }
|   Constraints T_Align T_Ne T_IntConstant  { $1->update(NEQ, $4); $$ = $1; }
;

FuncDef:
    T_Def T_Identifier '(' ArgLst ')' '{' Func_Stmts '}' { $$ = new FuncDefNode($2, $4, $7); }
;

IndexTaskMap:
    T_IndexTaskMap T_Identifier T_Identifier T_Identifier ';' { $$ = new IndexTaskMapNode($2, $3, $4); }
;

Assign_Stmt:
    T_Identifier '=' Expr ';'   { $$ = new AssignNode($1, $3); }
;

Func_Stmts:
    Func_Stmt               { FuncStmtsNode* fs = new FuncStmtsNode(); fs->stmtlst.push_back($1); $$ = fs; }
|   Func_Stmts Func_Stmt    { $1->stmtlst.push_back($2); }
;

Func_Stmt:
    Assign_Stmt   { $$ = $1; }
|   Return_Stmt   { $$ = $1; }
|   Print_Stmt    { $$ = $1; }
;

Return_Stmt:
    T_Return Expr ';'      { $$ = new ReturnNode($2); }
;

Print_Stmt:
    T_Print  '(' T_StringConstant Print_Args ')'  ';'  { $$ = new PrintNode($3, $4); }
;

Print_Args:
    /* empty */                 { $$ = new PrintArgsNode(); }
|   Print_Args ',' T_Identifier { $1->printargs.push_back(new IdentifierExprNode($3)); }
;

ArgLst:
    /* empty */             { $$ = new ArgLstNode(); }
|   ArgLst_                 { $$ = $1; }
;

ArgLst_:
    TYPE T_Identifier      { ArgNode* a = new ArgNode($1, $2);
                             ArgLstNode* b = new ArgLstNode(); 
                             b->arg_lst.push_back(a);
                             $$ = b; }
|   ArgLst_ ',' TYPE T_Identifier
                           { ArgNode* c = new ArgNode($3, $4);
                             $1->arg_lst.push_back(c);
                             $$ = $1; }
;


Expr:
    Expr '+' Expr           { $$ = new BinaryExprNode($1, PLUS, $3); }
|   Expr '-' Expr           { $$ = new BinaryExprNode($1, MINUS, $3); }
|   Expr '*' Expr           { $$ = new BinaryExprNode($1, MULTIPLY, $3); }
|   Expr '/' Expr           { $$ = new BinaryExprNode($1, DIVIDE, $3); }
|   Expr '%' Expr           { $$ = new BinaryExprNode($1, MOD, $3); }
|   Expr '>' Expr           { $$ = new BinaryExprNode($1, BIGGER, $3); }
|   Expr '<' Expr           { $$ = new BinaryExprNode($1, SMALLER, $3); }
|   '(' Expr ')'            { $$ = $2; }
|   Expr T_Ge Expr          { $$ = new BinaryExprNode($1, GE, $3); }
|   Expr T_Le Expr          { $$ = new BinaryExprNode($1, LE, $3); }
|   Expr T_Eq Expr          { $$ = new BinaryExprNode($1, EQ, $3); }
|   Expr T_Ne Expr          { $$ = new BinaryExprNode($1, NEQ, $3); }
|   Expr T_Or Expr          { $$ = new BinaryExprNode($1, OR, $3); }
|   Expr T_And Expr         { $$ = new BinaryExprNode($1, AND, $3); }
|   Expr '(' ExprN_1 ')'      { $$ = new FuncInvokeNode($1, $3); }
|   Expr '[' Expr ']'       { $$ = new IndexExprNode($1, $3); }
|   '-' Expr %prec '!'      { $$ = new NegativeExprNode($2); }
|   T_IntConstant           { $$ = new IntValNode($1); }
|   T_True                  { $$ = new BoolValNode(true); }
|   T_False                 { $$ = new BoolValNode(false); }
|   T_Identifier            { if (!strcmp($1, "Machine")) $$ = new MSpace(); else $$ = new IdentifierExprNode($1); }
|   '(' ExprN ')'           { $$ = $2; }
|   Proc                    { $$ = $1; }
|   Mem                     { $$ = $1; }
|   Expr '.' Prop           { $$ = new ObjectInvokeNode($1, $3); }
|   Expr '?' Expr ':' Expr %prec '?' { $$ = new TenaryExprNode($1, $3, $5); }
|   '*' Expr                { $$ = new UnpackExprNode($2); }
;

ExprN_1:
    Expr                     { TupleExprNode* t = new TupleExprNode(); t->exprlst.push_back($1); $$ = t; }
|   ExprN_1 ',' Expr         { $1->exprlst.push_back($3); $$ = $1; }

ExprN:
    Expr ',' Expr           { TupleExprNode* t = new TupleExprNode(); t->exprlst.push_back($1); t->exprlst.push_back($3); $$ = t; }
|   ExprN ',' Expr          { $1->exprlst.push_back($3); $$ = $1; }
;

Prop:
    T_Size                  { $$ = new APINode(SIZE); }
|   T_Split                 { $$ = new APINode(SPLIT); }
|   T_Merge                 { $$ = new APINode(MERGE); }
|   T_Swap                  { $$ = new APINode(SWAP); }
|   T_Slice                 { $$ = new APINode(SLICE); }
|   T_Reverse               { $$ = new APINode(REVERSE); }
|   T_Balance_split         { $$ = new APINode(BALANCE_SPLIT); }
|   T_Volume                { $$ = new APINode(VOLUME); }
|   T_Has                   { $$ = new APINode(HAS); }
;


TYPE:
    T_Int                   { $$ = new ArgTypeNode(INT); }
|   T_Bool                  { $$ = new ArgTypeNode(BOOL); }
|   T_IPoint                { $$ = new ArgTypeNode(IPOINT); }
|   T_ISpace                { $$ = new ArgTypeNode(ISPACE); }
|   T_MSpace                { $$ = new ArgTypeNode(MSPACE); }
;

ProcLst:
    Proc                    { ProcLstNode* b = new ProcLstNode(); b->proc_type_lst.push_back($1->proc_type); $$ = b; }
|   ProcLst ',' Proc        { $1->proc_type_lst.push_back($3->proc_type); $$ = $1; }
;

MemLst:
    Mem                     { MemLstNode* m = new MemLstNode(); m->mem_type_lst.push_back($1->mem_type); $$ = m; }
|   MemLst ',' Mem          { $1->mem_type_lst.push_back($3->mem_type); $$ = $1; }
;

Mem:
    T_SYSMEM                { $$ = new MemNode(SYSMEM); }
|   T_FBMEM                 { $$ = new MemNode(FBMEM); }
|   T_RDMEM                 { $$ = new MemNode(RDMEM); }
|   T_ZCMEM                 { $$ = new MemNode(ZCMEM); }
|   T_SOCKMEM               { $$ = new MemNode(SOCKMEM); }
;


Proc:
    T_CPU                   { $$ = new ProcNode(CPU); }
|   T_GPU                   { $$ = new ProcNode(GPU); }
|   T_IO                    { $$ = new ProcNode(IO); }
|   T_PY                    { $$ = new ProcNode(PY); }
|   T_PROC                  { $$ = new ProcNode(PROC); }
|   T_OMP                   { $$ = new ProcNode(OMP); }
;

%%


/* int main()
{
    extern FILE* yyin;
    yyin = fopen("in.py", "r");
    if (yyin == NULL)
    {
        std::cout << "Mapping policy file does not exist" << std::endl;
        assert(false);
    }
    yyparse();
    std::cout << root->stmt_list.size() << std::endl;
    // root->print();
    root->run();
} */

// flex scanner.l; bison -vdty parser.y;
