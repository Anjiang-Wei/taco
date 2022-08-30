/* A Bison parser, made by GNU Bison 3.0.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 1
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    T_Size = 258,
    T_Split = 259,
    T_Merge = 260,
    T_Swap = 261,
    T_Slice = 262,
    T_Reverse = 263,
    T_Balance_split = 264,
    T_Auto_split = 265,
    T_Volume = 266,
    T_Has = 267,
    T_Tuple = 268,
    T_For = 269,
    T_In = 270,
    T_Len = 271,
    T_Reverse_Dimension = 272,
    T_Positive_Dimension = 273,
    T_AOS = 274,
    T_SOA = 275,
    T_Compact = 276,
    T_Align = 277,
    T_CPU = 278,
    T_GPU = 279,
    T_IO = 280,
    T_PY = 281,
    T_PROC = 282,
    T_OMP = 283,
    T_SYSMEM = 284,
    T_FBMEM = 285,
    T_RDMEM = 286,
    T_ZCMEM = 287,
    T_SOCKMEM = 288,
    T_Int = 289,
    T_Bool = 290,
    T_IPoint = 291,
    T_ISpace = 292,
    T_MSpace = 293,
    T_Def = 294,
    T_Return = 295,
    T_True = 296,
    T_False = 297,
    T_Task = 298,
    T_Region = 299,
    T_Layout = 300,
    T_IndexTaskMap = 301,
    T_Print = 302,
    T_Instance = 303,
    T_Collect = 304,
    T_Le = 305,
    T_Ge = 306,
    T_Eq = 307,
    T_Ne = 308,
    T_And = 309,
    T_Or = 310,
    T_Identifier = 311,
    T_StringConstant = 312,
    T_IntConstant = 313
  };
#endif
/* Tokens.  */
#define T_Size 258
#define T_Split 259
#define T_Merge 260
#define T_Swap 261
#define T_Slice 262
#define T_Reverse 263
#define T_Balance_split 264
#define T_Auto_split 265
#define T_Volume 266
#define T_Has 267
#define T_Tuple 268
#define T_For 269
#define T_In 270
#define T_Len 271
#define T_Reverse_Dimension 272
#define T_Positive_Dimension 273
#define T_AOS 274
#define T_SOA 275
#define T_Compact 276
#define T_Align 277
#define T_CPU 278
#define T_GPU 279
#define T_IO 280
#define T_PY 281
#define T_PROC 282
#define T_OMP 283
#define T_SYSMEM 284
#define T_FBMEM 285
#define T_RDMEM 286
#define T_ZCMEM 287
#define T_SOCKMEM 288
#define T_Int 289
#define T_Bool 290
#define T_IPoint 291
#define T_ISpace 292
#define T_MSpace 293
#define T_Def 294
#define T_Return 295
#define T_True 296
#define T_False 297
#define T_Task 298
#define T_Region 299
#define T_Layout 300
#define T_IndexTaskMap 301
#define T_Print 302
#define T_Instance 303
#define T_Collect 304
#define T_Le 305
#define T_Ge 306
#define T_Eq 307
#define T_Ne 308
#define T_And 309
#define T_Or 310
#define T_Identifier 311
#define T_StringConstant 312
#define T_IntConstant 313

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE YYSTYPE;
union YYSTYPE
{
#line 26 "parser.y" /* yacc.c:1909  */

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
    class MemoryCollectNode* memorycollect;
    class ArgTypeNode* argtype;
    class AssignNode* assign;
    class ExprNode* expr;
    class IndexTaskMapNode* indextaskmap;
    class FuncDefNode* funcdef;
    class ArgLstNode* args;
    class TupleExprNode* exprn;
    class SliceExprNode* sliceexpr;
    class APINode* prop;
    class PrintArgsNode* printargs;
    class PrintNode* printstmt;
    class ReturnNode* returnstmt;
    class FuncStmtsNode* funcstmt;
    class ObjectInvokeNode* objinvoke;
    class InstanceLimitNode* instancelimit;
    class IdentifierLstNode* stringlist;

#line 203 "y.tab.h" /* yacc.c:1909  */
};
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */
