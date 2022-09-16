/* A Bison parser, made by GNU Bison 3.0.2.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 1 "parser.y" /* yacc.c:339  */

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

#line 79 "y.tab.c" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 1
#endif

/* In a future release of Bison, this section will be replaced
   by #include "y.tab.h".  */
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
    T_Greedy_split = 266,
    T_Volume = 267,
    T_Has = 268,
    T_Tuple = 269,
    T_For = 270,
    T_In = 271,
    T_Len = 272,
    T_Reverse_Dimension = 273,
    T_Positive_Dimension = 274,
    T_AOS = 275,
    T_SOA = 276,
    T_Compact = 277,
    T_Align = 278,
    T_CPU = 279,
    T_GPU = 280,
    T_IO = 281,
    T_PY = 282,
    T_PROC = 283,
    T_OMP = 284,
    T_SYSMEM = 285,
    T_FBMEM = 286,
    T_RDMEM = 287,
    T_ZCMEM = 288,
    T_SOCKMEM = 289,
    T_Int = 290,
    T_Bool = 291,
    T_IPoint = 292,
    T_ISpace = 293,
    T_MSpace = 294,
    T_Def = 295,
    T_Return = 296,
    T_True = 297,
    T_False = 298,
    T_Task = 299,
    T_Region = 300,
    T_Layout = 301,
    T_IndexTaskMap = 302,
    T_Print = 303,
    T_Instance = 304,
    T_Collect = 305,
    T_Le = 306,
    T_Ge = 307,
    T_Eq = 308,
    T_Ne = 309,
    T_And = 310,
    T_Or = 311,
    T_Identifier = 312,
    T_StringConstant = 313,
    T_IntConstant = 314
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
#define T_Greedy_split 266
#define T_Volume 267
#define T_Has 268
#define T_Tuple 269
#define T_For 270
#define T_In 271
#define T_Len 272
#define T_Reverse_Dimension 273
#define T_Positive_Dimension 274
#define T_AOS 275
#define T_SOA 276
#define T_Compact 277
#define T_Align 278
#define T_CPU 279
#define T_GPU 280
#define T_IO 281
#define T_PY 282
#define T_PROC 283
#define T_OMP 284
#define T_SYSMEM 285
#define T_FBMEM 286
#define T_RDMEM 287
#define T_ZCMEM 288
#define T_SOCKMEM 289
#define T_Int 290
#define T_Bool 291
#define T_IPoint 292
#define T_ISpace 293
#define T_MSpace 294
#define T_Def 295
#define T_Return 296
#define T_True 297
#define T_False 298
#define T_Task 299
#define T_Region 300
#define T_Layout 301
#define T_IndexTaskMap 302
#define T_Print 303
#define T_Instance 304
#define T_Collect 305
#define T_Le 306
#define T_Ge 307
#define T_Eq 308
#define T_Ne 309
#define T_And 310
#define T_Or 311
#define T_Identifier 312
#define T_StringConstant 313
#define T_IntConstant 314

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE YYSTYPE;
union YYSTYPE
{
#line 26 "parser.y" /* yacc.c:355  */

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

#line 270 "y.tab.c" /* yacc.c:355  */
};
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 285 "y.tab.c" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  34
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   530

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  80
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  31
/* YYNRULES -- Number of rules.  */
#define YYNRULES  121
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  220

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   314

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    71,     2,     2,     2,    70,     2,     2,
      74,    75,    68,    66,    61,    67,    76,    69,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    63,    77,
      64,    60,    65,    62,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    72,     2,    73,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    78,     2,    79,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   109,   109,   110,   114,   115,   116,   117,   118,   119,
     120,   121,   122,   126,   127,   131,   132,   136,   140,   144,
     148,   149,   153,   154,   158,   159,   160,   161,   162,   163,
     164,   165,   166,   167,   168,   169,   173,   177,   178,   182,
     186,   187,   191,   192,   193,   197,   201,   205,   206,   210,
     211,   215,   219,   227,   228,   229,   230,   231,   232,   233,
     234,   235,   236,   237,   238,   239,   240,   241,   242,   243,
     244,   246,   247,   248,   249,   250,   251,   252,   253,   254,
     255,   256,   260,   261,   262,   263,   266,   267,   270,   271,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   291,   292,   293,   294,   295,   299,   300,   304,
     305,   309,   310,   311,   312,   313,   318,   319,   320,   321,
     322,   323
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 1
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "T_Size", "T_Split", "T_Merge", "T_Swap",
  "T_Slice", "T_Reverse", "T_Balance_split", "T_Auto_split",
  "T_Greedy_split", "T_Volume", "T_Has", "T_Tuple", "T_For", "T_In",
  "T_Len", "T_Reverse_Dimension", "T_Positive_Dimension", "T_AOS", "T_SOA",
  "T_Compact", "T_Align", "T_CPU", "T_GPU", "T_IO", "T_PY", "T_PROC",
  "T_OMP", "T_SYSMEM", "T_FBMEM", "T_RDMEM", "T_ZCMEM", "T_SOCKMEM",
  "T_Int", "T_Bool", "T_IPoint", "T_ISpace", "T_MSpace", "T_Def",
  "T_Return", "T_True", "T_False", "T_Task", "T_Region", "T_Layout",
  "T_IndexTaskMap", "T_Print", "T_Instance", "T_Collect", "T_Le", "T_Ge",
  "T_Eq", "T_Ne", "T_And", "T_Or", "T_Identifier", "T_StringConstant",
  "T_IntConstant", "'='", "','", "'?'", "':'", "'<'", "'>'", "'+'", "'-'",
  "'*'", "'/'", "'%'", "'!'", "'['", "']'", "'('", "')'", "'.'", "';'",
  "'{'", "'}'", "$accept", "Program", "Stmt", "Identifier_star",
  "Identifier_List", "InstanceLimit", "MemoryCollect", "ProcCustom",
  "RegionCustom", "LayoutCustom", "Constraints", "FuncDef", "IndexTaskMap",
  "Assign_Stmt", "Func_Stmts", "Func_Stmt", "Return_Stmt", "Print_Stmt",
  "Print_Args", "ArgLst", "ArgLst_", "Expr", "SliceExpr", "ExprN_1",
  "ExprN", "Prop", "TYPE", "ProcLst", "MemLst", "Mem", "Proc", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
      61,    44,    63,    58,    60,    62,    43,    45,    42,    47,
      37,    33,    91,    93,    40,    41,    46,    59,   123,   125
};
# endif

#define YYPACT_NINF -141

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-141)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     458,   -26,    -7,    -7,    -7,    18,   -65,     3,    -7,    19,
     131,  -141,  -141,  -141,  -141,  -141,  -141,  -141,  -141,  -141,
    -141,     6,  -141,  -141,   141,    -7,    -7,    21,    49,   -46,
      75,   141,    -7,   130,  -141,  -141,   170,  -141,  -141,  -141,
    -141,  -141,  -141,   -51,  -141,   166,   117,    78,    62,    63,
      85,  -141,    82,    76,    72,  -141,  -141,  -141,  -141,  -141,
    -141,  -141,  -141,  -141,   130,   130,   130,   163,  -141,  -141,
    -141,  -141,  -141,  -141,  -141,   107,   122,   127,   141,  -141,
     270,   270,  -141,  -141,  -141,  -141,  -141,  -141,   -45,   109,
    -141,   130,    45,    45,   217,    29,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
      69,   130,   513,  -141,   118,   170,  -141,  -141,   -49,  -141,
     -48,   -15,     0,   130,   126,  -141,     2,   130,  -141,   130,
    -141,   412,   412,   425,   425,   399,   373,   243,   412,   412,
     386,   386,    45,    45,    45,   130,   269,   137,    12,   321,
      52,  -141,  -141,  -141,  -141,  -141,  -141,  -141,  -141,  -141,
    -141,  -141,  -141,  -141,    67,   154,   270,  -141,  -141,  -141,
    -141,  -141,  -141,  -141,   148,  -141,  -141,   321,  -141,   164,
     321,   321,   130,   321,   130,  -141,   130,  -141,  -141,   130,
    -141,    66,  -141,  -141,  -141,  -141,  -141,   161,   165,   167,
     177,   179,   188,   206,   347,   321,   321,   190,  -141,  -141,
    -141,  -141,  -141,  -141,  -141,  -141,   130,  -141,   295,  -141
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     2,     7,     8,     4,     5,     6,     9,    10,    11,
      12,     0,    13,    14,     0,     0,     0,    13,     0,     0,
       0,     0,     0,     0,     1,     3,    49,   116,   117,   118,
     119,   120,   121,     0,   107,     0,     0,     0,     0,     0,
       0,    47,     0,     0,     0,   111,   112,   113,   114,   115,
      72,    73,    74,    71,     0,     0,     0,     0,    77,    76,
     102,   103,   104,   105,   106,     0,    50,     0,     0,    19,
       0,     0,    24,    24,    15,    37,    38,    16,     0,     0,
      18,     0,    70,    80,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    39,     0,     0,    51,   108,     0,   109,
       0,     0,     0,     0,     0,    17,     0,     0,    60,     0,
      75,    62,    61,    63,    64,    66,    65,     0,    59,    58,
      53,    54,    55,    56,    57,    85,    86,     0,     0,    86,
       0,    90,    91,    92,    93,    94,    95,    96,    97,    98,
      99,   100,   101,    78,     0,     0,     0,    21,    20,    25,
      26,    27,    28,    29,     0,    23,    22,    48,    46,     0,
      88,    89,     0,    83,    84,    68,     0,    69,    67,     0,
      42,     0,    40,    43,    44,    52,   110,     0,     0,     0,
       0,     0,     0,     0,    79,    82,    87,     0,    36,    41,
      31,    33,    34,    35,    30,    32,     0,    45,     0,    81
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -141,  -141,   238,    84,  -141,  -141,  -141,  -141,  -141,  -141,
     168,  -141,  -141,  -140,  -141,    58,  -141,  -139,  -141,  -141,
    -141,   -64,  -141,   139,  -141,  -141,   138,  -141,   180,   -32,
      60
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    10,    11,    24,    29,    12,    13,    14,    15,    16,
     121,    17,    18,    19,   191,   192,   193,    20,    88,    75,
      76,    67,   147,   148,    95,   163,    77,    43,   118,    68,
      69
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      92,    93,    94,   169,   170,   171,   172,   173,   174,    30,
      78,    49,   166,   166,    83,    50,   123,   179,   169,   170,
     171,   172,   173,   174,   190,   194,    79,   126,   167,   168,
     124,    21,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   146,   149,   119,   119,
      22,   190,   194,    96,    97,    98,    99,   100,   101,   177,
      31,    23,   175,   180,   102,   181,   103,   104,   105,   106,
     107,   108,   109,   186,   110,    27,   111,   176,   112,    33,
      36,   183,    47,    54,    44,   187,    23,    25,    26,    28,
     129,    52,    32,    37,    38,    39,    40,    41,    42,    55,
      56,    57,    58,    59,   130,    81,    48,   189,   189,    45,
      46,    60,    61,   186,     6,     6,    53,   110,   204,   111,
     205,   112,   206,     9,     9,   207,    62,   188,    63,    80,
      82,    34,   145,    51,   196,    84,    64,    65,   117,    85,
      86,    89,    87,    66,    54,   208,    91,    55,    56,    57,
      58,    59,   218,    90,    37,    38,    39,    40,    41,    42,
      55,    56,    57,    58,    59,    37,    38,    39,    40,    41,
      42,     1,    60,    61,    22,     2,     3,     4,     5,     6,
       7,     8,   114,   115,   116,    23,   125,    62,     9,    63,
      37,    38,    39,    40,    41,    42,   164,    64,    65,   197,
     198,   199,   200,   178,    66,    70,    71,    72,    73,    74,
     185,   195,   201,   202,    96,    97,    98,    99,   100,   101,
     210,   203,   216,    22,   211,   102,   212,   103,   104,   105,
     106,   107,   108,   109,    23,   110,   213,   111,   214,   112,
     113,    96,    97,    98,    99,   100,   101,   215,    35,   209,
     150,   122,   102,   165,   103,   104,   105,   106,   107,   108,
     109,   120,   110,     0,   111,     0,   112,   217,    96,    97,
      98,    99,   100,   101,     0,     0,     0,     0,   127,   102,
       0,   103,   104,   105,   106,   107,   108,   109,     0,   110,
       0,   111,   128,   112,    96,    97,    98,    99,   100,   101,
      55,    56,    57,    58,    59,   102,   182,   103,   104,   105,
     106,   107,   108,   109,     0,   110,     0,   111,     0,   112,
      96,    97,    98,    99,   100,   101,     0,     0,     0,     0,
       0,   102,   184,   103,   104,   105,   106,   107,   108,   109,
       0,   110,     0,   111,     0,   112,    96,    97,    98,    99,
     100,   101,     0,     0,     0,     0,     0,   102,     0,   103,
     104,   105,   106,   107,   108,   109,     0,   110,     0,   111,
     219,   112,    96,    97,    98,    99,   100,   101,     0,     0,
       0,     0,     0,   102,     0,   103,   104,   105,   106,   107,
     108,   109,     0,   110,     0,   111,     0,   112,    96,    97,
      98,    99,   100,   101,     0,     0,     0,     0,     0,     0,
       0,   103,   104,   105,   106,   107,   108,   109,     0,   110,
       0,   111,     0,   112,    96,    97,    98,    99,   100,     0,
       0,     0,     0,     0,     0,     0,     0,   103,   104,   105,
     106,   107,   108,   109,     0,   110,     0,   111,     0,   112,
      96,    97,    98,    99,   107,   108,   109,     0,   110,     0,
     111,     0,   112,   103,   104,   105,   106,   107,   108,   109,
       0,   110,     0,   111,     0,   112,    96,    97,   105,   106,
     107,   108,   109,     0,   110,     0,   111,     0,   112,   103,
     104,   105,   106,   107,   108,   109,     0,   110,     1,   111,
       0,   112,     2,     3,     4,     5,     6,     7,     8,     0,
       0,     0,     0,     0,     0,     9,   151,   152,   153,   154,
     155,   156,   157,   158,   159,   160,   161,     0,     0,     0,
     162
};

static const yytype_int16 yycheck[] =
{
      64,    65,    66,    18,    19,    20,    21,    22,    23,    74,
      61,    57,    61,    61,    46,    61,    61,    15,    18,    19,
      20,    21,    22,    23,   164,   164,    77,    91,    77,    77,
      75,    57,    96,    97,    98,    99,   100,   101,   102,   103,
     104,   105,   106,   107,   108,   109,   110,   111,    80,    81,
      57,   191,   191,    51,    52,    53,    54,    55,    56,   123,
      57,    68,    77,   127,    62,   129,    64,    65,    66,    67,
      68,    69,    70,    61,    72,    57,    74,    77,    76,    60,
      74,   145,    61,    14,    24,    73,    68,     3,     4,     5,
      61,    31,     8,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    75,    45,    57,    41,    41,    25,
      26,    42,    43,    61,    48,    48,    32,    72,   182,    74,
     184,    76,   186,    57,    57,   189,    57,    75,    59,    45,
      46,     0,    63,    58,   166,    57,    67,    68,    78,    77,
      77,    59,    57,    74,    14,    79,    74,    30,    31,    32,
      33,    34,   216,    77,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    24,    25,    26,    27,    28,
      29,    40,    42,    43,    57,    44,    45,    46,    47,    48,
      49,    50,    75,    61,    57,    68,    77,    57,    57,    59,
      24,    25,    26,    27,    28,    29,    78,    67,    68,    51,
      52,    53,    54,    77,    74,    35,    36,    37,    38,    39,
      73,    57,    64,    65,    51,    52,    53,    54,    55,    56,
      59,    57,    16,    57,    59,    62,    59,    64,    65,    66,
      67,    68,    69,    70,    68,    72,    59,    74,    59,    76,
      77,    51,    52,    53,    54,    55,    56,    59,    10,   191,
     111,    83,    62,   115,    64,    65,    66,    67,    68,    69,
      70,    81,    72,    -1,    74,    -1,    76,    77,    51,    52,
      53,    54,    55,    56,    -1,    -1,    -1,    -1,    61,    62,
      -1,    64,    65,    66,    67,    68,    69,    70,    -1,    72,
      -1,    74,    75,    76,    51,    52,    53,    54,    55,    56,
      30,    31,    32,    33,    34,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    -1,    72,    -1,    74,    -1,    76,
      51,    52,    53,    54,    55,    56,    -1,    -1,    -1,    -1,
      -1,    62,    63,    64,    65,    66,    67,    68,    69,    70,
      -1,    72,    -1,    74,    -1,    76,    51,    52,    53,    54,
      55,    56,    -1,    -1,    -1,    -1,    -1,    62,    -1,    64,
      65,    66,    67,    68,    69,    70,    -1,    72,    -1,    74,
      75,    76,    51,    52,    53,    54,    55,    56,    -1,    -1,
      -1,    -1,    -1,    62,    -1,    64,    65,    66,    67,    68,
      69,    70,    -1,    72,    -1,    74,    -1,    76,    51,    52,
      53,    54,    55,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    64,    65,    66,    67,    68,    69,    70,    -1,    72,
      -1,    74,    -1,    76,    51,    52,    53,    54,    55,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    64,    65,    66,
      67,    68,    69,    70,    -1,    72,    -1,    74,    -1,    76,
      51,    52,    53,    54,    68,    69,    70,    -1,    72,    -1,
      74,    -1,    76,    64,    65,    66,    67,    68,    69,    70,
      -1,    72,    -1,    74,    -1,    76,    51,    52,    66,    67,
      68,    69,    70,    -1,    72,    -1,    74,    -1,    76,    64,
      65,    66,    67,    68,    69,    70,    -1,    72,    40,    74,
      -1,    76,    44,    45,    46,    47,    48,    49,    50,    -1,
      -1,    -1,    -1,    -1,    -1,    57,     3,     4,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    -1,    -1,    -1,
      17
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    40,    44,    45,    46,    47,    48,    49,    50,    57,
      81,    82,    85,    86,    87,    88,    89,    91,    92,    93,
      97,    57,    57,    68,    83,    83,    83,    57,    83,    84,
      74,    57,    83,    60,     0,    82,    74,    24,    25,    26,
      27,    28,    29,   107,   110,    83,    83,    61,    57,    57,
      61,    58,   110,    83,    14,    30,    31,    32,    33,    34,
      42,    43,    57,    59,    67,    68,    74,   101,   109,   110,
      35,    36,    37,    38,    39,    99,   100,   106,    61,    77,
      83,   110,    83,   109,    57,    77,    77,    57,    98,    59,
      77,    74,   101,   101,   101,   104,    51,    52,    53,    54,
      55,    56,    62,    64,    65,    66,    67,    68,    69,    70,
      72,    74,    76,    77,    75,    61,    57,   110,   108,   109,
     108,    90,    90,    61,    75,    77,   101,    61,    75,    61,
      75,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,    63,   101,   102,   103,   101,
     103,     3,     4,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    17,   105,    78,   106,    61,    77,    77,    18,
      19,    20,    21,    22,    23,    77,    77,   101,    77,    15,
     101,   101,    63,   101,    63,    73,    61,    73,    75,    41,
      93,    94,    95,    96,    97,    57,   109,    51,    52,    53,
      54,    64,    65,    57,   101,   101,   101,   101,    79,    95,
      59,    59,    59,    59,    59,    59,    16,    77,   101,    75
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    80,    81,    81,    82,    82,    82,    82,    82,    82,
      82,    82,    82,    83,    83,    84,    84,    85,    86,    87,
      88,    88,    89,    89,    90,    90,    90,    90,    90,    90,
      90,    90,    90,    90,    90,    90,    91,    92,    92,    93,
      94,    94,    95,    95,    95,    96,    97,    98,    98,    99,
      99,   100,   100,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   101,   101,   101,   101,   101,   101,   101,   101,
     101,   101,   102,   102,   102,   102,   103,   103,   104,   104,
     105,   105,   105,   105,   105,   105,   105,   105,   105,   105,
     105,   105,   106,   106,   106,   106,   106,   107,   107,   108,
     108,   109,   109,   109,   109,   109,   110,   110,   110,   110,
     110,   110
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     3,     3,     5,     4,     4,
       6,     6,     6,     6,     0,     2,     2,     2,     2,     2,
       4,     4,     4,     4,     4,     4,     8,     4,     4,     4,
       1,     2,     1,     1,     1,     3,     6,     0,     3,     0,
       1,     2,     4,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     4,     4,     4,
       2,     1,     1,     1,     1,     3,     1,     1,     3,     5,
       2,     8,     3,     2,     2,     1,     1,     3,     3,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     3,     1,
       3,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 109 "parser.y" /* yacc.c:1646  */
    { root = new ProgramNode(); root->stmt_list.push_back((yyvsp[0].stmt)); (yyval.program) = root; }
#line 1590 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 110 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].program)->stmt_list.push_back((yyvsp[0].stmt)); (yyval.program) = (yyvsp[-1].program); }
#line 1596 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 114 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].proccustom); }
#line 1602 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 115 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].regioncustom); }
#line 1608 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 116 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].layoutcustom); }
#line 1614 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 117 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].instancelimit); }
#line 1620 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 118 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].memorycollect); }
#line 1626 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 119 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].funcdef); }
#line 1632 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 120 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].indextaskmap); }
#line 1638 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 121 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].assign); }
#line 1644 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 122 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].printstmt); }
#line 1650 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 126 "parser.y" /* yacc.c:1646  */
    { (yyval.string) = (yyvsp[0].string); }
#line 1656 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 127 "parser.y" /* yacc.c:1646  */
    { (yyval.string) = "*"; }
#line 1662 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 131 "parser.y" /* yacc.c:1646  */
    { (yyval.stringlist) = new IdentifierLstNode((yyvsp[-2].string), (yyvsp[0].string)); }
#line 1668 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 132 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].stringlist)->append((yyvsp[0].string)); }
#line 1674 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 136 "parser.y" /* yacc.c:1646  */
    { (yyval.instancelimit) = new InstanceLimitNode((yyvsp[-3].string), (yyvsp[-2].proc), (yyvsp[-1].intVal)); }
#line 1680 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 140 "parser.y" /* yacc.c:1646  */
    { (yyval.memorycollect) = new MemoryCollectNode((yyvsp[-2].string), (yyvsp[-1].string)); }
#line 1686 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 144 "parser.y" /* yacc.c:1646  */
    { (yyval.proccustom) = new ProcCustomNode((yyvsp[-2].string), (yyvsp[-1].proclst)); }
#line 1692 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 148 "parser.y" /* yacc.c:1646  */
    { (yyval.regioncustom) = new RegionCustomNode((yyvsp[-4].string), (yyvsp[-3].string), (yyvsp[-2].proc), (yyvsp[-1].memlst)); }
#line 1698 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 149 "parser.y" /* yacc.c:1646  */
    { assert(strcmp((yyvsp[-2].string), "*") == 0); (yyval.regioncustom) = new RegionCustomNode((yyvsp[-4].string), (yyvsp[-3].string), new ProcNode(ALLPROC), (yyvsp[-1].memlst)); }
#line 1704 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 153 "parser.y" /* yacc.c:1646  */
    { (yyval.layoutcustom) = new LayoutCustomNode((yyvsp[-4].string), (yyvsp[-3].string), (yyvsp[-2].mem), (yyvsp[-1].constraints)); }
#line 1710 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 154 "parser.y" /* yacc.c:1646  */
    { assert(strcmp((yyvsp[-2].string), "*") == 0); (yyval.layoutcustom) = new LayoutCustomNode((yyvsp[-4].string), (yyvsp[-3].string), new MemNode(ALLMEM), (yyvsp[-1].constraints)); }
#line 1716 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 158 "parser.y" /* yacc.c:1646  */
    { (yyval.constraints) = new ConstraintsNode(); }
#line 1722 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 159 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("reverse"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1728 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 160 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("positive"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1734 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 161 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("aos"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1740 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 162 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("soa"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1746 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 163 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("compact"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1752 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 164 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(SMALLER, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1758 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 165 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(LE, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1764 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 166 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(BIGGER, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1770 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 167 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(GE, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1776 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 168 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(EQ, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1782 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 169 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(NEQ, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1788 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 173 "parser.y" /* yacc.c:1646  */
    { (yyval.funcdef) = new FuncDefNode((yyvsp[-6].string), (yyvsp[-4].args), (yyvsp[-1].funcstmt)); }
#line 1794 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 177 "parser.y" /* yacc.c:1646  */
    { (yyval.indextaskmap) = new IndexTaskMapNode((yyvsp[-2].string), (yyvsp[-1].string)); }
#line 1800 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 178 "parser.y" /* yacc.c:1646  */
    { (yyval.indextaskmap) =  new IndexTaskMapNode((yyvsp[-2].stringlist), (yyvsp[-1].string)); }
#line 1806 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 182 "parser.y" /* yacc.c:1646  */
    { (yyval.assign) = new AssignNode((yyvsp[-3].string), (yyvsp[-1].expr)); }
#line 1812 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 186 "parser.y" /* yacc.c:1646  */
    { FuncStmtsNode* fs = new FuncStmtsNode(); fs->stmtlst.push_back((yyvsp[0].stmt)); (yyval.funcstmt) = fs; }
#line 1818 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 187 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].funcstmt)->stmtlst.push_back((yyvsp[0].stmt)); }
#line 1824 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 191 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].assign); }
#line 1830 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 192 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].returnstmt); }
#line 1836 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 193 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].printstmt); }
#line 1842 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 197 "parser.y" /* yacc.c:1646  */
    { (yyval.returnstmt) = new ReturnNode((yyvsp[-1].expr)); }
#line 1848 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 201 "parser.y" /* yacc.c:1646  */
    { (yyval.printstmt) = new PrintNode((yyvsp[-3].string), (yyvsp[-2].printargs)); }
#line 1854 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 205 "parser.y" /* yacc.c:1646  */
    { (yyval.printargs) = new PrintArgsNode(); }
#line 1860 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 206 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].printargs)->printargs.push_back((yyvsp[0].expr)); }
#line 1866 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 210 "parser.y" /* yacc.c:1646  */
    { (yyval.args) = new ArgLstNode(); }
#line 1872 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 211 "parser.y" /* yacc.c:1646  */
    { (yyval.args) = (yyvsp[0].args); }
#line 1878 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 215 "parser.y" /* yacc.c:1646  */
    { ArgNode* a = new ArgNode((yyvsp[-1].argtype), (yyvsp[0].string));
                             ArgLstNode* b = new ArgLstNode(); 
                             b->arg_lst.push_back(a);
                             (yyval.args) = b; }
#line 1887 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 220 "parser.y" /* yacc.c:1646  */
    { ArgNode* c = new ArgNode((yyvsp[-1].argtype), (yyvsp[0].string));
                             (yyvsp[-3].args)->arg_lst.push_back(c);
                             (yyval.args) = (yyvsp[-3].args); }
#line 1895 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 227 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), PLUS, (yyvsp[0].expr)); }
#line 1901 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 228 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), MINUS, (yyvsp[0].expr)); }
#line 1907 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 229 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), MULTIPLY, (yyvsp[0].expr)); }
#line 1913 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 230 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), DIVIDE, (yyvsp[0].expr)); }
#line 1919 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 231 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), MOD, (yyvsp[0].expr)); }
#line 1925 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 232 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), BIGGER, (yyvsp[0].expr)); }
#line 1931 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 233 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), SMALLER, (yyvsp[0].expr)); }
#line 1937 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 234 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[-1].expr); }
#line 1943 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 235 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), GE, (yyvsp[0].expr)); }
#line 1949 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 236 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), LE, (yyvsp[0].expr)); }
#line 1955 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 237 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), EQ, (yyvsp[0].expr)); }
#line 1961 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 238 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), NEQ, (yyvsp[0].expr)); }
#line 1967 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 239 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), OR, (yyvsp[0].expr)); }
#line 1973 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 240 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), AND, (yyvsp[0].expr)); }
#line 1979 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 241 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new FuncInvokeNode((yyvsp[-3].expr), (yyvsp[-1].exprn)); }
#line 1985 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 242 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new IndexExprNode((yyvsp[-3].expr), (yyvsp[-1].sliceexpr)); }
#line 1991 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 243 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new IndexExprNode((yyvsp[-3].expr), (yyvsp[-1].exprn)); }
#line 1997 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 244 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new NegativeExprNode((yyvsp[0].expr)); }
#line 2003 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 246 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new IntValNode((yyvsp[0].intVal)); }
#line 2009 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 247 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BoolValNode(true); }
#line 2015 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 248 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BoolValNode(false); }
#line 2021 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 249 "parser.y" /* yacc.c:1646  */
    { if (!strcmp((yyvsp[0].string), "Machine")) (yyval.expr) = new MSpace(); else (yyval.expr) = new IdentifierExprNode((yyvsp[0].string)); }
#line 2027 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 250 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[-1].exprn)->Convert2TupleInt(); }
#line 2033 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 251 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[0].proc); }
#line 2039 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 252 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[0].mem); }
#line 2045 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 253 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new ObjectInvokeNode((yyvsp[-2].expr), (yyvsp[0].prop)); }
#line 2051 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 254 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new TenaryExprNode((yyvsp[-4].expr), (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 2057 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 255 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new UnpackExprNode((yyvsp[0].expr)); }
#line 2063 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 256 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new ForTupleExprNode((yyvsp[-5].expr), (yyvsp[-3].string), (yyvsp[-1].expr)); }
#line 2069 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 260 "parser.y" /* yacc.c:1646  */
    { (yyval.sliceexpr) = new SliceExprNode((yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 2075 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 261 "parser.y" /* yacc.c:1646  */
    { (yyval.sliceexpr) = new SliceExprNode(NULL, (yyvsp[0].expr)); }
#line 2081 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 262 "parser.y" /* yacc.c:1646  */
    { (yyval.sliceexpr) = new SliceExprNode((yyvsp[-1].expr), NULL); }
#line 2087 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 263 "parser.y" /* yacc.c:1646  */
    { (yyval.sliceexpr) = new SliceExprNode(NULL, NULL); }
#line 2093 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 266 "parser.y" /* yacc.c:1646  */
    { TupleExprNode* t = new TupleExprNode(); t->exprlst.push_back((yyvsp[0].expr)); (yyval.exprn) = t; }
#line 2099 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 267 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].exprn)->exprlst.push_back((yyvsp[0].expr)); (yyval.exprn) = (yyvsp[-2].exprn); }
#line 2105 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 270 "parser.y" /* yacc.c:1646  */
    { TupleExprNode* t = new TupleExprNode(); t->exprlst.push_back((yyvsp[-2].expr)); t->exprlst.push_back((yyvsp[0].expr)); (yyval.exprn) = t; }
#line 2111 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 271 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].exprn)->exprlst.push_back((yyvsp[0].expr)); (yyval.exprn) = (yyvsp[-2].exprn); }
#line 2117 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 275 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(SIZE); }
#line 2123 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 276 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(SPLIT); }
#line 2129 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 277 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(MERGE); }
#line 2135 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 278 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(SWAP); }
#line 2141 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 279 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(SLICE); }
#line 2147 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 280 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(REVERSE); }
#line 2153 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 281 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(BALANCE_SPLIT); }
#line 2159 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 282 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(AUTO_SPLIT); }
#line 2165 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 283 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(GREEDY_SPLIT); }
#line 2171 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 284 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(VOLUME); }
#line 2177 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 285 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(HAS); }
#line 2183 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 286 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(LEN); }
#line 2189 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 291 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(INT); }
#line 2195 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 292 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(BOOL); }
#line 2201 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 293 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(IPOINT); }
#line 2207 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 294 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(ISPACE); }
#line 2213 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 295 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(MSPACE); }
#line 2219 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 299 "parser.y" /* yacc.c:1646  */
    { ProcLstNode* b = new ProcLstNode(); b->proc_type_lst.push_back((yyvsp[0].proc)->proc_type); (yyval.proclst) = b; }
#line 2225 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 300 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].proclst)->proc_type_lst.push_back((yyvsp[0].proc)->proc_type); (yyval.proclst) = (yyvsp[-2].proclst); }
#line 2231 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 304 "parser.y" /* yacc.c:1646  */
    { MemLstNode* m = new MemLstNode(); m->mem_type_lst.push_back((yyvsp[0].mem)->mem_type); (yyval.memlst) = m; }
#line 2237 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 305 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].memlst)->mem_type_lst.push_back((yyvsp[0].mem)->mem_type); (yyval.memlst) = (yyvsp[-2].memlst); }
#line 2243 "y.tab.c" /* yacc.c:1646  */
    break;

  case 111:
#line 309 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(SYSMEM); }
#line 2249 "y.tab.c" /* yacc.c:1646  */
    break;

  case 112:
#line 310 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(FBMEM); }
#line 2255 "y.tab.c" /* yacc.c:1646  */
    break;

  case 113:
#line 311 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(RDMEM); }
#line 2261 "y.tab.c" /* yacc.c:1646  */
    break;

  case 114:
#line 312 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(ZCMEM); }
#line 2267 "y.tab.c" /* yacc.c:1646  */
    break;

  case 115:
#line 313 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(SOCKMEM); }
#line 2273 "y.tab.c" /* yacc.c:1646  */
    break;

  case 116:
#line 318 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(CPU); }
#line 2279 "y.tab.c" /* yacc.c:1646  */
    break;

  case 117:
#line 319 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(GPU); }
#line 2285 "y.tab.c" /* yacc.c:1646  */
    break;

  case 118:
#line 320 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(IO); }
#line 2291 "y.tab.c" /* yacc.c:1646  */
    break;

  case 119:
#line 321 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(PY); }
#line 2297 "y.tab.c" /* yacc.c:1646  */
    break;

  case 120:
#line 322 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(PROC); }
#line 2303 "y.tab.c" /* yacc.c:1646  */
    break;

  case 121:
#line 323 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(OMP); }
#line 2309 "y.tab.c" /* yacc.c:1646  */
    break;


#line 2313 "y.tab.c" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 326 "parser.y" /* yacc.c:1906  */



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
