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
    T_Volume = 265,
    T_Has = 266,
    T_Reverse_Dimension = 267,
    T_Positive_Dimension = 268,
    T_AOS = 269,
    T_SOA = 270,
    T_Compact = 271,
    T_Align = 272,
    T_CPU = 273,
    T_GPU = 274,
    T_IO = 275,
    T_PY = 276,
    T_PROC = 277,
    T_OMP = 278,
    T_SYSMEM = 279,
    T_FBMEM = 280,
    T_RDMEM = 281,
    T_ZCMEM = 282,
    T_SOCKMEM = 283,
    T_Int = 284,
    T_Bool = 285,
    T_IPoint = 286,
    T_ISpace = 287,
    T_MSpace = 288,
    T_Def = 289,
    T_Return = 290,
    T_True = 291,
    T_False = 292,
    T_Task = 293,
    T_Region = 294,
    T_Layout = 295,
    T_IndexTaskMap = 296,
    T_Print = 297,
    T_Instance = 298,
    T_Collect = 299,
    T_Le = 300,
    T_Ge = 301,
    T_Eq = 302,
    T_Ne = 303,
    T_And = 304,
    T_Or = 305,
    T_Identifier = 306,
    T_StringConstant = 307,
    T_IntConstant = 308
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
#define T_Volume 265
#define T_Has 266
#define T_Reverse_Dimension 267
#define T_Positive_Dimension 268
#define T_AOS 269
#define T_SOA 270
#define T_Compact 271
#define T_Align 272
#define T_CPU 273
#define T_GPU 274
#define T_IO 275
#define T_PY 276
#define T_PROC 277
#define T_OMP 278
#define T_SYSMEM 279
#define T_FBMEM 280
#define T_RDMEM 281
#define T_ZCMEM 282
#define T_SOCKMEM 283
#define T_Int 284
#define T_Bool 285
#define T_IPoint 286
#define T_ISpace 287
#define T_MSpace 288
#define T_Def 289
#define T_Return 290
#define T_True 291
#define T_False 292
#define T_Task 293
#define T_Region 294
#define T_Layout 295
#define T_IndexTaskMap 296
#define T_Print 297
#define T_Instance 298
#define T_Collect 299
#define T_Le 300
#define T_Ge 301
#define T_Eq 302
#define T_Ne 303
#define T_And 304
#define T_Or 305
#define T_Identifier 306
#define T_StringConstant 307
#define T_IntConstant 308

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
    class APINode* prop;
    class PrintArgsNode* printargs;
    class PrintNode* printstmt;
    class ReturnNode* returnstmt;
    class FuncStmtsNode* funcstmt;
    class ObjectInvokeNode* objinvoke;
    class InstanceLimitNode* instancelimit;

#line 256 "y.tab.c" /* yacc.c:355  */
};
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_Y_TAB_H_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 271 "y.tab.c" /* yacc.c:358  */

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
#define YYFINAL  32
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   424

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  74
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  29
/* YYNRULES -- Number of rules.  */
#define YYNRULES  110
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  197

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   308

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    65,     2,     2,     2,    64,     2,     2,
      68,    69,    62,    60,    55,    61,    70,    63,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    57,    71,
      58,    54,    59,    56,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    66,     2,    67,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    72,     2,    73,     2,     2,     2,     2,
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
      45,    46,    47,    48,    49,    50,    51,    52,    53
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   105,   105,   106,   110,   111,   112,   113,   114,   115,
     116,   117,   118,   122,   123,   126,   130,   134,   138,   139,
     143,   144,   148,   149,   150,   151,   152,   153,   154,   155,
     156,   157,   158,   159,   163,   167,   171,   175,   176,   180,
     181,   182,   186,   190,   194,   195,   199,   200,   204,   208,
     216,   217,   218,   219,   220,   221,   222,   223,   224,   225,
     226,   227,   228,   229,   230,   231,   232,   233,   234,   235,
     236,   237,   238,   239,   240,   241,   242,   243,   247,   248,
     251,   252,   256,   257,   258,   259,   260,   261,   262,   263,
     264,   269,   270,   271,   272,   273,   277,   278,   282,   283,
     287,   288,   289,   290,   291,   296,   297,   298,   299,   300,
     301
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 1
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "T_Size", "T_Split", "T_Merge", "T_Swap",
  "T_Slice", "T_Reverse", "T_Balance_split", "T_Volume", "T_Has",
  "T_Reverse_Dimension", "T_Positive_Dimension", "T_AOS", "T_SOA",
  "T_Compact", "T_Align", "T_CPU", "T_GPU", "T_IO", "T_PY", "T_PROC",
  "T_OMP", "T_SYSMEM", "T_FBMEM", "T_RDMEM", "T_ZCMEM", "T_SOCKMEM",
  "T_Int", "T_Bool", "T_IPoint", "T_ISpace", "T_MSpace", "T_Def",
  "T_Return", "T_True", "T_False", "T_Task", "T_Region", "T_Layout",
  "T_IndexTaskMap", "T_Print", "T_Instance", "T_Collect", "T_Le", "T_Ge",
  "T_Eq", "T_Ne", "T_And", "T_Or", "T_Identifier", "T_StringConstant",
  "T_IntConstant", "'='", "','", "'?'", "':'", "'<'", "'>'", "'+'", "'-'",
  "'*'", "'/'", "'%'", "'!'", "'['", "']'", "'('", "')'", "'.'", "';'",
  "'{'", "'}'", "$accept", "Program", "Stmt", "Identifier_star",
  "InstanceLimit", "MemoryCollect", "ProcCustom", "RegionCustom",
  "LayoutCustom", "Constraints", "FuncDef", "IndexTaskMap", "Assign_Stmt",
  "Func_Stmts", "Func_Stmt", "Return_Stmt", "Print_Stmt", "Print_Args",
  "ArgLst", "ArgLst_", "Expr", "ExprN_1", "ExprN", "Prop", "TYPE",
  "ProcLst", "MemLst", "Mem", "Proc", YY_NULLPTR
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
     305,   306,   307,   308,    61,    44,    63,    58,    60,    62,
      43,    45,    42,    47,    37,    33,    91,    93,    40,    41,
      46,    59,   123,   125
};
# endif

#define YYPACT_NINF -147

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-147)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     370,   -40,     8,     8,     8,     8,   -56,    16,     8,    17,
      14,  -147,  -147,  -147,  -147,  -147,  -147,  -147,  -147,  -147,
    -147,     0,  -147,  -147,   292,     8,     8,    21,    49,   292,
       8,    59,  -147,  -147,   150,  -147,  -147,  -147,  -147,  -147,
    -147,   -42,  -147,    71,    -2,    22,  -147,    44,    31,  -147,
    -147,  -147,  -147,  -147,  -147,  -147,  -147,  -147,    59,    59,
      59,    59,   101,  -147,  -147,  -147,  -147,  -147,  -147,  -147,
      50,    73,    78,   292,  -147,   208,   208,  -147,  -147,    67,
       7,    74,  -147,    66,    66,    66,   155,    62,    59,    59,
      59,    59,    59,    59,    59,    59,    59,    59,    59,    59,
      59,    59,    59,    59,   357,  -147,    63,   150,  -147,  -147,
      -5,  -147,    43,    -7,     4,  -147,   104,    85,  -147,    59,
    -147,    59,  -147,   324,   324,   337,   337,   311,   285,   181,
     324,   324,   354,   354,    66,    66,    66,   207,   233,    99,
    -147,  -147,  -147,  -147,  -147,  -147,  -147,  -147,  -147,  -147,
      65,   107,   208,  -147,  -147,  -147,  -147,  -147,  -147,  -147,
      94,  -147,  -147,  -147,  -147,   233,   233,    59,  -147,    59,
    -147,    59,  -147,    64,  -147,  -147,  -147,  -147,  -147,   113,
     117,   132,   140,   142,   144,   259,   233,   128,  -147,  -147,
    -147,  -147,  -147,  -147,  -147,  -147,  -147
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     2,     7,     8,     4,     5,     6,     9,    10,    11,
      12,     0,    13,    14,     0,     0,     0,     0,     0,     0,
       0,     0,     1,     3,    46,   105,   106,   107,   108,   109,
     110,     0,    96,     0,     0,     0,    44,     0,     0,   100,
     101,   102,   103,   104,    69,    70,    71,    68,     0,     0,
       0,     0,     0,    74,    73,    91,    92,    93,    94,    95,
       0,    47,     0,     0,    17,     0,     0,    22,    22,     0,
       0,     0,    16,    66,    77,    67,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    36,     0,     0,    48,    97,
       0,    98,     0,     0,     0,    35,     0,     0,    15,     0,
      57,     0,    72,    59,    58,    60,    61,    63,    62,     0,
      56,    55,    50,    51,    52,    53,    54,     0,    78,     0,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    75,
       0,     0,     0,    19,    18,    23,    24,    25,    26,    27,
       0,    21,    20,    45,    43,    80,    81,     0,    65,     0,
      64,     0,    39,     0,    37,    40,    41,    49,    99,     0,
       0,     0,     0,     0,     0,    76,    79,     0,    34,    38,
      29,    31,    32,    33,    28,    30,    42
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -147,  -147,   196,   100,  -147,  -147,  -147,  -147,  -147,   129,
    -147,  -147,  -146,  -147,    35,  -147,  -122,  -147,  -147,  -147,
     -58,  -147,  -147,  -147,   102,  -147,   136,   -29,    45
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,    10,    11,    24,    12,    13,    14,    15,    16,   113,
      17,    18,    19,   173,   174,   175,    20,    80,    70,    71,
      62,   139,    87,   149,    72,    41,   110,    63,    64
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      83,    84,    85,    86,   172,   155,   156,   157,   158,   159,
     160,    21,    28,    73,    32,    78,   155,   156,   157,   158,
     159,   160,    49,    50,    51,    52,    53,   172,   176,    74,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   111,   111,     1,    22,
     152,   176,     2,     3,     4,     5,     6,     7,     8,    22,
      23,   165,   116,   166,   161,     9,   153,    29,    34,    42,
      23,    31,    45,    79,    47,   162,   117,    35,    36,    37,
      38,    39,    40,    49,    50,    51,    52,    53,    76,    35,
      36,    37,    38,    39,    40,    54,    55,    81,   152,   171,
     171,    46,    82,    25,    26,    27,     6,     6,    30,   185,
      56,   186,    57,   187,   154,     9,     9,   121,   109,   106,
      58,    59,    22,   178,    60,    43,    44,    61,   107,   108,
      48,   122,   102,    23,   103,   150,   104,   188,   115,   179,
     180,   181,   182,    75,    77,   118,    88,    89,    90,    91,
      92,    93,   183,   184,   169,   163,   164,    94,   177,    95,
      96,    97,    98,    99,   100,   101,   190,   102,   170,   103,
     191,   104,   105,    88,    89,    90,    91,    92,    93,    65,
      66,    67,    68,    69,    94,   192,    95,    96,    97,    98,
      99,   100,   101,   193,   102,   194,   103,   195,   104,   196,
      88,    89,    90,    91,    92,    93,    33,   114,   189,   151,
     119,    94,   112,    95,    96,    97,    98,    99,   100,   101,
       0,   102,     0,   103,   120,   104,    88,    89,    90,    91,
      92,    93,    49,    50,    51,    52,    53,    94,   167,    95,
      96,    97,    98,    99,   100,   101,     0,   102,     0,   103,
       0,   104,    88,    89,    90,    91,    92,    93,     0,     0,
       0,     0,     0,    94,     0,    95,    96,    97,    98,    99,
     100,   101,     0,   102,   168,   103,     0,   104,    88,    89,
      90,    91,    92,    93,     0,     0,     0,     0,     0,    94,
       0,    95,    96,    97,    98,    99,   100,   101,     0,   102,
       0,   103,     0,   104,    88,    89,    90,    91,    92,    93,
      35,    36,    37,    38,    39,    40,     0,    95,    96,    97,
      98,    99,   100,   101,     0,   102,     0,   103,     0,   104,
      88,    89,    90,    91,    92,     0,     0,     0,     0,     0,
       0,     0,     0,    95,    96,    97,    98,    99,   100,   101,
       0,   102,     0,   103,     0,   104,    88,    89,    90,    91,
     140,   141,   142,   143,   144,   145,   146,   147,   148,    95,
      96,    97,    98,    99,   100,   101,     0,   102,     0,   103,
       0,   104,    88,    89,    97,    98,    99,   100,   101,     0,
     102,     0,   103,     0,   104,    95,    96,    97,    98,    99,
     100,   101,     0,   102,     1,   103,     0,   104,     2,     3,
       4,     5,     6,     7,     8,     0,    99,   100,   101,     0,
     102,     9,   103,     0,   104
};

static const yytype_int16 yycheck[] =
{
      58,    59,    60,    61,   150,    12,    13,    14,    15,    16,
      17,    51,    68,    55,     0,    44,    12,    13,    14,    15,
      16,    17,    24,    25,    26,    27,    28,   173,   150,    71,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    99,   100,   101,   102,   103,    75,    76,    34,    51,
      55,   173,    38,    39,    40,    41,    42,    43,    44,    51,
      62,   119,    55,   121,    71,    51,    71,    51,    68,    24,
      62,    54,    51,    51,    29,    71,    69,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    43,    18,
      19,    20,    21,    22,    23,    36,    37,    53,    55,    35,
      35,    52,    71,     3,     4,     5,    42,    42,     8,   167,
      51,   169,    53,   171,    71,    51,    51,    55,    73,    69,
      61,    62,    51,   152,    65,    25,    26,    68,    55,    51,
      30,    69,    66,    62,    68,    72,    70,    73,    71,    45,
      46,    47,    48,    43,    44,    71,    45,    46,    47,    48,
      49,    50,    58,    59,    55,    51,    71,    56,    51,    58,
      59,    60,    61,    62,    63,    64,    53,    66,    69,    68,
      53,    70,    71,    45,    46,    47,    48,    49,    50,    29,
      30,    31,    32,    33,    56,    53,    58,    59,    60,    61,
      62,    63,    64,    53,    66,    53,    68,    53,    70,    71,
      45,    46,    47,    48,    49,    50,    10,    78,   173,   107,
      55,    56,    76,    58,    59,    60,    61,    62,    63,    64,
      -1,    66,    -1,    68,    69,    70,    45,    46,    47,    48,
      49,    50,    24,    25,    26,    27,    28,    56,    57,    58,
      59,    60,    61,    62,    63,    64,    -1,    66,    -1,    68,
      -1,    70,    45,    46,    47,    48,    49,    50,    -1,    -1,
      -1,    -1,    -1,    56,    -1,    58,    59,    60,    61,    62,
      63,    64,    -1,    66,    67,    68,    -1,    70,    45,    46,
      47,    48,    49,    50,    -1,    -1,    -1,    -1,    -1,    56,
      -1,    58,    59,    60,    61,    62,    63,    64,    -1,    66,
      -1,    68,    -1,    70,    45,    46,    47,    48,    49,    50,
      18,    19,    20,    21,    22,    23,    -1,    58,    59,    60,
      61,    62,    63,    64,    -1,    66,    -1,    68,    -1,    70,
      45,    46,    47,    48,    49,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    58,    59,    60,    61,    62,    63,    64,
      -1,    66,    -1,    68,    -1,    70,    45,    46,    47,    48,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    58,
      59,    60,    61,    62,    63,    64,    -1,    66,    -1,    68,
      -1,    70,    45,    46,    60,    61,    62,    63,    64,    -1,
      66,    -1,    68,    -1,    70,    58,    59,    60,    61,    62,
      63,    64,    -1,    66,    34,    68,    -1,    70,    38,    39,
      40,    41,    42,    43,    44,    -1,    62,    63,    64,    -1,
      66,    51,    68,    -1,    70
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    34,    38,    39,    40,    41,    42,    43,    44,    51,
      75,    76,    78,    79,    80,    81,    82,    84,    85,    86,
      90,    51,    51,    62,    77,    77,    77,    77,    68,    51,
      77,    54,     0,    76,    68,    18,    19,    20,    21,    22,
      23,    99,   102,    77,    77,    51,    52,   102,    77,    24,
      25,    26,    27,    28,    36,    37,    51,    53,    61,    62,
      65,    68,    94,   101,   102,    29,    30,    31,    32,    33,
      92,    93,    98,    55,    71,    77,   102,    77,   101,    51,
      91,    53,    71,    94,    94,    94,    94,    96,    45,    46,
      47,    48,    49,    50,    56,    58,    59,    60,    61,    62,
      63,    64,    66,    68,    70,    71,    69,    55,    51,   102,
     100,   101,   100,    83,    83,    71,    55,    69,    71,    55,
      69,    55,    69,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    95,
       3,     4,     5,     6,     7,     8,     9,    10,    11,    97,
      72,    98,    55,    71,    71,    12,    13,    14,    15,    16,
      17,    71,    71,    51,    71,    94,    94,    57,    67,    55,
      69,    35,    86,    87,    88,    89,    90,    51,   101,    45,
      46,    47,    48,    58,    59,    94,    94,    94,    73,    88,
      53,    53,    53,    53,    53,    53,    71
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    74,    75,    75,    76,    76,    76,    76,    76,    76,
      76,    76,    76,    77,    77,    78,    79,    80,    81,    81,
      82,    82,    83,    83,    83,    83,    83,    83,    83,    83,
      83,    83,    83,    83,    84,    85,    86,    87,    87,    88,
      88,    88,    89,    90,    91,    91,    92,    92,    93,    93,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    94,    94,
      94,    94,    94,    94,    94,    94,    94,    94,    95,    95,
      96,    96,    97,    97,    97,    97,    97,    97,    97,    97,
      97,    98,    98,    98,    98,    98,    99,    99,   100,   100,
     101,   101,   101,   101,   101,   102,   102,   102,   102,   102,
     102
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     5,     4,     4,     6,     6,
       6,     6,     0,     2,     2,     2,     2,     2,     4,     4,
       4,     4,     4,     4,     8,     5,     4,     1,     2,     1,
       1,     1,     3,     6,     0,     3,     0,     1,     2,     4,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     4,     4,     2,     2,     1,     1,
       1,     1,     3,     1,     1,     3,     5,     2,     1,     3,
       3,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     3,     1,     3,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1
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
#line 105 "parser.y" /* yacc.c:1646  */
    { root = new ProgramNode(); root->stmt_list.push_back((yyvsp[0].stmt)); (yyval.program) = root; }
#line 1541 "y.tab.c" /* yacc.c:1646  */
    break;

  case 3:
#line 106 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].program)->stmt_list.push_back((yyvsp[0].stmt)); (yyval.program) = (yyvsp[-1].program); }
#line 1547 "y.tab.c" /* yacc.c:1646  */
    break;

  case 4:
#line 110 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].proccustom); }
#line 1553 "y.tab.c" /* yacc.c:1646  */
    break;

  case 5:
#line 111 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].regioncustom); }
#line 1559 "y.tab.c" /* yacc.c:1646  */
    break;

  case 6:
#line 112 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].layoutcustom); }
#line 1565 "y.tab.c" /* yacc.c:1646  */
    break;

  case 7:
#line 113 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].instancelimit); }
#line 1571 "y.tab.c" /* yacc.c:1646  */
    break;

  case 8:
#line 114 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].memorycollect); }
#line 1577 "y.tab.c" /* yacc.c:1646  */
    break;

  case 9:
#line 115 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].funcdef); }
#line 1583 "y.tab.c" /* yacc.c:1646  */
    break;

  case 10:
#line 116 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].indextaskmap); }
#line 1589 "y.tab.c" /* yacc.c:1646  */
    break;

  case 11:
#line 117 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].assign); }
#line 1595 "y.tab.c" /* yacc.c:1646  */
    break;

  case 12:
#line 118 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].printstmt); }
#line 1601 "y.tab.c" /* yacc.c:1646  */
    break;

  case 13:
#line 122 "parser.y" /* yacc.c:1646  */
    { (yyval.string) = (yyvsp[0].string); }
#line 1607 "y.tab.c" /* yacc.c:1646  */
    break;

  case 14:
#line 123 "parser.y" /* yacc.c:1646  */
    { (yyval.string) = "*"; }
#line 1613 "y.tab.c" /* yacc.c:1646  */
    break;

  case 15:
#line 126 "parser.y" /* yacc.c:1646  */
    { (yyval.instancelimit) = new InstanceLimitNode((yyvsp[-3].string), (yyvsp[-2].proc), (yyvsp[-1].intVal)); }
#line 1619 "y.tab.c" /* yacc.c:1646  */
    break;

  case 16:
#line 130 "parser.y" /* yacc.c:1646  */
    { (yyval.memorycollect) = new MemoryCollectNode((yyvsp[-2].string), (yyvsp[-1].string)); }
#line 1625 "y.tab.c" /* yacc.c:1646  */
    break;

  case 17:
#line 134 "parser.y" /* yacc.c:1646  */
    { (yyval.proccustom) = new ProcCustomNode((yyvsp[-2].string), (yyvsp[-1].proclst)); }
#line 1631 "y.tab.c" /* yacc.c:1646  */
    break;

  case 18:
#line 138 "parser.y" /* yacc.c:1646  */
    { (yyval.regioncustom) = new RegionCustomNode((yyvsp[-4].string), (yyvsp[-3].string), (yyvsp[-2].proc), (yyvsp[-1].memlst)); }
#line 1637 "y.tab.c" /* yacc.c:1646  */
    break;

  case 19:
#line 139 "parser.y" /* yacc.c:1646  */
    { assert(strcmp((yyvsp[-2].string), "*") == 0); (yyval.regioncustom) = new RegionCustomNode((yyvsp[-4].string), (yyvsp[-3].string), new ProcNode(ALLPROC), (yyvsp[-1].memlst)); }
#line 1643 "y.tab.c" /* yacc.c:1646  */
    break;

  case 20:
#line 143 "parser.y" /* yacc.c:1646  */
    { (yyval.layoutcustom) = new LayoutCustomNode((yyvsp[-4].string), (yyvsp[-3].string), (yyvsp[-2].mem), (yyvsp[-1].constraints)); }
#line 1649 "y.tab.c" /* yacc.c:1646  */
    break;

  case 21:
#line 144 "parser.y" /* yacc.c:1646  */
    { assert(strcmp((yyvsp[-2].string), "*") == 0); (yyval.layoutcustom) = new LayoutCustomNode((yyvsp[-4].string), (yyvsp[-3].string), new MemNode(ALLMEM), (yyvsp[-1].constraints)); }
#line 1655 "y.tab.c" /* yacc.c:1646  */
    break;

  case 22:
#line 148 "parser.y" /* yacc.c:1646  */
    { (yyval.constraints) = new ConstraintsNode(); }
#line 1661 "y.tab.c" /* yacc.c:1646  */
    break;

  case 23:
#line 149 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("reverse"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1667 "y.tab.c" /* yacc.c:1646  */
    break;

  case 24:
#line 150 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("positive"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1673 "y.tab.c" /* yacc.c:1646  */
    break;

  case 25:
#line 151 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("aos"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1679 "y.tab.c" /* yacc.c:1646  */
    break;

  case 26:
#line 152 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("soa"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1685 "y.tab.c" /* yacc.c:1646  */
    break;

  case 27:
#line 153 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].constraints)->update("compact"); (yyval.constraints) = (yyvsp[-1].constraints); }
#line 1691 "y.tab.c" /* yacc.c:1646  */
    break;

  case 28:
#line 154 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(SMALLER, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1697 "y.tab.c" /* yacc.c:1646  */
    break;

  case 29:
#line 155 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(LE, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1703 "y.tab.c" /* yacc.c:1646  */
    break;

  case 30:
#line 156 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(BIGGER, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1709 "y.tab.c" /* yacc.c:1646  */
    break;

  case 31:
#line 157 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(GE, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1715 "y.tab.c" /* yacc.c:1646  */
    break;

  case 32:
#line 158 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(EQ, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1721 "y.tab.c" /* yacc.c:1646  */
    break;

  case 33:
#line 159 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-3].constraints)->update(NEQ, (yyvsp[0].intVal)); (yyval.constraints) = (yyvsp[-3].constraints); }
#line 1727 "y.tab.c" /* yacc.c:1646  */
    break;

  case 34:
#line 163 "parser.y" /* yacc.c:1646  */
    { (yyval.funcdef) = new FuncDefNode((yyvsp[-6].string), (yyvsp[-4].args), (yyvsp[-1].funcstmt)); }
#line 1733 "y.tab.c" /* yacc.c:1646  */
    break;

  case 35:
#line 167 "parser.y" /* yacc.c:1646  */
    { (yyval.indextaskmap) = new IndexTaskMapNode((yyvsp[-3].string), (yyvsp[-2].string), (yyvsp[-1].string)); }
#line 1739 "y.tab.c" /* yacc.c:1646  */
    break;

  case 36:
#line 171 "parser.y" /* yacc.c:1646  */
    { (yyval.assign) = new AssignNode((yyvsp[-3].string), (yyvsp[-1].expr)); }
#line 1745 "y.tab.c" /* yacc.c:1646  */
    break;

  case 37:
#line 175 "parser.y" /* yacc.c:1646  */
    { FuncStmtsNode* fs = new FuncStmtsNode(); fs->stmtlst.push_back((yyvsp[0].stmt)); (yyval.funcstmt) = fs; }
#line 1751 "y.tab.c" /* yacc.c:1646  */
    break;

  case 38:
#line 176 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-1].funcstmt)->stmtlst.push_back((yyvsp[0].stmt)); }
#line 1757 "y.tab.c" /* yacc.c:1646  */
    break;

  case 39:
#line 180 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].assign); }
#line 1763 "y.tab.c" /* yacc.c:1646  */
    break;

  case 40:
#line 181 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].returnstmt); }
#line 1769 "y.tab.c" /* yacc.c:1646  */
    break;

  case 41:
#line 182 "parser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].printstmt); }
#line 1775 "y.tab.c" /* yacc.c:1646  */
    break;

  case 42:
#line 186 "parser.y" /* yacc.c:1646  */
    { (yyval.returnstmt) = new ReturnNode((yyvsp[-1].expr)); }
#line 1781 "y.tab.c" /* yacc.c:1646  */
    break;

  case 43:
#line 190 "parser.y" /* yacc.c:1646  */
    { (yyval.printstmt) = new PrintNode((yyvsp[-3].string), (yyvsp[-2].printargs)); }
#line 1787 "y.tab.c" /* yacc.c:1646  */
    break;

  case 44:
#line 194 "parser.y" /* yacc.c:1646  */
    { (yyval.printargs) = new PrintArgsNode(); }
#line 1793 "y.tab.c" /* yacc.c:1646  */
    break;

  case 45:
#line 195 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].printargs)->printargs.push_back(new IdentifierExprNode((yyvsp[0].string))); }
#line 1799 "y.tab.c" /* yacc.c:1646  */
    break;

  case 46:
#line 199 "parser.y" /* yacc.c:1646  */
    { (yyval.args) = new ArgLstNode(); }
#line 1805 "y.tab.c" /* yacc.c:1646  */
    break;

  case 47:
#line 200 "parser.y" /* yacc.c:1646  */
    { (yyval.args) = (yyvsp[0].args); }
#line 1811 "y.tab.c" /* yacc.c:1646  */
    break;

  case 48:
#line 204 "parser.y" /* yacc.c:1646  */
    { ArgNode* a = new ArgNode((yyvsp[-1].argtype), (yyvsp[0].string));
                             ArgLstNode* b = new ArgLstNode(); 
                             b->arg_lst.push_back(a);
                             (yyval.args) = b; }
#line 1820 "y.tab.c" /* yacc.c:1646  */
    break;

  case 49:
#line 209 "parser.y" /* yacc.c:1646  */
    { ArgNode* c = new ArgNode((yyvsp[-1].argtype), (yyvsp[0].string));
                             (yyvsp[-3].args)->arg_lst.push_back(c);
                             (yyval.args) = (yyvsp[-3].args); }
#line 1828 "y.tab.c" /* yacc.c:1646  */
    break;

  case 50:
#line 216 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), PLUS, (yyvsp[0].expr)); }
#line 1834 "y.tab.c" /* yacc.c:1646  */
    break;

  case 51:
#line 217 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), MINUS, (yyvsp[0].expr)); }
#line 1840 "y.tab.c" /* yacc.c:1646  */
    break;

  case 52:
#line 218 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), MULTIPLY, (yyvsp[0].expr)); }
#line 1846 "y.tab.c" /* yacc.c:1646  */
    break;

  case 53:
#line 219 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), DIVIDE, (yyvsp[0].expr)); }
#line 1852 "y.tab.c" /* yacc.c:1646  */
    break;

  case 54:
#line 220 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), MOD, (yyvsp[0].expr)); }
#line 1858 "y.tab.c" /* yacc.c:1646  */
    break;

  case 55:
#line 221 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), BIGGER, (yyvsp[0].expr)); }
#line 1864 "y.tab.c" /* yacc.c:1646  */
    break;

  case 56:
#line 222 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), SMALLER, (yyvsp[0].expr)); }
#line 1870 "y.tab.c" /* yacc.c:1646  */
    break;

  case 57:
#line 223 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[-1].expr); }
#line 1876 "y.tab.c" /* yacc.c:1646  */
    break;

  case 58:
#line 224 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), GE, (yyvsp[0].expr)); }
#line 1882 "y.tab.c" /* yacc.c:1646  */
    break;

  case 59:
#line 225 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), LE, (yyvsp[0].expr)); }
#line 1888 "y.tab.c" /* yacc.c:1646  */
    break;

  case 60:
#line 226 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), EQ, (yyvsp[0].expr)); }
#line 1894 "y.tab.c" /* yacc.c:1646  */
    break;

  case 61:
#line 227 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), NEQ, (yyvsp[0].expr)); }
#line 1900 "y.tab.c" /* yacc.c:1646  */
    break;

  case 62:
#line 228 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), OR, (yyvsp[0].expr)); }
#line 1906 "y.tab.c" /* yacc.c:1646  */
    break;

  case 63:
#line 229 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExprNode((yyvsp[-2].expr), AND, (yyvsp[0].expr)); }
#line 1912 "y.tab.c" /* yacc.c:1646  */
    break;

  case 64:
#line 230 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new FuncInvokeNode((yyvsp[-3].expr), (yyvsp[-1].exprn)); }
#line 1918 "y.tab.c" /* yacc.c:1646  */
    break;

  case 65:
#line 231 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new IndexExprNode((yyvsp[-3].expr), (yyvsp[-1].expr)); }
#line 1924 "y.tab.c" /* yacc.c:1646  */
    break;

  case 66:
#line 232 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new NegativeExprNode((yyvsp[0].expr)); }
#line 1930 "y.tab.c" /* yacc.c:1646  */
    break;

  case 67:
#line 233 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new ExclamationNode((yyvsp[0].expr)); }
#line 1936 "y.tab.c" /* yacc.c:1646  */
    break;

  case 68:
#line 234 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new IntValNode((yyvsp[0].intVal)); }
#line 1942 "y.tab.c" /* yacc.c:1646  */
    break;

  case 69:
#line 235 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BoolValNode(true); }
#line 1948 "y.tab.c" /* yacc.c:1646  */
    break;

  case 70:
#line 236 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BoolValNode(false); }
#line 1954 "y.tab.c" /* yacc.c:1646  */
    break;

  case 71:
#line 237 "parser.y" /* yacc.c:1646  */
    { if (!strcmp((yyvsp[0].string), "Machine")) (yyval.expr) = new MSpace(); else (yyval.expr) = new IdentifierExprNode((yyvsp[0].string)); }
#line 1960 "y.tab.c" /* yacc.c:1646  */
    break;

  case 72:
#line 238 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[-1].exprn)->Convert2TupleInt(); }
#line 1966 "y.tab.c" /* yacc.c:1646  */
    break;

  case 73:
#line 239 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[0].proc); }
#line 1972 "y.tab.c" /* yacc.c:1646  */
    break;

  case 74:
#line 240 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[0].mem); }
#line 1978 "y.tab.c" /* yacc.c:1646  */
    break;

  case 75:
#line 241 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new ObjectInvokeNode((yyvsp[-2].expr), (yyvsp[0].prop)); }
#line 1984 "y.tab.c" /* yacc.c:1646  */
    break;

  case 76:
#line 242 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new TenaryExprNode((yyvsp[-4].expr), (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 1990 "y.tab.c" /* yacc.c:1646  */
    break;

  case 77:
#line 243 "parser.y" /* yacc.c:1646  */
    { (yyval.expr) = new UnpackExprNode((yyvsp[0].expr)); }
#line 1996 "y.tab.c" /* yacc.c:1646  */
    break;

  case 78:
#line 247 "parser.y" /* yacc.c:1646  */
    { TupleExprNode* t = new TupleExprNode(); t->exprlst.push_back((yyvsp[0].expr)); (yyval.exprn) = t; }
#line 2002 "y.tab.c" /* yacc.c:1646  */
    break;

  case 79:
#line 248 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].exprn)->exprlst.push_back((yyvsp[0].expr)); (yyval.exprn) = (yyvsp[-2].exprn); }
#line 2008 "y.tab.c" /* yacc.c:1646  */
    break;

  case 80:
#line 251 "parser.y" /* yacc.c:1646  */
    { TupleExprNode* t = new TupleExprNode(); t->exprlst.push_back((yyvsp[-2].expr)); t->exprlst.push_back((yyvsp[0].expr)); (yyval.exprn) = t; }
#line 2014 "y.tab.c" /* yacc.c:1646  */
    break;

  case 81:
#line 252 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].exprn)->exprlst.push_back((yyvsp[0].expr)); (yyval.exprn) = (yyvsp[-2].exprn); }
#line 2020 "y.tab.c" /* yacc.c:1646  */
    break;

  case 82:
#line 256 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(SIZE); }
#line 2026 "y.tab.c" /* yacc.c:1646  */
    break;

  case 83:
#line 257 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(SPLIT); }
#line 2032 "y.tab.c" /* yacc.c:1646  */
    break;

  case 84:
#line 258 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(MERGE); }
#line 2038 "y.tab.c" /* yacc.c:1646  */
    break;

  case 85:
#line 259 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(SWAP); }
#line 2044 "y.tab.c" /* yacc.c:1646  */
    break;

  case 86:
#line 260 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(SLICE); }
#line 2050 "y.tab.c" /* yacc.c:1646  */
    break;

  case 87:
#line 261 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(REVERSE); }
#line 2056 "y.tab.c" /* yacc.c:1646  */
    break;

  case 88:
#line 262 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(BALANCE_SPLIT); }
#line 2062 "y.tab.c" /* yacc.c:1646  */
    break;

  case 89:
#line 263 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(VOLUME); }
#line 2068 "y.tab.c" /* yacc.c:1646  */
    break;

  case 90:
#line 264 "parser.y" /* yacc.c:1646  */
    { (yyval.prop) = new APINode(HAS); }
#line 2074 "y.tab.c" /* yacc.c:1646  */
    break;

  case 91:
#line 269 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(INT); }
#line 2080 "y.tab.c" /* yacc.c:1646  */
    break;

  case 92:
#line 270 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(BOOL); }
#line 2086 "y.tab.c" /* yacc.c:1646  */
    break;

  case 93:
#line 271 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(IPOINT); }
#line 2092 "y.tab.c" /* yacc.c:1646  */
    break;

  case 94:
#line 272 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(ISPACE); }
#line 2098 "y.tab.c" /* yacc.c:1646  */
    break;

  case 95:
#line 273 "parser.y" /* yacc.c:1646  */
    { (yyval.argtype) = new ArgTypeNode(MSPACE); }
#line 2104 "y.tab.c" /* yacc.c:1646  */
    break;

  case 96:
#line 277 "parser.y" /* yacc.c:1646  */
    { ProcLstNode* b = new ProcLstNode(); b->proc_type_lst.push_back((yyvsp[0].proc)->proc_type); (yyval.proclst) = b; }
#line 2110 "y.tab.c" /* yacc.c:1646  */
    break;

  case 97:
#line 278 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].proclst)->proc_type_lst.push_back((yyvsp[0].proc)->proc_type); (yyval.proclst) = (yyvsp[-2].proclst); }
#line 2116 "y.tab.c" /* yacc.c:1646  */
    break;

  case 98:
#line 282 "parser.y" /* yacc.c:1646  */
    { MemLstNode* m = new MemLstNode(); m->mem_type_lst.push_back((yyvsp[0].mem)->mem_type); (yyval.memlst) = m; }
#line 2122 "y.tab.c" /* yacc.c:1646  */
    break;

  case 99:
#line 283 "parser.y" /* yacc.c:1646  */
    { (yyvsp[-2].memlst)->mem_type_lst.push_back((yyvsp[0].mem)->mem_type); (yyval.memlst) = (yyvsp[-2].memlst); }
#line 2128 "y.tab.c" /* yacc.c:1646  */
    break;

  case 100:
#line 287 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(SYSMEM); }
#line 2134 "y.tab.c" /* yacc.c:1646  */
    break;

  case 101:
#line 288 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(FBMEM); }
#line 2140 "y.tab.c" /* yacc.c:1646  */
    break;

  case 102:
#line 289 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(RDMEM); }
#line 2146 "y.tab.c" /* yacc.c:1646  */
    break;

  case 103:
#line 290 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(ZCMEM); }
#line 2152 "y.tab.c" /* yacc.c:1646  */
    break;

  case 104:
#line 291 "parser.y" /* yacc.c:1646  */
    { (yyval.mem) = new MemNode(SOCKMEM); }
#line 2158 "y.tab.c" /* yacc.c:1646  */
    break;

  case 105:
#line 296 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(CPU); }
#line 2164 "y.tab.c" /* yacc.c:1646  */
    break;

  case 106:
#line 297 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(GPU); }
#line 2170 "y.tab.c" /* yacc.c:1646  */
    break;

  case 107:
#line 298 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(IO); }
#line 2176 "y.tab.c" /* yacc.c:1646  */
    break;

  case 108:
#line 299 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(PY); }
#line 2182 "y.tab.c" /* yacc.c:1646  */
    break;

  case 109:
#line 300 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(PROC); }
#line 2188 "y.tab.c" /* yacc.c:1646  */
    break;

  case 110:
#line 301 "parser.y" /* yacc.c:1646  */
    { (yyval.proc) = new ProcNode(OMP); }
#line 2194 "y.tab.c" /* yacc.c:1646  */
    break;


#line 2198 "y.tab.c" /* yacc.c:1646  */
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
#line 304 "parser.y" /* yacc.c:1906  */



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
