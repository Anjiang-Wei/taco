CPPFLAGS=-std=c++14 -O3

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

TACO_LIB_PATH=../build/lib/libtaco.so
TACO_INCLUDE_PATH=../include/
# xlC annoyingly has a bug that causes it to sneakily link in
# IBM's exception library even when it isn't used, so we have to
# add the link invocation for it...
IBM_LIB_PATH=/opt/ibm/lib/libibmc++.so.1

CLEANFILES=bin/* build/*

default: all

all: bin/petsc-converter bin/benchmark

bin/benchmark: make-dirs benchmark.cpp
	-$(PETSC_CXXCOMPILE_SINGLE) -c benchmark.cpp -o build/benchmark.o
	-${CLINKER} build/benchmark.o ${PETSC_KSP_LIB} ${IBM_LIB_PATH} -o bin/benchmark

bin/petsc-converter: make-dirs petsc-converter.cpp
	-$(PETSC_CXXCOMPILE_SINGLE) -I ${TACO_INCLUDE_PATH} -c petsc-converter.cpp -o build/petsc-converter.o
	-${CLINKER} build/petsc-converter.o ${PETSC_KSP_LIB} ${TACO_LIB_PATH} ${IBM_LIB_PATH} -o bin/petsc-converter

make-dirs: bin build
	mkdir -p bin build

petsc-clean:
	rm -f *.exec *.o
