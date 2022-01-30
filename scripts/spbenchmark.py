#!/usr/bin/env python3

import argparse
from enum import Enum, auto
import os
from pathlib import Path
import subprocess
import sys

from registry import *

def getTensorDir():
    if "TENSOR_DIR" not in os.environ:
        raise AssertionError("TENSOR_DIR must be set in the environment.")
    return os.environ["TENSOR_DIR"]

class BenchmarkKind(Enum):
    SpMV = auto()
    SpMSpV = auto()
    SpMM = auto()
    SDDMM = auto()
    SpAdd3 = auto()
    SpTTV = auto()
    SpMTTKRP = auto()
    SpInnerProd = auto()

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def names():
        return [str(b) for b in BenchmarkKind]

    @staticmethod
    def getByName(name):
        filtered = list(filter(lambda bk: str(bk) == name, list(BenchmarkKind)))
        assert(len(filtered) == 1)
        return filtered[0]

# Parent class for all benchmarks.
class Benchmark:
    def __init__(self, gpu, niter, warmup, extraArgs=[]):
        self.gpu = gpu
        self.warmup = warmup
        self.niter = niter
        # TODO (rohany): I'm not sure what the type of extraArgs is. Perhaps
        #  a list of arguments to directly pass to the binary?
        self.extraArgs = extraArgs

    def getCommand(self, tensor, benchKind, nodes):
        pass

class DISTALBenchmark(Benchmark):
    def getCommand(self, tensor, benchKind, nodes):
        # TODO (rohany): Standardize the DISTAL binary arguments more...
        # TODO (rohany): Handle getting rotated / input tensors for the multi-tensor arguments.
        args = {
            # TODO (rohany): Handle extra arguments...
            BenchmarkKind.SpMV: ["-csr", self.getDISTALTensor(tensor, "csr")],
            BenchmarkKind.SpMSpV: ["-csc", "?", "-spx", "?"],
            # TODO (rohany): Thread through the jdim here.
            BenchmarkKind.SpMM: ["-tensor", self.getDISTALTensor(tensor, 'csr')],
            # TODO (rohany): Thread through the jdim here.
            BenchmarkKind.SDDMM: ["-csr", self.getDISTALTensor(tensor, 'csr')],
            BenchmarkKind.SpAdd3: ["-tensorB", self.getDISTALTensor(tensor, 'csr'), 
                                   "-tensorC", self.getShiftedTensor(tensor, 'csr'), 
                                   "-tensorD", self.getShiftedTensor(tensor, 'csr')],
            BenchmarkKind.SpTTV: ["-tensor", self.getDISTALTensor(tensor, 'dss')],
            # TODO (rohany): Pass through the ldim here.
            BenchmarkKind.SpMTTKRP: ["-tensor", self.getDISTALTensor(tensor, 'dss')],
            # TODO (rohany): For some tensors, like the patents tensor, we might want to
            #  do a different format here.
            BenchmarkKind.SpInnerProd: ["-tensorB", self.getDISTALTensor(tensor, 'dss'), 
                                        "-tensorC", self.getShiftedTensor(tensor, 'dss')],
        }
        lassenPrefix = ["jsrun", "-b", "none", "-c", "ALL_CPUS", "-g", "ALL_GPUS", "-r", "1", "-n", str(nodes)]
        commonArgs = ["-n", str(self.niter), "-warmup", str(self.warmup)]
        assert(not self.gpu)
        legionArgs = ["-ll:ocpu", "2",
                      "-ll:othr", "18",
                      "-ll:onuma", "1",
                      "-ll:nsize", "75G",
                      "-ll:ncsize", "0",
                      "-ll:util", "2",
                      "-tm:numa_aware_alloc"]
        assert(benchKind in args)
        return lassenPrefix + [self.getBinary(benchKind)] + legionArgs + commonArgs + args[benchKind]

    def getBinary(self, benchKind):
        tacoDir = os.environ.get("TACO_BUILD_DIR", None)
        if (tacoDir is None):
            raise AssertionError("TACO_BUILD_DIR must be defined in environment.")
        if benchKind == BenchmarkKind.SpMSpV:
            binaryName = "spmv"
        else:
            binaryName = str(benchKind)
        if self.gpu:
            binaryName += "-cuda"
        return str(Path(tacoDir, "bin", binaryName))

    def getDISTALTensor(self, tensor, formatStr):
        name = f"{tensor.name}.{formatStr}.hdf5"
        path = Path(getTensorDir(), "distal", name)
        return str(path)

    def getShiftedTensor(self, tensor, formatStr):
        return self.getDISTALTensor(tensor, formatStr)

class CTFBenchmark(Benchmark):
    def getCommand(self, tensor, benchKind, nodes):
        args = {
            BenchmarkKind.SpMV: ["-bench", "spmv"],
            # TODO (rohany): IDK if this benchmark is legit cuz CTF doesn't have CSC.
            BenchmarkKind.SpMSpV: ["-bench", "spmspv", "-spmspvvec", "?"],
            # TODO (rohany): Thread through the jdim here.
            BenchmarkKind.SpMM: ["-bench", "spmm"],
            # TODO (rohany): Thread through the jdim here.
            BenchmarkKind.SDDMM: ["-bench", "sddmm"],
            BenchmarkKind.SpAdd3: ["-bench", "spadd3", "-tensorC", self.getShiftedTensor(tensor), "-tensorD", self.getShiftedTensor(tensor)],
            BenchmarkKind.SpTTV: ["-bench", "spttv"],
            # TODO (rohany): Pass through the ldim here.
            BenchmarkKind.SpMTTKRP: ["-bench", "spmttkrp"],
            BenchmarkKind.SpInnerProd: ["-bench", "spinnerprod", "-tensorC", self.getShiftedTensor(tensor)],
        }
        lassenPrefix = ["jsrun", "-b", "rs", "-c", "1", "-r", "40", "-n", str(40 * nodes)]
        commonArgs = ["-tensor", self.getCTFTensor(tensor), "-dims", ",".join([str(d) for d in tensor.dims]), "-n", str(self.niter), "-warmup", str(self.warmup)]
        ctfDir = os.environ.get("CTF_DIR", None)
        if (ctfDir is None):
            raise AssertionError("CTF_DIR must be defined in environment.")
        binary = Path(ctfDir, "bin", "spbench")
        return lassenPrefix + [str(binary)] + commonArgs + args[benchKind]

    def getCTFTensor(self, tensor):
        name = f"{tensor.name}.tns"
        path = Path(getTensorDir(), "coo-txt", name)
        return str(path)

    def getShiftedTensor(self, tensor):
        return self.getCTFTensor(tensor)

class PETScBenchmark(Benchmark):
    def getCommand(self, tensor, benchKind, nodes):
        args = {
            BenchmarkKind.SpMV: ["-bench", "spmv"],
            # TODO (rohany): Support passing through the JDim value.
            BenchmarkKind.SpMM: ["-bench", "spmm"],
            # TODO (rohany): Support getting the rotated tensors.
            BenchmarkKind.SpAdd3: ["-bench", "spadd3", 
                                   "-add3MatrixC", self.getShiftedMatrix(tensor),
                                   "-add3MatrixD", self.getShiftedMatrix(tensor)]
        }
        if benchKind not in args:
            raise AssertionError(f"Unsupported PETSc benchmark: {benchKind}")
        lassenPrefix = ["jsrun", "-n", str(40 * nodes), "-r", "40", "-c", "1", "-b", "rs"]
        commonArgs = ["-matrix", self.getPETScMatrix(tensor), "-n", str(self.niter), "-warmup", str(self.warmup)]
        petscDir = os.environ.get("PETSC_BUILD_DIR", None)
        if (petscDir is None):
            raise AssertionError("PETSC_BUILD_DIR must be defined in environment.")
        binary = Path(petscDir, "bin", "benchmark")
        assert(binary.exists())
        return lassenPrefix + [str(binary)] + commonArgs + args[benchKind]

    def getPETScMatrix(self, tensor):
        name = f"{tensor.name}.petsc"
        path = Path(getTensorDir(), "petsc", name)
        return str(path)

    def getShiftedMatrix(self, tensor):
        return self.getPETScMatrix(tensor)

class TrilinosBenchmark(Benchmark):
    def getCommand(self, tensor, benchKind, nodes):
        args = {
            BenchmarkKind.SpMV: ["--bench=spmv"],
            # TODO (rohany): Support passing through the JDim value.
            BenchmarkKind.SpMM: ["--bench=spmm"],
            # TODO (rohany): Support getting the rotated tensors.
            BenchmarkKind.SpAdd3: ["--bench=spadd3",
                                   f"--add3TensorC={self.getShiftedMatrix(tensor)}",
                                   f"--add3TensorD={self.getShiftedMatrix(tensor)}"]
        }
        if benchKind not in args:
            raise AssertionError(f"Unsupported Trilinos benchmark: {benchKind}")
        # TODO (rohany): Experiment with the optimal run configuration.
        lassenPrefix = ["jsrun", "-n", str(2 * nodes), "-r", "2", "-c", "20", "-b", "rs"]
        commonArgs = [f"--file={self.getTrilinosMatrix(tensor)}", f"--n={self.niter}", f"--warmup={self.warmup}"]
        trilinosDir = os.environ.get("TRILINOS_BUILD_DIR")
        if (trilinosDir is None):
            raise AssertionError("TRILINOS_BUILD_DIR must be defined in environment.")
        wrapper = Path(trilinosDir, "..", "trilinos_run_wrapper.sh")
        binary = Path(trilinosDir, "bin", "benchmark")
        assert(binary.exists())
        return lassenPrefix + [str(wrapper), str(binary)] + commonArgs + args[benchKind]

    def getTrilinosMatrix(self, tensor):
        name = f"{tensor.name}.mtx"
        path = Path(getTensorDir(), "coo-txt", name)
        return str(path)

    def getShiftedMatrix(self, tensor):
        return self.getTrilinosMatrix(tensor)

def executeCmd(cmd):
    cmdStr = " ".join(cmd)
    print("Executing command: {}".format(cmdStr))
    sys.stdout.flush()
    proc = subprocess.Popen(cmd)
    proc.wait()
    sys.stdout.flush()

def serializeBenchmark(system, benchKind, tensor, nodes):
    return f"BENCHID++{system}++{benchKind}++{tensor.name}++{nodes}"

def main():
    # Initialize the sparse tensor registry.
    registry = SparseTensorRegistry.initialize()

    parser = argparse.ArgumentParser()
    # TODO (rohany): Have an option to run all systems.
    parser.add_argument("system", type=str, choices=["DISTAL", "CTF", "PETSc", "Trilinos"])
    # TODO (rohany): Have option to run all benchmarks.
    parser.add_argument("bench", type=str, choices=BenchmarkKind.names())
    parser.add_argument("tensor", type=str, choices=registry.getAllNames() + ["all", "all-matrices", "all-3-tensors"])
    parser.add_argument("--nodes", type=int, nargs='+', help="Node counts to run out", default=[1])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dry-run", default=False, action="store_true")
    args = parser.parse_args()

    if args.system == "DISTAL":
        bencher = DISTALBenchmark(False, args.n, args.warmup)
    elif args.system == "CTF":
        bencher = CTFBenchmark(False, args.n, args.warmup)
    elif args.system == "PETSc":
        bencher = PETScBenchmark(False, args.n, args.warmup)
    elif args.system == "Trilinos":
        bencher = TrilinosBenchmark(False, args.n, args.warmup)
    else:
        assert(False)

    benchKind = BenchmarkKind.getByName(args.bench)

    if args.tensor == "all":
        tensors = registry.getAll()
    elif args.tensor == "all-matrices":
        tensors = [t for t in registry.getAll() if t.order == 2]
    elif args.tensor == "all-3-tensors":
        tensors = [t for t in registry.getAll() if t.order == 3]
    else:
        tensors = [registry.getByName(args.tensor)]

    for tensor in tensors:
        for n in args.nodes:
            cmd = bencher.getCommand(tensor, benchKind, n)
            if (args.dry_run):
                print(" ".join(cmd))
            else:
                print(serializeBenchmark(args.system, benchKind, tensor, n))
                executeCmd(cmd)

if __name__ == '__main__':
    main()
