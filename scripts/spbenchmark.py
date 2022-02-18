#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess
import sys

from registry import *
from benchkinds import *

def getTensorDir():
    if "TENSOR_DIR" not in os.environ:
        raise AssertionError("TENSOR_DIR must be set in the environment.")
    return os.environ["TENSOR_DIR"]

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
    def getCommand(self, tensor, benchKind, procs):
        # TODO (rohany): Standardize the DISTAL binary arguments more...
        args = {
            BenchmarkKind.SpMV: ["-csr", self.getDISTALTensor(tensor, "csr")] + (["-tm:align128"] if self.gpu else []),
            BenchmarkKind.SpMSpV: ["-csc", self.getDISTALTensor(tensor, "csc"), "-spx", self.getUniformVec(tensor)],
            # TODO (rohany): Thread through the jdim here.
            BenchmarkKind.SpMM: ["-tensor", self.getDISTALTensor(tensor, 'csr')] + (["-lg:eager_alloc_percentage", "5"] if self.gpu and tensor.name == "mycielskian19" else []),
            # TODO (rohany): Thread through the jdim here.
            BenchmarkKind.SDDMM: ["-csr", self.getDISTALTensor(tensor, 'csr')],
            BenchmarkKind.SpAdd3: ["-tensorB", self.getDISTALTensor(tensor, 'csr'), 
                                   "-tensorC", self.getShiftedTensor(tensor, 'csr', 0), 
                                   "-tensorD", self.getShiftedTensor(tensor, 'csr', 1),
                                   "-tm:untrack_valid_regions"] + (["-lg:eager_alloc_percentage", "5"] if self.gpu else []),
            BenchmarkKind.SpTTV: ["-tensor", self.getDISTALTensor(tensor, 'dss')] + (["-pos"] if self.gpu or tensor.name == "patents" else []),
            # TODO (rohany): Pass through the ldim here.
            BenchmarkKind.SpMTTKRP: ["-tensor", self.getDISTALTensor(tensor, 'dds' if tensor.name in ["patents"] and not self.gpu else "dss")] + (["-dds"] if tensor.name in ["patents"] and not self.gpu else []),
            # TODO (rohany): For some tensors, like the patents tensor, we might want to
            #  do a different format here.
            BenchmarkKind.SpInnerProd: ["-tensorB", self.getDISTALTensor(tensor, 'dds' if tensor.name in ["patents", "nell-2"] else "dss"), 
                                        "-tensorC", self.getShiftedTensor(tensor, 'dds' if tensor.name in ["patents", "nell-2"]  else "dss", 0)] + 
                                       (["-tm:untrack_valid_regions", "-dds"] if tensor.name in ["patents", "nell-2"] else []),
        }
        if self.gpu:
            # Lassen has 4 GPUs per node.
            nodes = (procs + 3) // 4
        else:
            nodes = procs
        lassenPrefix = ["jsrun", "-b", "none", "-c", "ALL_CPUS", "-g", "ALL_GPUS", "-r", "1", "-n", str(nodes)]
        commonArgs = ["-n", str(self.niter), "-warmup", str(self.warmup)]
        legionArgs = ["-ll:ocpu", "2",
                      "-ll:othr", "18",
                      "-ll:onuma", "1",
                      "-ll:nsize", "100G",
                      "-ll:ncsize", "0",
                      "-ll:util", "2",
                      "-tm:numa_aware_alloc"]
        if self.gpu:
            if nodes == 1:
                gpus = procs
            else:
                gpus = 4
            # Allocate as much GPU memory as physically possible.
            fbSize = "15.3G"
            if benchKind == BenchmarkKind.SpMV:
                # We'll allocate slightly less memory in this case for CuSparse, as the cusparseHandle_t
                # takes up a non-trivial amount of memory.
                fbSize = "15G"
            legionArgs += [
                "-ll:gpu", str(gpus),
                "-ll:fsize", fbSize,
                "-pieces", str(procs),
            ]

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

    def getShiftedTensor(self, tensor, formatStr, amount):
        name = f"{tensor.name}-rotated-{amount}.{formatStr}.hdf5"
        path = Path(getTensorDir(), "distal", name)
        return str(path)

    def getUniformVec(self, tensor):
      name = f"{tensor.name}-uniform-vec.vec.hdf5"
      path = Path(getTensorDir(), "distal", name)
      return str(path)

class CTFBenchmark(Benchmark):
    def getCommand(self, tensor, benchKind, nodes):
        # This will be disabled until CTF supports GPUs.
        assert(not self.gpu)
        args = {
            BenchmarkKind.SpMV: ["-bench", "spmv"],
            # TODO (rohany): IDK if this benchmark is legit cuz CTF doesn't have CSC.
            BenchmarkKind.SpMSpV: ["-bench", "spmspv", "-spmspvvec", "?"],
            # TODO (rohany): Thread through the jdim here.
            BenchmarkKind.SpMM: ["-bench", "spmm"],
            # TODO (rohany): Thread through the jdim here.
            BenchmarkKind.SDDMM: ["-bench", "sddmm"],
            BenchmarkKind.SpAdd3: ["-bench", "spadd3", "-tensorC", self.getShiftedTensor(tensor, 0), "-tensorD", self.getShiftedTensor(tensor, 1)],
            BenchmarkKind.SpTTV: ["-bench", "spttv"],
            # TODO (rohany): Pass through the ldim here.
            BenchmarkKind.SpMTTKRP: ["-bench", "spmttkrp"],
            BenchmarkKind.SpInnerProd: ["-bench", "spinnerprod", "-tensorC", self.getShiftedTensor(tensor, 0)],
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

    def getShiftedTensor(self, tensor, amount):
        name = f"{tensor.name}-rotated-{amount}.tns"
        path = Path(getTensorDir(), "coo-txt", name)
        return str(path)

class PETScBenchmark(Benchmark):
    def getCommand(self, tensor, benchKind, nodes):
        args = {
            BenchmarkKind.SpMV: ["-bench", "spmv"],
            # TODO (rohany): Support passing through the JDim value.
            BenchmarkKind.SpMM: ["-bench", "spmm"],
            BenchmarkKind.SpAdd3: ["-bench", "spadd3", 
                                   "-add3MatrixC", self.getShiftedMatrix(tensor, 0),
                                   "-add3MatrixD", self.getShiftedMatrix(tensor, 1)]
        }
        if benchKind not in args:
            raise AssertionError(f"Unsupported PETSc benchmark: {benchKind}")
        if self.gpu:
            physNodes = (nodes + 3) // 4
            if physNodes == 1:
                rsPerHost = nodes
            else:
                rsPerHost = 4
            lassenPrefix = ["jsrun", "-n", str(nodes), "-g", "1", "-r", str(rsPerHost), "-c", "10", "-b", "rs", "-M", "\"-gpu\""]
        else:
            lassenPrefix = ["jsrun", "-n", str(40 * nodes), "-r", "40", "-c", "1", "-b", "rs"]
        commonArgs = ["-matrix", self.getPETScMatrix(tensor), "-n", str(self.niter), "-warmup", str(self.warmup)]
        if self.gpu:
            commonArgs += ["-enable_gpu", "-vec_type", "cuda", "-mat_type", "aijcusparse"]
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

    def getShiftedMatrix(self, tensor, amount):
        name = f"{tensor.name}-rotated-{amount}.petsc"
        path = Path(getTensorDir(), "petsc", name)
        return str(path)

class TrilinosBenchmark(Benchmark):
    def getCommand(self, tensor, benchKind, nodes):
        args = {
            BenchmarkKind.SpMV: ["--bench=spmv"],
            # TODO (rohany): Support passing through the JDim value.
            BenchmarkKind.SpMM: ["--bench=spmm"],
            BenchmarkKind.SpAdd3: ["--bench=spadd3",
                                   f"--add3TensorC={self.getShiftedMatrix(tensor, 0)}",
                                   f"--add3TensorD={self.getShiftedMatrix(tensor, 1)}"]
        }
        if benchKind not in args:
            raise AssertionError(f"Unsupported Trilinos benchmark: {benchKind}")
        # TODO (rohany): Experiment with the optimal run configuration.
        if self.gpu:
            physNodes = (nodes + 3) // 4
            if physNodes == 1:
                rsPerHost = nodes
            else:
                rsPerHost = 4
            lassenPrefix = ["jsrun", "-n", str(nodes), "-g", "1", "-r", str(rsPerHost), "-c", "10", "-b", "rs", "-M", "\"-gpu\""]
        else:
            lassenPrefix = ["jsrun", "-n", str(2 * nodes), "-r", "2", "-c", "20", "-b", "rs"]
        commonArgs = [f"--file={self.getTrilinosMatrix(tensor)}", f"--n={self.niter}", f"--warmup={self.warmup}"]
        trilinosDir = os.environ.get("TRILINOS_BUILD_DIR")
        if (trilinosDir is None):
            raise AssertionError("TRILINOS_BUILD_DIR must be defined in environment.")
        wrapper = Path(trilinosDir, "..", "trilinos_run_wrapper.sh")
        if self.gpu:
            binary = Path(trilinosDir, "bin", "benchmark-cuda")
        else:
            binary = Path(trilinosDir, "bin", "benchmark")
        assert(binary.exists())
        return lassenPrefix + [str(wrapper), str(binary)] + commonArgs + args[benchKind]

    def getTrilinosMatrix(self, tensor):
        name = f"{tensor.name}.mtx"
        path = Path(getTensorDir(), "coo-txt", name)
        return str(path)

    def getShiftedMatrix(self, tensor, amount):
        name = f"{tensor.name}-rotated-{amount}.mtx"
        path = Path(getTensorDir(), "coo-txt", name)
        return str(path)

def executeCmd(cmd):
    cmdStr = " ".join(cmd)
    print("Executing command: {}".format(cmdStr))
    sys.stdout.flush()
    proc = subprocess.Popen(cmd)
    proc.wait()
    sys.stdout.flush()

def serializeBenchmark(system, benchKind, tensor, procs, procKind):
    assert(procKind in ["nodes", "gpus"])
    return f"BENCHID++{system}++{benchKind}++{tensor.name}++{procs}++{procKind}"

def makeBencher(system, args, gpu):
    if system == "DISTAL":
        bencher = DISTALBenchmark(gpu, args.n, args.warmup)
    elif system == "CTF":
        bencher = CTFBenchmark(gpu, args.n, args.warmup)
    elif system == "PETSc":
        bencher = PETScBenchmark(gpu, args.n, args.warmup)
    elif system == "Trilinos":
        bencher = TrilinosBenchmark(gpu, args.n, args.warmup)
    else:
        assert(False)
    return bencher

def main():
    # Initialize the sparse tensor registry.
    registry = SparseTensorRegistry.initialize()

    parser = argparse.ArgumentParser()
    systems = ["DISTAL", "CTF", "PETSc", "Trilinos"]
    parser.add_argument("system", type=str, choices=systems + ["all"])
    # TODO (rohany): Have option to run all benchmarks.
    parser.add_argument("bench", type=str, choices=BenchmarkKind.names())
    parser.add_argument("tensor", type=str, choices=registry.getAllNames() + ["all", "all-matrices", "all-3-tensors", "all-ctf-3-tensors"])
    parser.add_argument("--nodes", type=int, nargs='+', help="Node counts to run on", default=[1])
    parser.add_argument("--gpus", type=int, nargs='+', help="Number of GPUs to run on", default=[])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dry-run", default=False, action="store_true")
    args = parser.parse_args()

    useGPUs = len(args.gpus) != 0
    if useGPUs and args.nodes != [1]:
        print("Cannot set both --gpus and --nodes")
        return

    procs = args.nodes
    if useGPUs:
        procs = args.gpus

    benchers = []
    if args.system == "all":
        for system in systems:
            benchers.append(makeBencher(system, args, useGPUs))
    else:
        benchers.append(makeBencher(args.system, args, useGPUs))

    benchKind = BenchmarkKind.getByName(args.bench)

    if args.tensor == "all":
        tensors = registry.getAll()
    elif args.tensor == "all-matrices":
        tensors = [t for t in registry.getAll() if t.order == 2]
    elif args.tensor == "all-3-tensors":
        tensors = [t for t in registry.getAll() if t.order == 3]
    elif args.tensor == "all-ctf-3-tensors":
        maxint = 9223372036854775807
        def zipProd(l):
            tot = 1
            for e in l:
                tot *= e
            return tot
        tensors = [t for t in registry.getAll() if t.order == 3 and zipProd(t.dims) < maxint]
    else:
        tensors = [registry.getByName(args.tensor)]

    for bencher in benchers:
        for tensor in tensors:
            for n in procs:
                print(serializeBenchmark(args.system, benchKind, tensor, n, "gpus" if useGPUs else "nodes"))
                cmd = bencher.getCommand(tensor, benchKind, n)
                if (args.dry_run):
                    print(" ".join(cmd))
                else:
                    executeCmd(cmd)

if __name__ == '__main__':
    main()
