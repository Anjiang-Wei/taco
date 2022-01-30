#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from registry import *

# We require several environment variables to be set.
tensorDir = os.environ.get("TENSOR_DIR", None)
petscDir = os.environ.get("PETSC_BUILD_DIR", None)
tacoBuildDir = os.environ.get("TACO_BUILD_DIR", None)

if None in [tensorDir, petscDir, tacoBuildDir]:
    print("Must set all of TENSOR_DIR, PETSC_BUILD_DIR, TACO_BUILD_DIR in environment.")
    exit(1)

# Given a raw COO (mtx or tns) file, we need to do the following things.
# 1. Convert the COO file into an HDF5 COO file.
# 2. Convert the HDF5 COO file into desired DISTAL tensor format.
# 3. Convert the COO file into a .tns file for CTF.
# 4. Convert the COO file into the PETSc binary format.

def cooToDISTALCOO(name, path, overwrite):
    tth = Path(tacoBuildDir, "bin", "tensor_to_hdf5")
    # assert(tth.exists())
    outpath = Path(tensorDir, "coo-hdf5", f"{name}.hdf5")
    if outpath.exists() and not overwrite:
        return []
    return ["jsrun", "-n", "1", "-c", "ALL_CPUS", "-b", "none", "-r", "1",
           str(tth), "-tensor", path, "-o", str(outpath),
           "-ll:bgwork", "12", "-ll:csize", "100G"]

def DISTALCOOToDistalTensor(name, format, format_suffix, overwrite):
    formatconv = Path(tacoBuildDir, "bin", "format-converter")
    # assert(formatconv.exists())
    inpath = Path(tensorDir, "coo-hdf5", f"{name}.hdf5")
    outpath = Path(tensorDir, "distal", f"{name}.{format_suffix}.hdf5")
    if outpath.exists() and not overwrite:
        return []
    return ["jsrun", "-n", "1", "-c", "ALL_CPUS", "-b", "none", "-r", "1",
            str(formatconv), "-coofile", str(inpath), "-format", format, "-o", str(outpath),
            "-ll:bgwork", "12", "-ll:csize", "100G"]

def cooToTns(name, path, tmpPath, overwrite):
    taco = Path(tacoBuildDir, "bin", "taco")
    # assert(taco.exists())
    # First, we'll output the file to a local, temporary file, and the send
    # the output to the tensor dir.
    genTemp = [str(taco), f"-ctfMtxInput={path}", f"-ctfMtxOutput={tmpPath}"]
    outpath = Path(tensorDir, "coo-txt", f"{name}.tns")
    if outpath.exists() and not overwrite:
        return []
    copy = ["cp", tmpPath, str(outpath)]
    return [genTemp, copy]

def cooToPetsc(name, path, overwrite):
    converter = Path(petscDir, "bin", "petsc-converter")
    # assert(converter.exists())
    outpath = Path(tensorDir, "petsc", f"{name}.petsc")
    if outpath.exists() and not overwrite:
        return []
    return ["jsrun", "-n", "1", "-c", "ALL_CPUS", "-b", "none", "-r", "1", str(converter), "-matrix", path, "-o", str(outpath)]

def rotateTensor(name, path, dims, suffix, overwrite):
    taco = Path(tacoBuildDir, "bin", "taco")
    # assert(taco.exists())
    outpath = Path(tensorDir, "coo-txt", f"{name}-{suffix}.mtx")
    if outpath.exists() and not overwrite:
        return []
    return [str(taco), f"-rotateTensorOrder={dims}", f"-rotateInput={path}", f"-rotateOutput={outpath}"]

def generateUniformRandomVec(outpath, dim, overwrite):
    taco = Path(tacoBuildDir, "bin", "taco")
    # assert(taco.exists())
    if outpath.exists() and not overwrite:
        return []
    return [str(taco), f"-uniformVecDim={dim}", f"-uniformVecPercentage={0.10}", f"-uniformVecOutput={str(outpath)}"]

def executeCmd(cmd, dry_run):
    if len(cmd) == 0:
        return
    cmdStr = " ".join(cmd)
    print("Executing command: {}".format(cmdStr))
    sys.stdout.flush()
    if dry_run:
        return
    proc = subprocess.Popen(cmd)
    proc.wait()
    sys.stdout.flush()

def process_tensor(tensor, path, dims, args):
    print("Converting to DISTAL COO HDF5 file.")
    executeCmd(cooToDISTALCOO(tensor, path, args.overwrite), args.dry_run)

    print("Converting DISTAL COO to DISTAL tensor.")
    executeCmd(DISTALCOOToDistalTensor(tensor, args.distal_format, args.distal_format_suffix, args.overwrite), args.dry_run)

    if dims == 2 and "tns" not in path:
        print("Converting COO to TNS.")
        with tempfile.NamedTemporaryFile(delete=True, suffix='.tns') as tmp:
            cmds = cooToTns(tensor, path, tmp.name, args.overwrite)
            for cmd in cmds:
                executeCmd(cmd, args.dry_run)

    if dims == 2:
        print("Converting COO to PETSc.")
        executeCmd(cooToPetsc(tensor, path, args.overwrite), args.dry_run)

def main():
    registry = SparseTensorRegistry.initialize()
    parser = argparse.ArgumentParser()
    # Name of the tensor to use.
    parser.add_argument("tensor", type=str)
    parser.add_argument("distal_format", type=str)
    parser.add_argument("distal_format_suffix", type=str)
    parser.add_argument("--dry-run", default=False, action="store_true")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--tensorPath", type=str)
    args = parser.parse_args()

    if args.tensor == "all":
        tensors = registry.getAll()
    elif args.tensor == "all-matrices":
        tensors = [t for t in registry.getAll() if t.order == 2]
    elif args.tensor == "all-3-tensors":
        tensors = [t for t in registry.getAll() if t.order == 3]
    else:
        tensors = [registry.getByName(args.tensor)]

    for tensor in tensors:
        if tensor.order == 2:
            path = Path(tensorDir, "coo-txt", f"{tensor.name}.mtx")
        else:
            path = Path(tensorDir, "coo-txt", f"{tensor.name}.tns")

        process_tensor(tensor.name, str(path), tensor.order, args)

        # We'll rotate the tensor twice if it is a matrix (i.e. we need it for spadd3),
        # and only once for a tensor.
        executeCmd(rotateTensor(tensor.name, path, tensor.order, "rotated-0", args.overwrite), args.dry_run)
        rotatedName = f"{tensor.name}-rotated-0"
        outpath = Path(tensorDir, "coo-txt", f"{rotatedName}.mtx")
        process_tensor(rotatedName, str(outpath), tensor.order, args)
        if tensor.order == 2:
            executeCmd(rotateTensor(tensor.name, path, tensor.order, "rotated-1", args.overwrite), args.dry_run)
            rotatedName = f"{tensor.name}-rotated-1"
            outpath = Path(tensorDir, "coo-txt", f"{rotatedName}.mtx")
            process_tensor(rotatedName, str(outpath), tensor.order, args)

        # Generate a sparse vector for this tensor as well, if it is a matrix.
        if tensor.order == 2:
            vecname = f"{tensor.name}-uniform-vec"
            outpath = Path(tensorDir, "coo-txt", f"{vecname}.tns")
            executeCmd(generateUniformRandomVec(outpath, tensor.dims[1], args.overwrite), args.dry_run)
            # We now need to process the vector for DISTAL.
            executeCmd(cooToDISTALCOO(vecname, str(outpath), args.overwrite), args.dry_run)
            executeCmd(DISTALCOOToDistalTensor(vecname, "s", "vec", args.overwrite), args.dry_run)

if __name__ == '__main__':
    main()
