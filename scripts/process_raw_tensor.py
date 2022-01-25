#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

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

def cooToDISTALCOO(name, path):
    tth = Path(tacoBuildDir, "bin", "tensor_to_hdf5")
    assert(tth.exists())
    outpath = Path(tensorDir, "coo-hdf5", f"{name}.hdf5")
    return ["jsrun", "-n", "1", "-c", "ALL_CPUS", "-b", "none", "-r", "1",
           str(tth), "-tensor", path, "-o", str(outpath),
           "-ll:bgwork", "12", "-ll:csize", "100G"]

def DISTALCOOToDistalTensor(name, format, format_suffix):
    formatconv = Path(tacoBuildDir, "bin", "format-converter")
    assert(formatconv.exists())
    inpath = Path(tensorDir, "coo-hdf5", f"{name}.hdf5")
    outpath = Path(tensorDir, "distal", f"{name}.{format_suffix}.hdf5")
    return ["jsrun", "-n", "1", "-c", "ALL_CPUS", "-b", "none", "-r", "1",
            str(formatconv), "-coofile", str(inpath), "-format", format, "-o", str(outpath),
            "-ll:bgwork", "12", "-ll:csize", "100G"]

def cooToTns(name, path, tmpPath):
    taco = Path(tacoBuildDir, "bin", "taco")
    assert(taco.exists())
    # First, we'll output the file to a local, temporary file, and the send
    # the output to the tensor dir.
    genTemp = [str(taco), f"-ctfMtxInput={path}", f"-ctfMtxOutput={tmpPath}"]
    outpath = Path(tensorDir, "coo-txt", f"{name}.tns")
    copy = ["cp", tmpPath, str(outpath)]
    return [genTemp, copy]

def cooToPetsc(name, path):
    converter = Path(petscDir, "bin", "petsc-converter")
    # assert(converter.exists())
    outpath = Path(tensorDir, "petsc", f"{name}.petsc")
    return ["jsrun", "-n", "1", "-c", "ALL_CPUS", "-b", "none", "-r", "1", str(converter), "-matrix", path, "-o", str(outpath)]

def executeCmd(cmd, dry_run):
    cmdStr = " ".join(cmd)
    print("Executing command: {}".format(cmdStr))
    if dry_run:
        return
    proc = subprocess.Popen(cmd)
    proc.wait()
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    # Name of the tensor to use.
    parser.add_argument("tensor", type=str)
    parser.add_argument("tensorPath", type=str)
    parser.add_argument("dimensions", type=int)
    parser.add_argument("distal_format", type=str)
    parser.add_argument("distal_format_suffix", type=str)
    parser.add_argument("--dry-run", default=False, action="store_true")
    args = parser.parse_args()

    # TODO (rohany): Support running only one of the passes.
    tensor = args.tensor
    path = args.tensorPath
    dims = args.dimensions

    print("Converting to DISTAL COO HDF5 file.")
    executeCmd(cooToDISTALCOO(tensor, path), args.dry_run)

    print("Converting DISTAL COO to DISTAL tensor.")
    executeCmd(DISTALCOOToDistalTensor(tensor, args.distal_format, args.distal_format_suffix), args.dry_run)

    if dims == 2 and "tns" not in path:
        print("Converting COO to TNS.")
        with tempfile.NamedTemporaryFile(delete=True, suffix='.tns') as tmp:
            cmds = cooToTns(tensor, path, tmp.name)
            for cmd in cmds:
                executeCmd(cmd, args.dry_run)

    if dims == 2:
        print("Converting COO to PETSc.")
        executeCmd(cooToPetsc(tensor, path), args.dry_run)

if __name__ == '__main__':
    main()
