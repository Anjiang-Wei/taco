#!/usr/bin/env python3

# python3 scripts/install_lassen.py --openmp --sockets 2 --cuda --dim 3 --multi-node --threads 20 --no-tblis

import argparse
from contextlib import contextmanager
import os
import subprocess
import sys

@contextmanager
def pushd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

def run(*command, check=True, env=None):
    if env is None:
        env = {}
    env = {**os.environ, **env}
    print("Executing command: {}".format(" ".join(command)))
    subprocess.run(command, check=check, env=env)

def wget(url):
    run("wget", url)

def cmake(dir, defines={}, env=None):
    cmakeCmd = ["cmake", dir]
    for key, value in defines.items():
        assert isinstance(key, str)
        if isinstance(value, bool):
            cmakeCmd.append("-D{0}={1}".format(key, "ON" if value else "OFF"))
        elif isinstance(value, str) or isinstance(value, int):
            cmakeCmd.append("-D{0}={1}".format(key, value))
        else:
            raise TypeError("Unsupported type: {0}".format(type(value)))
    run(*cmakeCmd, env=env)

parser = argparse.ArgumentParser()
# Installation setup flags.
parser.add_argument("--deps-install-dir", default="deps-install", help="Path to install and build dependencies.")
parser.add_argument("--distal-build-dir", default="build", help="Path to build DISTAL within.")
# Configuration flags.
parser.add_argument("--openmp", default=False, action="store_true", help="Enable use of OpenMP threads.")
parser.add_argument("--sockets", default=None, type=int, help="Number of sockets on the CPU. Must be set if OpenMP is enabled.")
parser.add_argument("--cuda", default=False, action="store_true", help="Enable use of NVIDIA GPUs.")
parser.add_argument("--dim", default=None, type=int, help="Maximum tensor dimension supported by Legion.")
parser.add_argument("--multi-node", default=False, action="store_true", help="Enable distributed computations.")
parser.add_argument("--conduit", default="ibv", type=str, help="Network conduit (default ibv).")
parser.add_argument("--threads", default=1, type=int, help="Number of threads to use to build.")
parser.add_argument("--tblis", default=True, action="store_true", help="Enable TBLIS.")
parser.add_argument("--no-tblis", default=False, dest="tblis", action="store_false", help="Disable TBLIS.")
args = parser.parse_args()

# TODO (rohany): Make sure that this script is running from the root of the DISTAL repository.
distalRoot = os.getcwd()

# TODO (rohany): Update the script to be resilient to when parts of the installation have succeeded or not.

if args.openmp and args.sockets is None:
    print("--sockets must be provided if --openmp is set.")
    sys.exit(1)

# First, install dependencies.
os.makedirs(args.deps_install_dir, exist_ok=True)
makeInstallPath = os.path.abspath(os.path.join(args.deps_install_dir, "make-install"))
cmakeInstallPath = os.path.abspath(os.path.join(args.deps_install_dir, "cmake-install"))
with pushd(args.deps_install_dir):
    # Set up installation an installation path for all of the dependencies to install into.
    os.makedirs(makeInstallPath, exist_ok=True)
    os.makedirs(cmakeInstallPath, exist_ok=True)

    # HDF5.
    if not os.path.exists("hdf5-1.10.1"):
        wget("http://sapling.stanford.edu/~manolis/hdf/hdf5-1.10.1.tar.gz")
        run("mkdir", "hdf5-1.10.1")
        run("tar", "-xf", "hdf5-1.10.1.tar.gz", "-C", "hdf5-1.10.1", "--strip-components", "1")
    with pushd("hdf5-1.10.1"):
        run("./configure", "--prefix", makeInstallPath, "--enable-thread-safe", "--disable-hl")
        run("make", "-j{}".format(args.threads))
        run("make", "-j{}".format(args.threads), "install")

    # OpenBLAS.
    with pushd(os.path.join(distalRoot, "legion", "OpenBLAS")):
        env = {}
        if args.openmp:
            env["USE_OPENMP"] = "1"
            env["NUM_PARALLEL"] = str(args.sockets)
        run("make", "-j{}".format(args.threads), env=env)
        run("make", "-j{}".format(args.threads), "install", env={
            "PREFIX": makeInstallPath
        })

    # TBLIS.
    if args.tblis:
        with pushd(os.path.join(distalRoot, "deps", "tblis")):
            # BLAS is only used for the benchmark program, and hence disabled.
            cmd = ["./configure",
                   "--prefix", makeInstallPath,
                   "--enable-config=auto",
                   "--without-blas"]
            if args.openmp:
                cmd.append("--enable-thread-model=openmp")
            else:
                cmd.append("--enable-thread-model=none")
            run(*cmd)
            run("make", "-j{}".format(args.threads))
            run("make", "-j{}".format(args.threads), "install")

    # Legion.
    os.makedirs("legion-build", exist_ok=True)
    with pushd("legion-build"):
        # In order to get Legion to be a completely independent install, configuring the RPATH
        # through CMake is an option. See here for more details: 
        # https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#default-rpath-settings
        # I gave this an attempt but had to give up due to weird behavior when integrating the
        # shared library with the configured RPATH into DISTAL.
        cmakeDefs = {
            "BUILD_SHARED_LIBS": True,
            "CMAKE_CXX_FLAGS": "--std=c++11",
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_INSTALL_PREFIX": cmakeInstallPath,
            "Legion_USE_HDF5": True,
        }
        if args.openmp:
            cmakeDefs["Legion_USE_OpenMP"] = True
        if args.cuda:
            cmakeDefs["Legion_USE_CUDA"] = True
            cmakeDefs["Legion_CUDA_ARCH"] = "70"
        if args.dim is not None:
            cmakeDefs["Legion_MAX_DIM"] = args.dim
        if args.multi_node:
            cmakeDefs["Legion_NETWORKS"] = "gasnetex"
            cmakeDefs["Legion_EMBED_GASNet"] = True
            cmakeDefs["GASNet_CONDUIT"] = args.conduit
            cmakeDefs["Legion_EMBED_GASNet_GITREF"] = "045601071ecf5e7ca35accaeb6b8a9d302a8386f"
            # cmakeDefs["Legion_EMBED_GASNet_VERSION"] = "2021.3.0"

        cmake(os.path.join(distalRoot, "legion", "legion"), cmakeDefs, env={
            "HDF5_ROOT": makeInstallPath,
        })
        run("make", "-j{}".format(args.threads), "install")

# Finally build DISTAL.
os.makedirs(args.distal_build_dir, exist_ok=True)
with pushd(args.distal_build_dir):
    cmakeDefs = {
        "BLA_VENDOR": "OpenBLAS",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_MODULE_PATH": os.path.join(distalRoot, "cmake"),
        "CMAKE_PREFIX_PATH": ";".join([cmakeInstallPath, makeInstallPath]),
    }
    if args.openmp:
        cmakeDefs["OPENMP"] = True
    env = {
        "HDF5_ROOT": makeInstallPath,
        "LLNL_COMPUTE_NODES": "1",
    }
    if args.tblis:
        env["TBLIS_ROOT"] = makeInstallPath
    cmake(distalRoot, cmakeDefs, env=env)
    run("make", "-j{}".format(args.threads), "taco-test")
    run("./bin/taco-test", "--gtest_filter=distributed.cannonMM")
    run("make", "-j{}".format(args.threads), "cannonMM")
