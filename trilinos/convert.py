import scipy
import scipy.io
import os
import argparse
import subprocess

parser = argparse.ArgumentParser(description="Translate a MatrixMarket file into a Trilinos binary file.")
parser.add_argument("inFile", type=str, help="input file name")
parser.add_argument("outFile", type=str, help="output file name")
parser.add_argument("converter", type=str, help="path to format converter binary")
args = parser.parse_args()

# Load the input file.
print("Loading matrix from Matrix Market file.")
mat = scipy.io.mmread(args.inFile)
# Dump the matrix to a temporary output file.
tmpName = args.outFile + ".tmp.mtx"
print("Dumping matrix to temporary Matrix Market file.")
scipy.io.mmwrite(tmpName, mat, field="real", symmetry="general")
# Now invoke the format-conversion binary.
print("Invoking Trilinos binary converter.")
subprocess.run([args.converter, tmpName, args.outFile])
# Finally, delete the temporary file.
os.remove(tmpName)
print("Done!")
