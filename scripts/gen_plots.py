#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import pandas
import seaborn
import sys
import itertools

from registry import *

reg = SparseTensorRegistry.initialize()
filterFunc = lambda t : t.order == 2
nodes = [1, 2, 4, 8]
benchSet = {
    'DISTAL': set(itertools.product(reg.getAllNames(filterFunc=filterFunc), nodes)),
    'PETSc': set(itertools.product(reg.getAllNames(filterFunc=filterFunc), nodes)),
    'Trilinos': set(itertools.product(reg.getAllNames(filterFunc=filterFunc), nodes)),
    'CTF': set(itertools.product(reg.getAllNames(filterFunc=filterFunc), nodes)),
}
validMatrices = set(reg.getAllNames(filterFunc=filterFunc))

distalTimes = {}

data = pandas.read_csv(sys.argv[1])

procKey = "Nodes"
if procKey not in data.columns:
    procKey = "GPUs"

# Normalize the execution time to DISTAL.
for _, row in data.iterrows():
    if row["System"] == "DISTAL":
        time = row["Time (ms)"]
        distalTimes[(row["Benchmark"], row["Tensor"], row[procKey])] = time
normalizedData = []
for _, row in data.iterrows():
    if (row["Tensor"], row[procKey]) in benchSet[row["System"]]:
        benchSet[row["System"]].remove((row["Tensor"], row[procKey]))
    if row["System"] == "DISTAL":
        continue
    if row["Tensor"] not in validMatrices:
        continue
    normTime = distalTimes[(row["Benchmark"], row["Tensor"], row[procKey])]
    row["Time (ms)"] = row["Time (ms)"] / normTime
    normalizedData.append(list(row))

columns = list(data.columns)
columns[-1] = "Normalized Time to DISTAL"
normalized = pandas.DataFrame(data=normalizedData, columns=columns)
pandas.set_option("display.max_rows", None)
print(normalized)

for system, left in benchSet.items():
    if len(left) != 0:
        print(f"!!! System {system} is missing benchmarks: {left} !!!")

seaborn.boxplot(x=procKey, y="Normalized Time to DISTAL", hue="System", data=normalized, showfliers=False)
# seaborn.swarmplot(x="Nodes", y="Normalized Time to DISTAL", hue="System", data=normalized, dodge=True)
plt.axhline(y=1, color='r')
plt.yscale('log')
plt.show()
