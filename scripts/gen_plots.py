#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import pandas
import seaborn
import sys

distalTimes = {}

data = pandas.read_csv(sys.argv[1])

# Normalize the execution time to DISTAL.
for _, row in data.iterrows():
    if row["System"] == "DISTAL":
        time = row["Time (ms)"]
        distalTimes[(row["Benchmark"], row["Tensor"], row["Nodes"])] = time
normalizedData = []
for _, row in data.iterrows():
    if row["System"] == "DISTAL":
        continue
    normTime = distalTimes[(row["Benchmark"], row["Tensor"], row["Nodes"])]
    row["Time (ms)"] = row["Time (ms)"] / normTime
    normalizedData.append(list(row))

columns = list(data.columns)
columns[-1] = "Normalized Time to DISTAL"
normalized = pandas.DataFrame(data=normalizedData, columns=columns)
# pandas.set_option("display.max_rows", None)
# print(normalized)

seaborn.boxplot(x="Nodes", y="Normalized Time to DISTAL", hue="System", data=normalized, showfliers=False)
# seaborn.swarmplot(x="Nodes", y="Normalized Time to DISTAL", hue="System", data=normalized, dodge=True)
plt.axhline(y=1, color='r')
plt.show()
