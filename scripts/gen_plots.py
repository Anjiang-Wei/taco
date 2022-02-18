#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas
import seaborn
import sys
import itertools

from registry import *

reg = SparseTensorRegistry.initialize()
filterFunc = lambda t : t.order == 2
nodes = [1, 2, 4, 8, 16]
benchSet = {
    'DISTAL': set(itertools.product(reg.getAllNames(filterFunc=filterFunc), nodes)),
    'PETSc': set(itertools.product(reg.getAllNames(filterFunc=filterFunc), nodes)),
    'Trilinos': set(itertools.product(reg.getAllNames(filterFunc=filterFunc), nodes)),
    'CTF': set(itertools.product(reg.getAllNames(filterFunc=filterFunc), nodes)),
}
validMatrices = set(reg.getAllNames(filterFunc=filterFunc))
validTensors = set(reg.getAllNames(filterFunc=lambda t : t.order == 3))

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
    # if row["System"] == "DISTAL":
    #     continue
    # if row["System"] == "CTF":
    #     continue
    if row["Tensor"] not in validMatrices:
    # if row["Tensor"] not in validTensors:
        continue
    normTime = distalTimes[(row["Benchmark"], row["Tensor"], 1)]
    row["Time (ms)"] = normTime / row["Time (ms)"]
    # normTime = distalTimes[(row["Benchmark"], row["Tensor"], row[procKey])]
    # row["Time (ms)"] = row["Time (ms)"] / normTime
    normalizedData.append(list(row))

columns = list(data.columns)
columns[-1] = "Normalized Time to DISTAL"
normalized = pandas.DataFrame(data=normalizedData, columns=columns)
pandas.set_option("display.max_rows", None)
# print(normalized)

for system, left in benchSet.items():
    if len(left) != 0:
        # Do a groupby.
        map = {}
        for tensor, procs in sorted(left):
            if tensor not in map:
                map[tensor] = []
            map[tensor].append(procs)
        for tensor in map:
            map[tensor].sort()
        print(f"!!! System {system} is missing benchmarks: {map} !!!")


def brokenSpeedupPlot(data):
    palette = seaborn.color_palette()
    markers = ["o", "X", "d", "P"]
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)# , sharex=True)
    ax1 = seaborn.lineplot(ax=ax1, data=data[data["System"] != "CTF"], x="Nodes", y="Normalized Time to DISTAL", hue="System", style="System", err_style="band", ci=99, palette=palette[0:3], markers=markers[0:3])
    ax2 = seaborn.lineplot(ax=ax2, data=data[data["System"] == "CTF"], x="Nodes", y="Normalized Time to DISTAL", hue="System", style="System", err_style="band", ci=99, palette=palette[3:4], markers=markers[3:4])

    xpoints = [1, 16]
    ypoints = [1, 16]
    ax1.plot(xpoints, ypoints, color="black")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.set_yticks([0.5, 1, 2, 4, 8, 16])
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)

    ax1.get_xaxis().set_visible(False)
    ax1.set_ylabel("")
    ax2.set_ylabel("")
    # First argument is x position (bigger is farther right), second is y position (bigger is more up).
    fig.text(0.03, 0.50, 'Speedup over DISTAL 1 Node', va='center', rotation='vertical')

    xFormatter = ticker.FuncFormatter(lambda x, p : str(int(x)))
    ax1.get_xaxis().set_major_formatter(xFormatter)
    ax2.get_xaxis().set_major_formatter(xFormatter)

    fmt = ax1.get_yaxis().get_major_formatter()
    def axisFormatterFunc(x, p):
        if x >= 1:
            return str(int(x))
        return fmt.__call__(x, p)
    yFormatter = ticker.FuncFormatter(axisFormatterFunc)
    ax1.get_yaxis().set_major_formatter(yFormatter)
    ax2.get_yaxis().set_major_formatter(yFormatter)

    # Use the make subplots closer together.
    # plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    # TODO (rohany): Can set the location with an (x, y) pair.
    fig.legend(lines, labels, loc=(0.15, 0.68))
    # Add to remove the lines between each of the subplots.
    # ax1.spines["bottom"].set_visible(False)
    # ax2.spines["top"].set_visible(False)
    # Add the "line break" visual.
    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    # TODO (rohany): Change this to dump to a file?
    plt.show()

brokenSpeedupPlot(normalized)