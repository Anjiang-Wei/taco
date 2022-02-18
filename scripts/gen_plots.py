#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas
import seaborn
import sys
import itertools
import glob
import os
from pathlib import Path
import re

from registry import *
from benchkinds import *

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
benchOrds = {
    BenchmarkKind.SpMV: 2,
    BenchmarkKind.SpMM: 2,
    BenchmarkKind.SDDMM: 2,
    BenchmarkKind.SpAdd3: 2,
    BenchmarkKind.SpTTV: 3,
    BenchmarkKind.SpMTTKRP: 3,
}
timeRegex = re.compile("([0-9]*\.[0-9]+|[0-9]+)")

distalTimes = {}

def processRawExperimentOutput(file):
    procKind = "nodes"
    results = []
    filepath = Path(file)
    assert(filepath.exists())
    with filepath.open("r") as fo:
        lines = fo.readlines()
        currentBench = None
        currentResult = None
        for line in lines:
            line = line.strip()
            if line.startswith("BENCHID"):
                data = line.split("++")
                currentBench = data[1:5]
                currentBench[3] = int(currentBench[3])
                currentBench = tuple(currentBench)
                if (len(data) > 5):
                    procKind = data[-1]
            elif "Average" in line and "ms" in line:
                matched = timeRegex.findall(line)
                assert(len(matched) == 1)
                currentResult = float(matched[0])
                results.append((currentBench, currentResult))
    results.sort()
    if procKind == "nodes":
        header = ["System", "Benchmark", "Tensor", "Nodes", "Time (ms)"]
    else:
        header = ["System", "Benchmark", "Tensor", "GPUs", "Time (ms)"]
    pdData = []
    for row in results:
        keys, val = row
        pdData.append(list(keys) + [val])
    return pandas.DataFrame(pdData, columns=header)

def defaultLoadExperimentData(benchKind):
    tacoDir = os.getenv("TACO_DIR", default=None)
    assert(tacoDir is not None)
    benchKindStr = str(benchKind)
    files = glob.glob(f"{tacoDir}/sc-2022-result-logs/*-{benchKindStr}-*nodes*")
    dfs = []
    for f in files:
        dfs.append(processRawExperimentOutput(f))
    return pandas.concat(dfs)

def constructSpeedupData(raw, procKey, benchKind, maxProcs=16, dump=False):
    distalTimes = {}
    # Record all of DISTAL's times for look up later.
    for _, row in data.iterrows():
        if row["System"] == "DISTAL":
            time = row["Time (ms)"]
            distalTimes[(row["Benchmark"], row["Tensor"], row[procKey])] = time
    normalizedData = []
    validTensorSet = set(validMatrices)
    if (benchOrds[benchKind] == 3):
        validTensorSet = set(validTensors)
    for _, row in data.iterrows():
        if (row["Tensor"], row[procKey]) in benchSet[row["System"]]:
            benchSet[row["System"]].remove((row["Tensor"], row[procKey]))
        if (row["Tensor"] not in validTensorSet):
            continue
        normTime = distalTimes[(row["Benchmark"], row["Tensor"], 1)]
        row["Time (ms)"] = normTime / row["Time (ms)"]
        normalizedData.append(list(row))

    columns = list(data.columns)
    columns[-1] = "Normalized Time to DISTAL"
    normalized = pandas.DataFrame(data=normalizedData, columns=columns)
    if dump:
        pandas.set_option("display.max_rows", None)
        print(normalized)

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
    return normalized

rawPalette = seaborn.color_palette()
palette = {"DISTAL": rawPalette[0], "PETSc": rawPalette[1], "Trilinos": rawPalette[2], "CTF": rawPalette[3]}
markers = {"DISTAL": "o", "PETSc": "X", "Trilinos": "d", "CTF": "P"}
xFormatter = ticker.FuncFormatter(lambda x, p : str(int(x)))
def makeYAxisFormatter(ax):
    fmt = ax.get_yaxis().get_major_formatter()
    def axisFormatterFunc(x, p):
        if x >= 1:
            return str(int(x))
        return fmt.__call__(x, p)
    return axisFormatterFunc

# Much of the logic to generate the broken plot was taken from here:
# https://gist.github.com/pfandzelter/0ae861f0dee1fb4fd1d11344e3f85c9e.
def brokenSpeedupPlot(data, benchKind):
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios' : [2, 1]})
    ax1 = seaborn.lineplot(ax=ax1, data=data[data["System"] != "CTF"], x="Nodes", y="Normalized Time to DISTAL", hue="System", style="System", err_style="band", ci=99, palette=palette, markers=markers)
    ax2 = seaborn.lineplot(ax=ax2, data=data[data["System"] == "CTF"], x="Nodes", y="Normalized Time to DISTAL", hue="System", style="System", err_style="band", ci=99, palette=palette, markers=markers)

    xpoints = [1, 16]
    ypoints = [1, 16]
    idealLine, = ax1.plot(xpoints, ypoints, color="black")

    # xmin and xmax are fractions of the plot width, not data elements.
    ax1.axhline(y=1, color='gray', linestyle='dotted', xmin=0.05, xmax=0.95)

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

    ax1.get_xaxis().set_major_formatter(xFormatter)
    ax2.get_xaxis().set_major_formatter(xFormatter)

    yFormatter = ticker.FuncFormatter(makeYAxisFormatter(ax1))
    ax1.get_yaxis().set_major_formatter(yFormatter)
    ax2.get_yaxis().set_major_formatter(yFormatter)

    # Use the make subplots closer together.
    # plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    lines_labels = sorted([ax.get_legend_handles_labels()[::-1] for ax in fig.axes])
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)][::-1]
    # This pair of location coordinates must be in [0, 1).
    fig.legend([idealLine] + lines, ["Ideal"] + labels, loc=(0.15, 0.63))

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
    fig.suptitle(f"Speedup for {benchKind}", fontsize=16)
    # TODO (rohany): Change this to dump to a file?
    plt.show()

def speedupPlot(data, benchKind):
    ax = seaborn.lineplot(data=data, x="Nodes", y="Normalized Time to DISTAL", hue="System", style="System", err_style="band", ci=99, markers=markers, palette=palette)
    xpoints = [1, 16]
    ypoints = [1, 16]
    idealLine, = ax.plot(xpoints, ypoints, color="black")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)

    # Ticks must include [1, 2, 4, 8, 16] (or whatever the max proc count is).
    ticks = set(ax.get_yticks())
    for proc in [1, 2, 4, 8, 16]:
        ticks.add(proc)
    # For some reason the default ticks include some values way larger than
    # reasonable speedups.
    ticks = [t for t in sorted(list(ticks)) if t <= 16]
    ax.set_yticks(ticks)

    # xmin and xmax are fractions of the plot width, not data elements.
    ax.axhline(y=1, color='gray', linestyle='dotted', xmin=0.05, xmax=0.95)

    ax.get_legend().remove()
    lines_labels = sorted(zip(*ax.get_legend_handles_labels()[::-1]))
    lines = []
    labels = []
    for (la, li) in lines_labels:
        lines.append(li)
        labels.append(la)
    # This pair of location coordinates must be in [0, 1).
    plt.legend([idealLine] + lines, ["Ideal"] + labels, loc=(0.05, 0.70))

    yFormatter = ticker.FuncFormatter(makeYAxisFormatter(ax))
    ax.get_xaxis().set_major_formatter(xFormatter)
    ax.get_yaxis().set_major_formatter(yFormatter)
    plt.suptitle(f"Speedup for {benchKind}", fontsize=16)
    plt.show()

# TODO (rohany): Handle when some systems timeout / OOM.

for bench in [BenchmarkKind.SpMV, BenchmarkKind.SpMM, BenchmarkKind.SDDMM, BenchmarkKind.SpAdd3, BenchmarkKind.SpTTV, BenchmarkKind.SpMTTKRP]:
    data = defaultLoadExperimentData(bench)
    normalized = constructSpeedupData(data, "Nodes", bench)
    if bench in [BenchmarkKind.SpMV]:
        brokenSpeedupPlot(normalized, bench)
    else:
        speedupPlot(normalized, bench)
