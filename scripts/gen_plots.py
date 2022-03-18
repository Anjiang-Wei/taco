#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
import pandas
import seaborn
import sys
import itertools
import glob
import os
from pathlib import Path
import re
import numpy

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

def defaultLoadExperimentData(benchKind, kind="nodes", system="*"):
    tacoDir = os.getenv("TACO_DIR", default=None)
    assert(tacoDir is not None)
    benchKindStr = str(benchKind)
    files = glob.glob(f"{tacoDir}/sc-2022-result-logs/{system}-{benchKindStr}-*{kind}*")
    dfs = []
    for f in files:
        dfs.append(processRawExperimentOutput(f))
    return pandas.concat(dfs)

def constructSpeedupData(raw, procKey, benchKind, maxProcs=16, dump=False):
    distalTimes = {}
    # Record all of DISTAL's times for look up later.
    for _, row in raw.iterrows():
        if row["System"] == "DISTAL":
            time = row["Time (ms)"]
            distalTimes[(row["Benchmark"], row["Tensor"], row[procKey])] = time
    normalizedData = []
    validTensorSet = set(validMatrices)
    if (benchOrds[benchKind] == 3):
        validTensorSet = set(validTensors)
    for _, row in raw.iterrows():
        if (row["Tensor"], row[procKey]) in benchSet[row["System"]]:
            benchSet[row["System"]].remove((row["Tensor"], row[procKey]))
        if (row["Tensor"] not in validTensorSet):
            continue
        normTime = distalTimes[(row["Benchmark"], row["Tensor"], 1)]
        row["Time (ms)"] = normTime / row["Time (ms)"]
        normalizedData.append(list(row))

    columns = list(raw.columns)
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

def loadWeakScalingData():
    tacoDir = os.getenv("TACO_DIR", default=None)
    assert(tacoDir is not None)
    file = f"{tacoDir}/sc-2022-result-logs/spmv-banded-weak-scale-64-nodes.out"
    data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if "Average" in line and "ms" in line:
                matched = timeRegex.findall(line)
                assert(len(matched) == 1)
                currentResult = float(matched[0])
                data.append(currentResult)
    # We'll drop the 1 and 2 GPU times to get full 1->64 node results.
    columns = ["System", "Nodes", "Iter/Sec"]
    rows = []
    nodes = [1, 2, 4, 8, 16, 32, 64]
    # DISTAL CPU.
    for i, node in enumerate(nodes):
        # We'll compute iterations / sec.
        row = ["DISTAL", node, 1000.0 / data[i]]
        rows.append(row)
    data = data[len(nodes):]
    # Shave off 1 and 2 GPUs.
    data = data[2:]
    for i, node in enumerate(nodes):
        row = ["DISTAL-GPU", node, 1000.0 / data[i]]
        rows.append(row)
    data = data[len(nodes):]
    for i, node in enumerate(nodes):
        row = ["PETSc", node, 1000.0 / data[i]]
        rows.append(row)
    data = data[len(nodes):]
    # Shave off 1 and 2 GPUs.
    data = data[2:]
    for i, node in enumerate(nodes):
        row = ["PETSc-GPU", node, 1000.0 / data[i]]
        rows.append(row)
    data = data[len(nodes):]
    assert(len(data) == 0)
    result = pandas.DataFrame(rows, columns=columns)
    return result

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
def brokenSpeedupPlot(data, benchKind, outdir):
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, gridspec_kw={'height_ratios' : [2, 1]})
    ax1 = seaborn.lineplot(ax=ax1, data=data[data["System"] != "CTF"], x="Nodes", y="Normalized Time to DISTAL", hue="System", style="System", err_style="band", ci=99, palette=palette, markers=markers, dashes=False)
    ax2 = seaborn.lineplot(ax=ax2, data=data[data["System"] == "CTF"], x="Nodes", y="Normalized Time to DISTAL", hue="System", style="System", err_style="band", ci=99, palette=palette, markers=markers, dashes=False)

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
    ax2.set_xlabel("Nodes", fontsize=14)
    # First argument is x position (bigger is farther right), second is y position (bigger is more up).
    fig.text(0.04, 0.50, 'Speedup over DISTAL 1 Node', va='center', rotation='vertical', fontsize=14)

    ax1.get_xaxis().set_major_formatter(xFormatter)
    ax2.get_xaxis().set_major_formatter(xFormatter)

    yFormatter = ticker.FuncFormatter(makeYAxisFormatter(ax1))
    ax1.get_yaxis().set_major_formatter(yFormatter)
    ax2.get_yaxis().set_major_formatter(yFormatter)

    ax1.tick_params(axis='both', labelsize=14)
    ax2.tick_params(axis='both', labelsize=14)

    # Use the make subplots closer together.
    # plt.subplots_adjust(wspace=0, hspace=0)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    lines_labels = sorted([ax.get_legend_handles_labels()[::-1] for ax in fig.axes])
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)][::-1]
    # This pair of location coordinates must be in [0, 1).
    fig.legend([idealLine] + lines, ["Ideal"] + labels, loc=(0.15, 0.70), prop={'size': 11})

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
    fig.set_size_inches(8, 6)
    if outdir is not None:
        plt.savefig(str(Path(outdir, f"cpu-strong-scaling-{benchKind}.pdf")), bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def speedupPlot(data, benchKind, outdir):
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

    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel("Speedup over DISTAL 1 Node", fontsize=14)
    ax.set_xlabel("Nodes", fontsize=14)

    ax.get_legend().remove()
    lines_labels = sorted(zip(*ax.get_legend_handles_labels()[::-1]), key=lambda x: x[0])
    lines = []
    labels = []
    for (la, li) in lines_labels:
        lines.append(li)
        labels.append(la)
    # This pair of location coordinates must be in [0, 1).
    loc = (0.02, 0.73)
    if benchKind in [BenchmarkKind.SpTTV, BenchmarkKind.SpMTTKRP]:
        loc = (0.02, 0.80)
    plt.legend([idealLine] + lines, ["Ideal"] + labels, loc=loc, prop={'size': 11})

    yFormatter = ticker.FuncFormatter(makeYAxisFormatter(ax))
    ax.get_xaxis().set_major_formatter(xFormatter)
    ax.get_yaxis().set_major_formatter(yFormatter)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    if outdir is not None:
        plt.savefig(str(Path(outdir, f"cpu-strong-scaling-{benchKind}.pdf")), bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


# cmap will be shared across all of the systems. systemToColorMap will map the value
# of a particular system to an index in systemToColorMap.
def processHeatmapData(data, tensors, procs, systems, procKey, ratio=False):
    # Here, we need to construct an n-d array of tensors by procs and fill it in with data.
    # We'll also construct a sidecar array of labels that we'll display in the heatmap.
    heatmapData = numpy.ndarray((len(tensors), len(procs)), dtype=numpy.int32)
    heatmapLabels = numpy.ndarray((len(tensors), len(procs)), dtype=object)
    for i in range(0, len(tensors)):
        for j in range(0, len(procs)):
            tensor = tensors[i]
            proc = procs[j]
            times = []
            for system in systems:
                # Find an entry in data that matches the current tensor, system and proc.
                # If we can't find one, that means the system OOM'd or errored out. We'll
                # do this by just filtering the dataframe, hopefully it is not too slow.
                target = data[(data["System"] == system) & (data[procKey] == proc) & (data["Tensor"] == tensor)]
                if len(target) == 0:
                    times.append(None)
                else:
                    times.append(target.iloc[0]["Time (ms)"])
            minVal = None
            for t in times:
                if t is None:
                    continue
                elif minVal is None:
                    minVal = t
                else:
                    minVal = min(minVal, t)
            index = -1
            if minVal is not None:
                index = times.index(minVal)
            heatmapData[i, j] = len(systems) - 1 - index if index != -1 else index
            if ratio:
                assert(minVal is not None)
                assert(len(times) == 2)
                otherIndex = (index + 1) % 2
                speedup = times[otherIndex] / times[index]
                heatmapLabels[i, j] = "{:.1f}".format(speedup)
            else:
                labels = []
                for k in range(0, len(times)):
                    time = times[k]
                    if time is not None:
                        s = "{:.1f}".format(time)
                        if len(s) > 5 or True:
                            s = "{:.1e}".format(time).replace("e+0", "e+").replace("e+0", "")
                        if k == index:
                            BOLD = r'$\bf{'
                            END = '}$'
                            s = BOLD + s + END
                        labels.append(s)
                    else:
                        labels.append("DNC")
                heatmapLabels[i, j] = "\n".join(labels)
    return heatmapData, heatmapLabels


def constructHeatmapGroup(tag, benches, datas, procKeyPerBench, procsPerBench, tensors, systemsPerBench, palette, outdir, xLabels=None, xtickLabelList=None, includesFailure=True, ratio=False, failurePerBench=[], cbar=False):
    widths = [len(p) for p in procsPerBench]
    fig, axn = plt.subplots(1, len(benches), gridspec_kw={'width_ratios': widths})
    if len(benches) == 1:
        axn = [axn]
    for i in range(len(benches)):
        bench, data, procs, systems, procKey = benches[i], datas[i], procsPerBench[i], systemsPerBench[i], procKeyPerBench[i]
        if xLabels is None:
            xLabel = None
        else:
            xLabel = xLabels[i]
        if xtickLabelList is None:
            xtickLabels = None
        else:
            xtickLabels = xtickLabelList[i]
        if len(failurePerBench) != 0:
            includesFailure = failurePerBench[i]
        # Construct a color map out of the possible systems.
        colors = [palette["DNC"]] if includesFailure else []
        for s in reversed(systems):
            colors.append(palette[s])
        # TODO (rohany): I realized that I don't know what this last argument means...
        cmap = LinearSegmentedColormap.from_list("Custom", colors, len(colors))
        # Gross special case for SpTTV since the GPU always wins.
        if tag == "hot" and bench == BenchmarkKind.SpTTV:
            cmap = LinearSegmentedColormap.from_list("Custom", [palette["DISTAL-GPU"], palette["DISTAL-GPU"]], len(colors))
        heatmapData, heatmapLabels = processHeatmapData(data, tensors, procs, systems, procKey, ratio=ratio)
        seaborn.heatmap(
            heatmapData,
            ax=axn[i],
            annot=heatmapLabels,
            xticklabels=procs if xtickLabels is None else xtickLabels,
            yticklabels=tensors,
            square=True,
            fmt="",
            linewidths=1.0,
            cbar=False,
            cmap=cmap,
        )
        axn[i].set_title(f"{bench.fancyStr()}")

        labels = axn[i].get_xticklabels()
        axn[i].set_xticklabels(labels, rotation=0)

        if xLabel is not None:
            axn[i].set_xlabel(xLabel)
        else:
            axn[i].set_xlabel("GPUs")
        if i > 0:
            axn[i].yaxis.set_ticks([])

    if tag == "linalg":
        fig.set_size_inches(22, 10)
    elif tag == "sddmm":
        fig.set_size_inches(9, 7)

    if outdir is not None:
        plt.savefig(str(Path(outdir, f"{tag}-gpu-strong-scaling.pdf")), bbox_inches="tight")
        plt.clf()
    else:
        plt.show()

def constructHeatMap(bench, data, procKey, procs, tensors, systems, cmap, outdir, includesFailure=True, cbarLabels=None, cbarTicks=None, xLabel=None, xtickLabels=None):
    heatmapData, heatmapLabels = processHeatmapData(data, tensors, procs, systems, procKey)
    ax = seaborn.heatmap(
        heatmapData,
        annot=heatmapLabels,
        xticklabels=procs if xtickLabels is None else xtickLabels,
        yticklabels=tensors,
        square=True,
        fmt="",
        linewidths=1.0,
        cmap=cmap,
    )
    if cbarLabels is not None:
        ax.collections[0].colorbar.ax.set_yticks(cbarTicks)
        ax.collections[0].colorbar.ax.set_yticklabels(cbarLabels)
    else:
        # Some magic here to compute tick labels and offsets for where to put those
        # labels for colors in the heatmaps.
        numColors = len(systems)
        if includesFailure:
            numColors += 1
        lo = 0
        if includesFailure:
            lo = -1.0
        hi = float(len(systems) - 1)
        ticks = []
        size = (hi - lo) / numColors
        for i in range(0, numColors):
            ticks.append(lo + (size / 2) + i * size)
        ax.collections[0].colorbar.ax.set_yticks(ticks)
        ax.collections[0].colorbar.ax.set_yticklabels((["DNC"] if includesFailure else []) + systems)

    if xLabel is not None:
        ax.set_xlabel(xLabel)
    else:
        ax.set_xlabel(procKey)
    fig = plt.gcf()
    if bench in [BenchmarkKind.SpTTV, BenchmarkKind.SpMTTKRP]:
        fig.set_size_inches(6, 4)
    else:
        fig.set_size_inches(8, 10)
    plt.title(str(bench))

    plt.yticks(rotation=0) 
    if outdir is not None:
        plt.savefig(str(Path(outdir, f"gpu-strong-scaling-{bench}.pdf")), bbox_inches="tight")
        plt.clf()
    else:
        plt.show()

# TODO (rohany): Handle when some systems timeout / OOM.
parser = argparse.ArgumentParser()
parser.add_argument("kind", type=str, choices=["strong-scaling", "weak-scaling"], default="strong-scaling")
parser.add_argument("--outdir", type=str, default=None)
args = parser.parse_args()

if args.kind == "strong-scaling":
    for bench in [BenchmarkKind.SpMV, BenchmarkKind.SpMM, BenchmarkKind.SDDMM, BenchmarkKind.SpAdd3, BenchmarkKind.SpTTV, BenchmarkKind.SpMTTKRP]:
        data = defaultLoadExperimentData(bench)
        normalized = constructSpeedupData(data, "Nodes", bench)
        if bench in [BenchmarkKind.SpMV]:
            brokenSpeedupPlot(normalized, bench, args.outdir)
        else:
            speedupPlot(normalized, bench, args.outdir)

    # Generation of heatmap plots for select GPU results.
    matrices = sorted([str(t) for t in validMatrices])
    tensors = sorted([str(t) for t in validTensors])

    # For all of these color maps, we add on gray to the beginning to give the "DNC" label the same
    # color in all of the graphs.
    DNC = (0.5, 0.5, 0.5)
    DISTALBatched = (0.635, 0.80, 1.0)

    # Benchmarks where we don't have a comparison (i.e. we'll use ourselves).
    def loadGPUCPUData(bench):
        distalGPU = defaultLoadExperimentData(bench, kind="gpus", system="distal")
        distalCPU = defaultLoadExperimentData(bench, kind="nodes", system="distal")
        distalCPU["System"] = distalCPU["System"].apply(lambda x: "DISTAL-CPU")
        newDistalGPURows = []
        for _, row in distalGPU.iterrows():
            if row["GPUs"] // 4 > 0:
                newRow = ["DISTAL-GPU", row["Benchmark"], row["Tensor"], row["GPUs"] // 4, row["Time (ms)"]]
                newDistalGPURows.append(newRow)
        distalGPU = pandas.DataFrame(newDistalGPURows, columns=["System", "Benchmark", "Tensor", "Nodes", "Time (ms)"])
        return pandas.concat([distalGPU, distalCPU])

    # Grouping several plots together for easier presentation / space saving.
    benches = [
        BenchmarkKind.SpMV,
        BenchmarkKind.SpMM,
        BenchmarkKind.SpAdd3,
        BenchmarkKind.SDDMM,
    ]
    batchedData = defaultLoadExperimentData(BenchmarkKind.SpMM, kind="gpus", system="distalbatched")
    batchedData["System"] = batchedData["System"].apply(lambda x: "DISTAL-Batched")
    spmmData = pandas.concat([
        defaultLoadExperimentData(BenchmarkKind.SpMM, kind="gpus", system="distal"),
        defaultLoadExperimentData(BenchmarkKind.SpMM, kind="gpus", system="petsc"),
        defaultLoadExperimentData(BenchmarkKind.SpMM, kind="gpus", system="trilinos"),
        batchedData,
    ])
    datas = [
        defaultLoadExperimentData(BenchmarkKind.SpMV, kind="gpus"),
        spmmData,
        defaultLoadExperimentData(BenchmarkKind.SpAdd3, kind="gpus"),
        loadGPUCPUData(BenchmarkKind.SDDMM),
    ]
    procsPerBench = [
        [1, 2, 4, 8],
        [1, 2, 4, 8, 16, 32],
        [1, 2, 4, 8, 16, 32],
        [1, 2, 4, 8],
    ]
    systemsPerBench = [
        ["DISTAL", "Trilinos", "PETSc"],
        ["DISTAL", "DISTAL-Batched", "Trilinos", "PETSc"],
        ["DISTAL", "Trilinos"],
        ["DISTAL-GPU", "DISTAL-CPU"]
    ]
    procKeyPerBench = [
        "GPUs",
        "GPUs",
        "GPUs",
        "Nodes",
    ]
    failurePerBench = [
        True,
        True,
        True,
        False,
    ]
    xLabels = [
        "GPUs",
        "GPUs",
        "GPUs",
        "Nodes(GPUs)"
    ]
    xtickLabelList = [
        [1, 2, 4, 8],
        [1, 2, 4, 8, 16, 32],
        [1, 2, 4, 8, 16, 32],
        ["1(4)", "2(8)", "4(16)", "8(32)"],
    ]
    palette = {"DNC": DNC, "DISTAL": rawPalette[0], "Trilinos": rawPalette[1], "PETSc": rawPalette[2], "DISTAL-Batched": DISTALBatched, "DISTAL-GPU": rawPalette[0], "DISTAL-CPU": rawPalette[4]}
    constructHeatmapGroup("linalg", benches, datas, procKeyPerBench, procsPerBench, matrices, systemsPerBench, palette, args.outdir, failurePerBench=failurePerBench, xLabels=xLabels, xtickLabelList=xtickLabelList)

    # Another grouped heatmap.
    benches = [
        BenchmarkKind.SpTTV,
        BenchmarkKind.SpMTTKRP,
    ]
    datas = [
        loadGPUCPUData(BenchmarkKind.SpTTV),
        loadGPUCPUData(BenchmarkKind.SpMTTKRP),
    ]
    procsPerBench = [
        [1, 2, 4, 8],
        [1, 2, 4, 8],
    ]
    systemsPerBench = [
        ["DISTAL-GPU", "DISTAL-CPU"],
        ["DISTAL-GPU", "DISTAL-CPU"],
    ]
    procKeyPerBench = [
        "Nodes",
        "Nodes",
    ]
    xLabels = [
        "Nodes(GPUs)",
        "Nodes(GPUs)"
    ]
    xtickLabelList = [
        ["1(4)", "2(8)", "4(16)", "8(32)"],
        ["1(4)", "2(8)", "4(16)", "8(32)"],
    ]
    xLabel = "Nodes(GPUs)"
    nodeLabels = ["1(4)", "2(8)", "4(16)", "8(32)"]
    palette = {"DNC": DNC, "DISTAL-GPU": rawPalette[0], "DISTAL-CPU": rawPalette[4]}
    constructHeatmapGroup("hot", benches, datas, procKeyPerBench, procsPerBench, tensors, systemsPerBench, palette, args.outdir, includesFailure=False, xLabels=xLabels, xtickLabelList=xtickLabelList, ratio=True)
else:
    data = loadWeakScalingData()
    palette = {"DISTAL": rawPalette[0], "PETSc": rawPalette[1], "DISTAL-GPU": rawPalette[2], "PETSc-GPU": rawPalette[3]}
    markers = {"DISTAL": "o", "PETSc": "X", "DISTAL-GPU": "d", "PETSc-GPU": "P"}
    ax = seaborn.lineplot(data=data, x="Nodes", y="Iter/Sec", hue="System", style="System", markers=markers, palette=palette)
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(xFormatter)
    ticks = ax.get_xticks()
    labels = [f"{int(n)} ({int(n * 4)})" for n in ticks]
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 100])
    ax.set_ylabel("Throughput / Node (Iterations / second")
    ax.set_xlabel("Nodes (GPUs)")
    plt.title("SpMV Weak-Scaling on Synthetic Banded Matrices")
    labels, lines = ax.get_legend_handles_labels()[::-1]
    newLabels = [None] * len(labels)
    newLines = [None] * len(lines)
    def set(system, idx):
        i = labels.index(system)
        newLabels[idx] = labels[i]
        newLines[idx] = lines[i]
    set("DISTAL-GPU", 0)
    set("PETSc-GPU", 1)
    set("PETSc", 2)
    set("DISTAL", 3)
    if args.outdir is not None:
        plt.savefig(str(Path(args.outdir, f"weak-scaling-spmv.pdf")), bbox_inches="tight")
        plt.clf()
    else:
        plt.show()
