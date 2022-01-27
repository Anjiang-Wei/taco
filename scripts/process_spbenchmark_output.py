#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
import re
import sys

timeRegex = re.compile("([0-9]*\.[0-9]+|[0-9]+)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs='+')
    args = parser.parse_args()

    data = []
    for file in args.files:
        filepath = Path(file)
        assert(filepath.exists())
        with filepath.open("r") as fo:
            lines = fo.readlines()
            currentBench = None
            currentResult = None
            for line in lines:
                if line.startswith("BENCHID"):
                    data = line.split("++")
                    assert(len(data) == 5)
                    currentBench = tuple(data[1:])
                elif "Average" in line and "ms" in line:
                    matched = timeRegex.findall(line)
                    assert(len(matched) == 1)
                    currentResult = matched[0]
                    data.append((currentBench, currentResult))

    data.sort()
    # Dump the data out in a csv format.
    writer = csv.writer(sys.stdout)
    # TODO (rohany): This will likely change once I start adding GPUs to the benchmarks.
    header = ["System", "Benchmark", "Tensor", "Nodes", "Time (ms)"]
    writer.writerow(header)
    for d in data:
        keys, val = d
        writer.writerow(list(keys) + [val])

if __name__ == '__main__':
    main()
