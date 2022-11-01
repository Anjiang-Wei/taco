#!/usr/bin/env python3

# https://raw.githubusercontent.com/StanfordLegion/legion/papers/index-launch-sc21/language/sc21_scripts/parse_results.py

import csv
import glob
import os
import re

# On 16 nodes achieved GFLOPS per node: 8232.518652.
_content_re = re.compile(r'On +([0-9]+) nodes achieved GFLOPS per node: +([0-9.]+).', re.MULTILINE)
def parse_content(path):
    # only keep the last two performance results, in this order: Taco, DSL
    res = []
    with open(path, 'r') as f:
        content = f.readlines()
        for line in content:
            match = re.search(_content_re, line)
            if match is not None:
                res.append(match.groups())
    print(f"{path} has {len(res)} matches")
    if len(res) == 0:
        print(f"{path} error, assuming both fail")
        return None, None
    elif len(res) == 1:
        print(f"{path} error, assuming DSL fails")
        return res[-1], None
    return res[-2], res[-1]

def main():
    paths = glob.glob('*.out')
    content = []
    for path in paths:
        filename = os.path.basename(path)
        taco_perf, dsl_perf = parse_content(path)
        # 0:filename, 1:nodes, 2:taco_gflops, 3:dsl_gflops
        res = [filename, "Error", "Error", "Error"] # -999 represents error state
        if taco_perf is not None:
            res[1] = int(taco_perf[0]) # node count
            res[2] = float(taco_perf[1]) # gflops
        if dsl_perf is not None:
            if res[1] == int(dsl_perf[0]): # node count mismatch
                res[3] = float(dsl_perf[1]) # gflops
            else:
                print(f"Node count mismatch in {filename}")
        content.append(res)
    content.sort(key=lambda row: row[0]) # sort by filename

    import sys
    # with open(out_filename, 'w') as f:
    print("================================")
    out = csv.writer(sys.stdout)# , dialect='excel-tab') # f)
    out.writerow(['filename', 'nodes', 'taco_gflops', 'dsl_gflops'])
    out.writerows(content)

if __name__ == '__main__':
    main()
