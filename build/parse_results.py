#!/usr/bin/env python3

# https://raw.githubusercontent.com/StanfordLegion/legion/papers/index-launch-sc21/language/sc21_scripts/parse_results.py

import csv
import glob
import os
import re

# _filename_re = re.compile(r'out_([0-9]+)x([0-9]+)_r([0-9]+)[.]log')
_filename_re = re.compile(r'([0-9]+)[.]out')
def parse_basename(filename):
    match = re.match(_filename_re, filename)
    assert match is not None
    return match.groups()

# _content_re = re.compile(r'^ELAPSED TIME = +([0-9.]+) s$', re.MULTILINE)
# On 16 nodes achieved GFLOPS per node: 8232.518652.
_content_re = re.compile(r'On +([0-9]+) nodes achieved GFLOPS per node: +([0-9.]+).', re.MULTILINE)
def parse_content(path, choice):
    # first appearance, choice = 0 --> TacoMapper
    # second appearance, choice = 1 --> DSL Mapper
    res = None
    match_times = 0
    assert choice == 0 or choice == 1
    with open(path, 'r') as f:
        content = f.readlines()
        for line in content:
            match = re.search(_content_re, line)
            if match is not None:
                if match_times % 2 == choice:
                    res = match.groups()
                match_times += 1
        return res

# GFLOPS = 485417.560 GFLOPS
# un-used here
_content_re2 = re.compile(r'^GFLOPS = +([0-9.]+) GFLOPS$', re.MULTILINE)
def parse_content2(path):
    with open(path, 'r') as f:
        content = f.read()
        match = re.search(_content_re2, content)
        if match is None:
            return ('ERROR',)
        return match.groups()

def main():
    paths = glob.glob('*.out')
    content = []
    content2 = []
    for path in paths:
        filename = os.path.basename(path)
        file_c = parse_content(path, 0)
        if file_c is not None:
            content.append((filename, *file_c, "TACO"))
        file_c2 = parse_content(path, 1)
        if file_c2 is not None:
            content2.append((filename, *file_c2, "DSL"))
    content.sort(key=lambda row: (int(row[1]), row[0]))
    content2.sort(key=lambda row: (int(row[1]), row[0]))


    import sys
    # with open(out_filename, 'w') as f:
    out = csv.writer(sys.stdout)# , dialect='excel-tab') # f)
    out.writerow(['filename', 'nodes', 'gflops', 'tag'])
    out.writerows(content)
    out.writerows(content2)

if __name__ == '__main__':
    main()
