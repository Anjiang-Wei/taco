import sys
import subprocess

def get_job_num(fname):
    return fname.split("/")[-1].replace(".out", "")

def find_first_match(start_tag, end_tag, all_lines):
    res = []
    enter = False
    all_lines = all_lines.split("\n")
    for line in all_lines:
        if start_tag in line:
            enter = True
        if end_tag in line:
            enter = False
        if enter:
            res.append(line.strip())
    return res

def concat(lines):
    return "".join(lines)

import re
def find_all(lines, prefix):
    pat = r"40\*lassen\d+"
    list_of_str = re.findall(pat, lines)
    res = []
    for item in list_of_str:
        res.append(item.replace("40*", ""))
    return list(set(res))

if __name__ == "__main__":
    fname = sys.argv[1]
    job_num = get_job_num(fname)
    start_tag = f"Job <{job_num}>,"
    end_tag = "RUNLIMIT"
    p = subprocess.run(["bjobs", "-l"], capture_output=True, text=True)
    output = p.stdout
    lines = find_first_match(start_tag, end_tag, output)
    res = concat(lines)
    lassen_list = find_all(res, "lassen")
    print(" ".join(lassen_list))
