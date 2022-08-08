import re
import sys
from pprint import pprint

fail_first = False

def readlines(f):
    with open(f) as fin:
        return fin.readlines()

def process(one_line):
    if "PhysicalInstance" not in one_line:
        return one_line
    else:
        return re.sub(r'PhysicalInstance\[.*\]', r"PhysicalInstance[XXX]", one_line)

def find_file_line(fname, target_lines):
    all_lines = readlines(fname)
    res = []
    for i in range(len(all_lines)):
        if target_lines[0] in all_lines[i]:
            matched = True
            for j in range(len(target_lines)):
                if target_lines[j] not in process(all_lines[i+j]):
                    matched = False
                    break
            if matched:
                res.append(i)
    print(f"{len(res)} matches")
    assert(len(res) >= 1)
    return res

def filter_mapper(lines):
    ret = []
    for l in lines:
        l = l.strip()
        if "{mapper}" in l:
            l2 = re.sub(r'.*{mapper}:', "", l)
            ret.append(l2)
    return ret

def filter_keyword(lines, start_kw, continue_kw):
    res = {} # header -> [sentences]
    for i in range(len(lines)):
        if start_kw in lines[i]:
            assert("<" in lines[i])
            key = re.sub(r'<.*>' ,"", lines[i]).strip() # remove task_id, e.g., <80>
            value = []
            for j in range(i + 1, len(lines)):
                if continue_kw in lines[j]:
                    value.append(lines[j].strip())
                else:
                    break
            if key in res.keys():
                assert(value == res[key]) # it should always be the same
            else:
                res[key] = value
    return res

def filter_sharding(lines):
    start_kw = "SELECT_SHARDING_FUNCTOR for"
    continue_kw = "<-"
    return filter_keyword(lines, start_kw, continue_kw)

def filter_slicing(lines):
    start_kw = "SLICE_TASK for"
    continue_kw = "->"
    return filter_keyword(lines, start_kw, continue_kw)

def filter_maptask(lines):
    res = {} # header (including input) --> [(sentences), appear_times]}
    # because even the index_points are the same, the results can still differ
    start_kw = "MAP_TASK for"
    next_kw_lst = ["SLICE_TASK for", "SELECT_SHARDING_FUNCTOR for", "MAP_TASK for"]
    def judge_stage(line):
        for nxt_kw in next_kw_lst:
            if nxt_kw in line:
                return 0 # not MAP_TASK any more
        if "TARGET PROCS:" in line:
            return 1 # output mode
        return 2 # can be either in input or output (depending on the "state")
    for i in range(len(lines)):
        if start_kw in lines[i]:
            assert("<" in lines[i])
            header = re.sub(r'<.*>' ,"", lines[i]).strip() # remove task_id, e.g., <80>
            key = [header]
            value = []
            output_mode = False
            for j in range(i + 1, len(lines)):
                judge_result = judge_stage(lines[j])
                if judge_result == 0:
                    break
                elif judge_result == 1:
                    output_mode = True
                if output_mode:
                    value.append(process(lines[j].strip()))
                else:
                    key.append(process(lines[j].strip())) 
            key = tuple(key)
            value = tuple(value) # convert list to tuple
            if key in res.keys():
                assert(res[key][0] == value)
                res[key][1] += 1 # counting occuring times
            else:
                res[key] = [value, 1]
    return res

def print_maptask_diff(map1, map2):
    maptask_times1 = sum(map(lambda x: x[1], map1.values()))
    maptask_times2 = sum(map(lambda x: x[1], map2.values()))
    print(f"map_task invoked {maptask_times1}, {maptask_times2} times respectively")
    if maptask_times1 != maptask_times2 and fail_first:
        assert(False)
    for k in map1.keys():
        if k not in map2.keys():
            pprint(k)
            pprint(map1[k][0])
            print("only appears in", sys.argv[1], "for", map1[k][1], "times")
            print("line:", find_file_line(sys.argv[1], k))
            print("----------------------------------------")
            if fail_first:
                break
            else:
                continue
        v1 = map1[k][0] # tuple of sentences
        v2 = map2[k][0]
        for i in range(len(v1)):
            if v1[i][0] != v2[i][0]:
                pprint(k)
                pprint(v1[i][0])
                pprint(v2[i][0])
                print("line:", sys.argv[1], find_file_line(sys.argv[1], k))
                print("line:", sys.argv[2], find_file_line(sys.argv[2], k))
                print("----------------------------------------")
                if fail_first:
                    assert(False)
        assert(len(v1) == len(v2))
        assert(v1 == v2)
        if map1[k][1] != map2[k][1]:
            pprint(k)
            pprint(map1[k][0])
            print(f"appears in {sys.argv[1]} for {map1[k][1]} times", find_file_line(sys.argv[1], k))
            print(f"appears in {sys.argv[2]} for {map2[k][1]} times", find_file_line(sys.argv[2], k))
            print("----------------------------------------")
            if fail_first:
                assert(False)
        if fail_first:
            assert(map1[k] == map2[k])
    for k in map2.keys():
        if k not in map1.keys():
            pprint(k)
            pprint(map2[k][0])
            print("line:", find_file_line(sys.argv[2], k))
            print("only appears in", sys.argv[2], "for", map2[k][1], "times")
            print("----------------------------------------")
            if fail_first:
                break
    if (len(map1) != len(map2)):
        print("key numbers mismatch:", len(map1), len(map2))
    assert(len(map1) == len(map2))

if __name__ == "__main__":
    print("l1:", sys.argv[1])
    print("l2:", sys.argv[2])
    f1_line = filter_mapper(readlines(sys.argv[1]))
    f2_line = filter_mapper(readlines(sys.argv[2]))
    f1_sharding = filter_sharding(f1_line)
    f2_sharding = filter_sharding(f2_line)
    assert(f1_sharding == f2_sharding)
    print("pass sharding check:", len(f1_sharding))
    f1_slicing = filter_slicing(f1_line)
    f2_slicing = filter_slicing(f2_line)
    assert(f1_slicing == f2_slicing)
    print("pass slicing check:", len(f1_slicing))
    f1_maptask = filter_maptask(f1_line)
    f2_maptask = filter_maptask(f2_line)
    print_maptask_diff(f1_maptask, f2_maptask)
    assert(f1_maptask == f2_maptask)
    print("pass maptask check:", len(f1_maptask))
