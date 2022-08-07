import re
import sys
from pprint import pprint

def readlines(f):
    with open(f) as fin:
        return fin.readlines()

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
    res = {} # header --> {set of (sentences)}
    # because even the index_points are the same, the results can still differ
    start_kw = "MAP_TASK for"
    next_kw_lst = ["SLICE_TASK for", "SELECT_SHARDING_FUNCTOR for", "MAP_TASK for"]
    def same_kw(line):
        for nxt_kw in next_kw_lst:
            if nxt_kw in line:
                return False
        return True
    def process(one_line):
        if "PhysicalInstance" not in one_line:
            return one_line
        else:
            return re.sub(r'PhysicalInstance\[.*\]', r"PhysicalInstance[XXX]", one_line)
    for i in range(len(lines)):
        if start_kw in lines[i]:
            assert("<" in lines[i])
            key = re.sub(r'<.*>' ,"", lines[i]).strip() # remove task_id, e.g., <80>
            value = []
            for j in range(i + 1, len(lines)):
                if same_kw(lines[j]):
                    value.append(process(lines[j].strip()))
                else:
                    break
            value = tuple(value) # convert list to tuple
            if key in res.keys():
                res[key].add(value)
            else:
                res[key] = {value}
    return res

def print_maptask_diff(map1, map2):
    fail = False
    for k in map1.keys():
        if k not in map2.keys():
            print(k, "only appears in", sys.argv[1])
            break
        v1 = sorted(list(map1[k]))
        v2 = sorted(list(map2[k]))
        for i in range(len(v1)):
            if v1[i] != v2[i]:
                print(k)
                pprint(v1[i])
                pprint(v2[i])
                fail = True
                break # stop at first diff
        assert(len(v1) == len(v2))
        if fail:
            break
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
