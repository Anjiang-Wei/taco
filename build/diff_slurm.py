import re
import sys

def truncate(lines):
    ret = []
    for l in lines:
        l = l.strip()
        if "{mapper}" in l:
            l2 = re.sub(r'.*{mapper}', "", l)
            ret.append(l2)
    return ret


def readlines(f):
    with open(f) as fin:
        return fin.readlines()

def diff_lines(l1, l2):
    print(len(l1), len(l2))
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            print("equal", l1[i], i)
        else:
            print("diff", l1[i], "l1", i)
            print("diff", l2[i], "l2", i)


print("l1:", sys.argv[1])
print("l2:", sys.argv[2])
f1 = truncate(readlines(sys.argv[1]))
f2 = truncate(readlines(sys.argv[2]))
diff_lines(f1, f2)
