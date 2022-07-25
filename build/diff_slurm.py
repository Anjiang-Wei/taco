import re

def truncate(lines):
    ret = []
    for l in lines:
        l = l.strip()
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


f1 = truncate(readlines("slurm_taco.out"))
f2 = truncate(readlines("slurm_taco_error.out"))
diff_lines(f1, f2)
