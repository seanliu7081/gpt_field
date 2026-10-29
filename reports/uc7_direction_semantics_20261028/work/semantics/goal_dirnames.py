import re, sys
sys.path.insert(0, "/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/semantics")
from bddl_parse import all_bddls
def flat(g):
    if isinstance(g, str): return g
    return "(" + " ".join(flat(x) for x in g) + ")"
DW = r"(left|right|front|back|behind|middle|center)"
seen=set()
for d in all_bddls():
    g = flat(d["goal"]); i = " ".join(flat(s) for s in d["init"])
    if re.search(DW, g) or re.search(DW, i):
        key=(d["language"], g)
        print(f"{d['file'][:95]:95s}\n    lang={d['language']}\n    goal={g}\n    init-dir-terms={sorted(set(re.findall(r'[a-z_0-9]*'+DW+r'[a-z_0-9]*', i)))}")
