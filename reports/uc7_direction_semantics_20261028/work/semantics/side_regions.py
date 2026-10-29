import re, sys, collections
sys.path.insert(0, "/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/semantics")
from bddl_parse import all_bddls
def flat(g):
    if isinstance(g, str): return g
    return "(" + " ".join(flat(x) for x in g) + ")"
DW = r"(left|right|front|back|behind)"
rows = collections.defaultdict(list)
for d in all_bddls():
    g = flat(d["goal"]); i = " ".join(flat(s) for s in d["init"])
    for n, r in d.get("regions", {}).items():
        if not re.search(DW, n): continue
        full = r["target"] + "_" + n
        use = []
        if full in g: use.append("GOAL")
        if full in i: use.append("INIT")
        if not use: use.append("unused")
        geo = ""
        if "ranges" in r:
            off = d["offset"] or (0,0,0)
            x0,y0,x1,y1 = r["ranges"][0]
            geo = f"world_c=({(x0+x1)/2+off[0]:+.3f},{(y0+y1)/2+off[1]:+.3f}) half=({(x1-x0)/2:.3f},{(y1-y0)/2:.3f})"
        else:
            geo = "object-site"
        rows[(full, geo, "/".join(use))].append(d["file"].split("/")[0] + "/" + d["file"].split("/")[1][:60])
for (full, geo, use), files in sorted(rows.items()):
    print(f"{full:50s} {use:11s} {geo:45s} n={len(files)}  e.g. {files[0]}")
