import re, sys, json
sys.path.insert(0, "/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/semantics")
from bddl_parse import all_bddls
WORDS = r"\b(left|right|front|back|behind|middle|center|centre|between|next to)\b"
def flat(g):
    if isinstance(g, str): return g
    return "(" + " ".join(flat(x) for x in g) + ")"
for d in all_bddls():
    if not re.search(WORDS, d["language"], re.I): continue
    off = d["offset"] or (0,0,0)
    print("==", d["file"]); print("   lang:", d["language"]); print("   problem:", d["problem"], "offset", off)
    print("   goal:", flat(d["goal"]))
    for n, r in d.get("regions", {}).items():
        full = r["target"] + "_" + n
        if "ranges" in r:
            x0,y0,x1,y1 = r["ranges"][0]
            cx, cy = (x0+x1)/2+off[0], (y0+y1)/2+off[1]
            print(f"   region {full:55s} world_c=({cx:+.3f},{cy:+.3f}) half=({(x1-x0)/2:.3f},{(y1-y0)/2:.3f}) yaw={r.get('yaw')}")
        else:
            print(f"   region {full:55s} (object site on {r['target']})")
    print("   init:", " ".join(flat(s) for s in d["init"]))
    print("   fixtures:", d.get("fixtures")); print("   objects:", d.get("objects"))
