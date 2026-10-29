import re, sys, glob, os
L = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files"
def regions(txt):
    out = {}
    for m in re.finditer(r"\((\w+)\s*\(:target (\w+)\)\s*\(:ranges \(\s*\(([^)]*)\)", txt):
        name, target, rng = m.group(1), m.group(2), [float(v) for v in m.group(3).split()]
        x0, y0, x1, y1 = rng
        out[name] = (target, ((x0 + x1) / 2, (y0 + y1) / 2))
    return out
for f in sorted(glob.glob(f"{L}/*/*.bddl")):
    txt = open(f).read()
    lang = re.search(r"\(:language ([^)]*)\)", txt).group(1)
    if not re.search(r"\b(left|right|front|back|behind)\b", lang):
        continue
    goal = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", txt, re.S).group(1)
    goal = " ".join(goal.split())
    init = re.search(r"\(:init(.*?)\)\s*\(:goal", txt, re.S).group(1)
    regs = regions(txt)
    print("==", os.path.relpath(f, L)); print("   lang:", lang); print("   goal:", goal[:200])
    for n, (t, c) in regs.items():
        print(f"   region {n:45s} on {t:28s} centre=({c[0]:+.3f},{c[1]:+.3f})")
    print("   init:", " ".join(init.split())[:400])
