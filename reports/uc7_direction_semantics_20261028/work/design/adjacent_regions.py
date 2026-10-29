"""For goal regions named <anchor>_<dir>_region used in task goals: offset and size relative to the anchor's init region."""
import re, glob, os
L = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files"
def regions(txt):
    out = {}
    for m in re.finditer(r"\((\w+)\s*\(:target (\w+)\)\s*\(:ranges \(\s*\(([^)]*)\)", txt):
        x0, y0, x1, y1 = [float(v) for v in m.group(3).split()]
        out[m.group(1)] = (m.group(2), (x0, x1), (y0, y1))
    return out
seen = set()
for f in sorted(glob.glob(f"{L}/*/*.bddl")):
    txt = open(f).read()
    lang = re.search(r"\(:language ([^)]*)\)", txt).group(1)
    goal = " ".join(re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", txt, re.S).group(1).split())
    regs = regions(txt)
    init = re.search(r"\(:init(.*?)\)\s*\(:goal", txt, re.S).group(1)
    for name, (target, gx, gy) in regs.items():
        m = re.match(r"(\w+?)_(left|right|front|back)_region$", name)
        if not m or f"{target}_{name}" not in goal:
            continue
        anchor, d = m.group(1), m.group(2)
        anchor_reg = anchor + "_init_region"
        if anchor_reg not in regs:
            mm = re.search(rf"\(On {anchor}_1 \w+?_(\w+_region)\)", init)
            if mm and mm.group(1) in regs: anchor_reg = mm.group(1)
        if anchor_reg not in regs:
            print("no anchor for", lang, name); continue
        _, ax, ay = regs[anchor_reg]
        acx, acy = sum(ax)/2, sum(ay)/2
        key = (lang, name)
        if key in seen: continue
        seen.add(key)
        print(f"{lang[:75]:75s} | {d:5s} | {anchor_reg:24s} | goal-anchor x[{gx[0]-acx:+.3f},{gx[1]-acx:+.3f}] y[{gy[0]-acy:+.3f},{gy[1]-acy:+.3f}] | {os.path.basename(os.path.dirname(f))}")
