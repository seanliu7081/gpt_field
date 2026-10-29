"""Object-selection adjectives (the left/right/front/back X): compare goal object's init centre with its same-category siblings."""
import re, glob, os
L = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files"
def regions(txt):
    out = {}
    for m in re.finditer(r"\((\w+)\s*\(:target (\w+)\)\s*\(:ranges \(\s*\(([^)]*)\)", txt):
        x0, y0, x1, y1 = [float(v) for v in m.group(3).split()]
        out[m.group(1)] = (m.group(2), ((x0 + x1) / 2, (y0 + y1) / 2))
    return out
done=set()
for f in sorted(glob.glob(f"{L}/*/*.bddl")):
    txt = open(f).read()
    lang = re.search(r"\(:language ([^)]*)\)", txt).group(1)
    m = re.search(r"\b(?:the )?(\w+ \w+|\w+) (?:on the|at the|in the)? ?(left|right|front|back|middle)\b", lang)
    if not re.search(r"(on the (left|right)\b|at the (front|back)\b|the (left|right) (plate|bowl|moka pot)|(left|right) (bowl|plate))", lang):
        continue
    if lang in done: continue
    done.add(lang)
    regs = regions(txt)
    init = re.search(r"\(:init(.*?)\)\s*\(:goal", txt, re.S).group(1)
    goal = " ".join(re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", txt, re.S).group(1).split())
    pos = {}
    for m in re.finditer(r"\(On (\w+) (\w+)\)", init):
        obj, reg = m.group(1), m.group(2)
        for name in regs:
            if reg.endswith(name):
                pos[obj] = regs[name][1]
    print("==", lang, "||", goal)
    for obj, c in sorted(pos.items()):
        print(f"    {obj:28s} ({c[0]:+.3f},{c[1]:+.3f})")
