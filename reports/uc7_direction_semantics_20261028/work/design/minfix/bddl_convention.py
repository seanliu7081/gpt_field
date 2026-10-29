"""Offline convention check: for every LIBERO BDDL region named <anchor>_{left,right,front,back}_region,
compare its centre with the anchor's init region centre; report dominant world axis per word and suite.
LIBERO agentview: camera on +x at y=0 looking -x (problems/*.py set_camera) => viewer right=+y, front=+x."""
import re, glob, os, collections
L = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files"
expected = {"right": "+y", "left": "-y", "front": "+x", "back": "-x"}
def regions(txt):
    out = {}
    for m in re.finditer(r"\((\w+)\s*\(:target (\w+)\)\s*\(:ranges \(\s*\(([^)]*)\)", txt):
        x0, y0, x1, y1 = [float(v) for v in m.group(3).split()]
        out[m.group(1)] = ((x0 + x1) / 2, (y0 + y1) / 2)
    return out
per = collections.Counter(); files = collections.Counter(); bad = []
for f in sorted(glob.glob(f"{L}/*/*.bddl")):
    txt = open(f).read(); regs = regions(txt); suite = os.path.relpath(f, L).split("/")[0]
    init = re.search(r"\(:init(.*?)\)\s*\(:goal", txt, re.S).group(1)
    objpos = {}
    for m in re.finditer(r"\(On (\w+) (\w+)\)", init):
        for name, c in regs.items():
            if m.group(2).endswith(name): objpos[m.group(1)] = c
    hit = False
    for name, c in regs.items():
        m = re.match(r"(\w+?)_(left|right|front|back)_region$", name)
        if not m: continue
        anchor, word = m.groups()
        ac = regs.get(anchor + "_init_region") or next((v for k, v in objpos.items() if k.startswith(anchor)), None)
        if ac is None: continue
        dx, dy = c[0] - ac[0], c[1] - ac[1]
        if abs(dx) == abs(dy): continue
        axis = ("+x" if dx > 0 else "-x") if abs(dx) > abs(dy) else ("+y" if dy > 0 else "-y")
        per[(suite, word, axis)] += 1; hit = True
        if axis != expected[word]: bad.append((os.path.relpath(f, L), name, axis))
    files[suite] += hit
for k, v in sorted(per.items()): print(k, v)
print("files with anchored direction regions per suite:", dict(files))
print("disagreements with viewer convention:", bad if bad else 0)
