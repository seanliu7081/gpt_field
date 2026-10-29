"""Check LIBERO direction-word region conventions: offset of <anchor>_<dir>_region from the anchor's init region."""
import re, glob, os, collections
L = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files"
def regions(txt):
    out = {}
    for m in re.finditer(r"\((\w+)\s*\(:target (\w+)\)\s*\(:ranges \(\s*\(([^)]*)\)", txt):
        name, target, rng = m.group(1), m.group(2), [float(v) for v in m.group(3).split()]
        x0, y0, x1, y1 = rng
        out[name] = (target, ((x0 + x1) / 2, (y0 + y1) / 2), rng)
    return out
stats = collections.Counter()
rows = []
for f in sorted(glob.glob(f"{L}/*/*.bddl")):
    txt = open(f).read()
    regs = regions(txt)
    init = re.search(r"\(:init(.*?)\)\s*\(:goal", txt, re.S).group(1)
    # object -> init region
    obj_init = {}
    for m in re.finditer(r"\(On (\w+) (\w+)_(\w+_init_region|\w+_region)\)", init):
        pass
    for m in re.finditer(r"\(On (\w+) (\w+)\)", init):
        obj, reg = m.group(1), m.group(2)
        for name in regs:
            if reg.endswith(name):
                obj_init[obj] = regs[name][1]
    for name, (target, c, rng) in regs.items():
        m = re.match(r"(\w+?)_(left|right|front|back)_region$", name)
        if not m:
            continue
        anchor = m.group(1); d = m.group(2)
        # anchor centre: <anchor>_init_region or an object whose name starts with anchor
        ac = None; src = None
        if anchor + "_init_region" in regs:
            ac = regs[anchor + "_init_region"][1]; src = anchor + "_init_region"
        else:
            for obj, cc in obj_init.items():
                if obj.startswith(anchor):
                    ac = cc; src = obj
                    break
        if ac is None:
            continue
        dx, dy = c[0] - ac[0], c[1] - ac[1]
        axis = ("+x" if dx > 0 else "-x") if abs(dx) > abs(dy) else ("+y" if dy > 0 else "-y")
        if abs(dx) + abs(dy) < 1e-9:
            continue
        stats[(d, axis)] += 1
        rows.append((os.path.relpath(f, L), name, src, round(dx, 3), round(dy, 3), axis))
seen = set()
print("nonzero-offset instances (file, region):", sum(stats.values()))
for r in rows:
    key = (r[1], r[2], r[3], r[4])
    if key in seen: continue
    seen.add(key)
    print(r)
print()
for k, v in sorted(stats.items()):
    print(k, v)
