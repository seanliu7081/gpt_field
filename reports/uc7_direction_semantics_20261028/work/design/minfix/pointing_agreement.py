"""Offline prototype of the passive (non-gating) pointing cue: does the region's
external grid cell lie on the side of the anchor's cell that its relation word
names?  Image-space, sign-only (8x8 grid, cells 1-64 row-major from top-left).
right/left -> column sign; front/behind -> row sign (front = lower in the image
for a camera looking down at the support).  0/equal -> undetermined."""
import json, glob, os
RUNS = "/workspace/gpt_field/runs"
def rc(cell):
    return None if not cell else divmod(cell - 1, 8)
def side(relation, anchor_cell, region_cell, convention):
    a, b = rc(anchor_cell), rc(region_cell)
    if a is None or b is None:
        return "undetermined(cell 0)"
    drow, dcol = b[0] - a[0], b[1] - a[1]
    if convention == "viewer":   # proposed: right=image-right, front=nearer camera=lower rows
        want = {"right_of": dcol, "left_of": -dcol, "front_of": drow, "behind": -drow}[relation]
    else:                        # old constant in LIBERO: +x=toward camera (down), +y=image-right
        want = {"right_of": drow, "left_of": -drow, "behind": dcol, "front_of": -dcol}[relation]
    return "agrees" if want > 0 else "disagrees" if want < 0 else "undetermined(same row/col)"
seen = set()
for f in sorted(glob.glob(RUNS + "/**/approved_plan.json", recursive=True)):
    p = json.load(open(f))
    for r in p.get("regions", []) or []:
        if r["relation"] not in ("right_of", "left_of", "front_of", "behind"):
            continue
        parts = {x["id"]: x for x in p["parts"]}
        key = (os.path.relpath(f, RUNS).split("/")[0], r["id"])
        if key in seen: continue
        seen.add(key)
        a, e = parts[r["anchor"]], parts[r["id"]]
        print(f"{os.path.relpath(os.path.dirname(f), RUNS)[:62]:62s} {r['relation']:8s} anchor x={a['x']:2d} region x={e['x']:2d} "
              f"| viewer: {side(r['relation'], a['x'], e['x'], 'viewer'):26s} | old(LIBERO): {side(r['relation'], a['x'], e['x'], 'old')}")
        print(f"{'':62s} text: {e['text']}")
