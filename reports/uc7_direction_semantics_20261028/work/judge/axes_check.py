"""Read-only check: viewer axes from recorded external calibrations, plus the wrist quarter-turn a
calibration-backed external adapter would produce (same candidate rule as geometry.frame_alignment_turns)."""
import json, sys, re
import numpy as np

def strings(x):
    if isinstance(x, dict):
        for v in x.values(): yield from strings(v)
    elif isinstance(x, list):
        for v in x: yield from strings(v)
    elif isinstance(x, str):
        yield x

def geometry_from(obj):
    for s in strings(obj):
        if '"rotation_world_from_camera"' not in s: continue
        i = s.find('{')
        while i != -1:
            try:
                val, end = json.JSONDecoder().raw_decode(s[i:])
            except ValueError:
                i = s.find('{', i + 1); continue
            stack = [val]
            while stack:
                x = stack.pop()
                if isinstance(x, dict):
                    cams = x.get("cameras")
                    if isinstance(cams, dict) and "external" in cams:
                        return x
                    stack.extend(x.values())
                elif isinstance(x, list):
                    stack.extend(x)
            i = s.find('{', i + max(end, 1))
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            cams = x.get("cameras")
            if isinstance(cams, dict) and "external" in cams: return x
            stack.extend(x.values())
        elif isinstance(x, list): stack.extend(x)
    return None

def unit(v):
    v = np.asarray(v, float); n = float(np.linalg.norm(v)); return v / n if n > 1e-12 else None

for f in sys.argv[1:]:
    g = geometry_from(json.load(open(f)))
    print(f.split("runs/")[-1][:100])
    if g is None:
        print("  no geometry found"); continue
    ext = g["cameras"]["external"]; wr = g["cameras"].get("wrist")
    R = np.asarray(ext["rotation_world_from_camera"], float)
    right = unit(R[:2, 0]); away = np.array([-right[1], right[0]])
    away_view = unit(-R[:2, 2])
    print("  ext name", ext.get("name") or ext.get("camera"), "pose_source", ext.get("pose_source"), "pos", np.round(ext.get("position_world_m"), 3))
    print("  right", np.round(right, 4), "| away=up x right", np.round(away, 4), "| away=-R[:2,2] norm", np.round(away_view, 4),
          "| angle between", round(float(np.degrees(np.arccos(np.clip(away @ away_view, -1, 1)))), 2), "deg | |R[:2,2]|",
          round(float(np.linalg.norm(R[:2, 2])), 3))
    if wr is not None:
        Rw = np.asarray(wr["rotation_world_from_camera"], float)
        print("  wrist image-right", np.round(Rw[:, 0], 3), "dot ext right", round(float(unit(Rw[:2, 0]) @ right), 3))
