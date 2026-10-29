"""Roll / sign facts of the recorded external calibrations (reads request JSON files; no simulator)."""
import json, sys, glob, re
import numpy as np

def find_geometry(obj):
    if isinstance(obj, dict):
        if "cameras" in obj and isinstance(obj["cameras"], dict) and "external" in obj["cameras"]:
            return obj
        for v in obj.values():
            g = find_geometry(v)
            if g is not None: return g
    elif isinstance(obj, list):
        for v in obj:
            g = find_geometry(v)
            if g is not None: return g
    return None

def from_text(text):
    # request bodies may embed the fields as a JSON string inside the payload
    try:
        obj = json.loads(text)
    except Exception:
        return None
    g = find_geometry(obj)
    if g is not None: return g
    for s in re.findall(r'"\{.*?\}"', text[:0]):
        pass
    # search embedded JSON strings
    def walk(o):
        if isinstance(o, str) and '"cameras"' in o:
            try:
                return find_geometry(json.loads(o))
            except Exception:
                i = o.find('{"cameras"')
                return None
        if isinstance(o, dict):
            for v in o.values():
                r = walk(v)
                if r is not None: return r
        if isinstance(o, list):
            for v in o:
                r = walk(v)
                if r is not None: return r
        return None
    return walk(obj)

for path in sys.argv[1:]:
    text = open(path).read()
    g = from_text(text)
    if g is None:
        print(path, "no geometry found"); continue
    cam = g["cameras"]["external"]
    R = np.asarray(cam["rotation_world_from_camera"], float)
    r = R[:2, 0] / np.linalg.norm(R[:2, 0]); away = np.array([-r[1], r[0]])
    roll = np.degrees(np.arctan2(R[2, 0], R[2, 1]))
    print(f"{path.split('/runs/')[-1][:90]}\n   {cam['mujoco_name']} pose_source={cam['pose_source']} right={np.round(r,3)} away={np.round(away,3)} "
          f"roll=atan2(R20,R21)={roll:+.2f} deg  R[2,1]={R[2,1]:+.3f}  pos={np.round(cam['position_world_m'],3)}")
