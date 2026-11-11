"""Was H3's better first obstacle box consistent with the SAM 2D box of the operator-named obstacle?
Back-project the external bbox (bottom-centre and centre) at the model's own table belief and at true heights."""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/scripts")
from triangulate import ray
from pathlib import Path

def first_request(run):
    req = sorted((Path("/workspace/gpt_field/runs")/run/"trial"/"requests").glob("*.request.json"))[0]
    d = json.load(open(req))
    for c in d["payload"]["input"][-1]["content"]:
        if c.get("type") == "input_text" and c["text"].startswith("{"):
            return json.loads(c["text"])

def hit(cam, u, v, z):
    p, d = ray(cam, u, v); s = (z - p[2]) / d[2]; return p + s * d

cases = {
  "T04": dict(run="foundation_vision_t04_can_obstacle_20261105_001", key="extra_0", truth_lo=[-0.0545,-0.2454,0.4370], truth_hi=[0.0077,-0.1837,0.5132],
              boxes={"H3 call1": ([-0.085,-0.255,0.4],[0.015,-0.175,0.5]), "H1 call1": ([-0.015,-0.25,0.4],[0.065,-0.18,0.5]), "T20 call1": ([-0.015,-0.255,0.4],[0.065,-0.185,0.51])}),
  "T08": dict(run="foundation_vision_t08_bottle_obstacle_20261105_001", key="extra_0", truth_lo=[-0.0284,-0.0896,0.9002], truth_hi=[0.0278,-0.0529,1.0458],
              boxes={"H3 call1": ([-0.075,-0.105,0.82],[0.005,-0.05,1.025]), "H1 call1": ([-0.115,-0.105,0.8],[-0.055,-0.055,0.98]), "stove-off call1": ([-0.11,-0.125,0.79],[-0.045,-0.065,0.97])}),
}
for name, c in cases.items():
    t = first_request(c["run"])
    cam = t["geometry"]["cameras"]["external"]
    fv = t["context"]["foundation_vision"]
    obj = next(o for o in fv["objects"] if o["key"] == c["key"])
    bb = obj["views"]["external"]["candidates"][0]["bbox_xyxy_exclusive"]
    u_c = (bb[0] + bb[2]) / 2; v_b = bb[3]; v_c = (bb[1] + bb[3]) / 2; v_t = bb[1]
    tlo, thi = np.array(c["truth_lo"]), np.array(c["truth_hi"]); tc = (tlo + thi) / 2
    print(f"== {name}: external bbox {bb} (text '{obj['text']}'), truth centre {tc.round(3).tolist()}")
    for label, (lo, hi) in c["boxes"].items():
        lo, hi = np.array(lo), np.array(hi); bc = (lo + hi) / 2
        zb = lo[2]  # model's own table/support belief = box bottom
        foot = hit(cam, u_c, v_b, zb)        # bbox bottom-centre on model's support plane
        mid = hit(cam, u_c, v_c, bc[2])      # bbox centre at model's box mid-height
        # where the camera ray through bbox centre passes at the model box centre height, vs model box centre xy
        print(f"  {label:16s} box centre {bc.round(3).tolist()} | bbox-bottom ray at model z={zb:.2f}: {foot[:2].round(3).tolist()} (d={np.linalg.norm(foot[:2]-bc[:2])*100:.1f} cm) | "
              f"bbox-centre ray at box mid z={bc[2]:.3f}: {mid[:2].round(3).tolist()} (d={np.linalg.norm(mid[:2]-bc[:2])*100:.1f} cm) | box centre vs truth xy {np.linalg.norm(bc[:2]-tc[:2])*100:.1f} cm")
    # project the model box centres into the image to see pixel offset from the bbox centre
    R = np.array(cam["rotation_world_from_camera"]); P = np.array(cam["position_world_m"]); fx, fy = cam["focal_length_px"]; cx, cy = cam["principal_point_px"]
    def proj(X):
        dc = np.diag([1, -1, -1]) @ R.T @ (np.array(X) - P)
        return (fx * dc[0] / dc[2] + cx, fy * dc[1] / dc[2] + cy)
    print("  truth centre projects to px", np.round(proj(tc), 1).tolist(), "bbox centre px", [u_c, v_c])
    for label, (lo, hi) in c["boxes"].items():
        bc = (np.array(lo) + np.array(hi)) / 2
        print(f"  {label:16s} box centre projects to px", np.round(proj(bc), 1).tolist())
