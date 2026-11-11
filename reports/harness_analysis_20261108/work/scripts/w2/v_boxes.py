import json, glob, numpy as np
R = "/workspace/gpt_field/runs/"
def truth(run):
    lm = json.load(open(R + run + "/inputs/layout_manifest.json"))
    b = lm["can_after_hold_bounds"]; return np.array(b["lower_xyz_m"]), np.array(b["upper_xyz_m"])
def sample_cov(boxes, c, lo, hi, n=200000, seed=0):
    rng = np.random.default_rng(seed); P = lo + rng.random((n, 3)) * (hi - lo)
    ins = np.zeros(n, bool); ins_c = np.zeros(n, bool)
    for b in boxes:
        l, h = np.array(b["lower_xyz_m"]), np.array(b["upper_xyz_m"])
        ins |= np.all((P >= l) & (P <= h), axis=1); ins_c |= np.all((P >= l - c) & (P <= h + c), axis=1)
    return ins.mean(), ins_c.mean()
for run in ["foundation_vision_t04_can_obstacle_20261105_001", "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
            "foundation_vision_t08_bottle_obstacle_20261105_001", "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001"]:
    lo, hi = truth(run)
    for i, rq in enumerate(sorted(glob.glob(R + run + "/trial/requests/*.response.json")), 1):
        resp = json.load(open(rq))
        t = [c["text"] for o in resp["output"] if o.get("type") == "message" for c in o["content"] if c.get("type") == "output_text"][-1]
        om = json.loads(t).get("obstacle_map")
        if not om or not om.get("avoid_boxes"): continue
        cov, cov_c = sample_cov(om["avoid_boxes"], om["clearance_m"], lo, hi)
        # best single box shortfall
        print(f"{run[:52]:52s} call {i:2d} clearance={om['clearance_m']:.3f} union coverage={cov:.2f} with clearance={cov_c:.2f}")
        if i == 1 or cov < 0.6:
            for b in om["avoid_boxes"]:
                l, h = np.array(b["lower_xyz_m"]), np.array(b["upper_xyz_m"])
                ov = np.clip(np.minimum(h, hi) - np.maximum(l, lo), 0, None)
                if ov.prod() > 0 or np.linalg.norm((l + h)[:2] / 2 - (lo + hi)[:2] / 2) < 0.12:
                    print("      box", l.round(3).tolist(), h.round(3).tolist(), "truth", lo.round(3).tolist(), hi.round(3).tolist())
