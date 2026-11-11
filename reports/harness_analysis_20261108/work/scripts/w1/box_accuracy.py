"""Offline: how well do Astra's obstacle boxes cover the true added obstacle (layout truth AABB)?"""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/scripts')
from outputs import outputs

truth = {'T04': (np.array([-0.0545, -0.2454, 0.4370]), np.array([0.0077, -0.1837, 0.5132])),
         'T08': (np.array([-0.0284, -0.0896, 0.9002]), np.array([0.0278, -0.0529, 1.0458]))}
runs = {'H3 T04': ('foundation_vision_t04_can_obstacle_20261105_001', 'T04'), 'H1 T04': ('agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001', 'T04'),
        'H3 T08': ('foundation_vision_t08_bottle_obstacle_20261105_001', 'T08'), 'H1 T08': ('agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001', 'T08')}

def vol(lo, hi):
    d = np.clip(hi - lo, 0, None); return float(np.prod(d))

for label, (run, key) in runs.items():
    tlo, thi = truth[key]
    for name, obj, _ in outputs(run):
        if not obj or not obj.get('obstacle_map'): continue
        om = obj['obstacle_map']; c = om['clearance_m']
        best = None
        for b in om['avoid_boxes']:
            lo, hi = np.array(b['lower_xyz_m']), np.array(b['upper_xyz_m'])
            inter = vol(np.maximum(lo, tlo), np.minimum(hi, thi))
            inter_c = vol(np.maximum(lo - c, tlo), np.minimum(hi + c, thi))
            cov, cov_c = inter / vol(tlo, thi), inter_c / vol(tlo, thi)
            # worst-face shortfall (m) of box vs truth: how far truth sticks out of the box
            short = np.maximum(np.maximum(lo - tlo, 0), np.maximum(thi - hi, 0))
            if best is None or cov_c > best[1]:
                best = (cov, cov_c, short, b)
        cov, cov_c, short, b = best
        print(f"{label} {name[:6]} clearance={c:.3f} truth-coverage box={cov:.2f} box+clearance={cov_c:.2f} truth protrudes beyond box by xyz(max per axis)={np.round(short,3).tolist()} box={b['lower_xyz_m']}..{b['upper_xyz_m']}")
