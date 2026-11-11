import sys, json
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
import numpy as np
from v_common import *
run = RUNS["T04"]
rng = np.random.default_rng(0)
for rp in requests(run)[:1] + requests(run)[4:5]:
    f = last_user_fields(json.load(open(rp)))
    cams = {n: cam_from_geometry(f["geometry"], n) for n in ("external", "wrist")}
    for z0, z1 in ((0.457, 0.477), (0.56, 0.64), (0.44, 0.75)):
        P = np.column_stack((rng.uniform(-0.3, 0.15, 20000), rng.uniform(-0.45, 0.45, 20000), rng.uniform(z0, z1, 20000)))
        inv = {}
        for n, (K, E, _) in cams.items():
            uv, z = project(K, E, P)
            inv[n] = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= 511) & (uv[:, 1] >= 0) & (uv[:, 1] <= 511)
        print(f"step {f['step']} z {z0:.3f}-{z1:.3f}: in external {inv['external'].mean():.2f}, in wrist {inv['wrist'].mean():.2f}, in both {(inv['external'] & inv['wrist']).mean():.2f}")
