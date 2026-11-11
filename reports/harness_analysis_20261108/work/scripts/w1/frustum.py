import sys
sys.path.insert(0, '/workspace/gpt_field/src')
import numpy as np
from gpt_field.foundation_vision.bridge import load_manifest
from gpt_field.foundation_vision.geometry import _project
R = '/workspace/gpt_field/runs/'
m = load_manifest(R + 'foundation_vision_preparation_001/input/manifest.json')
frame = m.frames[0]; cams = {c.name: c for c in frame.cameras}
rng = np.random.default_rng(1)
TABLE_Z = 0.43676
for name, z0, z1 in (('2-4 cm above table', TABLE_Z + 0.02, TABLE_Z + 0.04), ('z 0.56-0.64', 0.56, 0.64), ('whole workspace z 0.44-0.75', 0.44, 0.75)):
    P = np.column_stack((rng.uniform(-0.3, 0.15, 20000), rng.uniform(-0.45, 0.45, 20000), rng.uniform(z0, z1, 20000)))
    ins = {}
    for n, c in cams.items():
        uv, zc = _project(P, c.intrinsics, c.world_to_camera)
        ins[n] = (zc > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= 511) & (uv[:, 1] >= 0) & (uv[:, 1] <= 511)
    print('%-28s in external frustum %.3f, in wrist frustum %.3f, in both %.3f' % (name, ins['external'].mean(), ins['wrist'].mean(), (ins['external'] & ins['wrist']).mean()))
