import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import feedback_state, kinematics
kin = kinematics()
for step in (8, 9, 18):
    s, q = feedback_state(step)
    b = kin.bind(s, q)
    print(step, 'rotvec', np.round(s[3:6], 4).tolist(), 'base_origin', np.round(b.world_from_base[:3, 3], 5).tolist(),
          'base_R_dev_from_I', round(float(np.abs(b.world_from_base[:3, :3] - np.eye(3)).max()), 5))
