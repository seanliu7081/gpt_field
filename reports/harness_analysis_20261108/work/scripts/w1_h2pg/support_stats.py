import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays
out = {}
for arrays in ['frame_000005_revision_000011.npz', 'frame_000010_revision_000022.npz']:
    record = record_by_arrays(arrays)
    scene, data, _ = restore(record)
    row = {}
    for name, v in scene.geometry._views.items():
        n = len(v.points_world)
        row[name] = {'valid_points': n,
                     'crossview_supported_frac': round(float(((v.support >= 2) & (v.contradictions == 0)).mean()), 4),
                     'contradicted_frac': round(float((v.contradictions > 0).mean()), 4),
                     'occluded_in_other_frac': round(float((v.occlusions > 0).mean()), 4),
                     'single_view_unchecked_frac': round(float(((v.support < 2) & (v.contradictions == 0) & (v.occlusions == 0)).mean()), 4)}
    out[arrays] = row
print(json.dumps(out, indent=1))
(Path(__file__).parent / 'support_stats_results.json').write_text(json.dumps(out, indent=1))
