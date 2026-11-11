"""Cross-view 'strong' fraction of depth points on the unnamed can obstacle (offline)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
src = (HERE / 'cf_v2.py').read_text().split('\nout = {}\n')[0]
ns = {'__file__': str(HERE / 'cf_v2.py')}
exec(compile(src, 'cf_v2_head', 'exec'), ns)
restore, step_inputs = ns['restore'], ns['step_inputs']
# can box in the 512x512 external image, read from the overlay (2x) image: x 105-235, y 555-715
x0, x1, y0, y1 = 52, 118, 277, 358
out = {}
for step in (2, 4, 9, 13, 18):
    rec, state, q = step_inputs(step)
    scene = restore(rec)
    view = scene.geometry._views['external']
    sel = (view.cols >= x0) & (view.cols < x1) & (view.rows >= y0) & (view.rows < y1)
    owners = np.asarray(scene.scene_point_object_ids, object)[:len(view.points_world)]
    sel &= owners == ''
    pts = view.points_world[sel]
    # keep pixels on the can (above the table): z above table + 1 cm
    table = np.percentile(scene.objects['seed_1'].points[:, 2], 50) if 'seed_1' in scene.objects and len(scene.objects['seed_1'].points) else .425
    above = pts[:, 2] > table + .01
    strong = (view.support[sel] >= 2) & (view.contradictions[sel] == 0)
    out[f'step{step}'] = {'can_box_unassigned_points_above_table': int(above.sum()),
                          'strong_fraction': float(strong[above].mean()) if above.sum() else None,
                          'support_ge2': int((view.support[sel][above] >= 2).sum()),
                          'contradicted': int((view.contradictions[sel][above] > 0).sum()),
                          'z_range_m': [float(pts[above, 2].min()), float(pts[above, 2].max())] if above.sum() else None}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
