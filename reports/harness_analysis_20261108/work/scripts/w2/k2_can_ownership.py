"""Do target masks claim obstacle (can) surface points? Cylinder test, evaluation-only truth."""
import json, sys
import numpy as np, cv2
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from k2_scene_audit import replica
import json as _j
M = _j.load(open('/workspace/gpt_field/output/foundation_harness_preparation_001/inputs/layout_manifest.json'))
poses = M['custom_after_passive_hold_object_poses']
can_c = np.array(poses['tomato_sauce_obstacle_1']['body_xyz_m'][:2]); can_r = 0.0311
mug_c = np.array(poses['porcelain_mug_1']['body_xyz_m'][:2])
out = []
for step in [int(x) for x in sys.argv[1:]] or [13, 15, 17, 19]:
    rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
    prov = replica(CandidatePlanner(kin), scene, state, q)
    P, O, view, keep, S = prov['points'], prov['owners'], prov['view'], prov['keep'], prov['strong']
    rad = np.linalg.norm(P[:, :2] - can_c, axis=1)
    mug_rad = np.linalg.norm(P[:, :2] - mug_c, axis=1)
    can = (rad <= can_r + .004) & (P[:, 2] > truth['table_top_z'] + .003) & (P[:, 2] < .5131 + .004) & (mug_rad > .06)
    res = {'step': step, 'frame': scene.frame_id}
    for v in ('external', 'wrist'):
        m = can & (view == v)
        owned = {k: int((m & (O == k)).sum()) for k in scene.objects if (m & (O == k)).sum()}
        res[v] = {'can_cylinder_points': int(m.sum()), 'owned_by_targets': owned,
                  'unowned_strong': int((m & (O == '') & S & keep).sum()), 'unowned_uncertain': int((m & (O == '') & ~S & keep).sum())}
    # planning surface of each target: does it include can-cylinder points?
    for k, obj in scene.objects.items():
        surf, ev = planning_surface(obj)
        if len(surf):
            rr = np.linalg.norm(surf[:, :2] - can_c, axis=1)
            mr = np.linalg.norm(surf[:, :2] - mug_c, axis=1)
            res.setdefault('planning_surface_can_points', {})[k] = int(((rr <= can_r + .004) & (mr > .06) & (surf[:, 2] > truth['table_top_z'] + .003)).sum())
            res.setdefault('planning_surface_status', {})[k] = [ev.get('status'), ev.get('reason'), len(surf), ev.get('component_count'), ev.get('selected_mass_fraction')]
    out.append(res); print(json.dumps(res), flush=True)
    # overlay of the white_mug external mask + can cylinder pixels
    if 'white_mug' in scene.objects and 'external' in scene.objects['white_mug'].masks:
        cam = next(c for c in scene.frame.cameras if c.name == 'external')
        o = cam.rgb.copy().astype(float)
        mk = scene.objects['white_mug'].masks['external']
        o[mk] = .4*o[mk] + .6*np.array([0, 120, 255])
        sel = can & (view == 'external') & (O == 'white_mug')
        o[prov['row'][sel], prov['col'][sel]] = [255, 0, 0]
        crop = o.astype(np.uint8)[200:380, 30:230]
        cv2.imwrite(str(W2 / f'white_mug_mask_vs_can_step{step}.png'), cv2.cvtColor(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
(W2 / 'k2_can_ownership.json').write_text(json.dumps(out, indent=1))
