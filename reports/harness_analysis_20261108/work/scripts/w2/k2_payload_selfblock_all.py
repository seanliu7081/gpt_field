"""For every saved frame and every mug target: unowned STRONG points inside the target's own payload hull (+6 mm),
i.e. what motion.py:278-293 would see at the close pose / lift start (payload = convex hull of planning_surface)."""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from gpt_field.foundation_harness.motion import _payload_shape
from gpt_field.foundation_harness.robot_geometry import CollisionShape
mugs = {'yellow and white mug': 'white_yellow_mug_1', 'white mug': 'porcelain_mug_1', 'textured ceramic mug': 'porcelain_mug_1'}
rows = []
done = set()
for step in range(20):
    rec = record_for_step(step)
    if rec['arrays'] in done: continue
    done.add(rec['arrays'])
    scene, _ = restore(rec); state, q = step_state(step)
    planner = CandidatePlanner(kin); planner._prepare(scene, state, q)
    for oid, obj in scene.objects.items():
        if obj.query not in mugs: continue
        pts, ev = planning_surface(obj)
        if len(pts) < 4:
            rows.append({'step': step, 'id': oid, 'payload': 'unknown', 'reason': ev.get('reason')}); continue
        local = _payload_shape(pts)
        hull = CollisionShape(local.name, 9, local.vertices, local.equations, np.eye(4), 'payload')
        ids = np.asarray(planner._tree.query_ball_point(hull.center, hull.radius + .006), int)
        inside = ids[hull.contains(planner._cloud[ids], .006)]
        unowned = inside[~np.isin(planner._owners[inside], [oid])]
        strong = unowned[planner._strong[unowned]]
        P = planner._cloud[strong]
        key = mugs[obj.query]
        intarget = int(in_box(P, truth[key]['lower'], truth[key]['upper'], .01).sum()) if len(P) else 0
        # share of the target's true-box strong points that are owned
        box = in_box(planner._cloud, truth[key]['lower'], truth[key]['upper'], .005) & (planner._cloud[:, 2] > truth['table_top_z'] + .003)
        st = box & planner._strong
        owned_share = float((st & (planner._owners == oid)).sum() / max(1, st.sum()))
        rows.append({'step': step, 'frame': scene.frame_id, 'id': oid, 'query': obj.query, 'status': obj.status,
                     'unowned_strong_in_payload_hull': int(len(strong)), 'of_which_in_true_target_box': intarget,
                     'owned_share_of_target_strong_points': round(owned_share, 3)})
for r in rows: print(json.dumps(r))
(W2 / 'k2_payload_selfblock_all.json').write_text(json.dumps(rows, indent=1))
