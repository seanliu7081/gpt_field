"""Evaluate the IN-PROGRESS keypose-2 self-filter / strong-obstacle split on saved T04 frames.
Uses a private copy of src/gpt_field/foundation_harness snapshotted 2026-11-08 00:02 UTC. Offline only."""
import sys, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'pkg_cur'))
import gpt_field
assert str(HERE / 'pkg_cur') in gpt_field.__file__, gpt_field.__file__
RUN = Path('/workspace/gpt_field/runs/foundation_harness_t04_can_obstacle_001')
V = RUN / 'trial/foundation_harness/vision'
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry
from gpt_field.foundation_harness.motion import CandidatePlanner, MotionConfig, PlanningFailure
from gpt_field.foundation_harness.robot_geometry import RobotCollisionModel
from gpt_field.robodojo_direct_sim import PandaKinematics
RECORDS = [json.loads(l) for l in (V / 'snapshots.jsonl').read_text().splitlines()]
EVENTS = [json.loads(l) for l in (RUN / 'trial/foundation_harness/events.jsonl').read_text().splitlines()]
ROBOT = json.loads((RUN / 'robot_kinematics.json').read_text())
kin = PandaKinematics.from_xml(ROBOT['robot_xml'], ROBOT['gripper_xml'])
BOXES = {'can': {'external': (52, 276, 118, 360), 'wrist': (420, 222, 512, 300)},
         'red_mug_distractor': {'external': (225, 172, 312, 278)}}

def restore(record):
    data = np.load(V / record['arrays'])
    cams, depths = [], {}
    for c in record['array_metadata']['cameras']:
        p, name = c['array_prefix'], c['view']
        cams.append(CameraFrame(name, data[p+'_rgb'], data[p+'_intrinsics'], data[p+'_world_to_camera'], 0.))
        depths[name] = DepthEstimate(name, data[p+'_depth_m'], data[p+'_depth_intrinsics'], data[p+'_world_to_camera'])
    frame = VisionFrame(record['snapshot']['frame_id'], 0., tuple(cams))
    geom = build_geometry(frame, depths, {})
    objects = {}
    for i, o in enumerate(record['snapshot']['objects']):
        objects[o['id']] = ObjectGeometry(o['id'], o['query'], data[f'object_{i}_points'].astype(float), data[f'object_{i}_normals'].astype(float), o['status'], evidence=o['evidence'])
    clouds, slices, off, trees = [], {}, 0, {}
    for name, view in geom._views.items():
        clouds.append(view.points_world); slices[name] = off; trees[name] = cKDTree(view.points_world); off += len(view.points_world)
    cloud = np.concatenate(clouds); owners = np.full(len(cloud), '', object)
    for key, obj in objects.items():
        for part in obj.evidence.get('point_support_ranges', []):
            pts = obj.points[part['start']:part['end']]
            if not len(pts): continue
            dd, ii = trees[part['view']].query(pts); ok = dd < 1e-6
            owners[slices[part['view']] + ii[ok]] = key
    return SceneSnapshot(frame.frame_id, 0., 0., objects, cloud, geom, frame, evidence=record['snapshot']['evidence'], scene_point_object_ids=owners)

out = []
coll = RobotCollisionModel(kin)
for arrays, step in [('frame_000005_revision_000011.npz', 4), ('frame_000010_revision_000022.npz', 9), ('frame_000019_revision_000041.npz', 18)]:
    record = next(r for r in RECORDS if r['arrays'] == arrays)
    fb = next(x for x in EVENTS if x['event'] == 'feedback' and x['step'] == step)
    state, q = np.array(fb['measured_state']), np.array(fb['joint_position'])
    scene = restore(record)
    planner = CandidatePlanner(kin)  # current defaults (allow_unknown_space=True)
    planner._prepare(scene, state, q)
    pts = np.asarray(scene.scene_points, float)
    kept = np.zeros(len(pts), bool)
    dd, ii = cKDTree(pts).query(planner._cloud); kept[ii[dd < 1e-9]] = True
    strong_full = np.zeros(len(pts), bool); strong_full[ii[dd < 1e-9]] = planner._strong[dd < 1e-9]
    slices = record['snapshot']['evidence']['scene_point_view_slices']
    row = {'arrays': arrays, 'step': step, 'self_filter': planner._self_filter_evidence, 'views': {}}
    shapes = coll.shapes(kin.bind(state, q), q, state[6:8])
    for cam in scene.frame.cameras:
        a, b = slices[cam.name]
        view = scene.geometry._views[cam.name]
        K, E = cam.intrinsics, cam.world_to_camera
        rows_, cols_ = view.rows, view.cols
        rays = np.column_stack((cols_, rows_, np.ones(len(rows_)))) @ np.linalg.inv(K).T
        z = view.estimate.depth_m[rows_, cols_]
        entry = np.full(len(rows_), np.inf)
        for s in shapes:
            Rcl = (E @ s.pose)[:3, :3]; tcl = (E @ s.pose)[:3, 3]
            n = s.equations[:, :3] @ Rcl.T; d = s.equations[:, 3] - n @ tcl
            den = rays @ n.T
            with np.errstate(divide='ignore', invalid='ignore'):
                t = -d[None, :] / den
            tin = np.where(den < 0, t, -np.inf).max(1); tout = np.where(den > 0, t, np.inf).min(1)
            hit = (tin <= tout) & (tout > 0)
            entry = np.where(hit & (np.maximum(tin, 0) < entry), np.maximum(tin, 0), entry)
        robot_px = np.isfinite(entry)
        err = z - entry
        k = kept[a:b]; st = strong_full[a:b]
        vrow = {'robot_pixels': int(robot_px.sum()),
                'robot_pixels_kept_after_new_self_filter': int((robot_px & k).sum()),
                'kept_robot_pixels_in_front_of_hull_gt_6mm': int((robot_px & k & (err < -.006)).sum()),
                'kept_robot_pixels_strong': int((robot_px & k & st).sum()),
                'all_kept_points': int(k.sum()), 'all_kept_strong': int((k & st).sum())}
        for obj, boxes in BOXES.items():
            if cam.name in boxes:
                x0, y0, x1, y1 = boxes[cam.name]
                inbox = (cols_ >= x0) & (cols_ < x1) & (rows_ >= y0) & (rows_ < y1) & ~robot_px
                vrow[f'{obj}_box_points'] = int(inbox.sum())
                vrow[f'{obj}_box_kept_strong_frac'] = round(float((inbox & k & st).sum() / max(1, (inbox & k).sum())), 3)
        row['views'][cam.name] = vrow
    start = planner._configuration(np.asarray(q), scene, np.asarray(state[6:8]), None)
    row['start_config'] = {kk: start.get(kk) for kk in ('safe', 'reason', 'part', 'unknown_samples', 'uncertain_obstacle_samples')}
    out.append(row); print(json.dumps(row, indent=1), flush=True)
(HERE / 'current_selffilter_results.json').write_text(json.dumps(out, indent=1))
