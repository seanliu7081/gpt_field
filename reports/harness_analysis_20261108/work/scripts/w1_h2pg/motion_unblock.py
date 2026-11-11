"""Which geometry fixes unblock moves from the measured start? Offline; frozen run code; saved artifacts only."""
import sys, json, math
from pathlib import Path
import numpy as np, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
import common
from common import record_by_arrays, feedback_state, kinematics, V
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry
from gpt_field.foundation_harness.motion import CandidatePlanner, MotionConfig, PlanningFailure
from gpt_field.foundation_harness.robot_geometry import RobotCollisionModel
OUT = Path(__file__).parent
kin = kinematics(); robot = RobotCollisionModel(kin)

def silhouettes(cameras, shapes, dilate_px):
    out = {}
    for cam in cameras:
        h, w = cam.rgb.shape[:2]; sil = np.zeros((h, w), np.uint8)
        K, E = cam.intrinsics, cam.world_to_camera
        for s in shapes:
            c = s.surface_points() @ E[:3, :3].T + E[:3, 3]; c = c[c[:, 2] > 1e-3]
            if len(c) < 3: continue
            uv = (c @ K.T)[:, :2] / c[:, 2:3]
            cv2.fillConvexPoly(sil, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        if dilate_px:
            sil = cv2.dilate(sil, np.ones((2*dilate_px+1, 2*dilate_px+1), np.uint8))
        out[cam.name] = sil.astype(bool)
    return out

def restore_masked(record, state, q, dilate_px=None):
    data = np.load(V / record['arrays'])
    cameras, depths = [], {}
    for c in record['array_metadata']['cameras']:
        p, name = c['array_prefix'], c['view']
        cameras.append(CameraFrame(name, data[p+'_rgb'], data[p+'_intrinsics'], data[p+'_world_to_camera'], 0.))
    sil = None
    if dilate_px is not None:
        shapes = robot.shapes(kin.bind(state, q), q, state[6:8]); sil = silhouettes(cameras, shapes, dilate_px)
    for c in record['array_metadata']['cameras']:
        p, name = c['array_prefix'], c['view']
        d = data[p+'_depth_m'].astype(float).copy()
        if sil is not None: d[sil[name]] = np.nan   # depth raster == original raster here
        depths[name] = DepthEstimate(name, d, data[p+'_depth_intrinsics'], data[p+'_world_to_camera'])
    frame = VisionFrame(record['snapshot']['frame_id'], 0., tuple(cameras))
    geom = build_geometry(frame, depths, {})
    objects = {}
    for i, obj in enumerate(record['snapshot']['objects']):
        objects[obj['id']] = ObjectGeometry(obj['id'], obj['query'], data[f'object_{i}_points'].astype(float), data[f'object_{i}_normals'].astype(float), obj['status'], evidence=obj['evidence'])
    clouds, slices, off, trees = [], {}, 0, {}
    for name, view in geom._views.items():
        clouds.append(view.points_world); slices[name] = off; trees[name] = cKDTree(view.points_world); off += len(view.points_world)
    cloud = np.concatenate(clouds); owners = np.full(len(cloud), '', object)
    for key, obj in objects.items():
        for part in obj.evidence.get('point_support_ranges', []):
            pts = obj.points[part['start']:part['end']]
            if not len(pts): continue
            dd, ii = trees[part['view']].query(pts)
            ok = dd < 1e-6
            owners[slices[part['view']] + ii[ok]] = key
    return SceneSnapshot(frame.frame_id, 0., 0., objects, cloud, geom, frame, evidence=record['snapshot']['evidence'], scene_point_object_ids=owners)

def prepare(planner, scene, state, q, margin):
    planner.bound = kin.bind(state, q); planner._scene_key = (id(scene), scene.frame_id)
    pts = np.asarray(scene.scene_points, float); owners = np.asarray(scene.scene_point_object_ids, object)
    ok = np.isfinite(pts).all(1); pts, owners = pts[ok], owners[ok]
    planner._initial_shapes = planner.robot.shapes(planner.bound, q, state[6:8])
    keep = np.ones(len(pts), bool); tree = cKDTree(pts)
    for s in planner._initial_shapes:
        ids = np.asarray(tree.query_ball_point(s.center, s.radius + margin), int)
        if len(ids): keep[ids[s.contains(pts[ids], margin) & (owners[ids] == '')]] = False
    planner._cloud, planner._owners = pts[keep], owners[keep]
    planner._tree = cKDTree(planner._cloud) if len(planner._cloud) else None
    planner._self_removed = int((~keep).sum())

MOVES = {'astra_step9_[0,-0.15,0.05]': [0, -.15, .05], 'astra_step13_[0,-0.05,0]': [0, -.05, 0], 'up_5cm': [0, 0, .05], 'down_5cm': [0, 0, -.05], 'x+5cm': [.05, 0, 0]}
results = []
for arrays, step in [('frame_000010_revision_000022.npz', 9), ('frame_000014_revision_000030.npz', 13)]:
    record = record_by_arrays(arrays); state, q = feedback_state(step)
    for variant, dil, margin in [('runtime_3mm', None, .003), ('consistent_6.5mm', None, .0065), ('hull_2cm', None, .02),
                                 ('image_silhouette_4px', 4, .003), ('image_silhouette_8px', 8, .003)]:
        scene = restore_masked(record, state, q, dil)
        for allow in (False, True):
            planner = CandidatePlanner(kin, MotionConfig(allow_unknown_space=allow))
            prepare(planner, scene, state, q, margin)
            start = planner._configuration(np.asarray(q), scene, np.asarray(state[6:8]), None)
            row = {'arrays': arrays, 'step': step, 'variant': variant, 'allow_unknown_space': allow, 'self_removed': planner._self_removed,
                   'start_config': {k: start.get(k) for k in ('safe', 'reason', 'part', 'unknown_samples')}, 'moves': {}}
            for name, disp in MOVES.items():
                prepare(planner, scene, state, q, margin)
                try:
                    plan = planner.plan({'kind': 'move', 'displacement': disp}, scene, state, q)
                    row['moves'][name] = 'PLANNED'
                except PlanningFailure as exc:
                    rej = exc.evidence.get('rejections', [{}])[-1]
                    ev = rej.get('evidence', {}) if isinstance(rej, dict) else {}
                    row['moves'][name] = f"{exc.reason}<-{rej.get('reason')}:{ev.get('reason')}:{ev.get('part', '')}:{ev.get('unknown_samples', '')}"
            results.append(row); print(json.dumps(row), flush=True)
(OUT / 'motion_unblock_results.json').write_text(json.dumps(results, indent=1))
