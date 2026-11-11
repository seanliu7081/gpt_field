"""Shared offline helpers for the keypose-2 perception/geometry audit.

Reads only saved run-001 public inputs (RGB, calibration, saved DA3 depth, saved
object points/evidence, measured robot state). Imports the CURRENT live modules,
after asserting they are byte-identical to the hash-pinned keypose-2 snapshot.
No simulator, model or GPU use.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

SNAP = Path('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/h2_keypose2_snapshot')
LIVE = Path('/workspace/gpt_field')
RUN = LIVE / 'runs/foundation_harness_t04_can_obstacle_001'
V = RUN / 'trial/foundation_harness/vision'
W2 = Path('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2')

def verify_snapshot():
    out = {}
    for f in sorted((SNAP / 'src/gpt_field/foundation_harness').glob('*.py')):
        rel = 'src/gpt_field/foundation_harness/' + f.name
        a = hashlib.sha256(f.read_bytes()).hexdigest()
        b = hashlib.sha256((LIVE / rel).read_bytes()).hexdigest()
        out[rel] = a == b
    if not all(out.values()):
        raise SystemExit(f'live tree diverged from snapshot: {out}')
    return out

verify_snapshot()
sys.path.insert(0, str(LIVE / 'src'))
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate  # noqa: E402
from gpt_field.foundation_vision.geometry import build_geometry  # noqa: E402
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry  # noqa: E402
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure  # noqa: E402
from gpt_field.foundation_harness.surface import planning_surface  # noqa: E402
from gpt_field.robodojo_direct_sim import PandaKinematics  # noqa: E402

records = [json.loads(l) for l in (V / 'snapshots.jsonl').read_text().splitlines()]
events = [json.loads(l) for l in (RUN / 'trial/foundation_harness/events.jsonl').read_text().splitlines()]
robot = json.loads((RUN / 'robot_kinematics.json').read_text())
kin = PandaKinematics.from_xml(robot['robot_xml'], robot['gripper_xml'])
truth = json.loads((W2 / 'truth_boxes.json').read_text())

def restore(record):
    """Same reconstruction as runs/.../analysis/reconstruct_rejections.py:restore, current classes."""
    data = np.load(V / record['arrays'])
    cameras, depths = [], {}
    for c in record['array_metadata']['cameras']:
        p, name = c['array_prefix'], c['view']
        cameras.append(CameraFrame(name, data[p + '_rgb'], data[p + '_intrinsics'], data[p + '_world_to_camera'], 0.))
        depths[name] = DepthEstimate(name, data[p + '_depth_m'], data[p + '_depth_intrinsics'], data[p + '_world_to_camera'])
    frame = VisionFrame(record['snapshot']['frame_id'], 0., tuple(cameras))
    geom = build_geometry(frame, depths, {})
    objects = {}
    for i, obj in enumerate(record['snapshot']['objects']):
        objects[obj['id']] = ObjectGeometry(obj['id'], obj['query'], data[f'object_{i}_points'], data[f'object_{i}_normals'],
                                            obj['status'], evidence=obj['evidence'])
    clouds, slices, offset, trees = [], {}, 0, {}
    for name, view in geom._views.items():
        clouds.append(view.points_world)
        slices[name] = offset
        trees[name] = cKDTree(view.points_world)
        offset += len(view.points_world)
    cloud = np.concatenate(clouds)
    owners = np.full(len(cloud), '', object)
    ambiguous = np.zeros(len(cloud), bool)
    parents = {k: v.evidence.get('parent_id') for k, v in objects.items()}
    def ancestors(k):
        result = set()
        while parents.get(k) is not None:
            k = parents[k]; result.add(k)
        return result
    ancestry = {k: ancestors(k) for k in objects}
    max_distance = 0.
    for key, obj in objects.items():
        for part in obj.evidence.get('point_support_ranges', []):
            points = obj.points[part['start']:part['end']]
            if not len(points):
                continue
            distances, indices = trees[part['view']].query(points)
            max_distance = max(max_distance, float(distances.max()))
            if distances.max() > 1e-6:
                raise ValueError('Saved object points not matched to source rays')
            positions = slices[part['view']] + indices
            existing = owners[positions].copy()
            assign = (existing == '') & ~ambiguous[positions]
            for owner in np.unique(existing[existing != '']):
                same = existing == owner
                if owner == key or owner in ancestry[key]:
                    continue
                if key in ancestry[owner]:
                    assign |= same
                else:
                    ambiguous[positions[same]] = True
            owners[positions[assign]] = key
    owners[ambiguous] = ''
    evidence = dict(record['snapshot']['evidence'])
    # scene_point_view_slices as perception.py:567-570 records them
    sl, off = {}, 0
    for name, view in geom._views.items():
        sl[name] = [off, off + len(view.points_world)]; off += len(view.points_world)
    evidence['scene_point_view_slices'] = sl
    scene = SceneSnapshot(frame.frame_id, 0., 0., objects, cloud, geom, frame, evidence=evidence, scene_point_object_ids=owners)
    return scene, max_distance

def step_state(step):
    # feedback step k carries the measured state of observation index k; observation 0 has no
    # feedback row, and the robot only held its pose, so step 1's measured state is used for it.
    fb = next(e for e in events if e['event'] == 'feedback' and e['step'] == max(step, 1))
    return np.asarray(fb['measured_state'], float), np.asarray(fb['joint_position'], float)

def record_for_step(step, last=True):
    rs = [r for r in records if r['snapshot']['evidence']['observation_index'] == step]
    return rs[-1] if last else rs[0]

def in_box(points, lower, upper, pad=0.):
    lower = np.asarray(lower) - pad; upper = np.asarray(upper) + pad
    return np.all((points >= lower) & (points <= upper), axis=1)
