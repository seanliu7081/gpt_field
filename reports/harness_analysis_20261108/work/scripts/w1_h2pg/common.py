"""Shared offline restore helpers (read-only use of saved run artifacts)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RUN = Path('/workspace/gpt_field/runs/foundation_harness_t04_can_obstacle_001')
sys.path.insert(0, str(RUN / 'source_at_launch/src'))

from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate  # noqa: E402
from gpt_field.foundation_vision.geometry import build_geometry  # noqa: E402
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry  # noqa: E402
from gpt_field.robodojo_direct_sim import PandaKinematics  # noqa: E402

V = RUN / 'trial/foundation_harness/vision'
RECORDS = [json.loads(line) for line in (V / 'snapshots.jsonl').read_text().splitlines()]
EVENTS = [json.loads(line) for line in (RUN / 'trial/foundation_harness/events.jsonl').read_text().splitlines()]
ROBOT = json.loads((RUN / 'robot_kinematics.json').read_text())


def kinematics():
    return PandaKinematics.from_xml(ROBOT['robot_xml'], ROBOT['gripper_xml'])


def record_by_arrays(name):
    return next(r for r in RECORDS if r['arrays'] == name)


def restore(record):
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
        objects[obj['id']] = ObjectGeometry(obj['id'], obj['query'], data[f'object_{i}_points'].astype(float),
                                            data[f'object_{i}_normals'].astype(float), obj['status'],
                                            evidence=obj['evidence'])
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
            k = parents[k]
            result.add(k)
        return result
    ancestry = {k: ancestors(k) for k in objects}
    point_view_index = {}
    for key, obj in objects.items():
        point_view_index[key] = []
        for part in obj.evidence.get('point_support_ranges', []):
            points = obj.points[part['start']:part['end']]
            if not len(points):
                continue
            distances, indices = trees[part['view']].query(points)
            point_view_index[key].append((part, indices))
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
    scene = SceneSnapshot(frame.frame_id, 0., 0., objects, cloud, geom, frame,
                          evidence=record['snapshot']['evidence'], scene_point_object_ids=owners)
    return scene, data, point_view_index


def feedback_state(step):
    fb = next(x for x in EVENTS if x['event'] == 'feedback' and x['step'] == step)
    return np.array(fb['measured_state']), np.array(fb['joint_position'])
