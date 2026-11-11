"""Shared offline helpers for the h2k2_planning_loop analysis (read-only inputs).

Imports the keypose-2 snapshot code through the overlay package copy in
w2/overlay/src (foundation_harness/*.py byte-identical to the hash-pinned
snapshot). No simulator, no model, no GPU.
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np

W2 = Path('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2')
WORK = W2 / 'h2k2_planning_loop_work'
sys.path.insert(0, str(W2 / 'h2k2_planning_loop_work/overlay/src'))

from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate  # noqa: E402
from gpt_field.foundation_vision.geometry import build_geometry  # noqa: E402
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry  # noqa: E402
from gpt_field.robodojo_direct_sim import PandaKinematics  # noqa: E402

RUN = Path('/workspace/gpt_field/runs/foundation_harness_t04_can_obstacle_001')
V = RUN / 'trial/foundation_harness/vision'
LAYOUT = Path('/workspace/gpt_field/output/layouts/libero_10_t4_can_obstacle_20261103_001/layout_manifest.json')

# Reuse the run-001 analysis restore() exactly as the keypose-2 prep check did.
_helper = RUN / 'analysis/reconstruct_rejections.py'
_tree = ast.parse(_helper.read_text())
_restore = next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == 'restore')
from scipy.spatial import cKDTree  # noqa: E402,F401  (used inside restore)
exec(compile(ast.Module(body=[_restore], type_ignores=[]), str(_helper), 'exec'))

records = [json.loads(l) for l in (V / 'snapshots.jsonl').read_text().splitlines()]
events = [json.loads(l) for l in (RUN / 'trial/foundation_harness/events.jsonl').read_text().splitlines()]
robot = json.loads((RUN / 'robot_kinematics.json').read_text())
kin = PandaKinematics.from_xml(robot['robot_xml'], robot['gripper_xml'])
layout = json.loads(LAYOUT.read_text())


def scene_at(step):
    feedback = next(e for e in events if e['event'] == 'feedback' and e['step'] == step)
    record = next(r for r in records if r['snapshot']['evidence']['observation_index'] == step)
    scene, err = restore(record)  # noqa: F821
    state = np.asarray(feedback['measured_state'])
    q = np.asarray(feedback['joint_position'])
    return scene, state, q, record, err


def truth_boxes():
    """Evaluator-side layout truth, used only for offline scoring."""
    can = layout['can']['collision_bounds']
    out = {'can': (np.array(can['lower_xyz_m']), np.array(can['upper_xyz_m']))}
    poses = layout['custom_initial_settled_object_poses']
    for name in ('porcelain_mug_1', 'white_yellow_mug_1', 'red_coffee_mug_1', 'plate_1', 'plate_2'):
        out[name + '_origin'] = np.array(poses[name]['body_xyz_m'])
    out['table_top_z'] = float(layout['can']['table_support_geometry']['upper_xyz_m'][2])
    return out
