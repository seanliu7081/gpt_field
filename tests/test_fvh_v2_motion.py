"""Small CPU checks for the actual-command filter; no simulator or model."""
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from gpt_field.contact_skills import StepView
from gpt_field.foundation_harness.robot_geometry import CollisionShape
from gpt_field.foundation_harness.types import SceneSnapshot
from gpt_field.foundation_vision_harness_V2.motion import CommandFilter, FilterConfig
from gpt_field.action import TRANSLATION_SCALE_M, ROTATION_SCALE_RAD
from gpt_field.robodojo_direct_sim import rotation_matrix


class CartesianChain:
    low, high = np.full(7, -5.), np.full(7, 5.)
    def bind(self, state, q):
        return self
    def forward(self, q):
        pose = np.eye(4)
        pose[:3, 3], pose[:3, :3] = q[:3], rotation_matrix(q[3:6])
        return pose, np.column_stack((np.eye(6), np.zeros(6)))
    def solve(self, *args, **kwargs):
        raise AssertionError("next-command filtering must not require a whole IK route")


class ToolOnlyRobot:
    def __init__(self):
        self.vertices = np.array([[x, y, z] for x in (-.01, .01)
            for y in (-.01, .01) for z in (-.01, .01)])
        self.equations = ConvexHull(self.vertices).equations
    def shapes(self, bound, q, fingers):
        return [CollisionShape("test_tool", 8, self.vertices, self.equations, bound.forward(q)[0], "tool")]
    @staticmethod
    def self_collision(shapes):
        return None


def setup(points=(), unresolved=()):
    stamp = perf_counter()
    points, unresolved = np.asarray(points, float).reshape(-1, 3), np.asarray(unresolved, float).reshape(-1, 3)
    scene = SceneSnapshot(1, stamp, stamp, {}, points, None, None,
        evidence={"revision": 1, "point_identity": np.full(len(points), "external"),
            "point_position_error_m": np.zeros(len(points)),
            "point_representation_radius_m": np.zeros(len(points)),
            "unresolved_points": unresolved, "unresolved_position_error_m": np.full(len(unresolved), .02),
            "unresolved_representation_radius_m": np.zeros(len(unresolved)),
            "unresolved_reason": np.full(len(unresolved), "robot_depth_conflict")},
        scene_point_object_ids=np.full(len(points), "obstacle"))
    view = StepView(SimpleNamespace(state=np.r_[np.zeros(6), .04, -.04],
        joint_position=np.zeros(7), joint_velocity=np.zeros(7)))
    filt = CommandFilter(CartesianChain(), FilterConfig(collision_margin_m=.001, joint_error_floor_rad=0.),
        robot_model=ToolOnlyRobot())
    return filt, view, scene


def test_nominal_command_needs_no_far_ik_and_learns_only_dispatched_response():
    filt, view, scene = setup()
    action = (.7, 0., 0., 0., 0., .1, -1.)
    result = filt.filter(action, view, scene, {})
    assert result["status"] == "execute" and result["action"] == action
    assert filt.stats["dispatched_commands"] == 0
    predicted = filt._pending_command["predicted_q"].copy()
    after = StepView(SimpleNamespace(state=np.r_[predicted[:6], .04, -.04], joint_position=predicted))
    assert filt.observe_transition(view, action, after)["status"] == "updated"
    assert filt.stats["dispatched_commands"] == 1
    filt.filter(action, after, scene, {})
    assert filt.stats["scene_preparations"] == 1


def test_unknown_nearby_slows_exact_translation_and_rotation_before_prediction():
    filt, view, scene = setup(unresolved=[[0., .025, 0.]])
    action = np.array([.8, 0., 0., 0., 0., .2, -1.])
    result = filt.filter(action, view, scene, {})
    assert result["status"] == "scaled"
    actual = np.asarray(result["action"])
    assert np.linalg.norm(actual[:3])*TRANSLATION_SCALE_M <= .005+1e-12
    assert np.linalg.norm(actual[3:6])*ROTATION_SCALE_RAD <= .03+1e-12
    assert actual[:6] == pytest.approx(action[:6]*result["evidence"]["scale"])
    assert tuple(actual) == filt._pending_command["action"]
    wrong = actual.copy(); wrong[0] += .001
    with pytest.raises(ValueError, match="exact dispatched"):
        filt.observe_transition(view, wrong, view)


def test_existing_external_overlap_can_escape_without_demanding_terminal_clearance():
    filt, view, scene = setup(points=[[-.005, 0., 0.]])
    result = filt.filter((.08, 0., 0., 0., 0., 0., -1.), view, scene, {})
    assert result["status"] == "execute"
    assert result["evidence"]["existing_external_conflicts"] == 1


def test_observed_foreground_is_kept_as_constraint_and_unknown_intrusion_requests_observation():
    filt, view, scene = setup(points=[[.009, 0., 0.]])
    action = (1., 0., 0., 0., 0., 0., -1.)
    result = filt.filter(action, view, scene, {})
    assert result["status"] == "candidate_unavailable" and result["action"] is None
    filt, view, scene = setup(unresolved=[[.009, 0., 0.]])
    result = filt.filter(action, view, scene, {})
    assert result["status"] == "reobserve" and result["action"] is None
    assert len(filt._points) == 0 and len(filt._unknown_points) == 1


def test_unknown_payload_extent_requests_observation_before_carrying():
    filt, view, scene = setup()
    held = {"object_id": "body", "extent_unknown": True}
    result = filt.filter((0., 0., .4, 0., 0., 0., 1.), view, scene, {},
        held=held)
    assert result["reason"] == "payload_extent_unknown" and result["action"] is None
    release = {"_candidate": {"skill": "set_gripper", "gripper": "open"}}
    result = filt.filter((0., 0., 0., 2e-9, 0., 0., -1.), view, scene, release, held=held)
    assert result["status"] == "execute" and result["evidence"]["release_geometry_unavailable"]
    assert held["extent_unknown"]  # Recovery never invents a carried-object shape.


def test_unresolved_contact_permission_is_scoped_to_owner_and_body_part():
    filt, view, scene = setup(unresolved=[[.009, 0., 0.]])
    scene.evidence["unresolved_owners"] = np.array(["target"])
    action = (1., 0., 0., 0., 0., 0., -1.)
    segment = {"object": "target", "contact_mode": "grasp", "contact_parts": ["finger", "pad"]}
    assert filt.filter(action, view, scene, segment)["status"] == "reobserve"
    segment["contact_parts"] = ["tool"]
    assert filt.filter(action, view, scene, segment)["status"] == "execute"
    # A fresh anonymous foreground observation has no intended-contact owner.
    scene.evidence["revision"] += 1
    scene.evidence["unresolved_owners"] = np.array([""])
    assert filt.filter(action, view, scene, segment)["status"] == "reobserve"


def test_filter_timing_counts_rejection_and_acceptance_without_dispatch():
    filt, view, scene = setup()
    rejected = filt.filter((0.,)*6+(2.,), view, scene, {})
    accepted = filt.filter((.1, 0., 0., 0., 0., 0., -1.), view, scene, {})
    assert rejected['status'] == 'candidate_unavailable'
    assert accepted['status'] == 'execute'
    assert rejected['evidence']['filter_wall_s'] >= 0.
    assert accepted['evidence']['filter_status'] == 'execute'
    summary = filt.timing_summary()['latency']
    assert summary['rejected']['count'] == summary['accepted']['count'] == 1
    assert summary['accepted']['p50_s'] == accepted['evidence']['filter_wall_s']
    assert summary['rejected']['total_s']+summary['accepted']['total_s'] == pytest.approx(filt.stats['filter_s'])
    assert filt.stats['dispatched_commands'] == 0


def test_disputed_measurement_is_not_an_occupied_surface_but_real_unknown_still_blocks():
    filt, view, scene = setup()
    scene.evidence.update(disputed_points=np.array([[.009, 0., 0.]]),
        disputed_reason=np.array(["crossview_depth_conflict"]),
        identity_counts={"external": 0, "unresolved": 0, "disputed": 1})
    action = (1., 0., 0., 0., 0., 0., -1.)
    result = filt.filter(action, view, scene, {})
    assert result["status"] == "execute"
    assert result["evidence"]["disputed_measurements"] == 1
    assert not result["evidence"]["unknown_space_is_free"]
    # A separate, credible identity-unknown patch at the same coordinates keeps
    # its normal motion constraint. Quarantine is not a global bypass switch.
    scene.evidence.update(revision=2, unresolved_points=np.array([[.009, 0., 0.]]),
        unresolved_position_error_m=np.array([.02]),
        unresolved_representation_radius_m=np.zeros(1),
        unresolved_reason=np.array(["identity_unresolved"]))
    result = filt.filter(action, view, scene, {})
    assert result["status"] == "reobserve" and result["action"] is None
