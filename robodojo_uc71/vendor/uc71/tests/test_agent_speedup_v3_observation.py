"""View proposal admission from public calibration; no simulator or model calls."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v3_observation import propose_evidence_views


def scenario(assessment, *, payload_unknown=False):
    state = np.zeros(8)
    state[2] = .8
    view = SimpleNamespace(state=state, position=state[:3], observation=SimpleNamespace(geometry={
        "image": {"width": 512, "height": 512}, "cameras": {"wrist": {
            "position_world_m": [0, 0, .8], "rotation_world_from_camera": np.eye(3).tolist(),
            "focal_length_px": [300, 300], "principal_point_px": [256, 256]}}}))
    class Geometry:
        entities = {}
        calls = []
        def _local_bodies(self, *args):
            return [{"half": np.array([.03, .03, .03])}], payload_unknown
        def preview_motion(self, view, target, quaternion, context, step):
            self.calls.append(deepcopy(context))
            return deepcopy(assessment)
    context = {"intent": "place", "target_id": "destination", "held_object_id": "payload",
               "commanded_gripper": "close", "allowed_contact_ids": ["destination"]}
    gap = {"coverage_witnesses": [{"world_point_m": [0, 0, .3], "uncertainty_m": None}]}
    return Geometry(), view, context, gap


def test_admitted_unknown_view_keeps_evidence_unknown_and_gripper_state():
    assessment = {"status": "unknown", "allowed_fraction": 1., "execution_allowed": True,
                  "execution_status": "admissible", "blocking_reasons": [],
                  "advisories": ["missing_free_region"], "physical_clearance_verified": False}
    geometry, view, context, gap = scenario(assessment, payload_unknown=True)
    before = deepcopy(context)
    result = propose_evidence_views(geometry, view, context, gap, 10)
    assert result["candidates"] and result["reason"] == "admissible_view_candidates"
    assert context == before and result["payload_geometry_unknown"] is True
    assert result["visibility_is_measured"] is False and result["original_intent_is_not_resumed"] is True
    for candidate in result["candidates"]:
        assert candidate["path_status"] == "unknown" and candidate["execution_allowed"] is True
        assert candidate["physical_clearance_verified"] is False
        assert candidate["advisories"] == ["missing_free_region"]
        assert candidate["execution"]["segments"][0]["gripper"] == "keep"
    assert all(item["commanded_gripper"] == "close" and item["allowed_contact_ids"] == [] for item in geometry.calls)


@pytest.mark.parametrize("assessment", [
    {"status": "supported", "allowed_fraction": 1., "execution_allowed": False,
     "blocking_reasons": ["unbounded_payload"]},
    {"status": "unknown", "allowed_fraction": 1.},
    {"status": "supported", "allowed_fraction": .4},
])
def test_no_view_executes_when_admission_denied_or_legacy_path_incomplete(assessment):
    geometry, view, context, gap = scenario(assessment, payload_unknown=True)
    result = propose_evidence_views(geometry, view, context, gap, 10)
    assert geometry.calls and not result["candidates"]
    assert result["rejected"]["motion"] > 0


def test_real_blocker_can_focus_existing_entity_without_inventing_free_space():
    assessment = {"status": "unknown", "execution_allowed": True,
                  "execution_status": "admissible", "blocking_reasons": []}
    geometry, view, context, _ = scenario(assessment)
    geometry.entities = {"obstacle": {"role": "obstacle", "attachment": "none",
                                     "position_m": [0, 0, .3], "uncertainty_m": .01}}
    gap = {"coverage_witnesses": [], "blocking_reasons": ["changed_geometry:obstacle"]}
    result = propose_evidence_views(geometry, view, context, gap, 10)
    assert result["candidates"] and result["focus_entity_id"] == "obstacle"
    assert result["visibility_is_measured"] is False
    assert "not_a_visible_material_point" in result["focus_source"]
    # An advisory alone does not create an unnecessary viewing request.
    advisory = {"coverage_witnesses": [], "blocking_reasons": [], "advisories": ["missing_free_region"]}
    assert not propose_evidence_views(geometry, view, context, advisory, 10)["candidates"]


@pytest.mark.parametrize("trigger", [
    {"blocking_reasons": ["measured_tracking_mismatch"]},
    {"tracking_status": "mismatch"},
    {"tracking_status": "lost"},
])
def test_tracking_failure_focuses_target_and_new_view_does_not_inherit_failed_tracker(trigger):
    admitted = {"status": "unknown", "execution_allowed": True,
                "execution_status": "admissible", "blocking_reasons": []}
    geometry, view, context, _ = scenario(admitted)
    geometry.entities = {"destination": {"role": "obstacle", "attachment": "none",
                                        "position_m": [0, 0, .3], "uncertainty_m": .01}}
    context["tracking"] = {"status": "mismatch", "position_progress_m": 0.}
    original = deepcopy(context)
    gap = {"coverage_witnesses": [], **trigger}
    before_gap = deepcopy(gap)
    result = propose_evidence_views(geometry, view, context, gap, 10)
    assert result["candidates"] and result["focus_entity_id"] == "destination"
    assert context == original and gap == before_gap
    assert all("tracking" not in item and item["intent"] == "observe" for item in geometry.calls)
    assert all(item["commanded_gripper"] == "close" for item in geometry.calls)
    assert all(not candidate["physical_clearance_verified"] for candidate in result["candidates"])


@pytest.mark.parametrize("admit", [True, False])
def test_tracking_failure_without_target_uses_public_endpoint_and_checks_new_path(admit):
    checked = {"status": "unknown", "execution_allowed": admit,
               "execution_status": "admissible" if admit else "blocked",
               "blocking_reasons": [] if admit else ["obstacle_core:wall:side"]}
    geometry, view, context, _ = scenario(checked)
    # A public synchronized mount with the grip site ahead of the wrist camera.
    view.observation.geometry["cameras"]["wrist"]["position_world_m"] = [0, 0, 1.3]
    context["tracking"] = {"status": "lost"}
    gap = {"coverage_witnesses": [], "tracking_status": "lost"}
    result = propose_evidence_views(geometry, view, context, gap, 10)
    assert bool(result["candidates"]) is admit
    assert geometry.calls and all("tracking" not in item for item in geometry.calls)
    assert result["focus_source"] == "observed_robot_endpoint_after_tracking_anomaly"
    assert result["witness_world_point_m"] == view.position.tolist()
    assert result["visibility_is_measured"] is False
    if not admit:
        assert result["rejected"]["motion"] > 0
