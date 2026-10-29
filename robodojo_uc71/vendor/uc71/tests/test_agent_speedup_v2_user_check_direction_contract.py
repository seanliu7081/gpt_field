"""uc7.1 direction semantics across frozen locks and actual offline requests."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, FINGERTIP_EDGE
from gpt_field.agent_speedup_v2_user_check_agent import AgentFirstV2Policy
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client
from gpt_field.agent_speedup_v2_user_check_client import (
    METRIC_INSTRUCTIONS, PLAN_METRIC_INSTRUCTIONS, PLANNING_WRIST_INSTRUCTIONS,
    PlanLock, ReviewedV2Client, ReviewedV2Policy, UserCheckV2Client,
)
from gpt_field.agent_speedup_v2_user_check_sim import CalibratedObservation
from gpt_field.agent_speedup_v2_user_check_spatial import build_view_frame
from gpt_field.contact_skills import StepView
from gpt_field.observation import image_to_data_url
from test_robocasa_user_check_geometry import STATE, metadata


TASK = "Put the package to the right of the plate."
PLAN = {
    "scene": "A package and a plate on a table.",
    "parts": [
        {"id": "A", "text": "package: body", "move": "free", "w": 35, "x": 0},
        {"id": "C", "text": "plate: flat surface", "move": "free", "w": 53, "x": 0},
        {"id": "E", "text": "table: surface to the right of the plate", "move": "fixed", "w": 55, "x": 0},
    ],
    "hold_true": [],
    "steps": [{"do": "place_on", "obj": "A", "grip": ["A"], "ref": "E", "sense": "any", "keep": [], "until": []}],
    "regions": [{"id": "E", "support": "observed_plane", "anchor": "C", "relation": "right_of", "frame": "world", "bounds_xy_m": None}],
    "placement_goals": [],
}


def calibrated_view():
    geometry = metadata()
    a = math.sqrt(.5)
    geometry["cameras"]["external"]["rotation_world_from_camera"] = [[0., -a, a], [1., 0., 0.], [0., a, a]]
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    image[32:64, 96:128] = [255, 128, 0]
    url = image_to_data_url(image, size=512, rotate=False)
    return StepView(CalibratedObservation(image=url, wrist_image=url, state=tuple(STATE),
                                          prompt=TASK, geometry=geometry))


class ExecutiveTransport:
    """Real client/history processing, deterministic response, no credentials/network."""
    offline_mock = True

    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(deepcopy(payload))
        answer = {"belief": "Placement remains unverified.", "plan": "Inspect the package.",
                  "why": "Read current visual evidence.", "type": "observe",
                  "motion": None, "macro": None, "execution": None}
        number = len(self.calls)
        return {"id": f"offline-direction-{number}", "status": "completed", "model": payload["model"],
                "reasoning": {"effort": "high"}, "offline_mock": True, "billing": "offline_mock",
                "output": [{"type": "message", "id": f"msg_direction_{number}", "role": "assistant",
                            "status": "completed", "content": [{"type": "output_text", "text": json.dumps(answer)}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


def executive(plan=PLAN):
    view = calibrated_view()
    frame = build_view_frame(view.observation.geometry)
    lock = PlanLock(deepcopy(plan), task_prompt=TASK, view_frame=frame)
    transport = ExecutiveTransport()
    client = AgentFirstV2Client(task_prompt=TASK, approved_plan=lock.plan(), lock=lock, transport=transport)
    policy = AgentFirstV2Policy(client, lock=lock)
    return view, lock, transport, client, policy


def current_fields(payload):
    message = next(row for row in reversed(payload["input"]) if row.get("role") == "user")
    return json.loads(message["content"][0]["text"])


def test_plan_lock_detaches_and_snapshots_frozen_frame_without_changing_plan_hash():
    frame = build_view_frame(calibrated_view().observation.geometry)
    lock = PlanLock(PLAN, view_frame=frame)
    expected = deepcopy(frame)
    frame["right_xy"][0] = 99
    assert lock.view_frame == expected
    snapshot = lock.snapshot()
    assert snapshot["view_frame"] == expected
    snapshot["view_frame"]["external_camera"]["position_world_m"][0] = 99
    assert lock.view_frame == expected
    assert lock.snapshot()["initial_approved_semantic_sha256"] == PlanLock(PLAN).snapshot()["initial_approved_semantic_sha256"]
    assert lock.plan() == PLAN


def test_plan_lock_validates_serialized_direction_frame_but_missing_frame_is_unavailable():
    frame = build_view_frame(calibrated_view().observation.geometry)
    frame["right_xy"] = [1., 0.]
    with pytest.raises(ValueError):
        PlanLock(PLAN, view_frame=frame)
    lock = PlanLock(PLAN)
    assert lock.view_frame["available"] is False
    assert lock.view_frame["reason"] == "no_external_calibration"


def test_frozen_frame_survives_scene_replacement_and_both_policy_reset_paths():
    view, lock, _, client, policy = executive()
    expected = deepcopy(lock.view_frame)
    assert policy.ctx.user_check_view_frame == expected
    policy.ctx.scene = object()
    assert policy._task_contract()["region_directions"]["regions"]["E"]["direction_world_xy"] == [0., 1.]
    policy.reset()
    assert policy.ctx.user_check_view_frame == expected
    policy.ctx.user_check_view_frame["right_xy"][0] = 99
    assert lock.view_frame == expected
    reviewed = ReviewedV2Client(task_prompt=TASK, approved_plan=PLAN, lock=lock, transport=ExecutiveTransport())
    legacy = ReviewedV2Policy(reviewed, lock=lock)
    assert legacy.ctx.user_check_view_frame == expected
    legacy.reset()
    assert legacy.ctx.user_check_view_frame == expected


def test_empty_regions_still_publish_frame_for_direction_named_object_parts():
    plan = {"scene": "One object.", "parts": [PLAN["parts"][0]], "hold_true": [],
            "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None, "sense": "any", "keep": [], "until": []}]}
    _, _, _, _, policy = executive(plan)
    directions = policy._task_contract()["region_directions"]
    assert directions["available"] and directions["regions"] == {}
    assert directions["frame"] == {"right_xy": [0., 1.], "away_xy": [-1., 0.]}


def test_actual_eleventh_request_keeps_frozen_directions_after_visual_history_trimming():
    view, lock, transport, client, policy = executive()
    expected = policy._task_contract()["region_directions"]
    for step in range(11):
        policy.step_index = step
        context = policy._context(view)
        # A stale caller cannot substitute a different frame; the client uses the lock.
        context["region_directions"] = {"available": False}
        result = client.request_agent(view, context)
        assert result["type"] == "observe"
        fields = current_fields(transport.calls[-1])
        assert fields["context"]["region_directions"] == expected
        if step == 0:
            assert fields["reviewed_goal_contract"]["task_contract"]["region_directions"] == expected
        else:
            assert "reviewed_goal_contract" not in fields
    final = transport.calls[-1]
    assert client.history_stats()["retained_image_pairs"] == 6
    assert "use it over any direction wording in reviewed" in final["instructions"]
    assert "frame is frozen from the external view at review" in final["instructions"]
    assert current_fields(final)["context"]["region_directions"]["regions"]["E"]["direction_world_xy"] == [0., 1.]
    assert client.budget_snapshot()["api_calls"] == 11
    assert lock.plan() == PLAN


@pytest.mark.parametrize("kind", ["plan", "replan", "repair", "review"])
def test_prompt_modes_use_appropriate_calibration_statement_and_no_old_axis_convention(kind):
    client = UserCheckV2Client(task_prompt=TASK)
    text = client.build_payload({"fields": {}, "images": []}, mode=kind)["instructions"]
    assert "right_of=+x" not in text and "WORLD axes" not in text
    assert (PLAN_METRIC_INSTRUCTIONS if kind in ("plan", "replan") else METRIC_INSTRUCTIONS) in text
    if kind in ("plan", "replan"):
        assert "you have no calibration" in text
        assert PLANNING_WRIST_INSTRUCTIONS in text
        assert "turned so that its left and right match" not in text
    else:
        assert "convert to world coordinates with calibration" in text


@pytest.mark.parametrize("turns", range(4))
def test_wrist_plan_label_matches_actual_turns_without_changing_pixels(turns):
    image = calibrated_view().external_rgb_upright
    client = UserCheckV2Client(task_prompt=TASK)
    baseline = AgentSpeedUpV2Client(task_prompt=TASK)
    args = dict(quarter_turns=turns, fingertip_edge=FINGERTIP_EDGE[turns])
    _, request, meta = client._plan_request(image, image, **args)
    _, original, _ = baseline._plan_request(image, image, **args)
    assert request["fields"]["wrist_quarter_turns"] == turns
    assert [url for _, url in request["images"]] == [url for _, url in original["images"]]
    assert meta["quarter_turns"] == turns
    label = request["images"][2][0]
    if turns == 0:
        assert "not turned; not aligned with image 1" in label
    else:
        assert f"turned {turns} quarter turns counter-clockwise" in label
        assert "not a perspective match" in label
    assert f"fingertips at the {FINGERTIP_EDGE[turns]} edge" in label
