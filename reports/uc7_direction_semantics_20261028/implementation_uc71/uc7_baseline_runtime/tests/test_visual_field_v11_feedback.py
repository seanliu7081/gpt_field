"""Feedback presentation shrinks without changing observations or permissions."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from gpt_field.visual_feedback_compact_v11 import compact_feedback, compact_payload_feedback
from test_visual_field_v7_protocol import current_record
from test_visual_field_v11_protocol import client, prediction
from test_visual_field_protocol import observation


def feedback():
    return {
        "controller_handoff":{"decision_rule":"Static instruction already in the prompt.",
                              "reference_repair_required":True, "recent_reference_failures":[{"reason":"track_lost"}]},
        "visual":{"attachment":{"state":"uncertain", "reason":"weak_fresh_motion", "uncertain_travel_remaining_m":.0012,
                                 "decision_contract":"Static instruction", "allowance_evidence":{"measured_gain":20}},
                  "reference_health":{"status":"budget_exhausted", "repair_required":True},
                  "response_evidence":{"state":"weakly_observable", "noise_threshold_px":1.2},
                  "response":{"status":"ready", "sample_count":24, "fit_samples":[[1,2,3]]},
                  "tracking":{"moving":{"tracked_point_uv":[.4,.3], "current_roi_uv":[.3,.2,.5,.4],
                                         "reliable":True, "residual_px":.4, "coordinate_semantics":"Static"}}},
        "subgoal_ledger":{"active":{"world_goal_xyz":[.03,0,0],"subgoal_id":"continue_same","local_complete":False}},
        "translation_guard":{"accepted":False,"reason":"projected_neighbor_conflict", "escape_remaining_m":.006,
                             "conflicting_neighbor_ids":["neighbor"], "current_tool_roi":[.3,.2,.5,.4]},
        "field":{"goal_xyz":[.03,0,0], "max_actions":80, "max_travel_m":.09,
                 "stop_reason":"field_command_below_useful_motion_floor", "inflated_regions":[{"id":"neighbor"}],
                 "region_feasibility":[{"id":"neighbor", "kind":"obstacles", "geometry_mode":"hard_tool_core_soft_uncertainty",
                    "nominal":{"lower_xyz":[.025,-.005,-.005], "upper_xyz":[.035,.005,.005]},
                    "tool_core":{"lower_xyz":[.02,-.01,-.01], "upper_xyz":[.04,.01,.01], "contains_current":False,"contains_goal":True},
                    "full_halo":{"lower_xyz":[.01,-.02,-.02],"upper_xyz":[.05,.02,.02],"contains_current":False,"contains_goal":True},
                    "margins_m":{"tool_radius":.005,"combined_uncertainty":.01}, "certified_occupied":False}],
                 "starvation_guard":{"minimum_useful_command_m":.001, "details":{"effective_cap_m":.0008}}},
        "new_unknown_warning":{"active":True,"reason":"preserve_new_diagnostics"},
    }


def test_active_warnings_budgets_correspondences_and_world_goals_remain_exact():
    raw = feedback(); before = copy.deepcopy(raw)
    packed = compact_feedback(raw, [0,0,0])
    assert raw == before
    for key in ("translation_guard", "subgoal_ledger", "new_unknown_warning"):
        assert packed[key] == raw[key]
    for key in ("reference_health", "response_evidence"):
        assert packed["visual"][key] == raw["visual"][key]
    for key in ("state", "reason", "uncertain_travel_remaining_m", "allowance_evidence"):
        assert packed["visual"]["attachment"][key] == raw["visual"]["attachment"][key]
    for key in ("tracked_point_uv", "current_roi_uv", "reliable", "residual_px"):
        assert packed["visual"]["tracking"]["moving"][key] == raw["visual"]["tracking"]["moving"][key]
    assert packed["controller_handoff"]["reference_repair_required"]
    assert packed["controller_handoff"]["recent_reference_failures"] == raw["controller_handoff"]["recent_reference_failures"]
    assert "fit_samples" not in packed["visual"]["response"]
    assert packed["visual"]["response"]["sample_count"] == 24
    region = packed["field"]["region_feasibility"][0]
    assert region["nominal"] == raw["field"]["region_feasibility"][0]["nominal"]
    assert region["tool_core"]["current_hypothesis_clearance_m"] == pytest.approx(.02)
    assert region["tool_core"]["goal_hypothesis_clearance_m"] == pytest.approx(-.01)
    assert region["tool_core"]["contains_goal"] is True
    assert "lower_xyz" not in region["tool_core"] and "inflated_regions" not in packed["field"]
    for key in ("stop_reason", "max_actions", "max_travel_m", "starvation_guard"):
        assert packed["field"][key] == raw["field"][key]


def test_client_compacts_wire_feedback_only_while_internal_full_feedback_remains_untouched():
    instance, transport = client(prediction())
    instance.observable_context = {"visual_feedback":feedback(), "refresh_reason":"projected_neighbor_conflict"}
    before = copy.deepcopy(instance.observable_context)
    obs = observation().to_dict()
    instance.predict(obs)
    record = current_record(transport.calls[0])
    assert instance.observable_context == before
    assert record["observation/state"] == obs["observation/state"]
    assert record["observation/joint_position"] == obs["observation/joint_position"]
    packed = record["observable_context"]["visual_feedback"]
    assert "decision_rule" not in packed["controller_handoff"]
    assert packed["translation_guard"] == before["visual_feedback"]["translation_guard"]
    assert record["observable_context"]["refresh_reason"] == "projected_neighbor_conflict"


def test_archived_v10_request_text_reduction_preserves_raw_images_state_outputs_and_file_hash():
    root = Path(__file__).resolve().parents[1]
    matches = list((root / "runs/visual_field_v10_zero_shot_001/trials/bowl_original_zero_hybrid_v10/api").glob("000009*.request.json"))
    if not matches: pytest.skip("Immutable V10 archive unavailable")
    path = matches[0]; digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.loads(path.read_text())["payload"]; original = copy.deepcopy(payload)
    packed = compact_payload_feedback(payload)
    assert payload == original and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert packed["instructions"] == payload["instructions"] and packed["text"] == payload["text"]
    assert len(packed["input"]) == len(payload["input"])
    before_chars = after_chars = 0
    for old, new in zip(payload["input"], packed["input"]):
        if old.get("role") != "user": assert old == new
        for a, b in zip(old.get("content", []), new.get("content", [])):
            if a.get("type") != "input_text": assert a == b
            else:
                before_chars += len(a["text"]); after_chars += len(b["text"])
                try: ra, rb = json.loads(a["text"]), json.loads(b["text"])
                except ValueError: assert a == b; continue
                for key in ("observation/state", "observation/joint_position", "observation/joint_velocity", "before_state", "after_state"):
                    if key in ra: assert ra[key] == rb[key]
    assert after_chars < .8 * before_chars
    assert compact_payload_feedback(packed) == packed  # Idempotent rendering, no progressive evidence loss.


def test_duplicate_regions_are_not_removed_when_nominal_geometry_is_unavailable():
    raw = {"field":{"inflated_regions":[{"id":"unknown","lower_xyz":[0,0,0],"upper_xyz":[1,1,1]}]}}
    assert compact_feedback(raw)["field"]["inflated_regions"] == raw["field"]["inflated_regions"]


def test_union_free_gap_is_not_changed_to_occupied_display_box_by_compaction():
    raw = feedback()
    field = raw["field"]
    field["goal_xyz"] = [.5, 0., 0.]
    region = field["region_feasibility"][0]
    region["union_bounds_are_display_only"] = True
    for label, margin in (("tool_core", 0.), ("full_halo", .1)):
        parts = [{"tool_part": "left", "lower_xyz": [-2.-margin,-1.,-1.], "upper_xyz": [-1.+margin,1.,1.]},
                 {"tool_part": "right", "lower_xyz": [1.-margin,-1.,-1.], "upper_xyz": [2.+margin,1.,1.]}]
        region[label+"_parts"] = parts
        # Both the current position and goal are INSIDE the display AABB,
        # but OUTSIDE the actual disconnected union used by the planner.
        region[label] = {"lower_xyz": [-2.-margin,-1.,-1.], "upper_xyz": [2.+margin,1.,1.],
            "contains_current": False, "contains_goal": False, "current_distance_to_union_m": 1.-margin}
    from gpt_field.robot_tool_geometry import world_aabbs
    envelope = {"source": "fixed_robot_collision_assets_and_measured_robot_state_only",
        "scene_geometry_certified": False, "orientation_margin_rad": .04, "full_finger_range": True,
        "parts": world_aabbs([0,0,1,0,0,0,.04,-.04], orientation_margin_rad=.04, full_finger_range=True)}
    field["robot_tool_envelope"] = envelope
    raw["selection"] = {"accepted": False, "diagnostics": {"robot_tool_envelope": copy.deepcopy(envelope)}}
    original = copy.deepcopy(raw)
    packed = compact_feedback(raw, [0., 0., 0.])
    compact = packed["field"]["region_feasibility"][0]
    assert raw == original and compact["nominal"] == region["nominal"]
    assert compact["margins_m"] == region["margins_m"]
    assert compact["union_bounds_are_display_only"] and compact["union_part_coordinates_omitted"]
    for label, margin in (("tool_core", 0.), ("full_halo", .1)):
        assert compact[label]["contains_current"] is compact[label]["contains_goal"] is False
        assert compact[label]["current_distance_to_union_m"] == 1.-margin
        assert compact[label]["goal_distance_to_union_m"] == pytest.approx(.5-margin)
        assert compact[label]["part_count"] == 2
        assert label+"_parts" not in compact
        assert "current_hypothesis_clearance_m" not in compact[label]
        assert "goal_hypothesis_clearance_m" not in compact[label]
        assert "lower_xyz" not in compact[label]
    for summary in (packed["field"]["robot_tool_envelope"], packed["selection"]["diagnostics"]["robot_tool_envelope"]):
        assert summary["source"] == envelope["source"] and not summary["scene_geometry_certified"]
        assert summary["orientation_margin_rad"] == .04 and summary["full_finger_range"]
        assert len(summary["parts"]) == 5 and summary["part_coordinates_omitted"]
        for source, part in zip(envelope["parts"], summary["parts"]):
            for key in ("name", "orientation_margin_m", "padding_m", "numerical_padding_m", "finger_range_m"):
                assert part[key] == source[key]
            assert "min_xyz" not in part and "offset_min_xyz" not in part
    assert packed["translation_guard"] == raw["translation_guard"]
    assert compact_feedback(packed, [0.,0.,0.]) == packed


def test_actual_v11_executor_summary_compacts_parts_but_preserves_union_evidence():
    from gpt_field.visual_field_v11 import MetricFieldExecutorV11
    from test_visual_field_v4 import box, state
    from test_visual_field_v6 import axis_intent
    from test_visual_field_v9 import assessment

    executor = MetricFieldExecutorV11()
    executor.configure(state(), axis_intent(obstacles=[box("neighbor",
        low=(.10,.10,-.02), high=(.12,.12,.02))]), scene_assessment=assessment("neighbor"))
    raw = {"field": executor.summary(), "translation_guard": feedback()["translation_guard"]}
    original = copy.deepcopy(raw)
    packed = compact_feedback(raw, state()[:3].tolist())
    old, new = raw["field"]["region_feasibility"][0], packed["field"]["region_feasibility"][0]
    for key in ("nominal", "margins_m", "geometry_mode", "current_only_in_uncertainty_halo",
                "hard_collision_hypothesis", "certified_occupied", "union_bounds_are_display_only"):
        assert new[key] == old[key]
    for key in ("tool_core", "full_halo"):
        for name in ("contains_current", "contains_goal", "current_distance_to_union_m"):
            assert new[key][name] == old[key][name]
        assert new[key]["part_count"] == 6
    assert raw == original and packed["translation_guard"] == raw["translation_guard"]
    assert len(json.dumps(packed)) < .65 * len(json.dumps(raw))
