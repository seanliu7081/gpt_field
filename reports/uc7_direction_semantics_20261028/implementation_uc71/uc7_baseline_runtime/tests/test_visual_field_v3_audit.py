"""Offline independent audit fixtures: no simulator, model, transport or actions."""
import base64
import copy
import importlib.util
import io
import json
from pathlib import Path

from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location("audit_visual_field_v3", Path(__file__).parents[1] / "scripts/audit_visual_field_v3.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reference():
    return {"moving_id": "tool_feature", "target_id": "destination_feature", "moving_kind": "tool",
            "tool_point_uv": [.35, .5], "moving_point_uv": [.35, .5], "target_point_uv": [.55, .5],
            "target_anchor_uv": [.7, .7], "moving_radius_uv": .06,
            "blocked_boxes_uv": [{"object_id": "plate", "lower_uv": [.48, .45], "upper_uv": [.65, .55]}],
            "unknown_boxes_uv": [{"object_id": "tool_occluded_table", "lower_uv": [.32, .48], "upper_uv": [.38, .52]}]}


def observation():
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), (55, 80, 30)).save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    return {"observation/image": url, "observation/wrist_image": url,
            "observation/state": [0.] * 8, "observation/joint_position": [0.] * 7,
            "observation/joint_velocity": [0.] * 7, "prompt": "synthetic audit fixture task"}


def recorded_run(tmp_path, *, blocks=2, last_field=True, interrupted=True):
    run = tmp_path / "recorded"
    run.mkdir()
    obs = observation()
    status = {"success": False, "libero_success": False, "collision": {"detected": False}}
    trace = [{"event": "reset", "observation": obs, "sim_status": status}]
    step = direct = field = 0
    for i in range(blocks):
        is_field = last_field and i == blocks - 1
        controller = "visual_field" if is_field else "direct"
        prediction = {"controller": controller, "phase": "approach", "target_id": "destination_feature",
                      "hypothesis_id": f"fixture_{i}", "reason": "Synthetic saved proposal; no model call occurred.",
                      "visual_references": {v: reference() for v in audit.VIEWS},
                      "direct_actions": [[.01, 0., 0., 0., 0., 0., 0.]] * 5 if not is_field else None,
                      "field_intent": {"goal_offset_m": [.01, 0., 0.], "orientation_target_xyzw": [0., 0., 0., 1.],
                                       "gripper_command": "hold"} if is_field else None}
        key = f"request_{i}"
        base = run / "api" / f"{i+1:06d}"
        request = {"request_key": key, "request_ordinal": i + 1, "billing": "chatgpt_subscription",
                   "payload": {"model": "gpt-6-astra", "reasoning": {"effort": "medium"}}}
        response = {"model": "gpt-6-astra", "reasoning": {"effort": "medium"},
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(prediction)}]}]}
        write(base.with_suffix(".request.json"), request)
        write(base.with_suffix(".response.json"), response)
        write(base.with_suffix(".result.json"), {"status": "valid", "budget": {"api_key_calls": 0}})
        for local_step in range(1 if is_field else 5):
            action = [0.] * 7 if is_field else [.01, 0., 0., 0., 0., 0., 0.]
            trace.extend([{"event": "action", "step": step, "action": action,
                "diagnostics": {"controller": controller, "chunk_index": i,
                                "prediction": prediction if local_step == 0 else None,
                                "request_key": key if local_step == 0 else None,
                                "selection": {"status": "obstacle_tracking_unavailable", "accepted": False} if is_field else None}},
                {"event": "executed", "step": step, "action": action, "observation": obs, "sim_status": status},
                {"event": "feedback", "step": step, "feedback": {"visual": {"views": {
                    "external": {"reliable": True}, "wrist": {"reliable": False}}, "action_to_eef": {"reliable": False}}}}])
            step += 1
            field += is_field
            direct += not is_field
    if interrupted:
        write(run / "api" / f"{blocks+1:06d}.request.json", {"request_key": "interrupted", "request_ordinal": blocks + 1,
              "billing": "chatgpt_subscription", "payload": {"model": "gpt-6-astra", "reasoning": {"effort": "medium"}}})
        write(run / "api" / f"{blocks+1:06d}.result.json", {"status": "interrupted", "budget": {"api_key_calls": 0}})
    write(run / "manifest.json", {"method_version": "v2", "controller": "hybrid", "shots": 0})
    write(run / "summary.json", {"status": "user_stop", "task_success": False, "libero_success": False,
          "steps": step, "field_actions": field, "direct_actions": direct, "local_servo_actions": 0,
          "control_elapsed_s": 100., "timing_components": {"video_render_s": 10.}, "control_without_video_s": 90.})
    (run / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in trace))
    return run


def test_conversion_preserves_regions_and_never_invents_semantic_permissions():
    source = reference()
    source["blocked_boxes_uv"][0]["object_id"] = "intended_support_contact_target"
    original = copy.deepcopy(source)
    converted = audit.legacy_reference_to_v3(source)
    assert source == original
    assert converted["blocked_boxes_uv"] == [{**original["blocked_boxes_uv"][0], "role": "obstacle"}]
    assert converted["unknown_boxes_uv"] == [{**original["unknown_boxes_uv"][0], "source": "occluded"}]
    assert converted["moving_footprint_uv"]["lower_uv"] == pytest.approx([.29, .44])
    assert "moving_radius_uv" not in converted


def test_footprint_is_not_clipped_and_unknown_texture_does_not_remove_mask():
    from gpt_field.visual_field_v3 import preflight_reference
    source = reference()
    source["moving_point_uv"] = [.02, .5]
    converted = audit.legacy_reference_to_v3(source)
    assert converted["moving_footprint_uv"]["lower_uv"][0] < 0
    result = preflight_reference(converted, (64, 64))
    assert "footprint_out_of_view" in [b["code"] for b in result["blockers"]]
    # Pure preflight has no image texture input; unknown masks stay geometric exclusions.
    source = reference()
    source["blocked_boxes_uv"] = []
    result = preflight_reference(audit.legacy_reference_to_v3(source), (64, 64))
    assert any(b["code"] == "start_unknown_conflict" and b["region_id"] == "tool_occluded_table" for b in result["blockers"])


def test_real_preflight_catches_all_84_synthetic_historical_blocks(tmp_path):
    run = recorded_run(tmp_path, blocks=84)
    before = audit.sha256(run / "trace.jsonl")
    report = audit.audit_run(run, tmp_path / "audit")
    assert report["observed"]["accepted_prediction_blocks"] == 84
    assert report["observed"]["requests_started"] == 85
    assert report["observed"]["executed_actions"] == 416
    assert report["observed"]["controller_counts"] == {"direct": 415, "visual_field": 1}
    cf = report["counterfactual"]
    assert cf["blocks_with_radius_only_conflicts"] == cf["blocks_with_legacy_grid40_conflicts"] == cf["blocks_with_v3_pre_execution_blockers"] == 84
    assert not cf["rescued_success_claim"] and cf["estimated_speedup"] is None
    assert cf["new_model_calls"] == cf["new_simulator_steps"] == cf["counterfactual_actions_executed"] == 0
    assert cf["legacy_controller_grid_size"] == 40 and cf["padding_uv"] == pytest.approx(1 / 39)
    assert report["observed"]["status"] == "user_stop" and report["observed"]["time_kind"] == "attempt"
    assert report["observed"]["control_without_video_s"] == 90
    assert not cf["per_view_model_persistence_replayed"] and not cf["local_recovery_execution_replayed"]
    assert audit.sha256(run / "trace.jsonl") == before


@pytest.mark.parametrize("tamper", ["prediction", "executed_action", "count", "fake_success", "api_key", "model_effort", "hidden_observation"])
def test_tampered_evidence_cannot_pass(tmp_path, tamper):
    run = recorded_run(tmp_path)
    if tamper in ("prediction", "executed_action", "hidden_observation"):
        rows = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
        if tamper == "prediction":
            next(x for x in rows if x["event"] == "action")["diagnostics"]["prediction"]["reason"] = "tampered"
        elif tamper == "executed_action":
            next(x for x in rows if x["event"] == "executed")["action"][0] = .9
        else:
            rows[0]["observation"]["depth"] = []
        (run / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    elif tamper in ("count", "fake_success"):
        summary = audit.read_json(run / "summary.json")
        if tamper == "count":
            summary["steps"] += 1
        else:
            summary.update(status="success", task_success=True, libero_success=True)
        write(run / "summary.json", summary)
    elif tamper == "api_key":
        write(run / "api/000001.result.json", {"status": "valid", "budget": {"api_key_calls": 1}})
    else:
        request = audit.read_json(run / "api/000001.request.json")
        request["payload"]["reasoning"]["effort"] = "high"
        write(run / "api/000001.request.json", request)
    with pytest.raises(ValueError):
        audit.audit_run(run, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_counterfactual_cannot_clear_masks_to_turn_conflict_into_success(tmp_path):
    run = recorded_run(tmp_path)

    def bad_preflight(ref, shape, grid_size):
        return {"blockers": [], "start_valid": True, "goal_valid": True}

    with pytest.raises(ValueError, match="silently accepted"):
        audit.audit_run(run, tmp_path / "audit", preflight=bad_preflight)


def test_preflight_input_mutation_is_detected(tmp_path):
    run = recorded_run(tmp_path)

    def bad_preflight(ref, shape, grid_size):
        ref["unknown_boxes_uv"].clear()
        return {"blockers": [{"code": "still_blocked"}]}

    with pytest.raises(ValueError, match="mutated"):
        audit.audit_run(run, tmp_path / "audit", preflight=bad_preflight)


def test_missing_reference_remains_missing_not_a_fake_geometry_conflict():
    from gpt_field.visual_field_v3 import preflight_reference
    assert audit.legacy_reference_to_v3(None) is None
    assert audit.legacy_intersections(None) == []
    assert preflight_reference(None, (64, 64))["available"] is False


def test_zero_radius_does_not_invent_observed_footprint():
    source = reference()
    source["moving_radius_uv"] = 0
    with pytest.raises(ValueError, match="do not invent"):
        audit.legacy_reference_to_v3(source)


def test_no_historical_or_existing_artifact_overwrite(tmp_path):
    run = recorded_run(tmp_path)
    for output in (run, run / "new_audit"):
        with pytest.raises(FileExistsError):
            audit.audit_run(run, output)
    report = tmp_path / "existing"
    report.mkdir()
    with pytest.raises(FileExistsError):
        audit.audit_run(run, report)


def synthetic_rgb(step=0):
    """Rendered test landmarks with stipulated motion; no robot simulator."""
    import numpy as np
    random = np.random.default_rng(8461)
    frame = np.full((128, 128, 3), 120, dtype=np.uint8)
    moving = random.integers(0, 256, (25, 25, 3), dtype=np.uint8)
    target = random.integers(0, 256, (25, 25, 3), dtype=np.uint8)
    frame[52:77, 20+step:45+step] = moving
    frame[52:77, 84:109] = target
    return frame


def synthetic_reference(step=0):
    x, y = (32 + step) / 127, 64 / 127
    return {"moving_id": "synthetic_tool", "target_id": "synthetic_target", "moving_kind": "tool",
            "moving_point_uv": [x, y], "tool_point_uv": [x, y], "target_anchor_uv": [96/127, y],
            "target_point_uv": [92/127, y],
            "moving_footprint_uv": {"lower_uv": [x-3/127, y-3/127], "upper_uv": [x+3/127, y+3/127]},
            "blocked_boxes_uv": [], "unknown_boxes_uv": [{"object_id": "textureless_unknown",
                "lower_uv": [.7, .1], "upper_uv": [.8, .2], "source": "occluded"}]}


def warm_synthetic_controller(*, missing_wrist=False):
    import numpy as np
    from gpt_field.visual_field_v3 import VisualFieldControllerV3
    controller = VisualFieldControllerV3(grid_size=40)
    q, state = np.zeros(7), np.zeros(8)
    seeds = {v: synthetic_reference() for v in audit.VIEWS}
    if missing_wrist:
        seeds["wrist"] = None
    controller.configure({v: synthetic_rgb() for v in audit.VIEWS}, seeds, q, state8=state)
    for step in range(1, 8):
        q[0] += .001
        state[0] += .002
        controller.observe({v: synthetic_rgb(step) for v in audit.VIEWS}, q, state8=state,
                           executed_action=[.05, 0., 0., 0., 0., 0., 0.])
    return controller, q, state


def test_real_per_view_fit_survives_other_view_reseed_without_clearing_unknown():
    controller, q, state = warm_synthetic_controller()
    summary = controller.summary()
    assert summary["view_models"]["external"]["eef_to_image"]["reliable"]
    assert summary["view_models"]["wrist"]["eef_to_image"]["reliable"]
    old_model = controller.views["external"]["feature_model"]
    old_epoch = summary["views"]["external"]["model_epoch"]
    old_wrist = summary["views"]["wrist"]["model_epoch"]
    refs = {v: synthetic_reference(7) for v in audit.VIEWS}
    refs["wrist"]["target_id"] = "explicitly_reseeded_wrist_target"
    refs["external"]["unknown_boxes_uv"] = []  # omission never proves free space
    refreshed = controller.configure({v: synthetic_rgb(7) for v in audit.VIEWS}, refs, q, state8=state)
    assert controller.views["external"]["feature_model"] is old_model
    assert refreshed["views"]["external"]["model_epoch"] == old_epoch
    assert refreshed["views"]["wrist"]["model_epoch"] != old_wrist
    assert refreshed["view_models"]["external"]["eef_to_image"]["reliable"]
    mask = refreshed["views"]["external"]["obstacle_tracks"][0]
    assert mask["object_id"] == "textureless_unknown" and mask["retained_after_omission"]
    assert mask["interior_track_required"] is False and mask["track_reliable"] is False
    assert not controller.views["external"]["field"].segment_valid([.75, .15], [.75, .15])


def test_missing_view_keeps_other_view_fit_and_omitted_constraints():
    controller, q, state = warm_synthetic_controller()
    old_model = controller.views["external"]["feature_model"]
    refreshed = controller.configure({v: synthetic_rgb(7) for v in audit.VIEWS},
                                     {"external": synthetic_reference(7), "wrist": None}, q, state8=state)
    assert controller.views["external"]["feature_model"] is old_model
    assert refreshed["view_models"]["external"]["eef_to_image"]["reliable"]
    assert refreshed["views"]["wrist"]["constraints_retained"]
    assert refreshed["views"]["wrist"]["obstacle_track_count"] == 1
    assert "wrist" in refreshed["recovery"]["missing_views"]


def observation_grant(*, wrist_corridor=True):
    return {"blocker_codes": ["response_unsupported", "moving_reference_unavailable"],
            "directions": [[.05, 0., 0.]], "max_steps": 3, "max_displacement_m": .01,
            "corridors_uv": {"external": {"lower_uv": [0., 0.], "upper_uv": [1., 1.]},
                             "wrist": {"lower_uv": [0., 0.], "upper_uv": [1., 1.]} if wrist_corridor else None}}


def test_synthetic_closed_loop_three_selected_actions_reduce_actual_image_error():
    """A stipulated RGB/state plant checks integration, not physical task performance."""
    import numpy as np
    controller, q, state = warm_synthetic_controller()
    start_error = controller.summary()["views"]["external"]["goal_error_px"]
    frame_step = 7
    for _ in range(3):
        decision = controller.select_action([.1, 0., 0., 0., 0., 0., 0.], q=q)
        assert decision["accepted"] and decision["status"] == "visual_distance_descent"
        assert decision["diagnostics"]["risk_unknown"] and not decision["diagnostics"]["clearance_certified"]
        # The test plant is explicitly stipulated as 20 pixels / normalized x action.
        pixels = int(round(20 * decision["action"][0]))
        assert pixels > 0 and np.allclose(decision["action"][1:], 0.)
        frame_step += pixels
        state[0] += pixels * .002
        q[0] += pixels * .001
        controller.observe({v: synthetic_rgb(frame_step) for v in audit.VIEWS}, q, state8=state,
                           executed_action=decision["action"])
    final = controller.summary()
    assert final["views"]["external"]["goal_error_px"] < start_error - 3
    assert final["task_success_certified"] is False


def test_grant_cannot_create_a_cold_start_response_model():
    import numpy as np
    from gpt_field.visual_field_v3 import VisualFieldControllerV3
    controller = VisualFieldControllerV3()
    q = np.zeros(7)
    controller.configure({v: synthetic_rgb() for v in audit.VIEWS},
                         {v: synthetic_reference() for v in audit.VIEWS}, q, state8=np.zeros(8))
    choice = controller.select_action([.05, 0., 0., 0., 0., 0., 0.], q=q, observation_intent=observation_grant())
    assert not choice["accepted"] and choice["status"] == "response_unsupported"
    assert choice["action"][:6] == [0.] * 6


def test_a_null_corridor_for_reliable_view_cannot_grant_a_probe():
    controller, q, state = warm_synthetic_controller()
    # Preserve the fitted directions but remove held-out validation, requiring a grant.
    for view in controller.views.values():
        view["feature_model"].validations = 0
    choice = controller.select_action([.05, 0., 0., 0., 0., 0., 0.], q=q,
                                     observation_intent=observation_grant(wrist_corridor=False))
    assert not choice["accepted"]
    assert choice["diagnostics"]["rejection_reasons"].get("probe_corridor_missing", 0) > 0


def test_never_observed_view_has_no_retained_scope_to_authorize_motion():
    controller, q, state = warm_synthetic_controller(missing_wrist=True)
    assert controller.summary()["view_models"]["external"]["eef_to_image"]["reliable"]
    choice = controller.select_action([.05, 0., 0., 0., 0., 0., 0.], q=q,
                                     observation_intent=observation_grant(wrist_corridor=False))
    assert not choice["accepted"] and choice["status"] == "unobserved_view_scope"


def test_missing_view_trust_expires_without_discarding_its_masks():
    controller, q, state = warm_synthetic_controller()
    controller.configure({v: synthetic_rgb(7) for v in audit.VIEWS},
                         {"external": synthetic_reference(7), "wrist": None}, q, state8=state)
    for _ in range(4):
        controller.observe({v: synthetic_rgb(7) for v in audit.VIEWS}, q, state8=state, executed_action=[0.] * 7)
    choice = controller.select_action([.05, 0., 0., 0., 0., 0., 0.], q=q,
                                     observation_intent=observation_grant(wrist_corridor=False))
    assert not choice["accepted"] and choice["status"] == "missing_view_trust_region"
    assert controller.summary()["views"]["wrist"]["obstacle_track_count"] == 1
