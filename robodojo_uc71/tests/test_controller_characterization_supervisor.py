"""Offline authorization/clearance/time tests; no native simulator is imported."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_controller_characterization as child
import provision_controller_characterization as parent


class Clock:
    def __init__(self): self.value = 0.; self.sleeps = []
    def monotonic(self): return self.value
    def sleep(self, duration): self.sleeps.append(duration); self.value += duration


def wait_fixture(tmp_path, monkeypatch, decision=None, drift=False):
    attempt = tmp_path/"attempt"; output = attempt/"worker/controller"; output.mkdir(parents=True)
    reviews = attempt/"operator_reviews"; reviews.mkdir(mode=0o750)
    os.chown(reviews, 0, 65534); os.chmod(reviews, 0o750)
    checkpoint = output/"stage_001"; checkpoint.mkdir()
    images = []
    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        path = checkpoint/(camera+".png"); path.write_bytes(b"synthetic non-image binding fixture")
        images.append({"camera":camera, "path":str(path.relative_to(output)), "sha256":child.digest(path)})
    capture = {"images":images}
    stage = {"index":1, "active_arm":"left"}
    preparation = {"plan_sha256":"synthetic-plan", "operator_review_directory":str(reviews)}
    clock = Clock(); monkeypatch.setattr(child.time, "monotonic", clock.monotonic); monkeypatch.setattr(child.time, "sleep", clock.sleep)
    reads = []
    def paused(env, counters):
        reads.append(1)
        return {"native_physics_time_s":1.0, "native_sim_step_calls":10,
            "native_physics_counter":10, "native_action_counter":[1],
            "articulation_dof_positions":{"left":[[.1 if drift and len(reads)>1 else 0.]*8], "right":[[0.]*8]}}
    monkeypatch.setattr(child, "paused_state", paused)
    written = []
    def runtime():
        if decision is not None and not written:
            request = output/"clearance_left/request.json"
            parent.write_operator_review(request, expected_request_sha256=child.digest(request),
                decision=decision, reason="synthetic-only visual-review fixture")
            written.append(1)
    return output, stage, preparation, capture, runtime, clock


def test_missing_clearance_stops_after_bounded_motionless_wait(tmp_path, monkeypatch):
    output, stage, prep, capture, verify, clock = wait_fixture(tmp_path, monkeypatch)
    with pytest.raises(TimeoutError):
        child.wait_for_operator_clearance(NS(take_action_cnt=[1]), {}, output, stage, capture, prep, lambda *a,**k:None, verify)
    evidence = json.loads((output/"clearance_left/wait_result.json").read_text())
    assert clock.value == 120 and len(clock.sleeps) == 120
    assert evidence["accepted"] is False and evidence["physics_unchanged"] is True
    assert evidence["simulation_updates_during_wait"] == 0


def test_fresh_root_owned_review_is_bound_to_exact_request(tmp_path, monkeypatch):
    output, stage, prep, capture, verify, clock = wait_fixture(tmp_path, monkeypatch, "clear_for_empty_gripper_timing")
    result = child.wait_for_operator_clearance(NS(take_action_cnt=[1]), {}, output, stage, capture, prep, lambda *a,**k:None, verify)
    review = Path(result["review_path"])
    assert review.parent.name == "operator_reviews" and review.stat().st_uid == 0 and review.stat().st_gid == 65534
    assert review.stat().st_mode & 0o777 == 0o640
    assert clock.sleeps == []


def test_ambiguous_review_stops_without_dispatch(tmp_path, monkeypatch):
    output, stage, prep, capture, verify, _ = wait_fixture(tmp_path, monkeypatch, "ambiguous_or_not_clear")
    with pytest.raises(RuntimeError, match="ambiguous"):
        child.wait_for_operator_clearance(NS(take_action_cnt=[1]), {}, output, stage, capture, prep, lambda *a,**k:None, verify)
    evidence = json.loads((output/"clearance_left/wait_result.json").read_text())
    assert evidence["accepted"] is False and evidence["failure"] == "RuntimeError"


def test_raw_joint_drift_invalidates_accepted_visual_review(tmp_path, monkeypatch):
    output, stage, prep, capture, verify, _ = wait_fixture(tmp_path, monkeypatch, "clear_for_empty_gripper_timing", drift=True)
    with pytest.raises(RuntimeError, match="Physics moved"):
        child.wait_for_operator_clearance(NS(take_action_cnt=[1]), {}, output, stage, capture, prep, lambda *a,**k:None, verify)
    assert json.loads((output/"clearance_left/wait_result.json").read_text())["physics_unchanged"] is False


def test_float32_accumulation_tolerance_is_explicit_and_bounded():
    assert parent.MAX_ACTIONS == child.MAX_ACTIONS == 960
    tolerance = child.accumulated_time_tolerance(960)
    assert 1.9e-6 < tolerance < 2e-6
    assert child.accumulated_time_tolerance(1) < 1.03e-7


def test_operator_review_cannot_be_rebound_to_other_images():
    request = {"plan_sha256":"x", "stage_index":3, "active_arm":"right",
        "native_action_counter":[20], "image_sha256":{"head.png":"a"}}
    review = {"schema":"robodojo-controller-clearance-review-1", "request_sha256":"hash",
        **request, "reviewer":"root/operator", "decision":"clear_for_empty_gripper_timing",
        "reason":"synthetic", "full_test_authorized":False, "benchmark_attempt_started":False, "object_truth_used":False}
    assert child.validate_operator_review(review, request, "hash") is True
    review["image_sha256"] = {"head.png":"b"}
    with pytest.raises(ValueError): child.validate_operator_review(review, request, "hash")


def test_diagnostic_scope_binds_program_and_first_probe(tmp_path, monkeypatch):
    program = tmp_path/"synthetic.py"; program.write_text("synthetic_only = True\n")
    monkeypatch.setattr(child, "program_files", lambda:[program])
    binding = {"gpu_uuid":"synthetic-gpu", "pci_bus_id":"0000:01:00.0", "kit_index":0, "cuda_ordinal":0}
    plan = {"binding":binding}; first = {"launch_result_sha256":"result", "analysis_sha256":"analysis"}
    auth = {"schema":"robodojo-controller-characterization-authorization-1", "plan_sha256":child.content_hash(plan),
        "gpu_uuid":binding["gpu_uuid"], "pci_bus_id":binding["pci_bus_id"], "attempt_output":str(tmp_path/"fresh_attempt"),
        "diagnostic_native_actions_authorized":True, "maximum_native_actions":960,
        "model_calls_authorized":False, "benchmark_attempt_started":False, "full_test_authorized":False,
        "automatic_retry":False, "automatic_tuning":False, "shutdown_mode":"stock-fast",
        "operator_visual_clearance_required":True, "operator_wait_timeout_s":120,
        "checkpoint_position_tolerance_m":.002, "checkpoint_orientation_tolerance_rad":.02,
        "open_finger_tolerance_m":.001, "first_probe_result_sha256":"result", "first_probe_analysis_sha256":"analysis",
        "program_sha256":{str(program):child.digest(program)}, "maximum_wall_time_s":1800,
        "authorization_basis":"synthetic test only", "user_message":"synthetic"}
    path=tmp_path/"scope.json"; path.write_text(json.dumps(auth))
    child.check_authorization(path, plan, binding, first_probe=first, launch=False, output_parent=tmp_path/"fresh_attempt")
    path.write_text(json.dumps({**auth, "maximum_native_actions": 1040}))
    with pytest.raises(ValueError, match="Exact separate"):
        child.check_authorization(path, plan, binding, first_probe=first, launch=False, output_parent=tmp_path/"fresh_attempt")
    path.write_text(json.dumps(auth))
    program.write_text("synthetic_only = False\n")
    with pytest.raises(ValueError, match="Exact separate"):
        child.check_authorization(path, plan, binding, first_probe=first, launch=False, output_parent=tmp_path/"fresh_attempt")
