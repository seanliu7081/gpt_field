"""CPU-only parent evidence checks; these tests create no native authorization."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import provision_controller_characterization as parent
from robodojo_uc71.controller import ArmState
from robodojo_uc71.native_action_observer import NativeActionObservationError


def mode_records(kind):
    contract = parent.diagnostic_contract(kind)
    counts = ([25, 40, 40, 25]*2 if contract["open_only"] else [25]*38+[10])
    plan = {"stages": [{"native_actions": n} for n in counts]}
    preparation = {"schema": contract["preparation_schema"],
        "diagnostic_native_actions_planned": contract["actions"]}
    result = {"schema": contract["result_schema"], "diagnostic_completed": not contract["open_only"],
        "diagnostic_native_actions": contract["actions"], "diagnostic_physics_substeps": contract["actions"]*10,
        "image_count": contract["images"], "native_take_action_count": [contract["actions"]]}
    ready = {"schema": contract["preclose_schema"], "diagnostic_native_actions": contract["actions"],
        "image_count": contract["images"]}
    if contract["open_only"]:
        flags = {"clearance_preview_completed": True, "controller_characterization_completed": False,
            "diagnostic_completed": False}
        result.update(flags); ready.update(flags)
        plan.update(candidate={"path": "synthetic-only", "sha256": "synthetic-not-a-real-candidate"},
            candidate_abort_envelopes={"joint_distance_from_home_rad": .35, "tool_distance_from_home_m": .065},
            endpoint_validation={"samples": 5, "position_tolerance_m": .002, "orientation_tolerance_rad": .02,
                "open_finger_tolerance_m": .001, "extra_settling_actions": 0})
        for record in (preparation, result, ready):
            record.update(diagnostic_kind=kind, open_only=True,
                candidate_abort_envelopes=deepcopy(plan["candidate_abort_envelopes"]))
        for record in (preparation, result):
            record.update(candidate=deepcopy(plan["candidate"]), endpoint_validation=deepcopy(plan["endpoint_validation"]))
    return plan, preparation, result, ready


def test_distinct_preview_contract_preserves_strict_default_full_counts():
    full = parent.diagnostic_contract("controller-characterization")
    preview = parent.diagnostic_contract("open-clearance-preview")
    assert (full["actions"], full["stages"], full["images"]) == (960, 39, 120)
    assert (preview["actions"], preview["stages"], preview["images"]) == (260, 8, 27)
    assert full["authorized_child_env"] != preview["authorized_child_env"]
    assert full["result_hash_key"] != preview["result_hash_key"]
    assert preview["max_wall_time_s"] == 900
    for kind in ("controller-characterization", "open-clearance-preview"):
        assert parent.verify_diagnostic_mode_evidence(*mode_records(kind), kind) == parent.diagnostic_contract(kind)


@pytest.mark.parametrize("kind,wrong", [("controller-characterization", "open-clearance-preview"),
                                      ("open-clearance-preview", "controller-characterization")])
def test_cross_mode_evidence_cannot_be_accepted(kind, wrong):
    with pytest.raises(ValueError):
        parent.verify_diagnostic_mode_evidence(*mode_records(wrong), kind)


@pytest.mark.parametrize("record_index", [1, 2, 3])
def test_preview_rejects_full_schema_in_each_evidence_document(record_index):
    records = mode_records("open-clearance-preview")
    records[record_index]["schema"] = mode_records("controller-characterization")[record_index]["schema"]
    with pytest.raises(ValueError):
        parent.verify_diagnostic_mode_evidence(*records, "open-clearance-preview")


@pytest.mark.parametrize("record_index,field,value", [
    (1, "diagnostic_native_actions_planned", 960), (2, "diagnostic_native_actions", 259),
    (2, "diagnostic_physics_substeps", 2590), (2, "native_take_action_count", [261]),
    (2, "image_count", 24), (3, "image_count", 30), (3, "diagnostic_native_actions", 261),
    (2, "diagnostic_completed", True), (3, "diagnostic_completed", True),
    (2, "controller_characterization_completed", True), (3, "controller_characterization_completed", True),
    (2, "clearance_preview_completed", 1), (3, "clearance_preview_completed", False),
])
def test_preview_requires_exact_counts_and_honest_completion_flags(record_index, field, value):
    records = mode_records("open-clearance-preview")
    records[record_index][field] = value
    with pytest.raises(ValueError):
        parent.verify_diagnostic_mode_evidence(*records, "open-clearance-preview")


def test_preview_cannot_add_settling_actions_or_stage():
    for change in ("action", "stage", "noninteger"):
        records = mode_records("open-clearance-preview")
        if change == "action": records[0]["stages"][-1]["native_actions"] += 1
        elif change == "stage": records[0]["stages"].append({"native_actions": 0})
        else: records[0]["stages"][-1]["native_actions"] = 25.0
        with pytest.raises(ValueError):
            parent.verify_diagnostic_mode_evidence(*records, "open-clearance-preview")


@pytest.mark.parametrize("record_index", [1, 2])
def test_preview_candidate_and_endpoint_rules_cannot_change(record_index):
    for field in ("candidate", "endpoint_validation", "candidate_abort_envelopes"):
        records = mode_records("open-clearance-preview")
        records[record_index][field] = {}
        with pytest.raises(ValueError, match="binding"):
            parent.verify_diagnostic_mode_evidence(*records, "open-clearance-preview")


def test_explicit_preview_envelope_must_match_every_document():
    records = mode_records("open-clearance-preview")
    for record in records:
        record["candidate_abort_envelopes"]["joint_distance_from_home_rad"] = .50
    parent.verify_diagnostic_mode_evidence(*records, "open-clearance-preview")
    for index in (1, 2, 3):
        changed = deepcopy(records)
        changed[index]["candidate_abort_envelopes"]["joint_distance_from_home_rad"] = .35
        with pytest.raises(ValueError, match="binding"):
            parent.verify_diagnostic_mode_evidence(*changed, "open-clearance-preview")


def test_parent_open_only_check_uses_actual_normalized_dispatch_values():
    action = {f"{arm}_arm_joint_state": [0.0]*6 for arm in ("left", "right")}
    action.update({f"{arm}_ee_joint_state": [1.0] for arm in ("left", "right")})
    parent.verify_preview_open_action(action)
    for arm in ("left", "right"):
        changed = deepcopy(action); changed[f"{arm}_ee_joint_state"] = [0.0]
        with pytest.raises(ValueError, match="non-open"):
            parent.verify_preview_open_action(changed)
    for bad in ([True], [float("nan")], [.5], [1.0, 1.0]):
        changed = deepcopy(action); changed["right_ee_joint_state"] = bad
        with pytest.raises(NativeActionObservationError):
            parent.verify_preview_open_action(changed)


def endpoint_fixture(monkeypatch):
    # Nonzero physical TCP offset catches accidental use of link6 as the tool.
    def states(measured, _geometry):
        result = {}
        for arm in ("left", "right"):
            row = measured["arms"][arm]
            result[arm] = ArmState([0.]*6, [0.]*6, row["link6_position"], row["quaternion"],
                np.eye(6), [-3.]*6, [3.]*6, [2.]*6, [.1, 0., 0.])
        return result
    monkeypatch.setattr(parent, "physical_arm_states", states)
    stage = {"active_arm": "left", "target_tool_pose_wxyz": [1., 2., 3., 1., 0., 0., 0.]}
    measurements = [{"arms": {arm: {"link6_position": [.9, 2., 3.], "quaternion": [1., 0., 0., 0.],
        "finger_positions": [.044, .044]} for arm in ("left", "right")}} for _ in range(5)]
    return measurements, stage


def test_endpoint_requires_five_measured_samples_with_physical_tcp_and_both_open(monkeypatch):
    measurements, stage = endpoint_fixture(monkeypatch)
    result = parent.verify_preview_endpoint(measurements, stage, {})
    assert result == {"passed": True, "samples_checked": 5, "extra_settling_actions": 0,
        "maximum_position_error_m": 0., "maximum_orientation_error_rad": 0., "maximum_open_finger_error_m": 0.}
    for count in (0, 1, 4):
        with pytest.raises(ValueError, match="five"):
            parent.verify_preview_endpoint(measurements[:count], stage, {})
    for index in range(5):
        changed = deepcopy(measurements); changed[index]["arms"]["left"]["link6_position"][0] += .00201
        with pytest.raises(ValueError, match="settle"):
            parent.verify_preview_endpoint(changed, stage, {})
    changed = deepcopy(measurements)
    angle = .02001
    changed[0]["arms"]["left"]["quaternion"] = [float(np.cos(angle/2)), 0., 0., float(np.sin(angle/2))]
    with pytest.raises(ValueError, match="settle"):
        parent.verify_preview_endpoint(changed, stage, {})
    for arm in ("left", "right"):
        for finger in (0, 1):
            changed = deepcopy(measurements); changed[0]["arms"][arm]["finger_positions"][finger] = .0429
            with pytest.raises(ValueError, match="physically open"):
                parent.verify_preview_endpoint(changed, stage, {})


def test_nonfinite_or_wrong_shape_open_fingers_rejected(monkeypatch):
    measurements, stage = endpoint_fixture(monkeypatch)
    for bad in ([float("nan"), .044], [.044], [.044, .044, .044]):
        changed = deepcopy(measurements); changed[-1]["arms"]["right"]["finger_positions"] = bad
        with pytest.raises(ValueError, match="physically open"):
            parent.verify_preview_endpoint(changed, stage, {})
