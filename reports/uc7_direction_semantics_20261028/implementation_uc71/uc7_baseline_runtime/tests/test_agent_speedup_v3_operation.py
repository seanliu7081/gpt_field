"""Persistent interaction lifecycle contracts; no simulator or model calls.

Real two-view optical-flow behavior is tested in test_agent_speedup_v3_interaction.
These fixtures isolate command acknowledgement and geometry acceptance, where
execution-route changes previously discarded evidence or advanced incorrectly.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gpt_field.agent_speedup_v3_operation import OperationRuntime


class Geometry:
    def __init__(self):
        self.entities = {name: {"id": name, "role": "object", "attachment": "none",
                                "geometry_revision": 1, "stale": False}
                         for name in ("package", "other")}
        self.entities["container"] = {"id": "container", "role": "destination_region"}
        self.raw_status = "unknown"
        self.accept = True
        self.attachment_probes = []
        self.release_probes = []
        self.confirmations = []
        self.transitions = []

    def observe_transition(self, before, after, step, command):
        self.transitions.append((step, command))
        if command == "open":
            for entity in self.entities.values():
                if entity.get("attachment") == "held":
                    entity["release_pending"] = True

    def begin_attachment_probe(self, ident, view, step):
        self.attachment_probes.append((ident, step))
        self.entities[ident].update(attachment="held", attachment_probe_unconfirmed=True,
                                   attachment_probe_shape_usable=True)
        return {"status": "unknown", "attachment_probe_shape_usable": True}

    def begin_release_probe(self, ident, view, step):
        self.release_probes.append((ident, step))
        self.entities[ident]["release_probe_started"] = True
        return {"status": "unknown", "release_pending": False}

    def confirm_attachment_evidence(self, evidence, view, step):
        self.confirmations.append(deepcopy(evidence))
        if not self.accept:
            return {"accepted": False, "reason": "unresolved_shape_or_neighbor_change"}
        entity = self.entities[evidence["target_id"]]
        entity.update(attachment="held" if evidence["kind"] == "hold" else "none",
                      attachment_probe_unconfirmed=False, release_pending=False)
        # Geometry may publish a fresh revision when accepting visual evidence;
        # that must not reset the very confirmation just obtained.
        entity["geometry_revision"] += 1
        return {"accepted": True, "kind": evidence["kind"], "target_id": evidence["target_id"]}


class Monitor:
    def __init__(self, geometry):
        self.geometry = geometry
        self.active = None
        self.compute_s = 0.
        self.observations = 0

    def begin(self, target_id, view, step, kind="hold"):
        self.active = {"kind": kind, "target_id": target_id, "reference_step": step, "last_step": step}
        self.compute_s += .25

    def adopt_reference(self, other):
        self.active = other.active

    def observe(self, view, step):
        if self.active is not None and step > self.active["last_step"]:
            self.active["last_step"] = step
            self.observations += 1

    def evidence(self, kind, target_id, view, step, since_step=0):
        self.observe(view, step)
        valid = (self.active is not None and self.active["kind"] == kind
                 and self.active["target_id"] == target_id and self.active["reference_step"] >= since_step)
        status = self.geometry.raw_status if valid else "unknown"
        return {"status": status, "kind": kind, "target_id": target_id,
                "reference_step": self.active["reference_step"] if self.active else None,
                "observed_at_step": step, "source": "public_two_view_feature_motion",
                "views": {name: {"status": status} for name in ("external", "wrist")},
                "attachment_verified": False, "task_success": "not_available_to_policy"}

    def performance_stats(self):
        return {"visual_interaction_s": self.compute_s,
                "visual_interaction_observations": self.observations}


def setup():
    geometry = Geometry()
    runtime = OperationRuntime(geometry, monitor=Monitor(geometry))
    view = SimpleNamespace(gap_mm=30.)
    context = {"intent": "grasp", "target_id": "package", "held_object_id": None}
    return runtime, geometry, view, context


def settle(runtime, view, context, first=1, last=12):
    for step in range(first, last+1):
        runtime.before_action(view, context, "close", step-1)
        runtime.observe_transition(view, view, context, "close", step)


def test_preparing_a_rejected_command_does_not_claim_close_or_payload():
    runtime, geometry, view, context = setup()
    runtime.before_action(view, context, "close", 0)
    runtime.before_action(view, context, "close", 0)
    assert runtime.counts["references_prepared"] == 1
    assert runtime.last_command == "open" and geometry.attachment_probes == []
    # The actual fallback action retains the previous open command.
    runtime.before_action(view, context, "open", 0)
    runtime.observe_transition(view, view, context, "open", 1)
    assert runtime.monitor.active is None
    assert runtime.counts["references_started"] == 0
    assert geometry.attachment_probes == [] and runtime.last_command == "open"
    assert runtime.performance_stats()["visual_interaction_s"] == pytest.approx(.25)


def test_close_settling_is_not_itself_a_grasp_and_acknowledgement_is_idempotent():
    runtime, geometry, view, context = setup()
    settle(runtime, view, context, last=11)
    assert geometry.attachment_probes == []
    runtime.observe_transition(view, view, context, "close", 11)
    assert runtime.counts["acknowledged_actions"] == 11
    settle(runtime, view, context, first=12, last=12)
    assert geometry.attachment_probes == [("package", 12)]
    assert runtime.evidence("hold", view, 12)["status"] == "unknown"
    assert runtime.counts["hold_supported"] == 0
    assert runtime.snapshot()["task_success"] == "not_available_to_policy"


def test_evidence_survives_execute_program_switch_and_confirmation_revision():
    runtime, geometry, view, context = setup()
    settle(runtime, view, context)
    original = runtime.operation_id
    geometry.raw_status = "supported"
    carrying = {"intent": "transit", "target_id": "container", "held_object_id": "package"}
    settle(runtime, view, carrying, first=13, last=15)
    assert runtime.evidence("hold", view, 15)["status"] == "supported"
    assert runtime.operation_id == original and runtime.bound_object_id == "package"
    assert len(geometry.attachment_probes) == len(geometry.confirmations) == 1
    assert runtime.counts["references_started"] == 1
    assert runtime.monitor.observations == 15


def test_raw_visual_support_cannot_override_rejected_geometry_confirmation():
    runtime, geometry, view, context = setup()
    settle(runtime, view, context)
    geometry.raw_status, geometry.accept = "supported", False
    settle(runtime, view, context, first=13, last=13)
    evidence = runtime.evidence("hold", view, 13)
    assert evidence["status"] == "unknown" and evidence["visual_status"] == "supported"
    assert evidence["geometry_confirmation"]["accepted"] is False
    assert evidence["reason"] == "geometry_confirmation_rejected"
    assert runtime.counts["confirmation_rejections"] == 1
    assert runtime.counts["hold_supported"] == 0
    geometry.accept = True
    settle(runtime, view, context, first=14, last=14)
    assert runtime.evidence("hold", view, 14)["status"] == "supported"


def test_release_keeps_manipulated_object_when_stage_target_is_container():
    runtime, geometry, view, context = setup()
    settle(runtime, view, context)
    geometry.raw_status = "supported"
    settle(runtime, view, context, first=13, last=13)
    ident = runtime.operation_id
    placing = {"intent": "place", "target_id": "container", "held_object_id": "package"}
    runtime.before_action(view, placing, "open", 13)
    assert not geometry.entities["package"].get("release_pending")
    runtime.observe_transition(view, view, placing, "open", 14)
    retreat = {"intent": "retreat", "target_id": "container", "held_object_id": None}
    runtime.observe_transition(view, view, retreat, "open", 15)
    evidence = runtime.evidence("release", view, 15)
    assert evidence["status"] == "supported" and evidence["target_id"] == "package"
    assert runtime.operation_id == ident and runtime.counts["references_started"] == 2
    assert runtime.evidence("release", view, 15, "container")["status"] == "unknown"
    assert runtime.evidence("hold", view, 15)["reason"] == "release_command_revoked_prior_hold_evidence"
    # Keeping a previous explicit held ID while opening must not create another
    # manipulation lifecycle after the visual release was already accepted.
    runtime.before_action(view, placing, "open", 15)
    runtime.observe_transition(view, view, placing, "open", 16)
    assert runtime.operation_id == ident and runtime.counts["release_supported"] == 1


def test_rejected_open_preparation_does_not_discard_existing_hold_reference():
    runtime, geometry, view, context = setup()
    settle(runtime, view, context)
    geometry.raw_status = "supported"
    settle(runtime, view, context, first=13, last=13)
    reference = runtime.monitor.active["reference_step"]
    runtime.before_action(view, context, "open", 13)
    runtime.before_action(view, context, "close", 13)
    runtime.observe_transition(view, view, context, "close", 14)
    assert runtime.monitor.active["reference_step"] == reference
    assert runtime.monitor.active["kind"] == "hold" and runtime.last_command == "close"
    assert runtime.evidence("hold", view, 14)["status"] == "supported"
    assert not geometry.entities["package"].get("release_pending")
    assert runtime.performance_stats()["visual_interaction_s"] == pytest.approx(.5)


def test_model_shape_refresh_preserves_operation_and_recovery_budget():
    runtime, geometry, view, context = setup()
    settle(runtime, view, context)
    original = runtime.operation_id
    assert runtime.claim_recovery("blocked", {"view": 1}, 12)["accepted"]
    geometry.entities["package"] = {"id": "package", "role": "object", "attachment": "none",
                                    "geometry_revision": 2, "stale": False}
    settle(runtime, view, context, first=13, last=13)
    assert runtime.operation_id == original
    assert len(geometry.attachment_probes) == 2
    assert not runtime.claim_recovery("blocked", {"view": 1}, 13)["accepted"]
    assert runtime.claim_recovery("blocked", {"view": 2}, 13)["accepted"]
    assert not runtime.claim_recovery("blocked", {"view": 3}, 13)["accepted"]


def test_new_unattached_model_hypothesis_cannot_erase_pending_physical_release():
    runtime, geometry, view, context = setup()
    settle(runtime, view, context)
    geometry.entities["package"] = {"id": "package", "role": "object", "attachment": "none",
                                    "geometry_revision": 2, "stale": False}
    placing = {"intent": "place", "target_id": "container", "held_object_id": None}
    runtime.before_action(view, placing, "open", 12)
    assert geometry.entities["package"]["attachment"] == "held"
    assert geometry.entities["package"]["attachment_probe_unconfirmed"] is True
    assert runtime.last_command == "close" and not geometry.entities["package"].get("release_pending")
    runtime.observe_transition(view, view, placing, "open", 13)
    assert geometry.entities["package"]["release_pending"] is True
    assert runtime.monitor.active["target_id"] == "package"
    assert runtime.evidence("release", view, 13)["status"] == "unknown"


def test_probe_authorization_requires_settling_and_cannot_renew_its_deadline():
    runtime, geometry, view, context = setup()
    lift = {**context, "held_object_id": "package"}
    assert "attachment_probe_authorized" not in runtime.probe_context(lift, 0, 20)
    settle(runtime, view, context)
    first = runtime.probe_context(lift, 12, 20)
    assert first["attachment_probe_target_id"] == "package"
    assert first["attachment_probe_start_step"] == 12 and first["attachment_probe_max_steps"] == 20
    later = runtime.probe_context(lift, 25, 120)
    assert later["attachment_probe_start_step"] == 12 and later["attachment_probe_max_steps"] == 20
    assert "attachment_probe_authorized" not in runtime.probe_context(lift, 32, 120)
    assert "attachment_probe_authorized" not in runtime.probe_context({**lift, "held_object_id": "other"}, 13, 20)
    assert "attachment_probe_authorized" not in lift


@pytest.mark.parametrize("intent, held_id", [("approach", None), ("manipulate", "package")])
def test_empty_hand_close_for_pushing_never_creates_attachment_from_target_or_held_claim(intent, held_id):
    runtime, geometry, view, context = setup()
    posture = {**context, "intent": intent, "held_object_id": held_id}
    settle(runtime, view, posture, last=16)
    push = {**posture, "intent": "manipulate"}
    settle(runtime, view, push, first=17, last=21)
    assert runtime.last_command == "close"
    assert geometry.attachment_probes == [] and geometry.release_probes == []
    assert all(item.get("attachment") != "held" for item in geometry.entities.values())
    assert runtime.monitor.active is None
    assert runtime.counts["references_prepared"] == runtime.counts["references_started"] == 0
    runtime.before_action(view, push, "open", 21)
    runtime.observe_transition(view, view, push, "open", 22)
    assert runtime.last_command == "open" and geometry.release_probes == []
    assert runtime.counts["release_supported"] == runtime.counts["hold_supported"] == 0


def test_real_grasp_after_empty_closed_posture_starts_new_reference_only_on_ack():
    runtime, geometry, view, grasp = setup()
    settle(runtime, view, {**grasp, "intent": "approach"}, last=20)
    assert runtime.last_command == "close" and runtime.monitor.active is None
    prepared = runtime.before_action(view, grasp, "close", 20)
    assert prepared["prepared"] and prepared["reference_step"] == 20
    assert runtime.counts["references_started"] == 0 and geometry.attachment_probes == []
    runtime.observe_transition(view, view, grasp, "close", 21)
    assert runtime.monitor.active["reference_step"] == 20
    assert runtime.snapshot()["current"]["close_reference_step"] == 20
    assert runtime.operations[runtime.bound_object_id]["command_steps"] == 1
    settle(runtime, view, grasp, first=22, last=31)
    assert geometry.attachment_probes == []
    settle(runtime, view, grasp, first=32, last=32)
    assert geometry.attachment_probes == [("package", 32)]
    assert runtime.counts["references_started"] == 1
    assert runtime.evidence("hold", view, 32)["status"] == "unknown"


def test_active_grasp_payload_survives_posture_intent_and_unrelated_model_held_claim():
    runtime, geometry, view, grasp = setup()
    settle(runtime, view, grasp)
    operation = runtime.operation_id
    posture = {"intent": "manipulate", "target_id": "other", "held_object_id": "other"}
    # A new shape estimate cannot erase the acknowledged grasp's possible
    # payload merely because the next operation calls its movement manipulation.
    geometry.entities["package"].update(attachment="none", geometry_revision=2)
    settle(runtime, view, posture, first=13, last=13)
    assert runtime.bound_object_id == "package" and runtime.operation_id == operation
    assert geometry.entities["package"]["attachment"] == "held"
    assert geometry.entities["other"]["attachment"] == "none"
    assert runtime.counts["references_started"] == 1
    assert runtime.monitor.active["target_id"] == "package"
    runtime.before_action(view, posture, "open", 13)
    runtime.observe_transition(view, view, posture, "open", 14)
    assert geometry.entities["package"]["release_pending"] is True
    release_reference = runtime.monitor.active["reference_step"]
    runtime.before_action(view, posture, "close", 14)
    runtime.observe_transition(view, view, posture, "close", 15)
    assert runtime.bound_object_id == "package" and runtime.operation_id == operation
    assert geometry.entities["package"]["release_pending"] is True
    assert runtime.monitor.active["kind"] == "release"
    assert runtime.monitor.active["reference_step"] == release_reference
    assert runtime.counts["references_started"] == 2
    assert runtime.counts["release_supported"] == 0


def test_prepared_grasp_is_discarded_when_actual_same_close_is_only_tool_posture():
    runtime, geometry, view, grasp = setup()
    runtime.before_action(view, grasp, "close", 0)
    runtime.observe_transition(view, view, {**grasp, "intent": "manipulate"}, "close", 1)
    assert runtime.last_command == "close" and runtime.monitor.active is None
    assert runtime.counts["references_started"] == 0
    assert runtime.snapshot()["current"]["close_reference_step"] is None
    assert geometry.attachment_probes == []
