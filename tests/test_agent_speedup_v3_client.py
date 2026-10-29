"""V3 schema and offline recovery contracts; no model, simulator or network calls."""
import copy
import json
from types import SimpleNamespace

import pytest

from gpt_field.agent_speedup_v2_user_check_agent_client import agent_schema as v7_agent_schema
from gpt_field.agent_speedup_v2_user_check_client import PlanLock
from gpt_field.agent_speedup_v3_client import (
    AgentSpeedUpV3Client, HISTORY_PROTOCOL, motion_context_schema,
    validate_motion_context, validate_v3_agent_decision, v3_agent_schema,
    v3_fragment_capabilities, current_decision_core,
)
from gpt_field.astra import SchemaError


TASK = "Hold the object."
PLAN = {"scene": "One object.", "parts": [
    {"id": "A", "text": "object: body", "move": "free", "w": 1, "x": 1}],
    "hold_true": [], "steps": [{"do": "hold", "obj": "A", "grip": [], "ref": None,
                                "sense": "any", "keep": [], "until": []}]}
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jXioAAAAASUVORK5CYII="


def context(**changes):
    value = {"intent": "observe", "target_id": None, "held_object_id": None,
             "allowed_contact_ids": [], "scene_inventory": {
                 "reviewed": False, "support_ids": [], "nearby_obstacle_ids": [], "unresolved": True}}
    value.update(changes)
    return value


def decision(**changes):
    value = {"belief": "Object location remains uncertain.", "plan": "Inspect before approach.",
             "why": "Get current images.", "type": "observe", "motion": None,
             "macro": None, "execution": None, "continuation": None, "program": None,
             "scene_update": {"entities": [], "invalidate_ids": [], "remove_ids": []},
             "motion_context": context()}
    value.update(changes)
    return value


def client():
    def forbidden_transport(*args, **kwargs):
        raise AssertionError("payload tests must never call a transport")
    forbidden_transport.offline_mock = True
    lock = PlanLock(PLAN, task_prompt=TASK)
    return AgentSpeedUpV3Client(task_prompt=TASK, approved_plan=PLAN, lock=lock,
                              transport=forbidden_transport)


def test_compact_output_instructions_cover_plan_and_executive_without_schema_changes():
    from gpt_field.agent_speedup_v3_client import AgentSpeedUpV3PlanClient, COMPACT_OUTPUT_INSTRUCTIONS, AGENT_INSTRUCTIONS
    from gpt_field.agent_speedup_v2_user_check_client import UserCheckV2Client
    plan = AgentSpeedUpV3PlanClient(task_prompt=TASK)
    baseline = UserCheckV2Client(task_prompt=TASK)
    assert plan._schemas == baseline._schemas
    assert all(value.endswith(COMPACT_OUTPUT_INSTRUCTIONS) for value in plan._instruction_text.values())
    assert AGENT_INSTRUCTIONS.endswith(COMPACT_OUTPUT_INSTRUCTIONS)


def test_additions_preserve_base_schema_and_single_action_validation():
    original = v7_agent_schema()
    extended = v3_agent_schema()
    assert set(extended["required"]) == set(original["required"]) | {"scene_update", "motion_context", "continuation", "program"}
    for key in original["properties"]:
        if key != "type":
            assert extended["properties"][key] == original["properties"][key]
    assert set(extended["properties"]["type"]["enum"]) == set(original["properties"]["type"]["enum"]) | {"program", "geometry_patch", "continue_intent", "acquire_evidence", "escalate"}
    raw = decision()
    accepted = validate_v3_agent_decision(raw)
    assert accepted == raw and accepted is not raw
    accepted["scene_update"]["invalidate_ids"].append("A")
    assert raw["scene_update"]["invalidate_ids"] == []
    assert v7_agent_schema() == original
    with pytest.raises(SchemaError):
        validate_v3_agent_decision(decision(macro={"step_index": 0, "budget_steps": 10,
                                                 "continuation": False}))


def test_program_is_exclusive_and_semantic_with_full_stage_context():
    program = {"stages": [{"motion_context": context(),
        "goal": {"kind": "hold", "reference_id": None, "quaternion_wxyz": None},
        "gripper": "keep", "profile": "contact", "verification": "arrival", "max_steps": 30}]}
    raw = decision(type="program", program=program)
    assert validate_v3_agent_decision(raw) == raw
    for invalid in (decision(program=program), decision(type="program"),
                    {**raw, "continuation": {"intent_id": "old", "resume": True}},
                    {**raw, "macro": {"step_index": 0, "budget_steps": 10, "continuation": False}}):
        with pytest.raises(SchemaError):
            validate_v3_agent_decision(invalid)
    program["stages"][0]["goal"]["offset_m"] = .03
    with pytest.raises(SchemaError):
        validate_v3_agent_decision(raw)


@pytest.mark.parametrize("alter", [
    lambda value: value.pop("scene_update"),
    lambda value: value.update(simulator_object_pose=[0., 0., 0.]),
    lambda value: value["scene_update"].update(source="measured"),
    lambda value: value.update(scene_update=None),
])
def test_rejects_missing_extensions_and_unapproved_truth_fields(alter):
    raw = decision()
    alter(raw)
    with pytest.raises(SchemaError):
        validate_v3_agent_decision(raw)


def test_motion_context_is_strict_bounded_and_not_an_attachment_assertion():
    raw = context(intent="place", target_id="support", held_object_id="A",
                  allowed_contact_ids=["support"])
    assert validate_motion_context(raw) == raw
    assert set(motion_context_schema()["required"]) == set(raw)
    for invalid in (context(intent="teleport"), context(target_id=" "),
                    context(allowed_contact_ids=["A", "A"]),
                    context(allowed_contact_ids=["A"] * 13),
                    context(allowed_contact_ids="A"),
                    {**raw, "grasp_verified": True}):
        with pytest.raises(ValueError):
            validate_motion_context(invalid)


@pytest.mark.parametrize("changes", [
    {"reviewed": 1}, {"unresolved": "false"}, {"support_ids": [" "]},
    {"nearby_obstacle_ids": ["milk", "milk"]}, {"nearby_obstacle_ids": "milk"},
    {"support_ids": [f"support_{i}" for i in range(13)]}, {"physically_verified": True},
])
def test_inventory_requires_explicit_booleans_and_bounded_entity_references(changes):
    raw = context()
    raw["scene_inventory"].update(changes)
    with pytest.raises(ValueError):
        validate_motion_context(raw)


def test_inventory_is_required_and_is_copied_without_certifying_completeness():
    raw = context()
    raw.pop("scene_inventory")
    with pytest.raises(ValueError):
        validate_motion_context(raw)
    raw = context(scene_inventory={"reviewed": True, "support_ids": ["table"],
                                   "nearby_obstacle_ids": ["milk"], "unresolved": False})
    accepted = validate_motion_context(raw)
    assert accepted == raw
    accepted["scene_inventory"]["support_ids"].append("shelf")
    assert raw["scene_inventory"]["support_ids"] == ["table"]
    # Existence, roles and relevant geometry are checked by GeometryMemory,
    # not certified by accepting a model's inventory declaration.


def test_payload_advertises_actual_v3_contract_without_mutating_request():
    executive = client()
    request = {"fields": {"context": {"capabilities": {"execute": {"version": "old"}},
                "native_evidence": {"shared_geometry": {"version": "estimate_snapshot", "entities": []}}}},
               "images": [("External", PNG), ("Wrist", PNG)]}
    before = copy.deepcopy(request)
    payload = executive.build_payload(request, mode="agent")
    assert request == before
    assert payload["model"] == "gpt-6-astra"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["store"] is False
    assert payload["text"]["format"]["name"] == "agent_speedup_v3"
    assert payload["text"]["format"]["schema"] == v3_agent_schema()
    fields = json.loads(payload["input"][-1]["content"][0]["text"])
    assert fields["context"]["capabilities"]["execute"] == v3_fragment_capabilities()
    assert fields["context"]["native_evidence"]["shared_geometry"]["version"] == "estimate_snapshot"
    assert "scene_update and motion_context in this SAME response" in payload["instructions"]
    assert "intermediate waypoint or early yield" in payload["instructions"]
    assert "model_visual_estimate" in payload["instructions"]
    assert executive.validate_response(decision()) == decision()
    assert executive.history_stats()["protocol"] == HISTORY_PROTOCOL


def test_six_paired_images_preserve_all_text_and_opaque_reasoning_history():
    executive = client()
    history = []
    for index in range(8):
        history.extend([
            {"role": "user", "content": [
                {"type": "input_text", "text": f"old observation {index}"},
                {"type": "input_image", "image_url": PNG},
                {"type": "input_image", "image_url": PNG}]},
            {"type": "reasoning", "encrypted_content": f"opaque-{index}"},
            {"role": "assistant", "content": [{"type": "output_text", "text": f"old decision {index}"}]},
        ])
    executive._agent_history = copy.deepcopy(history)
    payload = executive.build_payload({"fields": {}, "images": [("External", PNG), ("Wrist", PNG)]},
                                      mode="agent")
    assert len(payload["input"]) == len(history) + 1
    image_messages = [item for item in payload["input"] if item.get("role") == "user"
                      and any(part.get("type") == "input_image" for part in item["content"])]
    assert len(image_messages) == 6
    assert all(sum(part.get("type") == "input_image" for part in item["content"]) == 2
               for item in image_messages)
    text = json.dumps(payload["input"])
    for index in range(8):
        assert f"old observation {index}" in text
        assert f"old decision {index}" in text
        assert f"opaque-{index}" in text
    assert executive._agent_history == history


@pytest.mark.parametrize("route", ["geometry_patch", "continue_intent"])
def test_geometry_continuation_has_no_replacement_action(route):
    raw = decision(type=route, continuation={"intent_id": "paused-1", "resume": True})
    assert validate_v3_agent_decision(raw) == raw
    for changes in ({"continuation": None}, {"continuation": {"intent_id": "", "resume": True}},
                    {"continuation": {"intent_id": "paused-1", "resume": 1}},
                    {"execution": {"segments": []}}):
        with pytest.raises(SchemaError):
            validate_v3_agent_decision({**raw, **changes})
    if route == "continue_intent":
        with pytest.raises(SchemaError):
            validate_v3_agent_decision({**raw, "continuation": {"intent_id": "paused-1", "resume": False}})


def test_escalation_is_an_explicit_action_free_response():
    assert validate_v3_agent_decision(decision(type="escalate"))["type"] == "escalate"
    with pytest.raises(SchemaError):
        validate_v3_agent_decision(decision(type="escalate", continuation={"intent_id": "x", "resume": True}))


def test_current_requirements_survive_generic_compaction_without_history_changes(monkeypatch):
    from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client
    executive = client()
    requirements = [{"entity_id": f"part-{i}", "reason": "occluded_support",
                     "required_evidence": {"boundary": "current visible support", "related_ids": [f"wall-{i}"]}}
                    for i in range(24)]
    core = {"pending_intent": {"intent_id": "saved", "current_target": {"xyz_m": [1, 2, 3]}},
            "missing_evidence_requests": requirements, "requirements_truncated": False}
    context = {"native_evidence": {"current_decision": core}, "last_feedback": {"huge": "x" * 50000}}
    def inherited_request(self, view, context):
        assert "current_decision" not in context["native_evidence"]
        # Model the inherited generic compactor dropping nested causal details.
        return self.build_payload({"fields": {"context": {"native_evidence": {"detail_in_audit": True}}},
                                   "images": [("External", PNG), ("Wrist", PNG)]}, mode="agent")
    monkeypatch.setattr(AgentFirstV2Client, "request_agent", inherited_request)
    payload = executive.request_agent(object(), context)
    fields = json.loads(payload["input"][-1]["content"][0]["text"])
    assert fields["current_decision"] == core
    assert executive._current_decision_for_request is None
    assert context["native_evidence"]["current_decision"] == core
    assert executive._agent_history == []


def test_core_keeps_complete_current_requirements_without_resurrecting_old_stages():
    missing = [{"id": "support", "reason": "missing", "requirements": ["shape", "source", "view"]}]
    core = current_decision_core({"step": 12,
        "native_evidence": {"shared_geometry": {"last_motion": {"reason": "geometry_inventory_required",
            "missing_evidence_requests": missing}}},
        "last_feedback": {"missing_evidence_requests": ["obsolete_target"]}})
    assert core["missing_evidence_requests"] == missing
    assert core["requirements_truncated"] is False
    assert "obsolete_target" not in json.dumps(core)


def bare_policy():
    from gpt_field.agent_speedup_v3 import AgentSpeedUpV3Policy
    policy = AgentSpeedUpV3Policy.__new__(AgentSpeedUpV3Policy)
    shape = {"id": "A", "role": "object", "parent_id": None, "position_m": [0, 0, 0],
             "quaternion_wxyz": [1, 0, 0, 0], "boxes": [{"half_extents_m": [.02, .02, .02]}],
             "uncertainty_m": .01, "coverage": "partial", "attachment": "none", "anchors": None}
    policy.shared_geometry = SimpleNamespace(entities={"A": shape}, revision=1,
        last_motion={"status": "blocked", "reason": "geometry_inventory_required",
                     "evidence_ids": ["A"], "missing_evidence_requests": ["current_target_envelope:A"]})
    policy.operation_runtime = SimpleNamespace(snapshot=lambda: {"task_success": "not_available_to_policy"})
    policy.step_index = 4
    policy._pending_intent = None
    policy._repeat_guard = None
    policy._evidence_pause = None
    policy._need_agent = False
    policy._continuation_events = []
    policy._continuation_rejections = 0
    policy._intent_action = None
    policy._last_motion_tracking = {}
    policy.motion_context = context()
    policy._record_feedback = lambda value: setattr(policy, "feedback", value)
    return policy, SimpleNamespace(observation=SimpleNamespace(geometry=None))


def test_identical_copy_and_unrelated_edits_cannot_retry_same_blocked_action():
    from gpt_field.agent_speedup_v3 import _action_identity, _blocker_identity, EvidenceAcquisitionFailed
    policy, view = bare_policy()
    ctx = context(intent="grasp", target_id="A")
    action = decision(type="motion", motion={"xyz_m": [0, 0, 0]}, motion_context=ctx)
    key = _action_identity(action, ctx)
    policy._repeat_guard = {"action": key, "blocker": _blocker_identity(policy.shared_geometry.last_motion),
                           "evidence": policy._relevant_evidence(view, ctx), "rejections": 0}
    copied = {**policy.shared_geometry.entities["A"], "observation_step": 999, "updated_step": 999}
    unrelated = {**copied, "id": "unrelated", "position_m": [8, 8, 8]}
    patch = {"entities": [copied, unrelated], "invalidate_ids": [], "remove_ids": []}
    assert policy._reject_repeated(key, view, ctx, patch) is True
    assert policy.feedback["required_route"] == "acquire_evidence_or_change_intent"
    assert policy._need_agent is True
    with pytest.raises(EvidenceAcquisitionFailed, match="current_target_envelope:A"):
        policy._reject_repeated(key, view, ctx, patch)
    assert policy._evidence_pause["automatic_retry"] is False
    # A changed relevant model envelope is only new input; continuation still
    # requires the fragment's postpatch geometry guard, never a proof of safety.
    changed = {**copied, "position_m": [.03, 0, 0]}
    assert policy._reject_repeated(key, view, ctx, {"entities": [changed]}) is False


def test_patch_continuation_cannot_retarget_or_change_contact_policy():
    policy, view = bare_policy()
    ctx = context(intent="grasp", target_id="A", allowed_contact_ids=["A"])
    policy._pending_intent = {"intent_id": "frozen", "kind": "motion", "saved": {}, "motion_context": ctx}
    def forbidden_update(*args):
        raise AssertionError("invalid continuation must not change geometry")
    policy.shared_geometry.update_from_model = forbidden_update
    for changed in ({"target_id": "other"}, {"held_object_id": "other"}, {"allowed_contact_ids": ["A", "wall"]}, {"intent": "place"}):
        answer = decision(type="continue_intent", continuation={"intent_id": "frozen", "resume": True},
                          motion_context={**ctx, **changed})
        policy._continuation_rejections = 0
        policy._continue_intent(answer, view)
        assert policy.feedback["reason"] == "continuation_cannot_change_intent_target_held_or_contacts"
        assert policy._pending_intent["intent_id"] == "frozen"


def test_patch_then_continue_uses_saved_target_and_fresh_guard_factory(monkeypatch):
    import gpt_field.agent_speedup_v3 as module
    policy, view = bare_policy()
    ctx = context(intent="transit", target_id="A")
    saved = {"current_target": {"xyz_m": [.15, .2, .3], "quaternion_wxyz": [1, 0, 0, 0]},
             "remaining_watchdog_steps": 12, "gripper_command": "close"}
    policy._pending_intent = {"intent_id": "frozen", "kind": "motion", "saved": saved, "motion_context": ctx}
    policy._intent_action = {"type": "motion", "motion": {"xyz_m": [.15, .2, .3]}, "execution": None, "macro": None}
    policy.ctx = SimpleNamespace(memory={"user_check_gripper_command": "close"})
    calls = []
    def patch(update, current, step):
        calls.append("patch")
    policy.shared_geometry.update_from_model = patch
    policy.shared_geometry.preview_scene_update = lambda update, current, step: policy.shared_geometry
    class GuardedPrimitive:
        done = False
        @classmethod
        def from_continuation(cls, current, state, geometry, motion_context, **options):
            calls.append("current_guard")
            assert calls == ["patch", "current_guard"]
            assert state == saved and state is not saved
            assert motion_context == ctx
            assert options["keep_gripper"] == "close"
            return cls()
    monkeypatch.setattr(module, "AdaptiveLocalMotionPrimitive", GuardedPrimitive)
    answer = decision(type="continue_intent", continuation={"intent_id": "frozen", "resume": True}, motion_context=ctx)
    policy._continue_intent(answer, view)
    assert policy._pending_intent is None and policy._primitive.done is False
    assert policy._active_intent_id == "frozen"
    assert policy._continuation_events[-1]["current_guard_rechecked"] is True


def test_repeated_invalid_continuation_stops_without_more_model_roundtrips():
    from gpt_field.agent_speedup_v3 import EvidenceAcquisitionFailed
    policy, view = bare_policy()
    answer = decision(type="continue_intent", continuation={"intent_id": "stale", "resume": True})
    policy._continue_intent(answer, view)
    assert policy.feedback["reason"] == "fresh_matching_pending_intent_required"
    with pytest.raises(EvidenceAcquisitionFailed, match="fresh_matching_pending_intent_required"):
        policy._continue_intent(answer, view)


def test_policy_core_preserves_causal_geometry_and_tracking_diagnostics():
    policy, view = bare_policy()
    policy.shared_geometry.last_motion.update(
        stale_evidence_ids=["A"], motion_support_revoked=True, tracking_status="mismatch",
        limiting_pair={"body_id": "hand", "obstacle_id": "A", "clearance_estimate_m": -.001},
        selected_command_candidate=None, command_candidates=[{"accepted": False, "reason": "overlap"}])
    policy._last_motion_tracking = {"status": "mismatch", "translation_response_ratio": -.1}
    core = policy._decision_core()
    assert core["blocker"]["stale_evidence_ids"] == ["A"]
    assert core["blocker"]["motion_support_revoked"] is True
    assert core["blocker"]["limiting_pair"]["obstacle_id"] == "A"
    assert core["blocker"]["command_candidates"][0]["reason"] == "overlap"
    assert core["current_tracking"] == policy._last_motion_tracking
    assert core["missing_evidence_requests"] == ["current_target_envelope:A"]


class RecoveryClock:
    def __init__(self):
        self.now, self.waits = 100., []

    def __call__(self):
        return self.now

    def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay


class AvailabilityTransport:
    offline_mock = True

    def __init__(self, answer, *, failures=1, code="subscription_sharing_user_unavailable",
                 status=503, retry_after=None):
        self.answer, self.failures = answer, failures
        self.code, self.status, self.retry_after = code, status, retry_after
        self.calls = []

    def __call__(self, endpoint, payload, headers, timeout_s):
        from gpt_field.chatgpt_plan import _protocol_error
        self.calls.append(copy.deepcopy(payload))
        if len(self.calls) <= self.failures:
            error = _protocol_error("Temporary subscription lookup unavailable", http_status=self.status,
                                    error_code=self.code)
            error.retry_after_s = self.retry_after
            raise error
        return {"status": "completed", "model": "gpt-6-astra", "reasoning": {"effort": "high"},
                "output": [{"type": "reasoning", "encrypted_content": "accepted-once"},
                           {"type": "message", "role": "assistant", "content": [
                               {"type": "output_text", "text": json.dumps(self.answer)}]}],
                "usage": {"input_tokens": 10, "output_tokens": 5,
                          "output_tokens_details": {"reasoning_tokens": 1}}}


def availability_client(monkeypatch, transport, *, mode="agent", log_dir=None):
    import gpt_field.agent_speedup_v2_user_check_agent_client as inherited_agent
    from gpt_field.agent_speedup_v3_client import AgentSpeedUpV3PlanClient
    clock = RecoveryClock()
    monkeypatch.setattr(inherited_agent.time, "sleep", clock.sleep)
    monkeypatch.setattr(inherited_agent.random, "uniform", lambda low, high: .25)
    options = dict(task_prompt=TASK, transport=transport, clock=clock, log_dir=log_dir)
    if mode == "plan":
        return AgentSpeedUpV3PlanClient(**options), clock
    return AgentSpeedUpV3Client(approved_plan=PLAN, lock=PlanLock(PLAN, task_prompt=TASK), **options), clock


def recovery_request():
    return {"fields": {}, "images": [("External", PNG), ("Wrist", PNG)]}


@pytest.mark.parametrize("code", ["subscription_sharing_user_unavailable", "subscription_sharing_usage_unavailable"])
def test_availability_retry_reuses_identical_payload_and_commits_history_once(monkeypatch, tmp_path, code):
    from gpt_field.agent_speedup_v3_client import subscription_recovery_settings
    transport = AvailabilityTransport(decision(), code=code, retry_after=8.)
    executive, clock = availability_client(monkeypatch, transport, log_dir=tmp_path)
    assert executive.predict(recovery_request(), mode="agent") == decision()
    assert len(transport.calls) == 2 and transport.calls[0] == transport.calls[1]
    assert clock.waits == [8.]
    stats = executive.history_stats()
    assert stats["agent_decisions"] == 1 and stats["accepted_by_kind"]["agent"] == 1
    assert stats["calls_by_kind"]["agent"] == 2
    assert stats["transient_retry_requests"] == stats["subscription_availability_retry_requests"] == 1
    assert stats["transient_retry_wait_s"] == 8.
    assert stats["transient_retry_codes"] == {code: 1}
    assert executive._observation_count == 1
    assert sum(item.get("role") == "user" for item in executive._agent_history) == 1
    assert sum(item.get("type") == "reasoning" for item in executive._agent_history) == 1
    assert sum(item.get("role") == "assistant" for item in executive._agent_history) == 1
    logs = [json.loads(f.read_text()) for f in sorted(tmp_path.glob("*.request.json"))]
    assert [record["attempt"] for record in logs] == [1, 2]
    assert logs[0]["request_key"] == logs[1]["request_key"]
    assert logs[0]["payload"] == logs[1]["payload"]
    retry, = [json.loads(f.read_text()) for f in tmp_path.glob("*.transient_retry.json")]
    assert retry["subscription_recovery_settings"] == subscription_recovery_settings()
    assert retry["same_observation"] is True and retry["history_committed"] is False
    assert retry["actions_executed_from_failed_response"] == 0


@pytest.mark.parametrize("mode", ["agent", "plan"])
def test_availability_stops_after_three_attempts_without_history(monkeypatch, mode):
    from gpt_field.astra import APIError
    transport = AvailabilityTransport(decision() if mode == "agent" else PLAN, failures=99)
    executive, clock = availability_client(monkeypatch, transport, mode=mode)
    with pytest.raises(APIError):
        executive.predict(recovery_request(), mode=mode)
    assert len(transport.calls) == 3 and clock.waits == [2.25, 4.25]
    assert all(payload == transport.calls[0] for payload in transport.calls)
    assert executive._observation_count == 0
    assert not getattr(executive, "_agent_history", [])
    stats = executive.history_stats()
    assert stats["subscription_availability_retry_requests"] == 2
    assert stats["accepted_by_kind"][mode] == 0


def test_prepare_retry_preserves_plan_schema_and_compact_formatting(monkeypatch):
    from gpt_field.agent_speedup_v2_user_check_client import UserCheckV2Client
    from gpt_field.agent_speedup_v3_client import COMPACT_OUTPUT_INSTRUCTIONS
    transport = AvailabilityTransport(PLAN)
    executive, clock = availability_client(monkeypatch, transport, mode="plan")
    original = UserCheckV2Client(task_prompt=TASK, transport=transport)
    request = recovery_request()
    expected = original.build_payload(request, mode="plan")
    expected["instructions"] += "\n\n" + COMPACT_OUTPUT_INSTRUCTIONS
    assert executive.build_payload(request, mode="plan") == expected
    assert executive.predict(request, mode="plan") == PLAN
    assert len(transport.calls) == 2 and transport.calls[0] == transport.calls[1]
    assert clock.waits == [2.25]
    assert executive.history_stats()["subscription_availability_retry_requests"] == 1


@pytest.mark.parametrize("status,code", [
    (401, "subscription_sharing_user_unavailable"), (403, "subscription_sharing_user_unavailable"),
    (429, "subscription_sharing_usage_unavailable"), (500, "subscription_sharing_user_unavailable"),
    (503, "server_error"), (503, "unknown_error"), (503, "subscription_sharing_access_denied"),
])
def test_availability_does_not_expand_recovery_to_other_errors(monkeypatch, status, code):
    from gpt_field.astra import APIError
    transport = AvailabilityTransport(decision(), code=code, status=status)
    executive, clock = availability_client(monkeypatch, transport)
    with pytest.raises(APIError):
        executive.predict(recovery_request(), mode="agent")
    assert len(transport.calls) == 1 and not clock.waits
    assert executive.history_stats()["subscription_availability_retry_requests"] == 0


def test_availability_rejects_excessive_retry_after_and_committed_responses(monkeypatch):
    from gpt_field.astra import APIError
    from gpt_field.chatgpt_plan import _protocol_error
    transport = AvailabilityTransport(decision(), retry_after=61.)
    executive, clock = availability_client(monkeypatch, transport)
    with pytest.raises(APIError):
        executive.predict(recovery_request(), mode="agent")
    assert len(transport.calls) == 1 and not clock.waits
    error = _protocol_error("Temporary lookup", http_status=503,
                            error_code="subscription_sharing_user_unavailable")
    executive.last_response = {}
    assert executive._retry_cause(error, 0, {"transient": 0, "overload": 0}) is None
    executive.last_response = None
    executive._observation_count = 1
    assert executive._retry_cause(error, 0, {"transient": 0, "overload": 0}) is None


def bounded_stream_response(text, *, deltas=()):
    import io
    from email.message import Message
    item = {"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}]}
    events = [{"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
               "item_id": "msg_fixture", "delta": chunk} for chunk in deltas]
    events += [{"type": "response.output_item.done", "output_index": 0, "item": item},
               {"type": "response.completed", "response": {
                   "id": "resp_fixture", "status": "completed", "output": [], "model": "gpt-6-astra",
                   "reasoning": {"effort": "high"},
                   "usage": {"input_tokens": 10, "output_tokens": 5}}}]
    response = io.BytesIO("".join("data: " + json.dumps(event) + "\n\n" for event in events).encode())
    response.headers = Message()
    response.headers["Content-Type"] = "text/event-stream"
    return response


def test_default_v3_clients_install_bounded_transport_without_changing_v2():
    from gpt_field.agent_speedup_v3_client import AgentSpeedUpV3PlanClient
    from gpt_field.agent_speedup_v3_transport import V3BoundedSubscriptionTransport
    from gpt_field.agent_speedup_v2_user_check_client import UserCheckV2Client
    assert isinstance(AgentSpeedUpV3PlanClient(task_prompt=TASK).transport, V3BoundedSubscriptionTransport)
    assert isinstance(AgentSpeedUpV3Client(task_prompt=TASK, approved_plan=PLAN,
                       lock=PlanLock(PLAN, task_prompt=TASK)).transport, V3BoundedSubscriptionTransport)
    assert not isinstance(UserCheckV2Client(task_prompt=TASK).transport, V3BoundedSubscriptionTransport)


def test_bounded_transport_preserves_strings_escapes_and_counts_repeated_done_once(monkeypatch):
    import gpt_field.agent_speedup_v3_transport as module
    from gpt_field.astra import ENDPOINT
    text = json.dumps({"note": 'quoted \\" then ' + ' ' * 4096, "ok": True})
    response = bounded_stream_response(text, deltas=[text[i:i+7] for i in range(0, len(text), 7)])
    monkeypatch.setattr(module.protocol, "_open_public", lambda request, timeout: response)
    transport = module.V3BoundedSubscriptionTransport()
    result = transport(ENDPOINT, {"store": False, "stream": True}, {}, 10.)
    assert result["output"][0]["content"][0]["text"] == text
    assert transport.last_diagnostics["visible_characters"] == len(text)
    assert transport.last_diagnostics["longest_json_whitespace_run"] < 10
    assert result["_subscription_transport"]["sse_completion_verified"] is True


def test_pathological_json_whitespace_stops_without_history_or_retry(monkeypatch, tmp_path):
    import gpt_field.agent_speedup_v3_transport as module
    text = '{"scene": "ok",' + '\n ' * (module.MAX_JSON_WHITESPACE_RUN//2 + 1) + '"parts": []}'
    opened = []
    def open_response(request, timeout):
        opened.append(True)
        return bounded_stream_response(text, deltas=[text[:20], text[20:]])
    monkeypatch.setattr(module.protocol, "_open_public", open_response)
    transport = module.V3BoundedSubscriptionTransport()
    transport.offline_mock = True
    executive = AgentSpeedUpV3Client(task_prompt=TASK, approved_plan=PLAN, lock=PlanLock(PLAN, task_prompt=TASK),
                                    transport=transport, log_dir=tmp_path)
    with pytest.raises(module.SubscriptionOutputGuard) as caught:
        executive.predict(recovery_request(), mode="agent")
    assert caught.value.error_code == "subscription_output_whitespace_limit"
    assert len(opened) == 1 and executive._agent_history == [] and executive._observation_count == 0
    record, = [json.loads(path.read_text()) for path in tmp_path.glob("*.result.json")]
    assert record["v3_transport_diagnostics"]["longest_json_whitespace_run"] > module.MAX_JSON_WHITESPACE_RUN
    assert record["v3_transport_diagnostics"]["partial_output_accepted"] is False
    assert executive.history_stats()["transient_retry_requests"] == 0


def test_visible_output_limit_applies_inside_strings_and_to_full_snapshots(monkeypatch):
    import gpt_field.agent_speedup_v3_transport as module
    from gpt_field.astra import ENDPOINT
    monkeypatch.setattr(module, "MAX_VISIBLE_CHARACTERS", 64)
    response = bounded_stream_response(json.dumps({"value": "x" * 100}))
    monkeypatch.setattr(module.protocol, "_open_public", lambda request, timeout: response)
    transport = module.V3BoundedSubscriptionTransport()
    with pytest.raises(module.SubscriptionOutputGuard) as caught:
        transport(ENDPOINT, {"store": False, "stream": True}, {}, 10.)
    assert caught.value.error_code == "subscription_visible_output_limit"


def test_active_stream_cannot_extend_total_monotonic_deadline(monkeypatch):
    import io
    import signal
    from email.message import Message
    import gpt_field.agent_speedup_v3_transport as module
    from gpt_field.astra import ENDPOINT
    clock = RecoveryClock()
    class ActiveResponse(io.BytesIO):
        def read1(self, size):
            clock.now += .4
            return super().read1(10)
    response = ActiveResponse(b": heartbeat\n\n" * 100)
    response.headers = Message()
    response.headers["Content-Type"] = "text/event-stream"
    monkeypatch.setattr(module.protocol, "_open_public", lambda request, timeout: response)
    old_handler = signal.getsignal(signal.SIGALRM)
    transport = module.V3BoundedSubscriptionTransport(clock=clock)
    with pytest.raises(module.SubscriptionRequestDeadline) as caught:
        transport(ENDPOINT, {"store": False, "stream": True}, {}, 1.)
    assert caught.value.error_code == "subscription_request_deadline"
    assert transport.last_diagnostics["chunks"] > 0 and transport.last_diagnostics["deadline_exceeded"]
    assert signal.getsignal(signal.SIGALRM) == old_handler
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_alarm_interrupt_has_own_error_and_restores_handler(monkeypatch):
    import signal
    import gpt_field.agent_speedup_v3_transport as module
    from gpt_field.astra import ENDPOINT
    def blocked_open(request, timeout):
        signal.raise_signal(signal.SIGALRM)
        raise AssertionError("alarm must interrupt blocked transport before completion")
    monkeypatch.setattr(module.protocol, "_open_public", blocked_open)
    previous = signal.getsignal(signal.SIGALRM)
    transport = module.V3BoundedSubscriptionTransport()
    with pytest.raises(module.SubscriptionRequestDeadline):
        transport(ENDPOINT, {"store": False, "stream": True}, {}, 10.)
    assert transport.last_diagnostics["error_type"] == "SubscriptionRequestDeadline"
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_existing_alarm_prevents_network_without_replacing_timer(monkeypatch):
    import gpt_field.agent_speedup_v3_transport as module
    from gpt_field.astra import ENDPOINT
    monkeypatch.setattr(module.signal, "getitimer", lambda timer: (5., 0.))
    monkeypatch.setattr(module.protocol, "_open_public", lambda *args: pytest.fail("network must not start"))
    monkeypatch.setattr(module.signal, "setitimer", lambda *args: pytest.fail("existing timer must not change"))
    with pytest.raises(module.SubscriptionDeadlineUnavailable):
        module.V3BoundedSubscriptionTransport()(ENDPOINT, {"store": False, "stream": True}, {}, 10.)


def test_unterminated_oversized_event_is_bounded_before_json_parse(monkeypatch):
    import io
    from email.message import Message
    import gpt_field.agent_speedup_v3_transport as module
    from gpt_field.astra import ENDPOINT
    monkeypatch.setattr(module, "MAX_EVENT_BYTES", 64)
    response = io.BytesIO(b"data: " + b"x" * 100)
    response.headers = Message()
    response.headers["Content-Type"] = "text/event-stream"
    monkeypatch.setattr(module.protocol, "_open_public", lambda *args: response)
    with pytest.raises(module.SubscriptionOutputGuard) as caught:
        module.V3BoundedSubscriptionTransport()(ENDPOINT, {"store": False, "stream": True}, {}, 10.)
    assert caught.value.error_code == "subscription_event_size_limit"


def test_acquire_evidence_requires_preserved_intent_and_explicit_non_resume():
    raw = decision(type="acquire_evidence", continuation={"intent_id": "saved", "resume": False})
    assert validate_v3_agent_decision(raw) == raw
    for changes in ({"continuation": None},
                    {"continuation": {"intent_id": "saved", "resume": True}},
                    {"motion": {"xyz_m": [0., 0., 0.]}},
                    {"execution": {"segments": []}}):
        with pytest.raises(SchemaError):
            validate_v3_agent_decision({**raw, **changes})


def test_wire_projection_preserves_all_current_evidence_and_raw_accepted_history(monkeypatch, tmp_path):
    transport = AvailabilityTransport(decision(), failures=0)
    executive, _ = availability_client(monkeypatch, transport, log_dir=tmp_path)
    missing = [{"entity_id": "obstacle", "reason": "actual_overlap", "required_evidence": "fresh boundary"}]
    core = {"step": 1, "missing_evidence_requests": missing, "requirements_truncated": False,
            "operation_state": {"bound_object_id": "object", "phase": "release_pending",
                                "recent_events": [{"kind": "actual_open", "step": 1, "detail": "x" * 300}]},
            "blocking_reasons": ["unverified_release"], "advisories": ["coverage_unknown"]}
    request = {"fields": {"step": 1, "geometry": {"timestamp": 1, "calibration": [1, 2, 3]},
        "latest_instruction": "Keep the original object bound until release is accepted.",
        "context": {"native_evidence": {"current_decision": core,
                                       "operation_state": copy.deepcopy(core["operation_state"])},
                    "model_belief": {"belief": "redundant accepted response"},
                    "recent_feedback": [{"actual_step": 1, "reason": "close_is_posture"}]}},
        "images": [("External", PNG), ("Wrist", PNG)]}
    original = copy.deepcopy(request)
    executive.predict(request, mode="agent")
    first_history = executive.history_snapshot()
    raw = json.loads(first_history[0]["content"][0]["text"])
    wire = json.loads(transport.calls[0]["input"][0]["content"][0]["text"])
    assert raw["context"]["native_evidence"]["operation_state"] == core["operation_state"]
    assert wire["context"]["native_evidence"]["operation_state"] == {
        "same_observation_ref": "current_decision.operation_state"}
    assert wire["current_decision"] == core
    assert raw["geometry"] == wire["geometry"] == request["fields"]["geometry"]
    assert wire["latest_instruction"] == raw["latest_instruction"]
    assert request == original
    # On the next observation, only redundant historical capabilities/belief
    # disappear from the wire; prior physical transitions and blockers stay.
    second = copy.deepcopy(request)
    second["fields"]["step"] = 2
    executive.predict(second, mode="agent")
    old_wire = json.loads(transport.calls[-1]["input"][0]["content"][0]["text"])
    assert "capabilities" not in old_wire["context"] and "model_belief" not in old_wire["context"]
    assert old_wire["current_decision"] == core
    assert {"actual_step": 1, "reason": "close_is_posture"} in old_wire["context"]["outcomes_and_transitions"]
    assert executive.history_snapshot()[:len(first_history)] == first_history
    assert transport.calls[-1]["input"][1:3] == first_history[1:3]
    records = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*.request.json"))]
    assert records[-1]["original_observation"] == executive.history_snapshot()[-3]
    assert records[-1]["prompt_sizes"]["wire"]["image_count"] == 4
    assert records[-1]["prompt_sizes"]["wire"]["total_prompt_chars"] < records[-1]["prompt_sizes"]["before_wire_projection"]["total_prompt_chars"]


def test_projection_retry_commits_full_context_once_and_failed_call_preserves_history(monkeypatch):
    from gpt_field.astra import APIError
    transport = AvailabilityTransport(decision(), failures=1)
    executive, _ = availability_client(monkeypatch, transport)
    request = recovery_request()
    request["fields"]["context"] = {"capabilities": {"extra": "x" * 1000},
                                    "last_feedback": {"reason": "unresolved_boundary"}}
    executive.predict(request, mode="agent")
    accepted = executive.history_snapshot()
    assert transport.calls[0] == transport.calls[1]
    assert sum(item.get("role") == "user" for item in accepted) == 1
    assert json.loads(accepted[0]["content"][0]["text"])["context"]["capabilities"]["extra"] == "x" * 1000
    transport.failures = 99
    with pytest.raises(APIError):
        executive.predict(request, mode="agent")
    assert executive.history_snapshot() == accepted
    assert all(payload == transport.calls[2] for payload in transport.calls[2:])
