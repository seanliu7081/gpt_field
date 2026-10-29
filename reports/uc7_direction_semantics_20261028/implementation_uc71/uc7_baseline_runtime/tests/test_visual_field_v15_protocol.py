"""V15 delta only: unchanged contract, current diagnostic and valid blocked holds."""
import copy
import json

import pytest

from gpt_field.astra import SchemaError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v14 import output_schema as schema_v14
from gpt_field.visual_field_subscription_v15 import (
    VisualFieldSubscriptionClientV15, output_schema, shared_guidance, MOTION_LABEL, HISTORY_VERSION,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v14_protocol import prediction
from test_visual_field_v7_protocol import current_record, stream_error
from test_visual_feedback_compact_v14 import context


def client(*responses, **kwargs):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort": "medium"}, id=f"v15_{i}")
        for i, value in enumerate(responses)))
    return VisualFieldSubscriptionClientV15(token_provider=lambda: "offline-oauth", transport=transport, **kwargs), transport


def current_parts(payload):
    return next(message["content"] for message in reversed(payload["input"])
                if message.get("role") == "user" and message["content"][0].get("text", "").startswith('{"'))


def test_wire_is_identical_and_blocked_hold_is_valid_without_opening():
    assert output_schema() == schema_v14()
    value = prediction(phase="recover"); value["target_geometry"] = None
    value["direct_actions"] = [[0.] * 7 for _ in range(5)]
    instance, transport = client(value)
    result = instance.predict(observation().to_dict())
    assert result["direct_actions"] == value["direct_actions"]
    assert result["field_intent"] is result["gripper_intent"] is None
    text = shared_guidance(resolve_distance_task("bowl_plate"))
    for phrase in ("NON-NULL", "other two branches MUST be null", "Same", "same-direction recovery debt",
                   "recommended_local_max_travel_m", "side-effect-free", "does not automatically terminate"):
        assert phrase.lower() in text.lower()
    assert instance.history_stats()["protocol"] == HISTORY_VERSION
    assert transport.calls[0]["text"]["format"]["name"] == "visual_field_rgb_joints_v15"


@pytest.mark.parametrize("controller,key", [("visual_field", "field_intent"), ("direct", "direct_actions"), ("gripper", "gripper_intent")])
def test_selected_null_branch_still_rejected(controller, key):
    value = prediction(controller=controller, phase="grasp" if controller == "gripper" else "approach")
    value[key] = None
    instance, transport = client()
    with pytest.raises(SchemaError): instance.validate_response(value, None)
    assert not transport.calls


def test_current_motion_diagnostic_replaces_only_derived_overlay_and_keeps_all_live_facts():
    instance, transport = client(prediction())
    ctx = context(); ctx["visual_feedback"]["visual"]["reference_health"]["repair_required"] = True
    ctx["visual_feedback"]["motion_evidence"] = {"status": "ambiguous", "descriptor_bounds_px": [1, 2, 4, 6]}
    before = copy.deepcopy(ctx); instance.observable_context = ctx
    obs = observation().to_dict(); seen = []
    diagnostic = observation(9).image
    instance.motion_diagnostic_provider = lambda current: seen.append(copy.deepcopy(current)) or diagnostic
    instance.predict(obs)
    parts = current_parts(transport.calls[0])
    images = [part["image_url"] for part in parts if part["type"] == "input_image"]
    assert images == [obs["observation/image"], obs["observation/wrist_image"], diagnostic]
    assert sum(part.get("text") == MOTION_LABEL for part in parts) == 1
    assert seen == [obs] and ctx == before
    feedback = current_record(transport.calls[0])["observable_context"]["visual_feedback"]
    for key in ("translation_guard", "recovery_options", "reference_support", "motion_evidence"):
        assert feedback[key] == ctx["visual_feedback"][key]
    assert "NOT distal-tool segmentation" in MOTION_LABEL


def test_no_healthy_diagnostic_call_and_no_stale_motion_image_rollover():
    instance, transport = client(prediction(), prediction(), prediction())
    obs0, obs1, obs2 = (observation(i).to_dict() for i in range(3))
    diagnostic = observation(9).image; seen = []
    def provider(obs):
        seen.append(obs["observation/image"])
        return diagnostic if obs["observation/image"] == obs1["observation/image"] else None
    instance.motion_diagnostic_provider = provider
    instance.predict(obs0)
    assert not seen
    instance.observable_context = context()
    instance.predict(obs1); instance.predict(obs2)
    assert seen == [obs1["observation/image"], obs2["observation/image"]]
    assert any(part.get("image_url") == diagnostic for part in current_parts(transport.calls[1]))
    assert not any(part.get("image_url") == diagnostic for part in current_parts(transport.calls[2]))
    assert not any(part.get("text") == MOTION_LABEL for part in current_parts(transport.calls[2]))
    assert sum(part.get("type") == "input_image" for part in current_parts(transport.calls[0])) == 2


def test_mixed_bounded_repairs_keep_same_observation_medium_subscription_and_raw_output(tmp_path):
    bad = prediction(); bad["direct_actions"] = None
    good = prediction(); original = copy.deepcopy(good)
    instance, transport = client(stream_error(), bad, good, prediction(), log_dir=tmp_path)
    instance.predict(observation().to_dict()); instance.predict(observation(1).to_dict())
    assert len(transport.calls) == instance.budget_snapshot()["api_calls"] == 4
    assert instance.budget_snapshot()["api_key_calls"] == 0
    first = [current_record(payload) for payload in transport.calls[:3]]
    assert len({record["observation_index"] for record in first}) == 1
    assert all(record["observation/state"] == first[0]["observation/state"] for record in first)
    assert all(payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"} for payload in transport.calls)
    raw = [json.loads(part["text"]) for message in transport.calls[-1]["input"] if message.get("role") == "assistant"
           for part in message.get("content", []) if part["type"] == "output_text"]
    assert raw == [original]
    assert len(list(tmp_path.glob('*.response.json'))) == 3
