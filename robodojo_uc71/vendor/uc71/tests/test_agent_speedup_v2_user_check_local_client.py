"""Exercise actual variant request/response handling without provider calls."""
import json

import pytest

from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, FixedAnswers, ScriptedV2Transport
from gpt_field.agent_speedup_v2_user_check_client import PlanLock, ReviewedV2Client, UserCheckV2Client
from gpt_field.agent_speedup_v2_user_check_motion import motion_context, parse_reviewed_repair
from gpt_field.astra import SchemaError
from test_agent_speedup_v2 import TASK2, TASK2_PLAN
from test_agent_speedup_v2_client import image
from test_agent_speedup_v2_user_check_motion import _view, _answer
from gpt_field.agent_speedup_v2_plan import parse_plan, plan_status_block
from gpt_field.agent_speedup_v2_types import StepState


def test_all_variant_modes_use_high_and_keep_baseline_unchanged():
    transport = ScriptedV2Transport(FixedAnswers())
    baseline = AgentSpeedUpV2Client(task_prompt=TASK2, transport=transport)
    draft = UserCheckV2Client(task_prompt=TASK2, transport=transport)
    lock = PlanLock(TASK2_PLAN, task_prompt=TASK2)
    reviewed = ReviewedV2Client(task_prompt=TASK2, transport=transport,
                                approved_plan=lock.plan(), lock=lock)
    request = {"fields": {}, "images": []}
    for mode in baseline._instruction_text:
        assert baseline.build_payload(request, mode=mode)["reasoning"]["effort"] == "medium"
        for client in (draft, reviewed):
            payload = client.build_payload(request, mode=mode)
            assert payload["reasoning"]["effort"] == "high"
            assert "Never give directions" not in payload["instructions"]
            assert "never how to move" not in payload["instructions"]
    assert UserCheckV2Client.reasoning_effort == "high"
    assert "local_motion" not in json.dumps(baseline._schemas["repair"])
    assert reviewed._schemas["plan"] == baseline._schemas["plan"]


def test_actual_repair_request_carries_measurements_and_parses_numeric_target():
    proposed = _answer()
    transport = ScriptedV2Transport(FixedAnswers(repairs=[proposed]), effort="high")
    lock = PlanLock(TASK2_PLAN, task_prompt=TASK2)
    lock._local_motion_context = motion_context(_view())
    lock._local_motion_feedback = {"status": "segment_complete", "executed_steps": 2}
    client = ReviewedV2Client(task_prompt=TASK2, transport=transport,
                              approved_plan=lock.plan(), lock=lock)
    plan = parse_plan(lock.plan())
    states = [StepState.pending(i, step) for i, step in enumerate(plan.steps)]
    raw = client.request_repair(image(1), image(2), quarter_turns=0,
                               status_block=plan_status_block(plan, states, holding=None),
                               last_outcome=None, holding=None)
    assert parse_reviewed_repair(raw).to_json() == proposed
    payload = transport.calls[-1]
    fields = json.loads(payload["input"][0]["content"][0]["text"])
    assert fields["local_control"]["context"]["xyz_m"] == [.1, -.1, .9]
    assert fields["local_control"]["last_feedback"]["executed_steps"] == 2
    assert payload["reasoning"]["effort"] == "high"
    assert payload["text"]["format"]["strict"] is True
    assert sum(item["type"] == "input_image" for item in payload["input"][0]["content"]) == 2
    # Changing only the action cannot smuggle a numeric target into native retry.
    with pytest.raises(SchemaError):
        client.validate_response({**proposed, "action": "retry"}, client.grid)
