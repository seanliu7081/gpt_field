from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest

from robodojo_uc71.client import NativeExecutiveClient, NativePlanClient, executive_instructions
from robodojo_uc71.planning import PUBLIC_HOME_RULE, planning_images, rgb_image
from robodojo_uc71.history import FROZEN_SOURCE
from gpt_field.astra import APIError, AstraConfig
from gpt_field.agent_speedup_v2_user_check_transport import UserCheckSubscriptionTransport
from client_fixtures import TASK, OfflineTransport, observation, plan, frame, current_fields


def executive(tmp_path, transport=None):
    transport = transport or OfflineTransport()
    return NativeExecutiveClient(task_prompt=TASK, approved_plan=plan(), direction_frame=frame(),
                                 transport=transport, log_dir=tmp_path), transport


def test_planning_has_six_native_clean_grid_images_and_no_execution_sensors(tmp_path):
    transport = OfflineTransport(plan())
    client = NativePlanClient(task_prompt=TASK, transport=transport, log_dir=tmp_path)
    answer = client.request_plan(observation()["images"])
    assert answer == plan()
    payload = transport.calls[0]
    assert payload["reasoning"] == {"effort": "high"}
    assert PUBLIC_HOME_RULE in payload["instructions"]
    assert "79 mm" not in payload["instructions"]
    content = payload["input"][0]["content"]
    images = [row for row in content if row["type"] == "input_image"]
    assert len(images) == 6
    assert all(rgb_image(row["image_url"]).shape == (480, 640, 3) for row in images)
    fields = json.loads(content[0]["text"])
    assert not {"robot", "calibration", "state", "joint_position_rad", "direction_frame"} & fields.keys()
    assert fields["task"] == TASK
    assert "max_output_tokens" not in payload


def test_eight_accepted_turns_retain_six_complete_sets_all_text_and_opaque_items(tmp_path):
    client, transport = executive(tmp_path)
    initial = deepcopy(observation()["robot"])
    for step in range(8):
        client.request_agent(observation(step), feedback={"status": "succeeded", "reason": "observation_only", "position_error_m": step * .001})
    payload = transport.calls[-1]
    users = [row for row in payload["input"] if row.get("role") == "user"]
    assert len(users) == 8
    counts = [sum(part["type"] == "input_image" for part in row["content"]) for row in users]
    assert counts == [0, 0, 3, 3, 3, 3, 3, 3]
    assert json.loads(users[0]["content"][0]["text"])["robot"] == initial
    assert "reviewed_goal_contract" in json.loads(users[0]["content"][0]["text"])
    assert all(json.loads(users[i]["content"][0]["text"])["context"]["last_feedback"]["position_error_m"] == i * .001 for i in range(8))
    assert sum(row.get("type") == "reasoning" for row in payload["input"]) == 7
    assert client.history_stats()["retained_visual_sets"] == 6
    assert client.budget_snapshot()["api_key_calls"] == 0
    assert client.budget_snapshot()["subscription_inference_calls"] == 8
    assert current_fields(payload)["context"]["region_directions"]["direction_semantics"] == "external_view_v1"


def test_failure_does_not_commit_history_or_advance_observation(tmp_path):
    client, transport = executive(tmp_path)
    client.request_agent(observation(0))
    before = client.history_snapshot()
    failure = APIError("offline invalid request")
    failure.error_code = "invalid_request_error"
    transport.failure = failure
    with pytest.raises(APIError):
        client.request_agent(observation(1))
    assert client.history_snapshot() == before
    assert client._last_step == 0
    transport.failure = None
    client.request_agent(observation(1))
    assert client._last_step == 1


def test_privileged_feedback_nested_fields_and_changed_tools_rejected(tmp_path):
    client, transport = executive(tmp_path)
    for key in ("reward", "remaining_steps", "object_poses", "contacts"):
        with pytest.raises(ValueError):
            client.agent_request(observation(), feedback={"detail": {key: 1}})
    assert transport.calls == []


def test_unknown_truth_keys_and_arbitrary_feedback_text_rejected(tmp_path):
    client, transport = executive(tmp_path)
    for bad in ({"debug": {"bread_xyz": [0, 1, 2]}},
                {"status": "failed", "reason": "bread is at secret coordinates"},
                {"status": "failed", "reason": "segment_stall", "measurements": [{"bread_xyz": [1, 2, 3]}]}):
        with pytest.raises(ValueError):
            client.agent_request(observation(), feedback=bad)
    with pytest.raises(ValueError):
        NativeExecutiveClient(task_prompt=TASK, approved_plan=plan(), direction_frame=frame(),
                              transport=transport, controller_capabilities={"debug": {"bread_xyz": [0, 1, 2]}})
    with pytest.raises(ValueError):
        client.agent_request(observation(), execution={"status": "running", "debug": [0, 1, 2]})
    assert transport.calls == []


def test_static_camera_binding_allows_only_wrist_extrinsic_changes(tmp_path):
    client, transport = executive(tmp_path)
    client.request_agent(observation(0))
    changed = observation(1)
    changed["calibration"]["cameras"]["left_wrist"]["world_from_optical"][0][3] += .01
    client.request_agent(changed)
    for modify in (lambda c: c["cameras"]["right_wrist"]["intrinsics"][0].__setitem__(0, 410.),
                   lambda c: c["cameras"]["left_wrist"].__setitem__("pose_source", "different_source"),
                   lambda c: c["cameras"]["head"]["world_from_optical"][0].__setitem__(3, .02)):
        changed = observation(2)
        modify(changed["calibration"])
        with pytest.raises(ValueError):
            client.agent_request(changed)


def test_other_models_rejected_by_both_clients(tmp_path):
    config = AstraConfig(model="different-model")
    with pytest.raises(ValueError):
        NativePlanClient(config, task_prompt=TASK, transport=OfflineTransport(plan()))
    with pytest.raises(ValueError):
        NativeExecutiveClient(config, task_prompt=TASK, approved_plan=plan(), direction_frame=frame(), transport=OfflineTransport())


def test_fixed_tool_provenance_rejected_after_accepted_observation(tmp_path):
    client, _ = executive(tmp_path)
    client.request_agent(observation(0))
    altered = observation(1)
    altered["calibration"]["tools"]["left"]["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="calibration changed"):
        client.request_agent(altered)


def test_execution_status_must_reference_the_last_accepted_decision(tmp_path):
    client, _ = executive(tmp_path)
    accepted = client.request_agent(observation(0))
    snapshot = {"decision": accepted, "progress": {"status": "succeeded", "reason": "observation_only"}}
    assert client.agent_request(observation(1), execution=snapshot)["fields"]["context"]["execution"] == snapshot
    snapshot["decision"]["why"] = "Unaccepted fabricated content"
    with pytest.raises(ValueError, match="prior accepted"):
        client.agent_request(observation(1), execution=snapshot)


def test_off_main_thread_fails_before_any_transport_and_production_deadline_is_frozen(tmp_path):
    client, transport = executive(tmp_path)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(client.request_agent, observation())
        with pytest.raises(RuntimeError, match="main thread"):
            future.result()
    assert transport.calls == []
    production = NativeExecutiveClient(task_prompt=TASK, approved_plan=plan(), direction_frame=frame())
    assert isinstance(production.transport, UserCheckSubscriptionTransport)
    assert production.transport_configuration["request_deadline_s"] == 180
    import gpt_field
    assert str(FROZEN_SOURCE) in gpt_field.__file__


def test_native_prompt_has_no_panda_controller_claim_and_no_macro_action():
    text = executive_instructions()
    assert "one Franka" not in text and "20 Hz" not in text and "gripper0_grip_site" not in text
    assert "macro: motion=" not in text and "25 Hz" in text and PUBLIC_HOME_RULE in text


def test_native_action_counter_and_raw_payload_bypasses_rejected(tmp_path):
    client, transport = executive(tmp_path)
    with pytest.raises(ValueError, match="native action counts"):
        client.agent_request(observation(40))
    request = client.agent_request(observation())
    request["fields"]["reward"] = 1
    with pytest.raises(ValueError):
        client.build_payload(request, mode="agent")
    planner = NativePlanClient(task_prompt=TASK, transport=OfflineTransport(plan()))
    request = planner.planning_request(observation()["images"])
    request["fields"]["robot"] = observation()["robot"]
    with pytest.raises(ValueError, match="sensor"):
        planner.build_payload(request, mode="plan")
    assert transport.calls == []
