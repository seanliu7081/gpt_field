"""Offline behavioral tests for the separate direct-action history baseline."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gpt_field.astra import APIError, AstraConfig, BudgetExceeded, SchemaError
from gpt_field.direct_history import (
    DirectHistoryAstraClient, DirectHistoryPolicy, HISTORY_PROTOCOL, PROMPT_VERSION, TASK,
)
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.policy import Decision
from gpt_field.schema import GridSpec


def observation(index=0, state=None):
    image = image_to_data_url(np.full((2, 2, 3), index % 255, dtype=np.uint8), size=2)
    wrist = image_to_data_url(np.full((2, 2, 3), (index + 100) % 255, dtype=np.uint8), size=2)
    return PolicyObservation(image, wrist, tuple(state or [0.1, 0.2, 1.0, 0., 0., 0., .04, -.04]), TASK)


def prediction(gripper=1.0):
    return {"actions": [[.2, 0., -.1, 0., .1, 0., gripper] for _ in range(5)]}


def envelope(data=None, **updates):
    result = {"id": "resp_offline", "status": "completed", "output": [
        {"type": "reasoning", "id": "rs_offline", "summary": [], "encrypted_content": "opaque-fixture"},
        {"type": "message", "id": "msg_offline", "role": "assistant", "phase": "final_answer", "content": [
            {"type": "output_text", "text": json.dumps(prediction() if data is None else data)}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50,
                  "input_tokens_details": {"cached_tokens": 20},
                  "output_tokens_details": {"reasoning_tokens": 30}}}
    result.update(updates)
    return result


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(copy.deepcopy(payload))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


def client(fake, **kwargs):
    return DirectHistoryAstraClient(api_key="offline-secret-fixture", transport=fake, **kwargs)


def execute(policy, before, index, *, realized=True):
    decision = policy.act(before)
    state = list(before.state)
    if realized:
        for axis in range(3):
            state[axis] += decision.action[axis] * .025
        state[6] -= .001
        state[7] += .001
    after = observation(index + 1, state)
    feedback = policy.observe_transition(before, decision, after)
    return after, decision, feedback


class DirectHistoryClientTests(unittest.TestCase):
    def test_preview_requires_no_credentials_no_api_and_has_own_provenance(self):
        fake = FakeTransport()
        instance = DirectHistoryAstraClient(api_key=None, transport=fake)
        payload = instance.build_payload(observation().to_dict())
        self.assertEqual(fake.calls, [])
        self.assertFalse(payload["store"])
        self.assertEqual(payload["service_tier"], "default")
        self.assertNotIn("previous_response_id", payload)
        self.assertEqual(payload["include"], ["reasoning.encrypted_content"])
        self.assertIn(PROMPT_VERSION, payload["instructions"])
        self.assertIn("NOT seven robot joint angles", payload["instructions"])
        self.assertIn("0.25 seconds", payload["instructions"])
        self.assertEqual(payload["text"]["format"]["name"], "astra_direct_history_v1")
        self.assertEqual(payload["text"]["format"]["schema"]["required"], ["actions"])
        self.assertEqual(instance.history_snapshot(), [])

    def test_actual_requests_are_manual_history_with_last_two_image_pairs(self):
        fake = FakeTransport(envelope(), envelope(), envelope())
        instance = client(fake)
        policy = DirectHistoryPolicy(instance)
        obs = observation()
        first_images = (obs.image, obs.wrist_image)
        for step in range(11):
            obs, _, _ = execute(policy, obs, step)
        self.assertEqual(len(fake.calls), 3)
        third = fake.calls[2]
        images = [part["image_url"] for msg in third["input"] if isinstance(msg.get("content"), list)
                  for part in msg["content"] if part["type"] == "input_image"]
        self.assertEqual(len(images), 4)
        self.assertFalse(any(image in images for image in first_images))
        observations, feedback, actions = [], [], []
        for message in third["input"]:
            if message.get("type") == "reasoning":
                continue
            if message["role"] == "assistant":
                actions.append(json.loads(message["content"][0]["text"]))
                self.assertEqual(message["phase"], "final_answer")
                continue
            record = json.loads(message["content"][0]["text"])
            (observations if record["type"] == "observation" else feedback).append(record)
        self.assertEqual([x["observation_index"] for x in observations], [0, 1, 2])
        self.assertEqual([x["step"] for x in feedback], list(range(10)))
        self.assertEqual(actions, [prediction(), prediction()])
        self.assertEqual(instance.history_stats()["retained_image_pairs"], 2)
        self.assertEqual(instance.history_stats()["reasoning_replay"], "opaque_encrypted_output_items")
        reasoning = [msg for msg in third["input"] if msg.get("type") == "reasoning"]
        self.assertEqual(reasoning, [envelope()["output"][0]] * 2)
        self.assertEqual(third["input"][1:3], envelope()["output"])

    def test_retention_one_and_preview_are_side_effect_free(self):
        instance = client(FakeTransport(envelope()), image_history=1)
        instance.predict(observation().to_dict())
        before = instance.history_snapshot()
        payload = instance.build_payload(observation(1).to_dict())
        self.assertEqual(instance.history_snapshot(), before)
        count = sum(part["type"] == "input_image" for msg in payload["input"]
                    if isinstance(msg.get("content"), list) for part in msg["content"])
        self.assertEqual(count, 2)
        before[0]["content"].clear()
        self.assertNotEqual(instance.history_snapshot(), before)

    def test_invalid_retry_never_duplicates_or_commits_bad_output(self):
        fake = FakeTransport(envelope({"actions": []}), envelope())
        instance = client(fake)
        instance.predict(observation().to_dict())
        self.assertEqual(fake.calls[0], fake.calls[1])
        self.assertEqual(instance.history_stats()["observations"], 1)
        self.assertEqual(len(instance.history_snapshot()), 3)
        self.assertEqual(instance.budget_snapshot()["api_calls"], 2)
        self.assertEqual(instance.budget_snapshot()["output_tokens"], 100)

    def test_two_invalid_responses_leave_transcript_unchanged(self):
        fake = FakeTransport(envelope(), envelope({}), envelope({}))
        instance = client(fake)
        instance.predict(observation().to_dict())
        history = instance.history_snapshot()
        with self.assertRaises(SchemaError):
            instance.predict(observation(1).to_dict())
        self.assertEqual(instance.history_snapshot(), history)
        self.assertEqual(instance.budget_snapshot()["api_calls"], 3)

    def test_http_error_no_retry_no_history_commit(self):
        fake = FakeTransport(APIError("HTTP failure"))
        instance = client(fake)
        with self.assertRaises(APIError):
            instance.predict(observation().to_dict())
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(instance.history_snapshot(), [])
        self.assertEqual(instance.budget_snapshot()["unknown_usage_calls"], 1)

    def test_incomplete_and_refusal_outputs_are_not_executed(self):
        refusal = envelope()
        refusal["output"][1]["content"] = [{"type": "refusal", "refusal": "No"}]
        for response in (envelope(status="incomplete"), refusal):
            with self.subTest(response=response):
                fake = FakeTransport(response)
                instance = client(fake)
                with self.assertRaises(APIError):
                    instance.predict(observation().to_dict())
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(instance.history_snapshot(), [])

    def test_unsettled_interrupt_keeps_reservation_and_writes_error_record(self):
        config = AstraConfig(max_cost_usd=150, input_price_per_million=10,
                             output_price_per_million=50, cached_input_price_per_million=1,
                             cache_write_price_per_million=12.5)
        with tempfile.TemporaryDirectory() as directory:
            instance = client(FakeTransport(KeyboardInterrupt()), config=config, log_dir=directory)
            with self.assertRaises(KeyboardInterrupt):
                instance.predict(observation().to_dict())
            budget = instance.budget_snapshot()
            self.assertEqual(budget["api_calls"], 1)
            self.assertEqual(budget["unknown_usage_calls"], 1)
            self.assertEqual(budget["cost_usd_upper_bound"], budget["reservation_per_call_usd"])
            self.assertEqual(instance.history_snapshot(), [])
            record = json.loads(next(Path(directory).glob("*.result.json")).read_text())
            self.assertEqual(record["error_type"], "KeyboardInterrupt")

    def test_budget_prevents_dispatch_without_mutating_history(self):
        fake = FakeTransport()
        instance = client(fake, config=AstraConfig(max_api_calls=0))
        with self.assertRaises(BudgetExceeded):
            instance.predict(observation().to_dict())
        self.assertEqual(fake.calls, [])
        self.assertEqual(instance.history_snapshot(), [])

    def test_request_logs_and_hash_use_baseline_provenance_no_credentials(self):
        fake = FakeTransport(envelope())
        with tempfile.TemporaryDirectory() as directory:
            instance = client(fake, log_dir=directory)
            instance.predict(observation().to_dict())
            request = json.loads(next(Path(directory).glob("*.request.json")).read_text())
            self.assertEqual(request["prompt_version"], PROMPT_VERSION)
            self.assertEqual(request["history_protocol"], HISTORY_PROTOCOL)
            self.assertEqual(request["request_key"], instance.request_key(fake.calls[0], GridSpec(), "direct"))
            all_logs = "".join(path.read_text() for path in Path(directory).glob("*.json"))
            self.assertNotIn("offline-secret-fixture", all_logs)
            self.assertNotIn("astra-potential-v1", all_logs)

    def test_extra_oracle_data_wrong_task_wrong_mode_rejected_before_call(self):
        fake = FakeTransport()
        instance = client(fake)
        for field in ("object_pose", "contact", "success", "reward"):
            data = observation().to_dict()
            data[field] = "oracle-canary"
            with self.assertRaises(ValueError):
                instance.predict(data)
        wrong = observation().to_dict()
        wrong["prompt"] = "some other task"
        with self.assertRaises(ValueError):
            instance.predict(wrong)
        with self.assertRaises(ValueError):
            instance.predict(observation().to_dict(), mode="field")
        self.assertEqual(fake.calls, [])

    def test_invalid_actions_are_rejected_without_clipping(self):
        cases = []
        for value in (1.01, -1.01, True, float("nan"), float("inf")):
            bad = prediction()
            bad["actions"][0][0] = value
            cases.append(bad)
        cases.extend(({"actions": [[0.] * 7] * 4}, {"actions": [[0.] * 6] * 5},
                      {"actions": [[0.] * 7] * 5, "note": "unexpected"}))
        for data in cases:
            with self.subTest(data=data):
                instance = client(FakeTransport(envelope(data), envelope(data)))
                with self.assertRaises(SchemaError):
                    instance.predict(observation().to_dict())
                self.assertEqual(instance.history_snapshot(), [])

    def test_invalid_retention_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DirectHistoryAstraClient(image_history=value)


class DirectHistoryPolicyTests(unittest.TestCase):
    def test_exact_chunk_execution_close_reaches_action_without_override(self):
        data = {"actions": [[0., 0., 0., 0., 0., 0., x] for x in (-1., 0., 1., .4, -.2)]}
        fake = FakeTransport(envelope(data), envelope())
        policy = DirectHistoryPolicy(client(fake))
        obs = observation()
        actions = []
        for step in range(6):
            obs, decision, feedback = execute(policy, obs, step)
            actions.append(list(decision.action))
            self.assertEqual(decision.diagnostics["refreshed"], step % 5 == 0)
            self.assertEqual(decision.diagnostics["chunk_index"], step // 5)
            self.assertEqual(decision.diagnostics["chunk_step"], step % 5)
            self.assertEqual(feedback["normalized_action"], actions[-1])
        self.assertEqual(actions[:5], data["actions"])
        self.assertEqual(len(fake.calls), 2)

    def test_numeric_feedback_reports_command_vs_actual_movement(self):
        policy = DirectHistoryPolicy(client(FakeTransport(envelope())))
        _, _, feedback = execute(policy, observation(), 0)
        np.testing.assert_allclose(feedback["desired_translation_delta_m"], [.01, 0., -.005])
        np.testing.assert_allclose(feedback["desired_world_rotation_delta_rad"], [0., .05, 0.])
        np.testing.assert_allclose(feedback["measured_translation_delta_m"], [.005, 0., -.0025])
        np.testing.assert_allclose(feedback["gripper_joint_delta"], [-.001, .001])
        self.assertAlmostEqual(feedback["translation_tracking_error_m"], np.linalg.norm([.005, 0, -.0025]))
        self.assertEqual(feedback["rolling_progress"]["gripper_close_steps"], 1)

    def test_stagnation_feedback_is_descriptive_without_changing_commands(self):
        fake = FakeTransport(*(envelope() for _ in range(6)))
        policy = DirectHistoryPolicy(client(fake))
        obs = observation()
        for step in range(27):
            obs, decision, feedback = execute(policy, obs, step, realized=False)
            self.assertEqual(list(decision.action), prediction()["actions"][step % 5])
        progress = feedback["rolling_progress"]
        self.assertEqual(progress["window_size"], 25)
        self.assertEqual(progress["translation_command_steps"], 25)
        self.assertEqual(progress["low_motion_translation_steps"], 25)
        self.assertEqual(progress["measured_translation_path_m"], 0)
        self.assertEqual(progress["gripper_close_steps"], 25)
        self.assertEqual(len(fake.calls), 6)
        self.assertEqual(policy.client.history_stats()["feedback_records"], 27)

    def test_observation_feedback_order_is_enforced(self):
        policy = DirectHistoryPolicy(client(FakeTransport(envelope())))
        obs = observation()
        decision = policy.act(obs)
        with self.assertRaisesRegex(RuntimeError, "observe_transition"):
            policy.act(obs)
        with self.assertRaises(ValueError):
            policy.observe_transition(observation(5), decision, obs)
        with self.assertRaises(ValueError):
            policy.observe_transition(obs, Decision(decision.action, decision.diagnostics), obs)
        after = observation(1)
        policy.observe_transition(obs, decision, after)
        with self.assertRaises(RuntimeError):
            policy.observe_transition(obs, decision, after)
        with self.assertRaisesRegex(ValueError, "preceding transition"):
            policy.act(observation(2))
        self.assertEqual(policy.act(after).diagnostics["step"], 1)

    def test_task_switch_rejected_even_with_queued_actions(self):
        policy = DirectHistoryPolicy(client(FakeTransport(envelope())))
        obs, _, _ = execute(policy, observation(), 0)
        with self.assertRaises(ValueError):
            policy.act(replace(obs, prompt="another task"))
        self.assertEqual(policy.step_index, 1)

    def test_reset_clears_episode_history_pending_and_queue_but_keeps_ledger(self):
        instance = client(FakeTransport(envelope(), envelope()))
        policy = DirectHistoryPolicy(instance)
        obs, _, _ = execute(policy, observation(), 0)
        policy.act(obs)
        budget = instance.budget_snapshot()
        policy.reset()
        self.assertEqual(instance.history_snapshot(), [])
        self.assertEqual(instance.budget_snapshot()["api_calls"], budget["api_calls"])
        self.assertEqual(policy.step_index, 0)
        self.assertEqual(policy.chunk, [])
        self.assertIsNone(policy.latest_feedback)
        self.assertEqual(policy.act(observation()).diagnostics["step"], 0)
        self.assertEqual(instance.budget_snapshot()["api_calls"], 2)

    def test_feedback_chunk_bound_uses_client_setting_and_preserves_direct_default(self):
        instance = client(FakeTransport(envelope()))
        policy = DirectHistoryPolicy(instance)
        _, _, feedback = execute(policy, observation(), 0)
        with self.assertRaisesRegex(ValueError, "5-action chunk"):
            instance.record_transition({**feedback, "chunk_step": 5})
        instance.chunk_size = 5
        instance.field_update_steps = 30
        instance.record_transition({**feedback, "step": 29, "chunk_step": 29})
        with self.assertRaisesRegex(ValueError, "30-action chunk"):
            instance.record_transition({**feedback, "step": 30, "chunk_step": 30})

    def test_feedback_rejects_evaluator_fields_and_snapshot_mutation(self):
        instance = client(FakeTransport(envelope()))
        policy = DirectHistoryPolicy(instance)
        _, _, feedback = execute(policy, observation(), 0)
        for extra in ("success", "object_pose", "contact", "reward"):
            contaminated = {**feedback, extra: 1}
            with self.assertRaises(ValueError):
                instance.record_transition(contaminated)
        contaminated = copy.deepcopy(feedback)
        contaminated["rolling_progress"]["success"] = 1
        with self.assertRaises(ValueError):
            instance.record_transition(contaminated)
        feedback["normalized_action"][0] = 999
        self.assertNotEqual(feedback, policy.latest_feedback)
        self.assertNotIn("oracle-canary", json.dumps(instance.history_snapshot()))


if __name__ == "__main__":
    unittest.main()
