"""Behavioral checks for the history-only field-planner ablation; no live API."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gpt_field.astra import APIError, AstraClient, AstraConfig, BudgetExceeded, SchemaError
from gpt_field.direct_history import DirectHistoryPolicy
from gpt_field.field_history import (
    FieldHistoryAstraClient, FieldHistoryPolicy, HISTORY_PROTOCOL, PROMPT_VERSION, TASK,
)
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig
from gpt_field.policy import Decision, FieldPolicy
from gpt_field.schema import GridSpec


def observation(index=0, state=None):
    image = image_to_data_url(np.full((2, 2, 3), index % 255, dtype=np.uint8), size=2)
    wrist = image_to_data_url(np.full((2, 2, 3), (index + 100) % 255, dtype=np.uint8), size=2)
    return PolicyObservation(image, wrist, tuple(state or [.1, .2, 1., 0., 0., 0., .04, -.04]), TASK)


def prediction(gripper="open", motion="move", stage="approach", axis=2):
    return {"stage": stage, "orientation_target_xyzw": [1., 0., 0., 0.],
            "gripper_command": gripper, "motion_mode": motion,
            "values": GridSpec().coordinates([0., 0., 0.])[:, axis].tolist(),
            "blocked_indices": [], "unknown_indices": []}


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
    return FieldHistoryAstraClient(api_key="offline-secret-fixture", transport=fake, **kwargs)


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


class FieldHistoryClientTests(unittest.TestCase):
    def test_preview_keeps_original_schema_and_field_semantics_without_api(self):
        fake = FakeTransport()
        instance = FieldHistoryAstraClient(api_key=None, transport=fake)
        payload = instance.build_payload(observation().to_dict())
        original = AstraClient(api_key=None).build_payload(observation().to_dict(), GridSpec())
        self.assertEqual(payload["text"]["format"]["schema"], original["text"]["format"]["schema"])
        self.assertEqual(payload["text"]["format"]["name"], "astra_field_history_v1")
        self.assertIn(PROMPT_VERSION, payload["instructions"])
        self.assertIn("State is " + original["instructions"].split("State is ", 1)[1], payload["instructions"])
        self.assertNotIn("No previous conversation", payload["instructions"])
        self.assertIn("its own preceding observation", payload["instructions"])
        self.assertIn("not the current observation", payload["instructions"])
        self.assertIn("0.05-meter controller", payload["instructions"])
        self.assertIn("0.5-radian world-frame", payload["instructions"])
        self.assertIn("Gripper +1 closes", payload["instructions"])
        self.assertIn("initially open", payload["instructions"])
        self.assertFalse(payload["store"])
        self.assertEqual(payload["service_tier"], "default")
        self.assertEqual(payload["include"], ["reasoning.encrypted_content"])
        self.assertNotIn("previous_response_id", payload)
        self.assertEqual(fake.calls, [])
        self.assertEqual(instance.history_snapshot(), [])

    def test_full_fields_feedback_and_opaque_outputs_with_latest_two_images(self):
        predictions = [prediction(), prediction("close", "hold", "grasp"), prediction("hold")]
        fake = FakeTransport(*(envelope(data) for data in predictions))
        instance = client(fake)
        policy = FieldHistoryPolicy(instance)
        obs = observation()
        first_images = (obs.image, obs.wrist_image)
        observation_origins = []
        for step in range(11):
            if step % 5 == 0:
                observation_origins.append(list(obs.state[:3]))
            obs, _, _ = execute(policy, obs, step)
        self.assertEqual(len(fake.calls), 3)
        third = fake.calls[2]
        images = [part["image_url"] for msg in third["input"] if isinstance(msg.get("content"), list)
                  for part in msg["content"] if part["type"] == "input_image"]
        self.assertEqual(len(images), 4)
        self.assertTrue(all(image not in images for image in first_images))
        observations, feedback, fields = [], [], []
        for message in third["input"]:
            if message.get("type") == "reasoning":
                continue
            if message["role"] == "assistant":
                fields.append(json.loads(message["content"][0]["text"]))
                self.assertEqual(message["phase"], "final_answer")
                continue
            record = json.loads(message["content"][0]["text"])
            (observations if record["type"] == "observation" else feedback).append(record)
        self.assertEqual(fields, predictions[:2])
        self.assertEqual([x["observation_index"] for x in observations], [0, 1, 2])
        self.assertEqual([x["observation/state"][:3] for x in observations], observation_origins)
        self.assertEqual([x["step"] for x in feedback], list(range(10)))
        self.assertEqual(third["input"][1:3], envelope(predictions[0])["output"])
        self.assertEqual([x for x in third["input"] if x.get("type") == "reasoning"],
                         [envelope()["output"][0]] * 2)
        self.assertEqual(instance.history_stats()["retained_image_pairs"], 2)
        self.assertEqual(instance.history_stats()["feedback_records"], 11)

    def test_payload_preview_and_returned_snapshots_do_not_mutate_history(self):
        instance = client(FakeTransport(envelope()), image_history=1)
        instance.predict(observation().to_dict())
        before = instance.history_snapshot()
        payload = instance.build_payload(observation(1).to_dict())
        self.assertEqual(instance.history_snapshot(), before)
        self.assertEqual(sum(part["type"] == "input_image" for msg in payload["input"]
                             if isinstance(msg.get("content"), list) for part in msg["content"]), 2)
        before[0]["content"].clear()
        self.assertNotEqual(instance.history_snapshot(), before)

    def test_invalid_retry_uses_same_input_and_only_commits_accepted_field(self):
        fake = FakeTransport(envelope({}), envelope())
        instance = client(fake)
        instance.predict(observation().to_dict())
        self.assertEqual(fake.calls[0], fake.calls[1])
        self.assertEqual(instance.history_stats()["observations"], 1)
        self.assertEqual(len(instance.history_snapshot()), 3)
        self.assertEqual(instance.budget_snapshot()["api_calls"], 2)
        self.assertEqual(instance.history_snapshot()[2], envelope()["output"][1])

    def test_schema_exhaustion_keeps_existing_history(self):
        instance = client(FakeTransport(envelope(), envelope({}), envelope({})))
        instance.predict(observation().to_dict())
        before = instance.history_snapshot()
        with self.assertRaises(SchemaError):
            instance.predict(observation(1).to_dict())
        self.assertEqual(instance.history_snapshot(), before)
        self.assertEqual(instance.budget_snapshot()["api_calls"], 3)

    def test_failed_incomplete_and_refused_responses_never_commit(self):
        refusal = envelope()
        refusal["output"][1]["content"] = [{"type": "refusal", "refusal": "No"}]
        for response in (APIError("HTTP failure"), envelope(status="incomplete"), refusal):
            with self.subTest(response=response):
                fake = FakeTransport(response)
                instance = client(fake)
                with self.assertRaises(APIError):
                    instance.predict(observation().to_dict())
                self.assertEqual(instance.history_snapshot(), [])
                self.assertEqual(len(fake.calls), 1)

    def test_budget_denial_does_not_dispatch_or_change_history(self):
        fake = FakeTransport()
        instance = client(fake, config=AstraConfig(max_api_calls=0))
        with self.assertRaises(BudgetExceeded):
            instance.predict(observation().to_dict())
        self.assertEqual(fake.calls, [])
        self.assertEqual(instance.history_snapshot(), [])

    def test_interrupt_reservation_and_variant_request_log_are_preserved(self):
        config = AstraConfig(max_cost_usd=300, input_price_per_million=10,
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
            request = json.loads(next(Path(directory).glob("*.request.json")).read_text())
            self.assertEqual(request["prompt_version"], PROMPT_VERSION)
            result = json.loads(next(Path(directory).glob("*.result.json")).read_text())
            self.assertEqual(result["error_type"], "KeyboardInterrupt")

    def test_log_hash_and_metadata_use_field_history_provenance(self):
        fake = FakeTransport(envelope())
        with tempfile.TemporaryDirectory() as directory:
            instance = client(fake, log_dir=directory)
            instance.predict(observation().to_dict())
            request = json.loads(next(Path(directory).glob("*.request.json")).read_text())
            self.assertEqual(request["prompt_version"], PROMPT_VERSION)
            self.assertEqual(request["history_protocol"], HISTORY_PROTOCOL)
            self.assertEqual(request["request_key"], instance.request_key(fake.calls[0], GridSpec(), "field"))
            logs = "".join(path.read_text() for path in Path(directory).glob("*.json"))
            self.assertNotIn("offline-secret-fixture", logs)
            self.assertNotIn("astra-direct-history-v1", logs)
            self.assertNotIn("astra-potential-v1", logs)

    def test_oracle_inputs_wrong_task_mode_and_grid_are_rejected_before_dispatch(self):
        fake = FakeTransport()
        instance = client(fake)
        for name in ("object_pose", "contact", "success", "reward"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                instance.predict({**observation().to_dict(), name: "oracle-canary"})
        with self.assertRaises(ValueError):
            instance.predict(replace(observation(), prompt="another task").to_dict())
        with self.assertRaises(ValueError):
            instance.predict(observation().to_dict(), mode="direct")
        for grid in (GridSpec(size=3), GridSpec(spacing_m=.01), GridSpec(frame="robot")):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                instance.predict(observation().to_dict(), grid)
        self.assertEqual(fake.calls, [])


class FieldHistoryPolicyTests(unittest.TestCase):
    def test_actions_and_every_local_plan_match_original_field_policy(self):
        predictions = [prediction(), prediction("close", "hold", "grasp"),
                       prediction("hold", "move", "lift", axis=0), prediction("open")]
        fake = FakeTransport(*(envelope(data) for data in predictions))
        original_fake = FakeTransport(*(envelope(data) for data in predictions))
        policy = FieldHistoryPolicy(client(fake))
        original = FieldPolicy(AstraClient(api_key="offline", transport=original_fake))
        self.assertEqual(policy.grid, GridSpec())
        self.assertEqual(policy.planner.config, PlannerConfig())
        obs = observation()
        actions = []
        for step in range(16):
            original_decision = original.act(obs)
            after, decision, feedback = execute(policy, obs, step)
            self.assertEqual(decision.action, original_decision.action)
            self.assertEqual(decision.diagnostics["plan"], original_decision.diagnostics["plan"])
            self.assertEqual(decision.diagnostics["field_center"], original_decision.diagnostics["field_center"])
            self.assertEqual(decision.diagnostics["status"], original_decision.diagnostics["status"])
            self.assertEqual(decision.diagnostics["refreshed"], step % 5 == 0)
            self.assertEqual(decision.diagnostics["method"], "field_history")
            self.assertEqual(decision.diagnostics["chunk_step"], step % 5)
            self.assertEqual(decision.diagnostics["chunk_index"], step // 5)
            self.assertEqual(feedback["normalized_action"], list(decision.action))
            self.assertLessEqual(np.linalg.norm(decision.action[3:6]) * .5, .15000000001)
            actions.append(decision.action)
            obs = after
        self.assertEqual([action[6] for action in actions], [-1.] * 5 + [1.] * 10 + [-1.])
        self.assertTrue(all(action[:3] == (0., 0., 0.) for action in actions[5:10]))
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual(len(original_fake.calls), 4)

    def test_feedback_reuses_exact_direct_history_implementation_and_numeric_schema(self):
        self.assertIs(FieldHistoryPolicy.observe_transition, DirectHistoryPolicy.observe_transition)
        policy = FieldHistoryPolicy(client(FakeTransport(envelope(prediction("close")))))
        _, decision, feedback = execute(policy, observation(), 0)
        np.testing.assert_allclose(feedback["desired_translation_delta_m"], [0, 0, -.01])
        np.testing.assert_allclose(feedback["measured_translation_delta_m"], [0, 0, -.005])
        np.testing.assert_allclose(feedback["desired_world_rotation_delta_rad"], np.array(decision.action[3:6]) * .5)
        np.testing.assert_allclose(feedback["gripper_joint_delta"], [-.001, .001])
        self.assertAlmostEqual(feedback["translation_tracking_error_m"], .005)
        self.assertEqual(feedback["rolling_progress"]["gripper_close_steps"], 1)

    def test_stagnation_is_reported_without_forced_close_phase_or_recovery(self):
        fake = FakeTransport(*(envelope() for _ in range(6)))
        policy = FieldHistoryPolicy(client(fake))
        obs = observation()
        for step in range(27):
            obs, decision, feedback = execute(policy, obs, step, realized=False)
            np.testing.assert_allclose(decision.action[:3], (0., 0., -.2), atol=1e-15)
            self.assertEqual(decision.action[6], -1.)
            self.assertEqual(decision.diagnostics["stage"], "approach")
        progress = feedback["rolling_progress"]
        self.assertEqual(progress["window_size"], 25)
        self.assertEqual(progress["low_motion_translation_steps"], 25)
        self.assertEqual(progress["translation_command_steps"], 25)
        self.assertEqual(progress["measured_translation_path_m"], 0)
        self.assertEqual(len(fake.calls), 6)

    def test_pending_action_feedback_and_observation_order_are_enforced(self):
        policy = FieldHistoryPolicy(client(FakeTransport(envelope())))
        before = observation()
        decision = policy.act(before)
        with self.assertRaisesRegex(RuntimeError, "observe_transition"):
            policy.act(before)
        with self.assertRaises(ValueError):
            policy.observe_transition(observation(5), decision, before)
        with self.assertRaises(ValueError):
            policy.observe_transition(before, Decision(decision.action, decision.diagnostics), before)
        after = observation(1)
        policy.observe_transition(before, decision, after)
        with self.assertRaises(RuntimeError):
            policy.observe_transition(before, decision, after)
        with self.assertRaisesRegex(ValueError, "preceding transition"):
            policy.act(observation(2))
        self.assertEqual(policy.act(after).diagnostics["step"], 1)

    def test_reset_clears_field_gripper_and_history_but_preserves_spend(self):
        instance = client(FakeTransport(envelope(prediction("close")), envelope(prediction("hold"))))
        policy = FieldHistoryPolicy(instance)
        obs, _, _ = execute(policy, observation(), 0)
        policy.act(obs)
        before = instance.budget_snapshot()
        policy.reset()
        self.assertEqual(instance.history_snapshot(), [])
        self.assertEqual(instance.budget_snapshot()["api_calls"], before["api_calls"])
        self.assertEqual(policy.step_index, 0)
        self.assertIsNone(policy.field)
        self.assertIsNone(policy.latest_feedback)
        self.assertEqual(policy._progress, [])
        decision = policy.act(observation())
        self.assertEqual(decision.action[6], -1.)
        self.assertEqual(decision.diagnostics["step"], 0)
        self.assertEqual(instance.budget_snapshot()["api_calls"], 2)

    def test_no_oracle_feedback_and_no_shared_mutable_feedback(self):
        instance = client(FakeTransport(envelope()))
        policy = FieldHistoryPolicy(instance)
        obs, _, feedback = execute(policy, observation(), 0)
        for name in ("reward", "success", "object_pose", "contact"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                instance.record_transition({**feedback, name: "oracle-canary"})
        corrupted = copy.deepcopy(feedback)
        corrupted["rolling_progress"]["success"] = True
        with self.assertRaises(ValueError):
            instance.record_transition(corrupted)
        feedback["normalized_action"][0] = 999
        self.assertNotEqual(policy.latest_feedback, feedback)
        decision = policy.act(obs)
        decision.diagnostics["feedback"]["normalized_action"][0] = 888
        self.assertNotEqual(policy.latest_feedback["normalized_action"][0], 888)
        self.assertNotIn("oracle-canary", json.dumps(instance.history_snapshot()))

    def test_failed_refresh_does_not_advance_policy_or_create_pending_action(self):
        policy = FieldHistoryPolicy(client(FakeTransport(envelope({}), envelope({}))))
        with self.assertRaises(SchemaError):
            policy.act(observation())
        self.assertEqual(policy.step_index, 0)
        self.assertIsNone(policy._pending)
        self.assertIsNone(policy.field)
        self.assertEqual(policy.client.history_snapshot(), [])

    def test_task_cannot_change_between_field_refreshes(self):
        policy = FieldHistoryPolicy(client(FakeTransport(envelope())))
        obs, _, _ = execute(policy, observation(), 0)
        with self.assertRaises(ValueError):
            policy.act(replace(obs, prompt="another task"))
        self.assertEqual(policy.step_index, 1)


if __name__ == "__main__":
    unittest.main()
