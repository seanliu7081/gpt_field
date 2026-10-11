"""Offline client tests: no credentials, network, SDK, or simulator required."""

import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gpt_field.astra import (
    APIError, AstraClient, AstraConfig, BudgetExceeded, MissingCredentials,
    SchemaError, validate_prediction,
)
from gpt_field.schema import GridSpec


def observation():
    # A real 1x1 PNG; protocol tests do not inspect image content.
    image = "data:image/png;base64," + (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX2cAAAAASUVORK5CYII="
    )
    return {"observation/image": image, "observation/wrist_image": image,
            "observation/state": [0.0] * 8, "prompt": "Move the cup onto the plate."}


def prediction(grid):
    return {"stage": "approach", "orientation_target_xyzw": [0., 0., 0., 1.],
            "gripper_command": "hold", "motion_mode": "move",
            "values": [float(i) / 10 for i in range(grid.count)],
            "blocked_indices": [], "unknown_indices": []}


def envelope(data, **updates):
    result = {"id": "resp_fake", "status": "completed",
              "output": [{"type": "message", "role": "assistant", "content": [
                  {"type": "output_text", "text": json.dumps(data)}]}],
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
        self.calls.append((endpoint, copy.deepcopy(payload), dict(headers), timeout_s))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


class AstraTests(unittest.TestCase):
    def setUp(self):
        self.grid = GridSpec(size=3, spacing_m=0.02, frame="world")

    def client(self, transport, **kwargs):
        return AstraClient(api_key="test-only-not-a-key", transport=transport, **kwargs)

    def test_missing_key_fails_before_transport(self):
        fake = FakeTransport()
        with patch.dict("os.environ", {}, clear=True):
            client = AstraClient(transport=fake)
        with self.assertRaisesRegex(MissingCredentials, "OPENAI_API_KEY"):
            client.predict(observation(), self.grid)
        self.assertEqual(fake.calls, [])

    def test_payload_is_stateless_strict_and_image_only_allowlist(self):
        fake = FakeTransport(envelope(prediction(self.grid)))
        client = self.client(fake)
        self.assertEqual(client.predict(observation(), self.grid), prediction(self.grid))
        endpoint, payload, headers, timeout = fake.calls[0]
        self.assertEqual(endpoint, "https://api.openai.com/v1/responses")
        self.assertEqual(payload["model"], "gpt-6-astra")
        self.assertEqual(payload["service_tier"], "default")
        self.assertFalse(payload["store"])
        self.assertNotIn("previous_response_id", payload)
        self.assertEqual(len(payload["input"]), 1)
        self.assertEqual(len([x for x in payload["input"][0]["content"] if x["type"] == "input_image"]), 2)
        fmt = payload["text"]["format"]
        self.assertTrue(fmt["strict"])
        self.assertFalse(fmt["schema"]["additionalProperties"])
        self.assertEqual(fmt["schema"]["properties"]["values"]["minItems"], self.grid.count)
        self.assertIn("unitless", payload["instructions"])
        self.assertEqual(client.budget_snapshot()["output_tokens"], 50)
        self.assertEqual(client.budget_snapshot()["reasoning_tokens"], 30)

    def test_observation_extra_state_cannot_leak(self):
        fake = FakeTransport()
        client = self.client(fake)
        obs = observation()
        obs["object_pose"] = [1., 2., 3.]
        with self.assertRaisesRegex(ValueError, "four allowed keys"):
            client.predict(obs, self.grid)
        self.assertEqual(fake.calls, [])

    def test_state_and_image_validation_precedes_calls(self):
        for field, value in (("observation/state", [float("nan")] * 8),
                             ("observation/state", [False] * 8),
                             ("observation/image", "https://example.com/image.png"),
                             ("observation/image", "data:image/png;base64,broken!"),
                             ("observation/image", "data:image/png;base64," + base64.b64encode(b"not PNG").decode())):
            with self.subTest(field=field, value=value):
                fake = FakeTransport()
                obs = observation()
                obs[field] = value
                with self.assertRaises(ValueError):
                    self.client(fake).predict(obs, self.grid)
                self.assertEqual(fake.calls, [])

    def test_invalid_output_retries_once_with_identical_current_observation(self):
        bad = prediction(self.grid)
        bad["values"] = [0.]
        fake = FakeTransport(envelope(bad), envelope(prediction(self.grid)))
        client = self.client(fake)
        client.predict(observation(), self.grid)
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(fake.calls[0][1], fake.calls[1][1])
        self.assertEqual(client.budget_snapshot()["api_calls"], 2)
        self.assertEqual(client.budget_snapshot()["output_tokens"], 100)

    def test_two_schema_failures_are_terminal(self):
        bad = envelope(prediction(self.grid))
        bad["output"][0]["content"][0]["text"] = "not json"
        fake = FakeTransport(bad, bad)
        with self.assertRaises(SchemaError):
            self.client(fake).predict(observation(), self.grid)
        self.assertEqual(len(fake.calls), 2)

    def test_no_silent_numeric_fix_or_unexpected_fields(self):
        cases = []
        for key, value in (("values", [float("nan")] * self.grid.count),
                           ("values", [True] * self.grid.count),
                           ("gripper_command", "grasp"),
                           ("blocked_indices", [self.grid.count]),
                           ("blocked_indices", {}),
                           ("unknown_indices", ""),
                           ("orientation_target_xyzw", [0, 0, 0, 0]),
                           ("motion_mode", "wait")):
            bad = prediction(self.grid)
            bad[key] = value
            cases.append(bad)
        extra = prediction(self.grid)
        extra["explanation"] = "unexpected"
        cases.append(extra)
        for bad in cases:
            with self.subTest(data=bad):
                with self.assertRaises(SchemaError):
                    validate_prediction(bad, self.grid)

    def test_duplicate_json_keys_are_rejected(self):
        response = envelope(prediction(self.grid))
        response["output"][0]["content"][0]["text"] = '{"actions": [], "actions": []}'
        fake = FakeTransport(response, response)
        with self.assertRaisesRegex(SchemaError, "duplicate"):
            self.client(fake).predict(observation(), self.grid, "direct")
        self.assertEqual(len(fake.calls), 2)

    def test_refusal_incomplete_and_http_errors_are_not_retried(self):
        refusal = envelope(prediction(self.grid))
        refusal["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
        failures = (refusal, envelope({}, status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
                    APIError("HTTP 429", response_body='{"error":"rate limited"}'),
                    OSError("offline"))
        for response in failures:
            with self.subTest(response=response):
                fake = FakeTransport(response)
                with self.assertRaises(APIError):
                    self.client(fake).predict(observation(), self.grid)
                self.assertEqual(len(fake.calls), 1)

    def test_retry_must_reserve_another_api_call(self):
        fake = FakeTransport(envelope({}))
        client = self.client(fake, config=AstraConfig(max_api_calls=1))
        with self.assertRaisesRegex(BudgetExceeded, "max_api_calls"):
            client.predict(observation(), self.grid)
        self.assertEqual(len(fake.calls), 1)

    def test_zero_call_budget_makes_no_request(self):
        fake = FakeTransport()
        with self.assertRaises(BudgetExceeded):
            self.client(fake, config=AstraConfig(max_api_calls=0)).predict(observation(), self.grid)
        self.assertEqual(fake.calls, [])

    def prices(self, **kwargs):
        # Explicit invented rates for arithmetic tests; no assertions about live prices.
        settings = {"input_price_per_million": 10., "output_price_per_million": 50.,
                    "cached_input_price_per_million": 1., "cache_write_price_per_million": 12.5}
        settings.update(kwargs)
        return AstraConfig(**settings)

    def test_cost_cap_requires_explicit_prices_and_safe_context_bound(self):
        for config in (AstraConfig(max_cost_usd=100),
                       self.prices(max_cost_usd=100, max_input_tokens=1000),
                       self.prices(max_cost_usd=100, model="some-other-model")):
            fake = FakeTransport()
            with self.assertRaises(ValueError):
                self.client(fake, config=config).predict(observation(), self.grid)
            self.assertEqual(fake.calls, [])

    def test_cost_reservation_prevents_dispatch_and_uses_actual_usage_afterwards(self):
        fake = FakeTransport()
        with self.assertRaises(BudgetExceeded):
            self.client(fake, config=self.prices(max_cost_usd=0.01)).predict(observation(), self.grid)
        self.assertEqual(fake.calls, [])
        fake = FakeTransport(envelope(prediction(self.grid)))
        client = self.client(fake, config=self.prices(max_cost_usd=50))
        client.predict(observation(), self.grid)
        self.assertAlmostEqual(client.budget_snapshot()["cost_usd_upper_bound"],
                               (100 * 12.5 + 50 * 50) / 1_000_000)
        # Reasoning tokens are already included in 50 output tokens, not added twice.
        self.assertEqual(client.budget_snapshot()["reasoning_tokens"], 30)

    def test_unknown_usage_retains_full_reservation(self):
        fake = FakeTransport(APIError("network outcome unknown"))
        client = self.client(fake, config=self.prices(max_cost_usd=50))
        with self.assertRaises(APIError):
            client.predict(observation(), self.grid)
        budget = client.budget_snapshot()
        self.assertAlmostEqual(budget["cost_usd_upper_bound"], budget["reservation_per_call_usd"])
        self.assertEqual(budget["unknown_usage_calls"], 1)
        with self.assertRaises(BudgetExceeded):
            client.predict(observation(), self.grid)
        self.assertEqual(len(fake.calls), 1)

    def test_schema_retry_also_obeys_cost_reservation(self):
        fake = FakeTransport(envelope({}))
        # Leave less than the actual first request's charge after the reservation.
        config = self.prices(max_cost_usd=(1_050_000 * 12.5 * 2 + 8192 * 50 * 1.5) / 1_000_000)
        client = self.client(fake, config=config)
        with self.assertRaises(BudgetExceeded):
            client.predict(observation(), self.grid)
        self.assertEqual(len(fake.calls), 1)

    def test_wall_budget_limits_timeout_and_discards_late_output(self):
        now = [10.]
        fake = FakeTransport(envelope(prediction(self.grid)))
        client = self.client(fake, config=AstraConfig(timeout_s=180, max_wall_time_s=8), clock=lambda: now[0])
        now[0] = 13.
        client.predict(observation(), self.grid)
        self.assertEqual(fake.calls[0][3], 5.)
        now[0] = 18.
        with self.assertRaises(BudgetExceeded):
            client.predict(observation(), self.grid)
        self.assertEqual(len(fake.calls), 1)

        now[0] = 20.
        def slow_transport(*args):
            now[0] += 10.
            return envelope(prediction(self.grid))
        client = self.client(slow_transport, config=AstraConfig(max_wall_time_s=5), clock=lambda: now[0])
        with self.assertRaisesRegex(BudgetExceeded, "waiting"):
            client.predict(observation(), self.grid)
        self.assertEqual(client.budget_snapshot()["input_tokens"], 100)

    def test_raw_logs_and_identity_do_not_reuse_stochastic_response(self):
        fake = FakeTransport(envelope(prediction(self.grid)), envelope(prediction(self.grid)))
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client(fake, log_dir=tmp)
            client.predict(observation(), self.grid)
            key = client.last_request_key
            client.predict(observation(), self.grid)
            self.assertEqual(key, client.last_request_key)
            self.assertEqual(len(fake.calls), 2)
            requests = sorted(Path(tmp).glob("*.request.json"))
            responses = sorted(Path(tmp).glob("*.response.json"))
            self.assertEqual(len(requests), 2)
            self.assertEqual(len(responses), 2)
            serialized = "".join(p.read_text() for p in Path(tmp).glob("*.json"))
            self.assertNotIn("test-only-not-a-key", serialized)
            self.assertEqual(json.loads(responses[0].read_text())["usage"]["output_tokens"], 50)
        payload = client.build_payload(observation(), self.grid)
        changed = copy.deepcopy(payload)
        changed["model"] = "an-explicit-other-model"
        self.assertNotEqual(client.request_key(payload, self.grid, "field"),
                            client.request_key(changed, self.grid, "field"))
        changed = copy.deepcopy(payload)
        changed["input"][0]["content"][0]["text"] += "modified task"
        self.assertNotEqual(client.request_key(payload, self.grid, "field"),
                            client.request_key(changed, self.grid, "field"))

    def test_waypoint_and_direct_modes(self):
        waypoint = prediction(self.grid)
        waypoint.pop("values")
        waypoint["waypoint_xyz"] = [0.1, 0.2, 0.3]
        direct = {"actions": [[0.] * 7 for _ in range(5)]}
        fake = FakeTransport(envelope(waypoint), envelope(direct))
        client = self.client(fake)
        self.assertEqual(client.predict(observation(), self.grid, "waypoint"), waypoint)
        self.assertEqual(client.predict(observation(), self.grid, "direct"), direct)
        for actions in ([[0.] * 7] * 4, [[2.] * 7] * 5, [[True] * 7] * 5):
            with self.assertRaises(SchemaError):
                validate_prediction({"actions": actions}, self.grid, "direct")

    def test_resume_preserves_budget_and_conservatively_charges_crash_gap(self):
        config = self.prices(max_api_calls=3, max_cost_usd=50)
        fake = FakeTransport(envelope(prediction(self.grid)), envelope(prediction(self.grid)))
        with tempfile.TemporaryDirectory() as tmp:
            first = self.client(fake, config=config, log_dir=tmp)
            first.predict(observation(), self.grid)
            snapshot = first.budget_snapshot()
            first.predict(observation(), self.grid)  # Crash before this ledger is persisted.
            resumed_transport = FakeTransport()
            resumed = self.client(resumed_transport, config=config, log_dir=tmp)
            resumed.load_budget_snapshot(snapshot)
            self.assertEqual(resumed.budget_snapshot()["api_calls"], 1)
            self.assertEqual(resumed.reconcile_request_logs(), 1)
            state = resumed.budget_snapshot()
            self.assertEqual(state["api_calls"], 2)
            self.assertEqual(state["unknown_usage_calls"], 1)
            self.assertAlmostEqual(state["cost_usd_upper_bound"],
                                   snapshot["cost_usd_upper_bound"] + state["reservation_per_call_usd"])
            self.assertEqual(resumed.reconcile_request_logs(), 0)
            with self.assertRaises(BudgetExceeded):
                resumed.predict(observation(), self.grid)
            self.assertEqual(resumed_transport.calls, [])

    def test_resume_rejects_changed_config_or_missing_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.client(FakeTransport(envelope(prediction(self.grid))), log_dir=tmp)
            first.predict(observation(), self.grid)
            snapshot = first.budget_snapshot()
            changed = self.client(FakeTransport(), config=AstraConfig(max_api_calls=10), log_dir=tmp)
            with self.assertRaisesRegex(ValueError, "configuration"):
                changed.load_budget_snapshot(snapshot)
            resumed = self.client(FakeTransport(), log_dir=tmp)
            resumed.load_budget_snapshot(snapshot)
            next(Path(tmp).glob("*.request.json")).unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resumed.reconcile_request_logs()

    def test_long_context_pricing_applies_to_full_request_and_reservation(self):
        expected_reservation = (1_050_000 * 12.5 * 2 + 8192 * 50 * 1.5) / 1_000_000
        for inp, input_multiplier, output_multiplier in ((272_000, 1, 1),
                                                        (272_001, 2, 1.5),
                                                        (1_050_000, 2, 1.5)):
            with self.subTest(input_tokens=inp):
                response = envelope(prediction(self.grid))
                response["usage"]["input_tokens"] = inp
                fake = FakeTransport(response)
                client = self.client(fake, config=self.prices(max_cost_usd=50))
                self.assertAlmostEqual(client.budget_snapshot()["reservation_per_call_usd"], expected_reservation)
                client.predict(observation(), self.grid)
                self.assertAlmostEqual(client.budget_snapshot()["cost_usd_upper_bound"],
                                       (inp * 12.5 * input_multiplier + 50 * 50 * output_multiplier) / 1_000_000)
        fake = FakeTransport()
        client = self.client(fake, config=self.prices(max_cost_usd=25))
        with self.assertRaisesRegex(BudgetExceeded, "long-context"):
            client.predict(observation(), self.grid)
        self.assertEqual(fake.calls, [])

    def test_resume_rejects_obsolete_pricing_ledger(self):
        client = self.client(FakeTransport(), config=self.prices(max_cost_usd=50))
        snapshot = client.budget_snapshot()
        snapshot["snapshot_version"] = 1
        with self.assertRaisesRegex(ValueError, "version"):
            client.load_budget_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
