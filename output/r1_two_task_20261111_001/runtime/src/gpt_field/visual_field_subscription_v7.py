"""Opt-in actionable reference repair and bounded subscription stream recovery.

V6 wire validation, persistent references and controller meanings are unchanged.
One observation permits at most one semantic repair and one missing-completion
retry, sharing the same request budget and never executing a partial response.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .action import _rotvec_quaternion
from .astra import APIError, SchemaError, _canonical
from .hybrid_subscription import ChatGPTPlanHybridClient
from .observation import PolicyObservation
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v6 import VisualFieldSubscriptionClientV6, output_schema
from .visual_reference_overlay import append_reference_diagnostic

VERSION = "visual-field-rgb-joints-v7-reference-health-bounded-stream-retry-v1"
HISTORY_VERSION = "visual-v7-reference-health-recent-two-turns-v1"


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError("Visual v7 is zero-shot Hybrid only")
    return (
        f"Task: {task.prompt}. ZERO-SHOT; no demonstration. Inputs are two RGB views, measured EEF "
        "xyz/axis-angle/two-finger state, seven arm joint positions/velocities and observable feedback. "
        "No depth, supplied calibration, object truth, contacts, reward or evaluator success. Preserve "
        "identities. Avoid unintended robot/payload collision and obstacle displacement; necessary "
        "finger interaction and intended support contact are allowed. Account for arm, fingers and "
        "payload extent. Image overlap never certifies clearance or completion. "
        "ROUTING: visual_field performs substantial noncontact movement; direct supplies exactly five "
        "actions for terminal contact correction, reorientation or a short attachment test; gripper "
        "settles an explicit open/close at the current pose. Populate only the selected branch; others "
        "null. A tracking stop is not evidence of physical misalignment or reason to retreat. "
        "FIELD: goal_offset_m translates EEF from its CURRENT measured position in robot/world axes, "
        "+/-0.08 m/axis. Local 33-cubed, 0.005 m goal-distance planning avoids your obstacle hypotheses "
        "and follows measured poses without repeated model calls. orientation_command and "
        "gripper_command must be hold. motion_axes=xyz allows spatial motion; xy holds initial height "
        "and requires zero goal z; z holds initial XY and requires zero goal x/y. Do not force a route "
        "outside these constraints. Use max_actions=80 and command_limit_m=0.012..0.02 in visible clear "
        "space; use shorter bounded intents and 0.003..0.006 near interaction. max_actions=10..120; "
        "max_travel_m<=0.16. Arrival or contradictory evidence ends reuse early. Supply honest tolerance, "
        "radius and uncertainty. Field phases: approach, transport, align, release, recover. Field "
        "release requires z-only motion and held grip. "
        "GEOMETRY: obstacles/unknown_regions are RGB hypotheses, world-axis-aligned boxes with stable "
        "unique IDs, center_offset_m from the SAME current EEF, positive half_extents_m and uncertainty_m. "
        "Local planning inflates by radius/uncertainty. Preserve static world positions as EEF moves; "
        "do not delete obstacles, lower uncertainty or invent occlusion walls to force a route. "
        "REFERENCE CONTRACT: normalized UV has top-left origin, u right/v down. Each external/wrist "
        "reference selects seed or retain. Seed supplies physical points with nullable ordered "
        "positive-area [u0,v0,u1,v1] ROIs containing their points. Prefer textured rigid regions. "
        "Retain requires seed=null and EXACT existing identities/moving_kind; it reuses tracked "
        "coordinates, never guessed replacements. Retain cannot restore lost tracks. External moving "
        "point belongs to actual tool/payload; target anchor is a distinct stationary landmark. "
        "Payload requires an independent tool point/track. External goal_offset_uv is desired moving "
        "projection minus anchor projection; wrist desired_target_uv is desired target bearing. "
        "Wrist target_id always matches top-level target_id; Field external target_id also matches. "
        "Direct/gripper may use a separate stationary anchor while interacting with the moving payload. "
        "Field needs external references; wrist may be null. Preserve payload/tool correspondences "
        "through destination changes. Seed resets evidence. "
        "REFERENCE HEALTH: reliable patch matching does not prove the patch moves with the controlled "
        "body. Read visual_feedback.visual.reference_health and current tracked positions. If "
        "repair_required, change the physical correspondence or choose a bounded informative view; "
        "retain, renaming and repeated tiny calibration do not repair it. For tool tracking select "
        "the distal moving gripper housing/finger, not fixed robot base. An optional annotated current "
        "external image labels M=declared moving feature, T=stationary anchor, G=independent tool. "
        "These labels are prior claims, not verified object identities. Check them against both views. "
        "New useful references can gather response evidence during normal task progress. "
        "INTERACTION: perfect 2D centering high above a target is unnecessary and can mislead through "
        "parallax. After coarse alignment use bounded Field approach and reobserve. Navigation centering "
        "or stalled descent does not establish a graspable feature between the jaws. Use both views "
        "to select a graspable surface/handle/rim and any necessary Direct5 contact-pose correction "
        "before closure. Gripper is allowed only in grasp/release: command=open|close, max_actions=10..30. "
        "Measured finger settling is not attachment. After closing, use a short Direct test with "
        "independent payload/tool references; Field transport requires observed co-motion and payload "
        "moving_kind. Unobserved co-motion after a tiny/noisy test is inconclusive: if both views and "
        "finger state still support a grasp, extend one bounded test or change its observable direction. "
        "Reopen and change grasp geometry on visible miss/slip evidence, not insufficient samples. "
        "PLACEMENT: compare payload/support using both views and observed grasp offset. Use Field "
        "align/xy for needed horizontal correction FIRST, then Field release/z for bounded support "
        "approach. Direct release requires zero XYZ in all five actions; reorientation may remain. "
        "Choose gripper open from current support evidence; reassess after separation. No automatic "
        "release or stage completion. "
        "DIRECT: five unchanged [dx,dy,dz,rx,ry,rz,grip] actions in [-1,1], 20 Hz; translation 1 requests "
        "0.05 m from measured pose, rotation 1 requests 0.5 rad world axis-angle. Grip +1 closes, -1 "
        "opens, 0 holds remembered intent. Actual motion may be smaller. Model waiting advances no "
        "physics. Strict JSON, concise reasons, no completion flag. Two recent turns plus observable "
        "memory are supplied; raw records remain local."
    )


class VisualFieldSubscriptionClientV7(VisualFieldSubscriptionClientV6):
    history_protocol = HISTORY_VERSION

    def reset_history(self):
        super().reset_history()
        self._stream_retry_requests = 0

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v7", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return append_reference_diagnostic(payload, observation, self.observable_context)

    def _record_stream_retry(self):
        """Persist safe local metadata alongside the original failed attempt."""
        record = {
            "protocol": VERSION, "status": "bounded_retry_scheduled",
            "error_code": "stream_missing_completion", "observation_index": self._observation_count,
            "failed_request_key": self.last_request_key, "failed_request_ordinal": self._api_calls,
            "same_observation": True, "actions_executed_from_failed_response": 0,
            "max_stream_retries_per_observation": 1, "max_total_requests_per_observation": 3,
            "budget": self.budget_snapshot(),
        }
        path = (Path(self.last_log_path).with_suffix(".v7_retry.json")
                if self.last_log_path is not None else None)
        self._write_log(path, record)

    def predict(self, observation, grid=None, mode="hybrid"):
        # Call the single-attempt primitive directly: nesting V5.predict would
        # reset its semantic-repair allowance after a stream retry. Both retry
        # allowances below belong to this same observation, at most three calls.
        # The policy's outer timer includes all attempts and every raw attempt
        # reserves budget, logs independently and charges unknown stream usage.
        with self._history_lock:
            original_context = self.observable_context
            initial_observation_count = self._observation_count
            semantic_used = stream_used = False
            self._v5_repair_record = None
            try:
                while True:
                    try:
                        result = ChatGPTPlanHybridClient.predict(self, observation, grid, mode)
                        break
                    except SchemaError:
                        if semantic_used or self._v5_repair_record is None:
                            raise
                        semantic_used = True
                        self._semantic_repair_requests += 1
                        self.observable_context = copy.deepcopy(self.observable_context)
                        self.observable_context["output_repair"] = copy.deepcopy(self._v5_repair_record)
                        self._v5_repair_record = None
                    except APIError as exc:
                        # A completed response followed by an API/model check
                        # failure is not eligible, even if a caller labels it
                        # with this code. Never retry HTTP/auth/quota failures.
                        if (stream_used or getattr(exc, "error_code", None) != "stream_missing_completion"
                                or getattr(exc, "http_status", None) is not None
                                or self.last_response is not None
                                or self._observation_count != initial_observation_count):
                            raise
                        stream_used = True
                        self._stream_retry_requests += 1
                        self._record_stream_retry()
                        self.observable_context = copy.deepcopy(self.observable_context)
                        self.observable_context["transport_retry"] = {
                            "attempt": 1, "error_code": "stream_missing_completion",
                            "instruction": "The preceding stream ended without a completed response. "
                                "No actions from it were executed. This is the SAME observation, "
                                "not new physical evidence. Return a complete response; partial "
                                "output is unavailable. This is the only missing-stream retry.",
                        }
            finally:
                self.observable_context = original_context
                self._v5_repair_record = None
        if result["controller"] == "visual_field":
            state = PolicyObservation.from_dict(observation).state
            intent = result["field_intent"]
            intent.pop("orientation_command")
            intent["orientation_target_xyzw"] = _rotvec_quaternion(np.asarray(state[3:6], float)).tolist()
        return result

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=HISTORY_VERSION, stream_retry_requests=self._stream_retry_requests,
                      max_requests_per_observation=3)
        return result
