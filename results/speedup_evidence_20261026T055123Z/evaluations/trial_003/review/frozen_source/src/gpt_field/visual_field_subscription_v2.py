"""Opt-in zero-shot Hybrid v2 contract; existing v1 transport and prompts stay intact."""
from __future__ import annotations

import copy
import json

from .astra import SchemaError, _canonical
from .visual_field_subscription import (
    VisualFieldSubscriptionClient, _array, _validate,
    output_schema as output_schema_v1, shared_guidance as guidance_v1,
)

VERSION = "visual-field-rgb-joints-v2"
HISTORY_VERSION = "visual-v2-recent-two-turns-observable-block-memory-v1"


def output_schema(controller="hybrid"):
    if controller != "hybrid":
        raise ValueError("Visual v2 is an opt-in Hybrid-only experiment")
    result = output_schema_v1("hybrid")
    for view in ("external", "wrist"):
        reference = result["properties"]["visual_references"]["properties"][view]["anyOf"][0]
        reference["properties"]["target_anchor_uv"] = _array(2, 0, 1)
        if "target_anchor_uv" not in reference["required"]:
            reference["required"].append("target_anchor_uv")
    return result


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError("Visual v2 rollout is zero-shot; demonstrations are not accepted")
    return guidance_v1(task, False) + (
        "V2 ANCHOR CONTRACT: target_anchor_uv is a real, currently visible, textured landmark on the "
        "target object/surface, with stable target_id. target_point_uv is the desired projection of "
        "the SAME moving point, and can be empty space or a textureless virtual location. The local "
        "controller tracks target_anchor_uv and preserves target_point_uv minus target_anchor_uv; it "
        "never matches a patch at a virtual goal. These are different meanings even when the two UV "
        "values happen to coincide. Keep moving_id and target_id tied to the same physical landmarks "
        "across observations; do not silently switch to a nearby object or rename a failed hypothesis. "
        "V2 MEASURED RECOVERY: unavailable tracks or image-response models require information, not "
        "another nominal completion statement. Supply corrected visible anchors or choose a short "
        "Direct probe justified by BOTH current images: small translation at stable orientation and "
        "gripper state, then compare measured EEF/joint motion and image response. Keep per-action "
        "translation norm at most 0.1 normalized (5 mm controller delta target) for such probes. "
        "Do not sweep unknown space, assume global upward is clear, or repeat an unsupported direction. "
        "A drawer can require a different view/direction; there is no automatic upward recovery. "
        "If task_stagnation.active is true, address its specific missing measurement: select a bounded "
        "nonzero information-gathering viewpoint/alignment action or a genuinely changed grasp/interaction "
        "hypothesis supported by RGB, and state what observable difference you expect. Do not respond "
        "with the same all-zero verify/hold or merely a new hypothesis_id. If movement is not visibly "
        "justified, say exactly which anchor/occlusion prevents it and revise the visual references; "
        "uncertainty must remain explicit. Stage labels and 'finished' claims never resolve a stall. "
        "V2 ATTACHMENT: local Field transport or held-object alignment requires independently tracked "
        "payload and gripper co-motion during a measured test motion. Use moving_kind=payload and "
        "the same physical object feature for the image goal; a tool feature cannot substitute for "
        "the held object's placement. Missing co-motion requests a short model-selected Direct test "
        "lift/regrasp or better observation, not unverified Field transport. Loss of identity or slip "
        "invalidates prior evidence. V2 PLACEMENT: observed image alignment after opening and visible "
        "object/gripper separation over several observations are only image evidence. They do not "
        "prove 3D support, containment, release stability or task success. When placement evidence is "
        "uncertain, actively improve the observation or correct alignment; repeated holds do not "
        "increase confidence by themselves. No task-completion flag or episode cap is used. "
    )


def _text_record(message):
    content = message.get("content", [])
    if message.get("role") != "user" or not isinstance(content, list):
        return None
    for part in content:
        if part.get("type") == "input_text":
            try:
                value = json.loads(part.get("text", ""))
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and "type" in value:
                return value
    return None


def compact_history(messages):
    """Retain two full recent turns, summarize older measured blocks, leave logs alone."""
    observations = [i for i, message in enumerate(messages)
                    if (_text_record(message) or {}).get("type") == "observation"]
    if len(observations) <= 2:
        return copy.deepcopy(messages)
    cutoff = observations[-2]
    old, recent = messages[:cutoff], messages[cutoff:]
    memory = {"type": "visual_history_summary", "protocol": HISTORY_VERSION,
              "observations_summarized": 0, "execution_blocks_summarized": 0,
              "executed_actions_summarized": 0, "measured_path_m": 0., "commanded_path_m": 0.,
              "first_observation_index": None, "last_observation_index": None,
              "blocks_by_phase": {}, "last_observable_block": None,
              "task_success_certified": False,
              "retention": "Two complete recent request turns and all subsequent feedback retained. "
                           "Older model outputs and observations are represented by observable block memory; "
                           "raw requests, responses and transitions remain in local logs."}
    for message in old:
        record = _text_record(message)
        if record is None:
            continue
        kind = record.get("type")
        if kind == "visual_history_summary":
            for key in ("observations_summarized", "execution_blocks_summarized", "executed_actions_summarized",
                        "measured_path_m", "commanded_path_m"):
                memory[key] += record[key]
            memory["first_observation_index"] = record["first_observation_index"]
            memory["last_observation_index"] = record["last_observation_index"]
            memory["blocks_by_phase"] = copy.deepcopy(record["blocks_by_phase"])
            memory["last_observable_block"] = copy.deepcopy(record["last_observable_block"])
        elif kind == "observation":
            memory["observations_summarized"] += 1
            index = record["observation_index"]
            if memory["first_observation_index"] is None:
                memory["first_observation_index"] = index
            memory["last_observation_index"] = index
        elif kind == "visual_execution_block":
            memory["execution_blocks_summarized"] += 1
            memory["executed_actions_summarized"] += record["steps"]
            for key in ("measured_path_m", "commanded_path_m"):
                memory[key] += record[key]
            phase = record["phase"]
            memory["blocks_by_phase"][phase] = memory["blocks_by_phase"].get(phase, 0) + 1
            memory["last_observable_block"] = {key: copy.deepcopy(record[key]) for key in
                ("chunk_index", "last_step", "after_state", "phase", "target_id", "hypothesis_id",
                 "controller", "refresh_reason", "task_stagnation", "placement_evidence") if key in record}
    return [{"role": "user", "content": [{"type": "input_text", "text": _canonical(memory)}]}] + copy.deepcopy(recent)


_FORBIDDEN = {"object_poses", "object_positions", "depth", "camera_calibration", "environment_calibration",
              "contacts", "contact", "reward", "rewards", "libero_success", "evaluator", "simulator_status",
              "success", "task_complete", "segmentation", "ground_truth"}


def _observable_only(value):
    if isinstance(value, dict):
        if set(value) & _FORBIDDEN:
            raise ValueError("Privileged evaluator/environment fields cannot enter visual v2 context")
        if value.get("task_success_certified", False) is not False or value.get("clearance_certified", False) is not False:
            raise ValueError("Observable feedback cannot certify task success or 3D clearance")
        for item in value.values():
            _observable_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _observable_only(item)


class VisualFieldSubscriptionClientV2(VisualFieldSubscriptionClient):
    history_protocol = HISTORY_VERSION

    def __init__(self, config=None, *, controller="hybrid", task_key="bowl_plate", demonstration=None, **kwargs):
        if controller != "hybrid" or demonstration is not None:
            raise ValueError("Visual v2 is zero-shot Hybrid only; no Direct arm or demonstration")
        super().__init__(config, controller=controller, task_key=task_key, demonstration=None, **kwargs)

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        _observable_only(self.observable_context)
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec, False) + (
            "HYBRID V2: visual_field supplies field_intent and direct_actions=null; direct supplies "
            "field_intent=null and exactly five normalized actions. Use Direct for contact and short "
            "observation probes. Moving Direct contact chunks execute unchanged. Only an already "
            "observed persistent stall can interrupt the unused suffix of an entirely zero-action "
            "chunk; that suffix is logged as unexecuted. Local Field actions require observed response "
            "support, reliable image tracks and payload attachment where needed; there is no blind "
            "bootstrap. Field intent is a local Cartesian prior, not calibrated scene geometry. "
            "Field translates while holding the current orientation: orientation residuals up to "
            "0.01 rad use zero rotation; use Direct for larger reorientation before resuming Field. "
            "Refresh/recovery thresholds are local control decisions, never task stop conditions. "
            "History retains two complete recent request turns and deterministic cumulative observable "
            "execution-block memory. Active failed-hypothesis, recovery and placement evidence is "
            "supplied again in current observable_context; complete raw records remain local. "
            "Return only strict JSON; never output a task-completion flag.")
        payload["input"] = compact_history(payload["input"])
        payload["text"]["format"].update(name="visual_field_rgb_joints_v2", schema=output_schema())
        return payload

    def validate_response(self, data, grid):
        _validate(data, output_schema())
        legacy = copy.deepcopy(data)
        for reference in legacy["visual_references"].values():
            if reference is not None:
                reference.pop("target_anchor_uv")
        # Retain v1 branch/quaternion/box semantics without accepting its old seed contract.
        super().validate_response(legacy, grid)
        return copy.deepcopy(data)

    def record_block(self, record):
        _observable_only(record)
        super().record_block(record)

    def history_stats(self):
        result = super().history_stats()
        result.update(protocol=HISTORY_VERSION, text_history="recent_two_complete_turns_plus_observable_block_memory",
                      raw_logs="retained_unmodified", task_success_certified=False)
        return result
