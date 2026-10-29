"""Local ARX payload adaptation over the frozen UC7.1 OAuth client.

Construction/payload inspection is offline. Only request_plan/request_agent send
requests, and both require the POSIX main thread. No simulator is imported.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import difflib
import json
import threading

from .history import require_frozen_core, trim_visual_sets
from .planning import (NativePlanLock, PUBLIC_HOME_RULE, PLAN_VERSION, canonical_sha256, direction_frame_from_calibration,
                       native_plan_schema, planning_images, png_url, rgb_image, validate_plan)
from .schema import (CAMERAS, ExecutionLimits, SCHEMA_VERSION, capabilities, decision_schema,
                     public_json, validate_decision, validate_observation, validate_robot, validate_calibration,
                     validate_feedback, validate_controller_capabilities, calibration_binding)

require_frozen_core()
from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client
from gpt_field.agent_speedup_v2_user_check_client import UserCheckV2Client, user_check_worked_example
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client, AGENT_INSTRUCTIONS
from gpt_field.agent_speedup_v2_user_check_context import size
from gpt_field.agent_speedup_v2_user_check_spatial import region_directions, validate_view_frame
from gpt_field.agent_speedup_v2_user_check_transport import configure_client_transport
from gpt_field.astra import SchemaError
from gpt_field.ordered_transport import ordered_sse_transport, ordered_dumps

CLIENT_VERSION = "robodojo-uc71-client-1"
HISTORY_VERSION = "uc71-all-text-six-native-three-view-sets-1"


def _replace(text, old, new):
    if old not in text:
        raise RuntimeError("Frozen prompt no longer matches the reviewed adapter transformation")
    return text.replace(old, new)


def executive_instructions(limits=ExecutionLimits()):
    text = AGENT_INSTRUCTIONS
    text = _replace(text, "one Franka Panda with a parallel-jaw\nPanda gripper.",
                    "two native ARX X5 arms, each with a parallel-jaw gripper.")
    text = _replace(text, "The two images are external\nand wrist, upright and uncropped.",
                    "The three images are head, left wrist and right wrist, upright, unflipped and uncropped native 640x480 RGB.")
    text = text.replace("observation pairs", "synchronized three-view observation sets")
    text = _replace(text, "motion, macro and execution. All keys are required.",
        "arm_id, motion and execution. All keys are required. Select arm_id=left or right for execute/motion; all segments belong to that one arm. For observe/finish arm_id=null. The other arm holds its measured handoff reference and keeps its requested gripper command.")
    start = text.index("macro: motion=null, execution=null,")
    end = text.index("observe: motion=null", start)
    text = text[:start] + text[end:]
    text = text.replace("macro=null, ", "").replace(", macro=null", "")
    text = text.replace("not resume a native macro or obtain a native verified holding flag", "not obtain a native verified holding flag")
    text = text.replace("absolute right_hand wxyz", "absolute selected tool-frame wxyz")
    text = text.replace("absolute world grip-site target", "absolute world selected tool-frame target")
    text = text.replace("execution.segments contains 1..6", f"execution.segments contains 1..{limits.max_segments}")
    text = text.replace("max_steps is 1..120 per stage, with sum(max_steps)<=240", f"max_steps is 1..{limits.max_segment_steps} per stage, with sum(max_steps)<={limits.max_execution_steps}")
    text = text.replace("At 20 Hz", "At 25 Hz").replace("steps is 1..5 at 20 Hz", f"steps is 1..{limits.direct_max_steps} at 25 Hz")
    text = _replace(text, "profile=transit allows at most 0.30 m / 1.2 rad from a stage's measured start;\nprofile=contact allows at most 0.08 m / 0.5 rad. The unchanged OSC feedback loop\ncaps each action at 0.02 m translation (0.005 m for contact) and 0.1 rad rotation.",
        f"profile=transit allows at most {limits.transit_translation_m:g} m / {limits.transit_rotation_rad:g} rad from a stage's measured start;\nprofile=contact allows at most {limits.contact_translation_m:g} m / {limits.contact_rotation_rad:g} rad. The robot-only controller uses bounded measured-error joint targets; its declared capabilities give the reviewed physical limits.")
    text = _replace(text, "locates gripper0_grip_site between the fingers, not the hand origin. The optional\nquaternion_wxyz is an absolute robot0_right_hand world orientation, or null to",
        "locates the selected arm's calibrated tool frame, not link6. The optional\nquaternion_wxyz is an absolute tool-frame world orientation, or null to")
    text = _replace(text, "The target is at most 0.05 m translation and 0.35 rad\nrotation", f"The target is at most {limits.direct_translation_m:g} m translation and {limits.direct_rotation_rad:g} rad\nrotation")
    text = _replace(text, "pose error, capped at 0.02 m translation and 0.1 rad rotation.",
                    "pose error through the declared bounded robot-only controller.")
    text = text.replace("external image (image 1)", "head image (image 1)")
    text = text.replace("external camera", "head camera").replace("external view", "head view")
    text = text.replace("Never judge them in the wrist image.", "Never redefine them using either wrist image.")
    start = text.index("Camera position_world_m and rotation_world_from_camera")
    text = text[:start] + """calibration.cameras[name].world_from_optical maps optical camera coordinates to the fixed world frame. Optical +X is image-right, +Y image-down, +Z forward. For Z>0, pixels follow [u,v,1] proportional to K*[X,Y,Z]; K is intrinsics. Use each image's synchronized transform. Pixels specify rays, not known 3-D points; infer depth from views and observed motion. Do not use image axes as world axes or assume image center is a grasp point.
calibration.tools contains fixed tool/gripper calibration only. tool_pose_wxyz already locates that calibrated tool, so do not add its link6 offset again. finger_position_m contains both measured joints. finger_gap_m is their sum, a joint-separation proxy, NOT a calibrated inner-pad clearance or proof of attachment. Keep each arm's commanded gripper setting separate from measured fingers.
Native Panda macros and their geometric observers are unavailable in this explicitly reduced pilot. Use generic fragments/direct motions; no task-specific grasp, insertion, lever or homing routine is supplied.
The first execution observation contains the measured starting robot poses and remains in this attempt's text history. """ + PUBLIC_HOME_RULE + """
The step field labels model observations; it is not an episode action counter. No remaining native budget or evaluator progress is supplied. Tool geometry does not prove clearance or contact. Judge both arms, carried objects and nearby objects in all three views. Keep task reasoning concise and output JSON only."""
    return text.strip()


def prompt_diff(limits=ExecutionLimits()):
    return "".join(difflib.unified_diff(AGENT_INSTRUCTIONS.splitlines(True),
        executive_instructions(limits).splitlines(True), fromfile="frozen_uc71", tofile="native_dual_arx"))


def _main_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Subscription requests must run on the parent POSIX main thread")


def _transport(client, request_deadline_s, stream_progress):
    if client.config.model != "gpt-6-astra":
        raise ValueError("The native pilot requires exactly gpt-6-astra")
    if request_deadline_s != 180:
        raise ValueError("The reviewed pilot uses a 180-second per-request deadline")
    if client.transport is ordered_sse_transport:
        return configure_client_transport(client, request_deadline_s=180, stream_progress=stream_progress)
    if getattr(client.transport, "offline_mock", False) is not True:
        raise ValueError("Only frozen production OAuth transport or explicit offline mock is allowed")
    return {"enabled": False, "offline_mock": True, "request_deadline_s": 180}


class NativePlanClient(UserCheckV2Client):
    prompt_version = PLAN_VERSION

    def __init__(self, *args, request_deadline_s=180, stream_progress=True, **kwargs):
        if kwargs.get("reasoning_effort", "high") != "high":
            raise ValueError("The native pilot requires high reasoning effort")
        kwargs["reasoning_effort"] = "high"
        super().__init__(*args, **kwargs)
        text = self._instruction_text["plan"]
        text = _replace(text, "a robot arm with one parallel-jaw gripper (about 79 mm maximum opening) that starts with the hand pointing straight down.",
                        "two native ARX X5 arms with parallel-jaw grippers; the images show their actual starting orientations.")
        text = _replace(text, "Four images, all upright (not mirrored): 1 the external camera, clean; 2 the same with an 8x8 grid of numbered cells; 3 the wrist camera, clean; 4 the same with the grid.",
                        "Six native images, upright and unmirrored: 1 head clean, 2 head grid, 3 left wrist clean, 4 left wrist grid, 5 right wrist clean, 6 right wrist grid.")
        text = _replace(text, "w is the part's cell in image 4 and x its cell in image 2, 0 when it is not visible in that image.",
                        "x is the part's head cell in image 2, w its left-wrist cell in image 4, and r its right-wrist cell in image 6. Each is 0 when the part is not visible in that view.")
        example = user_check_worked_example()
        revised_example = deepcopy(example)
        for part in revised_example["parts"]:
            part["r"] = 0
        text = text.replace(ordered_dumps(example), ordered_dumps(revised_example))
        text += "\nBoth wrist images are unrotated native views; neither is perspective-aligned to the head image. The robot's own fingers are not task-object parts. " + PUBLIC_HOME_RULE
        self._instruction_text["plan"] = text
        self._schemas["plan"] = native_plan_schema()
        self.transport_configuration = _transport(self, request_deadline_s, stream_progress)

    def build_payload(self, request, grid=None, mode="plan"):
        if mode != "plan" or not isinstance(request, dict) or set(request) != {"fields", "images"}:
            raise ValueError("Only the allowlisted initial-planning request is supported")
        if set(request["fields"]) != {"holding", "status", "image_instructions", "wrist_quarter_turns"}:
            raise ValueError("Initial planning cannot receive execution sensor fields")
        if request["fields"]["holding"] is not None or request["fields"]["status"] is not None:
            raise ValueError("Initial planning has no runtime holding/status evidence")
        if len(request["images"]) != 6:
            raise ValueError("Initial planning requires three clean and three grid images")
        for _, image in request["images"]:
            rgb_image(image)
        public_json(request["fields"])
        return super().build_payload(request, grid, mode)

    def validate_response(self, data, grid=None):
        if self._request_kind != "plan":
            raise SchemaError("This client supports only the initial planning request")
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        try:
            return validate_plan(data, semantic=False)
        except (ValueError, TypeError, KeyError) as exc:
            raise SchemaError(f"plan output: {exc}") from None

    def planning_request(self, images, *, wrist_quarter_turns=None):
        pictures = planning_images(images, wrist_quarter_turns=wrist_quarter_turns)
        return {"fields": {"holding": None, "status": None,
            "image_instructions": "Head clean/grid, left wrist clean/grid, right wrist clean/grid; 8x8 row-major cells; native unmirrored RGB.",
            "wrist_quarter_turns": {"left_wrist": 0, "right_wrist": 0}}, "images": pictures}

    def request_plan(self, images, *, wrist_quarter_turns=None):
        _main_thread()
        request = self.planning_request(images, wrist_quarter_turns=wrist_quarter_turns)
        self.last_request_meta = {"kind": "plan", "schema_version": PLAN_VERSION, "planning_images": 6}
        return self.predict(request, mode="plan")

    def request_key(self, payload, grid=None, mode="plan"):
        if mode != "plan":
            raise ValueError("Only initial plan is supported")
        return canonical_sha256({"wire": ordered_dumps(payload), "client": CLIENT_VERSION, "schema": PLAN_VERSION})


class NativeExecutiveClient(AgentFirstV2Client):
    prompt_version = "robodojo-uc71-executive-prompt-1"
    history_protocol = HISTORY_VERSION

    def __init__(self, *args, approved_plan, direction_frame, limits=None,
                 controller_capabilities=None, request_deadline_s=180, stream_progress=True, **kwargs):
        self.limits = limits or ExecutionLimits()
        self.controller_capabilities = validate_controller_capabilities(controller_capabilities or {})
        self._tool_binding = None
        self._last_step = None
        self._last_decision = None
        self._native_instructions = executive_instructions(self.limits)
        if "lock" in kwargs:
            raise ValueError("The native client constructs its own validated plan lock")
        lock = NativePlanLock(approved_plan, task_prompt=kwargs["task_prompt"], direction_frame=direction_frame)
        if kwargs.get("reasoning_effort", "high") != "high":
            raise ValueError("The native pilot requires high reasoning effort")
        kwargs["reasoning_effort"] = "high"
        super().__init__(*args, approved_plan=approved_plan, lock=lock, **kwargs)
        self.transport_configuration = _transport(self, request_deadline_s, stream_progress)

    def reset_history(self):
        super().reset_history()
        self._last_step = None
        self._tool_binding = None
        self._last_decision = None

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        result["retained_visual_sets"] = result.pop("retained_image_pairs", 0)
        result["views_per_visual_set"] = 3
        return result

    def check_observation(self, observation):
        return validate_observation(observation)

    def _execution_snapshot(self, value):
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != {"decision", "progress"}:
            raise ValueError("Execution requires the prior accepted decision and typed robot progress")
        command = validate_decision(value["decision"], self.limits)
        if self._last_decision is None or command != self._last_decision:
            raise ValueError("Execution command is not the prior accepted model decision")
        return {"decision": command, "progress": validate_feedback(value["progress"])}

    def _capabilities(self):
        result = capabilities(self.limits)
        result["controller"] = deepcopy(self.controller_capabilities)
        return result

    def _reviewed_contract(self):
        task = self.plan_lock.task_contract.to_dict()
        task["region_directions"] = region_directions(self.plan_lock.view_frame, self.plan_lock.plan().get("regions", []))
        return {"initial_plan": self.plan_lock.plan(), "task_contract": task,
                "authority": "goals_and_applicable_constraints; execution_methods_may_change"}

    def build_payload(self, request, grid=None, mode="plan"):
        if mode != "agent":
            raise ValueError("The native executive supports only agent decisions")
        if grid is not None and grid != self.grid:
            raise ValueError("Unexpected legacy grid")
        if not isinstance(request, dict) or set(request) != {"fields", "images"}:
            raise ValueError("Expected allowlisted public fields and three camera images")
        fields = request["fields"]
        required = {"step", "robot", "calibration", "context"}
        optional = {"reviewed_goal_contract"}
        if not isinstance(fields, dict) or not required <= set(fields) or set(fields) - required - optional:
            raise ValueError("Unexpected executive request fields")
        public_json(fields)
        if type(fields["step"]) is not int or fields["step"] != self._agent_decisions:
            raise ValueError("Invalid model observation index")
        validate_robot(fields["robot"])
        validate_calibration(fields["calibration"])
        if direction_frame_from_calibration(fields["calibration"]) != self.plan_lock.view_frame:
            raise ValueError("Head calibration differs from the reviewed frame")
        if self._tool_binding is not None and canonical_sha256(calibration_binding(fields["calibration"])) != self._tool_binding:
            raise ValueError("Fixed sensor/tool calibration changed")
        if not self._agent_history:
            if fields.get("reviewed_goal_contract") != self._reviewed_contract():
                raise ValueError("First request must contain exactly the approved goal contract")
        elif "reviewed_goal_contract" in fields:
            raise ValueError("The reviewed contract is retained from the initial accepted turn")
        context = fields["context"]
        expected_context = {"measured_state", "model_belief", "last_feedback", "execution", "capabilities",
                            "macro", "native_evidence", "task_success_source", "region_directions"}
        if not isinstance(context, dict) or set(context) != expected_context:
            raise ValueError("Executive context contains non-allowlisted fields")
        validate_feedback(context["last_feedback"])
        self._execution_snapshot(context["execution"])
        expected_directions = region_directions(self.plan_lock.view_frame, self.plan_lock.plan().get("regions", []))
        if (context["measured_state"] != {"robot": fields["robot"]}
                or context["model_belief"] != getattr(self, "_agent_latest_belief", None)
                or context["capabilities"] != self._capabilities()
                or context["region_directions"] != expected_directions
                or context["macro"] != {"available": False} or context["native_evidence"] != {"available": False}
                or context["task_success_source"] != "independent evaluator; unknown to policy"):
            raise ValueError("Executive context must match measured state and frozen client-owned metadata")
        if len(request["images"]) != 3:
            raise ValueError("Every executive visual set must contain three cameras")
        for name, (label, image) in zip(CAMERAS, request["images"]):
            if label != f"{name.replace('_', ' ')} native RGB, observation {fields['step']}":
                raise ValueError("Camera labels must match the declared synchronized view set")
            rgb_image(image)
        content = AgentSpeedUpV2Client._content(self, request, mode)
        self._request_kind = "agent"
        history = trim_visual_sets(self._agent_history + [{"role": "user", "content": content}])
        return {"model": self.config.model, "service_tier": "default", "store": False,
            "instructions": self._native_instructions, "input": history,
            "text": {"format": {"type": "json_schema", "name": "robodojo_uc71_dual_arm",
                                  "strict": True, "schema": decision_schema(self.limits)}},
            "stream": True, "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"]}

    def validate_response(self, data, grid=None):
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        try:
            return validate_decision(data, self.limits)
        except (ValueError, TypeError, KeyError) as exc:
            raise SchemaError(f"agent output: {exc}") from None

    def request_key(self, payload, grid=None, mode="agent"):
        return canonical_sha256({"wire": ordered_dumps(payload), "client": CLIENT_VERSION,
            "prompt": self.prompt_version, "history": HISTORY_VERSION, "schema": SCHEMA_VERSION,
            "limits": asdict(self.limits)})

    def agent_request(self, observation, *, feedback=None, execution=None):
        """Build a previewable allowlisted request without credentials or network."""
        obs = validate_observation(observation)
        self.plan_lock.check_active()
        if obs["step"] != self._agent_decisions:
            raise ValueError("step must equal the accepted model-decision count; native action counts are prohibited")
        if "direction_frame" in obs and validate_view_frame(obs["direction_frame"]) != self.plan_lock.view_frame:
            raise ValueError("Observation attempts to redefine reviewed direction semantics")
        if direction_frame_from_calibration(obs["calibration"]) != self.plan_lock.view_frame:
            raise ValueError("Native fixed head calibration differs from the reviewed frame")
        binding = canonical_sha256(calibration_binding(obs["calibration"]))
        if self._tool_binding is not None and self._tool_binding != binding:
            raise ValueError("Fixed sensor/tool calibration changed within the attempt")
        directions = region_directions(self.plan_lock.view_frame, self.plan_lock.plan().get("regions", []))
        context = {"measured_state": {"robot": obs["robot"]},
            "model_belief": deepcopy(getattr(self, "_agent_latest_belief", None)),
            "last_feedback": validate_feedback(feedback), "execution": self._execution_snapshot(execution),
            "capabilities": self._capabilities(),
            "macro": {"available": False}, "native_evidence": {"available": False},
            "task_success_source": "independent evaluator; unknown to policy",
            "region_directions": directions}
        fields = {"step": obs["step"], "robot": obs["robot"], "calibration": obs["calibration"], "context": context}
        if not self._agent_history:
            fields["reviewed_goal_contract"] = self._reviewed_contract()
        public_json(fields)
        if size(fields) > 56000:
            raise ValueError("Current public observation exceeds the frozen text-size boundary")
        images = [(f"{name.replace('_', ' ')} native RGB, observation {obs['step']}", png_url(obs["images"][name]))
                  for name in CAMERAS]
        return {"fields": fields, "images": images}

    def request_agent(self, observation, *, feedback=None, execution=None):
        _main_thread()
        request = self.agent_request(observation, feedback=feedback, execution=execution)
        self.last_request_meta = {"kind": "agent", "step": observation["step"],
                                  "protocol": HISTORY_VERSION, "image_history_limit": 6, "views": 3}
        result = self.predict(request, mode="agent")
        self._last_step = observation["step"]
        self._last_decision = deepcopy(result)
        self._tool_binding = canonical_sha256(calibration_binding(observation["calibration"]))
        return result
