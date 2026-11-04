"""UC7.1 arm fragments plus explicit, bounded native mobile actions.

No simulator imports or task-specific object coordinates occur in this module.
The arm7 decision and the additional mobile command are logged separately and
acknowledged at the same physical endpoint by the runner.
"""
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from gpt_field.agent_speedup_v2_user_check_agent import AgentFirstV2Policy
from gpt_field.agent_speedup_v2_user_check_agent_client import (
    AgentFirstV2Client, agent_schema, validate_agent_decision,
)
from gpt_field.agent_speedup_v2_user_check_client import UserCheckV2Client
from gpt_field.astra import SchemaError
from gpt_field.contact_skills import Primitive

VERSION = "rc365-uc71-mobile-1"
MACRO_REASON = "LIBERO native macros have not been validated for the mobile PandaOmron."
ROBOT_CONTEXT = """
ROBOCASA365 PandaOmron. Arm motion uses world-frame grip-site targets at
gripper0_right_grip_site; the arm controller retains UC7.1 transit/contact
fragments. The synchronized external camera is mounted on the robot. Camera
poses change with base/torso motion; always use the current calibration for
current images, while reviewed directional meanings retain their initial frame.
Native LIBERO macros are unavailable. Use execute, motion, observe or mobile.

Every response additionally requires mobile=null except for type=mobile.
For type=mobile set motion, macro, execution to null and supply
mobile={base:[forward,sideways,yaw],torso:up,steps:N}. All four commands are
normalized native controller inputs in [-0.2,0.2], and N is 1..5 at 20Hz.
Base translation axes follow the current robot base heading, as described in
measured_state.mobile; positive yaw is counterclockwise. Positive torso raises
the torso; its native delta scale is 0.05m, so 0.2 requests at most 0.01m per
step. Base commands are normalized native inputs; the controller also applies
the verified actuator ranges reported in measured_state.mobile. Do not treat
the normalized yaw value as an equal angular velocity. These are COMMAND scales,
not measured displacement guarantees. The arm follows the base in native mobile
mode, with zero arm delta and the previous finger command; no grasp/open/close
is implicit. Do not combine mobile and arm motion in the same response.
Inspect new RGB and measured robot feedback after every mobile chunk. Keep
the hand/carried object clear before moving the base or torso. Move only when
the visible free route supports it, and shorten or stop near obstacles. There
is no autonomous navigation planner, scene-truth clearance, or full-body
collision guarantee. Use mobile to reposition within visible space; never
infer that an arm failure proves a base path is safe. After moving, reassess
the obstacle map from current RGB. Feedback describes robot motion only.
""".strip()


def mobile_schema():
    value = {"type": "number", "minimum": -.2, "maximum": .2}
    props = {"base": {"type": "array", "items": value, "minItems": 3, "maxItems": 3},
             "torso": deepcopy(value),
             "steps": {"type": "integer", "minimum": 1, "maximum": 5}}
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def validate_mobile(value):
    if not isinstance(value, dict) or set(value) != {"base", "torso", "steps"}:
        raise ValueError("mobile requires exactly base, torso, steps")
    base = value["base"]
    if not isinstance(base, (list, tuple)) or len(base) != 3:
        raise ValueError("mobile.base needs three normalized commands")
    for number in [*base, value["torso"]]:
        if (isinstance(number, (bool, np.bool_)) or not isinstance(number, Real)
                or not math.isfinite(number) or abs(number) > .2):
            raise ValueError("mobile inputs must be finite real values in [-0.2,0.2]")
    if type(value["steps"]) is not int or not 1 <= value["steps"] <= 5:
        raise ValueError("mobile.steps must be an integer in 1..5")
    return {"base": [float(x) for x in base], "torso": float(value["torso"]),
            "steps": value["steps"]}


def rc365_schema():
    schema = agent_schema()
    schema["properties"]["type"]["enum"].append("mobile")
    schema["properties"]["mobile"] = {"anyOf": [mobile_schema(), {"type": "null"}]}
    schema["required"].append("mobile")
    return schema


def validate_decision(value):
    if not isinstance(value, dict) or set(value) != set(rc365_schema()["required"]):
        raise SchemaError("RoboCasa365 output requires exactly the schema fields, including mobile")
    original = deepcopy(value)
    arm = deepcopy(value)
    mobile = arm.pop("mobile")
    try:
        if arm.get("type") == "mobile":
            if any(arm.get(k) is not None for k in ("motion", "macro", "execution")):
                raise ValueError("mobile cannot also request an arm action or macro")
            original["mobile"] = validate_mobile(mobile)
            arm["type"] = "observe"
        elif mobile is not None:
            raise ValueError("mobile must be null for other action types")
        clean = validate_agent_decision(arm)
        original["obstacle_map"] = clean["obstacle_map"]
        return original
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"RoboCasa365 mobile command: {exc}") from None


class RoboCasa365PlanClient(UserCheckV2Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for kind in ("plan", "replan"):
            self._instruction_text[kind] += (
                "\nThe robot is a mobile PandaOmron with a Panda arm, base and torso. "
                "Plan the official language goal from the images. Base and torso "
                "repositioning are available during execution; no fixed task route "
                "or object coordinates are provided. This planning request has no camera calibration.")


class RoboCasa365Client(AgentFirstV2Client):
    def build_payload(self, request, grid=None, mode="plan"):
        payload = super().build_payload(request, grid, mode)
        if mode == "agent":
            payload["instructions"] = payload["instructions"].replace(
                "gripper0_grip_site", "gripper0_right_grip_site") + "\n" + ROBOT_CONTEXT
            payload["text"]["format"]["schema"] = rc365_schema()
            payload["text"]["format"]["name"] = "robocasa365_uc71_mobile"
        return payload

    def validate_response(self, data, grid=None):
        if self._request_kind != "agent":
            return super().validate_response(data, grid)
        self.last_output_key_order = list(data) if isinstance(data, dict) else None
        return validate_decision(data)


class MobilePrimitive(Primitive):
    def __init__(self, view, command, gripper):
        super().__init__("robocasa_mobile")
        self.command = validate_mobile(command)
        self.gripper = 1. if gripper == "close" else -1.
        self.start = deepcopy(view.observation.mobile_state)
        self.last = self.start
        self.mobile_command = {k: deepcopy(self.command[k]) for k in ("base", "torso")}

    def _next(self, view):
        return [0., 0., 0., 0., 0., 0., self.gripper]

    def next_action(self, view):
        # Native mobile mode handles goal refreshing. The arm-only Primitive
        # refresh adds a tiny rotation which must never enter a mobile command.
        if self.done:
            raise RuntimeError("mobile primitive has finished")
        if self._pending is not None:
            raise RuntimeError("observe_transition required before next mobile action")
        action = tuple(self._next(view))
        self._pending = (view.state.copy(), action)
        return action

    def _observe(self, before, action, after):
        self.last = deepcopy(after.observation.mobile_state)
        evidence = {"command": self.command, "before_mobile_state": self.start,
                    "after_mobile_state": self.last,
                    "semantic_result": "unknown; native command duration only"}
        if self.actions_executed >= self.command["steps"]:
            self._finish("succeeded", "bounded_mobile_chunk_complete", evidence)
        return evidence


class RoboCasa365Policy(AgentFirstV2Policy):
    def reset(self):
        self.mobile_actions = 0
        super().reset()

    def _capabilities(self):
        value = super()._capabilities()
        value["native_macros"] = {"implemented": [], "available": False, "reason": MACRO_REASON}
        value["mobile"] = {"version": VERSION, "steps": [1, 5], "input_bounds": [-.2, .2],
                           "gripper": "keep", "arm_delta": "zero_in_native_mobile_mode",
                           "collision_check": "none; use RGB", "schema": mobile_schema()}
        return value

    def _context(self, view):
        value = super()._context(view)
        value["measured_state"]["mobile"] = deepcopy(view.observation.mobile_state)
        return value

    def _apply_agent(self, answer, view):
        if answer["type"] == "macro":
            self._record_feedback({"action": "macro", "status": "unavailable",
                                   "reason": MACRO_REASON, "executed_steps": 0})
            self._need_agent = True
            return
        if answer["type"] != "mobile":
            return super()._apply_agent(answer, view)
        self._obstacle_map.update(answer.get("obstacle_map"), step=self.step_index)
        self._abandon_macro("model selected native mobile motion")
        self._mode = "direct"
        self._segment_start = {"step": self.step_index, "action_type": "mobile"}
        self._primitive = MobilePrimitive(view, answer["mobile"],
                                         self.ctx.memory["user_check_gripper_command"])

    def act(self, observation):
        decision = super().act(observation)
        acting = self._pending[3]
        if isinstance(acting, MobilePrimitive):
            decision.diagnostics["mobile_command"] = deepcopy(acting.mobile_command)
            decision.diagnostics["execution_kind"] = "mobile_native:bounded_chunk"
        return decision

    def observe_transition(self, before, decision, after):
        mobile = "mobile_command" in decision.diagnostics
        result = super().observe_transition(before, decision, after)
        if mobile:
            self.mobile_actions += 1
            self._direct_actions -= 1
            result["mobile_command"] = deepcopy(decision.diagnostics["mobile_command"])
            result["mobile_state"] = deepcopy(after.mobile_state)
        return result

    def _complete_direct(self):
        if isinstance(self._primitive, MobilePrimitive):
            outcome = self._primitive.outcome()
            self._record_feedback({"action": "mobile", "status": "chunk_complete",
                                   "reason": outcome["reason"], "measurement": outcome["evidence"],
                                   "executed_steps": outcome["actions"],
                                   "semantic_result": "unknown; inspect fresh RGB"})
            self._primitive = None
            self._need_agent = True
            return
        return super()._complete_direct()

    def agent_stats(self):
        result = super().agent_stats()
        result["counts"]["mobile_actions"] = self.mobile_actions
        result["versions"]["robocasa365_mobile"] = VERSION
        return result
