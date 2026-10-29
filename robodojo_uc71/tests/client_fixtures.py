from copy import deepcopy
import json

import numpy as np

from robodojo_uc71.planning import direction_frame_from_calibration

TASK = "Pick up two slices of bread, place them into the toaster, and press the lever down."


def plan():
    return {"scene": "A package and a tray.", "parts": [
        {"id": "A", "text": "package: whole body", "move": "free", "x": 2, "w": 3, "r": 0},
        {"id": "B", "text": "tray: open cavity", "move": "fixed", "x": 7, "w": 9, "r": 12}],
        "hold_true": [], "steps": [{"do": "place_in", "obj": "A", "grip": [], "ref": "B",
                                     "sense": "any", "keep": [], "until": []}],
        "regions": [], "placement_goals": []}


def calibration():
    camera = {"width": 640, "height": 480, "intrinsics": [[400., 0., 319.5], [0., 400., 239.5], [0., 0., 1.]],
              "world_from_optical": [[1., 0., 0., 0.], [0., 0., 1., -.4], [0., -1., 0., 1.3], [0., 0., 0., 1.]],
              "pose_source": "synchronized_robot_pose_fixed_mount"}
    tool = {"link_name": "link6", "link6_from_tool": np.eye(4).tolist(),
            "finger_gap_formula": "sum_of_measured_finger_joint_positions",
            "finger_gap_semantics": "joint_separation_proxy", "source_sha256": "a" * 64}
    return {"version": "robodojo-uc71-calibration-1", "cameras": {name: deepcopy(camera) for name in
            ("head", "left_wrist", "right_wrist")}, "tools": {name: deepcopy(tool) for name in ("left", "right")}}


def observation(step=0):
    arm = {"tool_pose_wxyz": [0., -.2, .9, 1., 0., 0., 0.], "joint_position_rad": [0.] * 6,
           "joint_velocity_rad_s": [0.] * 6, "finger_position_m": [.03, .03],
           "finger_gap_m": .06, "gripper_command": "open"}
    array = np.zeros((480, 640, 3), dtype=np.uint8)
    array[20:50, 100:150] = [10, 30, 200]
    return {"step": step, "images": {name: array.copy() for name in ("head", "left_wrist", "right_wrist")},
            "robot": {name: deepcopy(arm) for name in ("left", "right")}, "calibration": calibration()}


def frame():
    return direction_frame_from_calibration(calibration())


def decision():
    return {"belief": "Attachment is unknown.", "plan": "Inspect fresh views.", "why": "Obtain robot observations.",
            "type": "observe", "arm_id": None, "execution": None, "motion": None}


class OfflineTransport:
    offline_mock = True

    def __init__(self, answer=None):
        self.answer = decision() if answer is None else answer
        self.calls = []
        self.failure = None

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(deepcopy(payload))
        if self.failure is not None:
            raise self.failure
        number = len(self.calls)
        return {"id": f"offline-{number}", "status": "completed", "model": payload["model"],
                "reasoning": {"effort": "high"}, "offline_mock": True, "billing": "offline_mock",
                "output": [{"type": "reasoning", "id": f"reason-{number}", "summary": [],
                            "encrypted_content": f"opaque-fixture-{number}"},
                           {"type": "message", "id": f"msg-{number}", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": json.dumps(self.answer)}]}],
                "usage": {"input_tokens": 100, "output_tokens": 20,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 5}}}


def current_fields(payload):
    turn = next(row for row in reversed(payload["input"]) if row.get("role") == "user")
    return json.loads(turn["content"][0]["text"])
