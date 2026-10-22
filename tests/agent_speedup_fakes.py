"""Test-only fakes for agent_SpeedUp_v1: kinematic plant plus pinhole wrist camera.

Uses ORACLE scene geometry (mark positions, surfaces) that the policy never
receives. Never import from src. Conventions follow the real stack: state8 is
[grip xyz, right_hand world rotvec, q1, q2]; actions are OSC deltas scaled by
0.05 m / 0.5 rad with world-frame rotation; images are delivered in the LEGACY
orientation (upright[:, ::-1]); upright image right = R[:,1], up = R[:,0];
wrist camera at hand origin + 0.05 R[:,0], grip site at hand origin + 0.097 R[:,2].
"""

from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image

from gpt_field.action import _quaternion_product, _rotvec_quaternion
from gpt_field.direct_history import TASK
from gpt_field.observation import PolicyObservation, quaternion_to_axisangle
from gpt_field.robot_tool_geometry import _rotation_matrix

SIZE = 512
FOCAL_PX = 256 / math.tan(math.radians(37.5))
CENTER = (SIZE - 1) / 2
OPEN_GAP_M = 0.0774
EMPTY_GAP_M = 0.0012
DRAWER_PROMPT = "open the top drawer and put the bowl inside"


def png_url(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_EXTERNAL_URL = None


def external_url() -> str:
    global _EXTERNAL_URL
    if _EXTERNAL_URL is None:
        _EXTERNAL_URL = png_url(np.full((SIZE, SIZE, 3), 120, np.uint8))
    return _EXTERNAL_URL


class FakeWristSim:
    """Kinematic arm with realized-translation ratio, supporting surfaces, a
    rate-limited gripper and a wrist camera that renders textured marks."""

    def __init__(self, *, eef=(0.10, 0.02, 1.00), rotvec=(math.pi, 0.0, 0.0), marks=((0.10, 0.0, 0.90),),
                 translation_ratio=0.26, rotation_ratio=1.0, floor_z=None, plane=None,
                 close_rate_m=0.004, object_width_m=None, grasp_predicate=None,
                 slip_after_lift_m=None, deflect=None, occlude_rows=None, prompt=TASK,
                 seed=0, mark_radius_m=0.006, background_std=3.0, external_px_per_m=None,
                 tilt_beyond_x=None, bars=(), render_marks=True):
        self.state = np.array([*eef, *rotvec, OPEN_GAP_M / 2, -OPEN_GAP_M / 2], dtype=float)
        self.marks = [np.asarray(mark, float).copy() for mark in marks]
        # Uniform bars (p0, p1, half_width_m): NCC localizes them only across their length (drawer handle).
        self.bars = [(np.asarray(a, float), np.asarray(b, float), float(w)) for a, b, w in bars]
        self.render_marks = bool(render_marks)    # False: marks are ORACLE points only (e.g. a bar centre)
        self.translation_ratio, self.rotation_ratio = float(translation_ratio), float(rotation_ratio)
        self.floor_z, self.plane = floor_z, plane
        self.close_rate_m = float(close_rate_m)
        self.object_width_m, self.grasp_predicate = object_width_m, grasp_predicate
        self.slip_after_lift_m, self.deflect = slip_after_lift_m, deflect
        self.occlude_rows, self.prompt = occlude_rows, prompt
        self.mark_radius_m = float(mark_radius_m)
        rng = np.random.default_rng(seed)
        ys, xs = np.mgrid[0:SIZE, 0:SIZE]
        base = 90 + 60 * (xs / SIZE) + 30 * (ys / SIZE) + rng.normal(0, background_std, (SIZE, SIZE))
        self._background = np.clip(base, 0, 255).astype(np.uint8)
        self._textures = [(rng.random((16, 16)) > .5) for _ in self.marks]
        # Optional ORACLE external camera facing the robot: world +y -> image right, +x -> image down.
        self.external_px_per_m = external_px_per_m
        self._external_background = np.clip(110 + rng.normal(0, 25, (SIZE, SIZE)), 0, 255).astype(np.uint8)
        self._hand_texture = np.where(rng.random((48, 48)) > .5, 240, 20).astype(np.uint8)
        # Optional workspace edge: beyond this x the hand pitches about world y and +x motion collapses.
        self.tilt_beyond_x = tilt_beyond_x
        self.attached = False
        self.lost = False
        self.attach_z = None
        self.contact = False
        self.steps = 0
        self.actions = []

    # geometry -----------------------------------------------------------
    @property
    def R(self):
        return _rotation_matrix(self.state[3:6])

    def camera(self):
        R = self.R
        hand = self.state[:3] - 0.097 * R[:, 2]
        return hand + 0.05 * R[:, 0], R

    def project(self, point):
        cam, R = self.camera()
        d = np.asarray(point, float) - cam
        a, b, z = d @ R[:, 1], d @ R[:, 0], d @ R[:, 2]
        if z <= 1e-6:
            return None, float(z)
        return np.array([CENTER + FOCAL_PX * a / z, CENTER - FOCAL_PX * b / z]), float(z)

    def grip_axis_lateral_error(self, point):
        """Lateral distance (m) between the grip axis and a world point, in the R1/R0 plane."""
        R = self.R
        d = np.asarray(point, float) - self.state[:3]
        return float(np.hypot(d @ R[:, 1], d @ R[:, 0]))

    @property
    def gap_mm(self):
        return float((self.state[6] - self.state[7]) * 1000)

    # rendering ----------------------------------------------------------
    def wrist_upright(self):
        gray = self._background.astype(np.int16).copy()
        for mark, texture in zip(self.marks if self.render_marks else (), self._textures):
            uv, depth = self.project(mark)
            if uv is None:
                continue
            radius = int(np.clip(FOCAL_PX * self.mark_radius_m / depth, 5, 45))
            u, v = int(round(uv[0])), int(round(uv[1]))
            y0, y1, x0, x1 = max(v - radius, 0), min(v + radius + 1, SIZE), max(u - radius, 0), min(u + radius + 1, SIZE)
            if y0 >= y1 or x0 >= x1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            inside = (yy - v) ** 2 + (xx - u) ** 2 <= radius ** 2
            cells = texture[((yy - v + radius) * 16 // (2 * radius + 1)).clip(0, 15),
                            ((xx - u + radius) * 16 // (2 * radius + 1)).clip(0, 15)]
            patch = np.where(cells, 230, 25)
            region = gray[y0:y1, x0:x1]
            region[inside] = patch[inside]
        for start, end, half_width in self.bars:
            for t in np.linspace(0.0, 1.0, 241):
                uv, depth = self.project(start + t * (end - start))
                if uv is None:
                    continue
                radius = max(2, int(round(FOCAL_PX * half_width / depth)))
                u, v = int(round(uv[0])), int(round(uv[1]))
                y0, y1, x0, x1 = max(v - radius, 0), min(v + radius + 1, SIZE), max(u - radius, 0), min(u + radius + 1, SIZE)
                if y0 < y1 and x0 < x1:
                    yy, xx = np.mgrid[y0:y1, x0:x1]
                    region = gray[y0:y1, x0:x1]
                    region[(yy - v) ** 2 + (xx - u) ** 2 <= radius ** 2] = 215
        if self.occlude_rows is not None:
            gray[int(self.occlude_rows):, :] = 15
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    def external_upright(self):
        gray = self._external_background.copy()
        u = int(round(256 + self.external_px_per_m * self.state[1]))
        v = int(round(200 + self.external_px_per_m * (self.state[0] - 0.10)))
        y0, x0 = max(v - 24, 0), max(u - 24, 0)
        y1, x1 = min(v + 24, SIZE), min(u + 24, SIZE)
        if y0 < y1 and x0 < x1:
            gray[y0:y1, x0:x1] = self._hand_texture[y0 - (v - 24):y1 - (v - 24), x0 - (u - 24):x1 - (u - 24)]
        return np.repeat(gray[:, :, None], 3, axis=2)

    def observation(self):
        legacy = np.ascontiguousarray(self.wrist_upright()[:, ::-1])
        external = (external_url() if self.external_px_per_m is None
                    else png_url(np.ascontiguousarray(self.external_upright()[:, ::-1])))
        return PolicyObservation(external, png_url(legacy), tuple(float(v) for v in self.state), self.prompt)

    # dynamics -----------------------------------------------------------
    def reset(self):
        return self.observation()

    def step(self, action):
        a = np.asarray(action, float)
        if a.shape != (7,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1 + 1e-9):
            raise ValueError("invalid fake action")
        self.actions.append(a.tolist())
        old = self.state[:3].copy()
        move = self.translation_ratio * 0.05 * a[:3]
        pitch = 0.0
        if self.tilt_beyond_x is not None and old[0] + move[0] > self.tilt_beyond_x and move[0] > 0:
            # past the edge: outward motion realizes 2% and the hand pitches outward (live run 001)
            pitch = 0.04 + 2.0 * (old[0] + move[0] - self.tilt_beyond_x)
            move[0] *= 0.02
        pos = old + move
        self.contact = False
        if self.floor_z is not None and pos[2] < self.floor_z:
            pos[2], self.contact = self.floor_z, True
        if self.plane is not None:
            normal, offset = np.asarray(self.plane[0], float), float(self.plane[1])
            violation = normal @ pos - offset
            if violation < 0:
                pos, self.contact = pos - violation * normal, True
        if self.attached:
            self.marks = [mark + (pos - old) for mark in self.marks]
        self.state[:3] = pos
        q = _quaternion_product(_rotvec_quaternion(self.rotation_ratio * 0.5 * a[3:6] + np.array([0.0, -pitch, 0.0])),
                                _rotvec_quaternion(self.state[3:6]))
        self.state[3:6] = quaternion_to_axisangle(q / np.linalg.norm(q))
        gap = self.state[6] - self.state[7]
        if a[6] > 0:
            holding = self.attached or (not self.lost and self.object_width_m is not None
                                        and (self.grasp_predicate is None or self.grasp_predicate(self)))
            target = self.object_width_m if holding else EMPTY_GAP_M
        elif a[6] < 0:
            target, self.attached, self.lost = OPEN_GAP_M, False, False
        else:
            target = gap
        gap += float(np.clip(target - gap, -2 * self.close_rate_m, 2 * self.close_rate_m))
        if a[6] > 0 and self.object_width_m is not None and abs(gap - self.object_width_m) < 1e-12 and not self.lost:
            if not self.attached:
                self.attach_z = float(pos[2])
            self.attached = True
        if (self.attached and self.slip_after_lift_m is not None
                and pos[2] - self.attach_z > self.slip_after_lift_m):
            self.attached, self.lost = False, True
        self.state[6], self.state[7] = gap / 2, -gap / 2
        if self.deflect is not None:
            event = self.deflect(self)
            if event is not None:
                finger, mm = event
                if finger == 1:
                    self.state[6] -= mm / 1000
                else:
                    self.state[7] += mm / 1000
        self.steps += 1
        return self.observation()


def collision_status(steps=0, success=False, task_id=0):
    return {"task_id": task_id, "episode_index": 0, "steps": steps, "success": success, "terminated": False,
            "libero_success": success,
            "collision": {"enabled": True, "mode": "task_manipulation_v1", "detected": False,
                          "stop_on_collision": False, "collision_policy": "record_only",
                          "success_qualification": "raw_libero_goal", "first_event": None,
                          "libero_success": success}}


class FakeLiberoSim:
    """Runner-facing simulator fake with the speed-first record-only status contract."""

    instances = []

    def __init__(self, config=None, *, wrist=None, success_at=None, task_id=0, prompt=TASK):
        self.config = config
        self.wrist = wrist or FakeWristSim(prompt=prompt)
        self.success_at, self.task_id, self.prompt = success_at, task_id, prompt
        self.status = collision_status(task_id=task_id)
        self.metadata = {"fake": True}
        self.actions = []
        self.closed = False
        FakeLiberoSim.instances.append(self)

    def start(self):
        return self

    def list_tasks(self):
        return [{"task_id": self.task_id, "prompt": self.prompt}]

    def reset(self, task_id, episode_index):
        self.status = collision_status(task_id=task_id)
        return self.wrist.observation()

    def step(self, action):
        self.actions.append(list(action))
        obs = self.wrist.step(action)
        success = self.success_at is not None and len(self.actions) >= self.success_at
        self.status = collision_status(len(self.actions), success, self.task_id)
        return obs, success

    def close(self):
        self.closed = True


def _rotvec_from_quaternion(q):
    q = np.asarray(q, float) / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-12:
        return np.zeros(3)
    return q[:3] / s * 2 * math.atan2(s, q[3])


class DynWristSim(FakeWristSim):
    """Second-order OSC plant (spring-damper per 2 ms substep) fitted to recorded 512-px LIBERO traces:
    translation from rest realizes 0.115/0.209/0.234 -> 0.256; the arm keeps rotating after the command stops."""

    def __init__(self, *args, w_t=11.6, z_t=1.0, w_r=11.75, z_r=1.2, couple=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.vel, self.omega = np.zeros(3), np.zeros(3)
        self.w_t, self.z_t, self.w_r, self.z_r = w_t, z_t, w_r, z_r
        self.couple = couple    # translational acceleration -> wrist rotation (the arm swings when a move stops)

    def step(self, action):
        a = np.asarray(action, float)
        if a.shape != (7,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1 + 1e-9):
            raise ValueError("invalid fake action")
        self.actions.append(a.tolist())
        old = self.state[:3].copy()
        goal = old + 0.05 * a[:3]
        q = _rotvec_quaternion(self.state[3:6])
        q_goal = _quaternion_product(_rotvec_quaternion(0.5 * a[3:6]), q)
        pos, vel, omega = old.copy(), self.vel.copy(), self.omega.copy()
        for _ in range(25):
            acc = self.w_t ** 2 * (goal - pos) - 2 * self.z_t * self.w_t * vel
            vel = vel + acc * 0.002
            pos = pos + vel * 0.002
            if self.floor_z is not None and pos[2] < self.floor_z:
                pos[2], vel[2] = self.floor_z, max(vel[2], 0.0)
            error = _rotvec_from_quaternion(_quaternion_product(q_goal, np.r_[-q[:3], q[3]]))
            omega = omega + (self.w_r ** 2 * error - 2 * self.z_r * self.w_r * omega
                             + self.couple * np.cross([0.0, 0.0, 1.0], acc)) * 0.002
            q = _quaternion_product(_rotvec_quaternion(omega * 0.002), q)
            q = q / np.linalg.norm(q)
        self.vel, self.omega = vel, omega
        if self.attached:
            self.marks = [mark + (pos - old) for mark in self.marks]
        self.state[:3] = pos
        self.state[3:6] = quaternion_to_axisangle(q)
        gap = self.state[6] - self.state[7]
        target = OPEN_GAP_M if a[6] < 0 else EMPTY_GAP_M if a[6] > 0 else gap
        gap += float(np.clip(target - gap, -2 * self.close_rate_m, 2 * self.close_rate_m))
        self.state[6], self.state[7] = gap / 2, -gap / 2
        self.steps += 1
        return self.observation()
