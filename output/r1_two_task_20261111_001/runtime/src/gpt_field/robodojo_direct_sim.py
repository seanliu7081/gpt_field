"""Robot-only Panda kinematics for the RoboDojo-style absolute-motion baseline.

This module never imports MuJoCo or reads the live scene. It parses only a robot
XML chain and the gripper's fixed grip-site offset. The world mounting transform
is recovered from measured robot joints and measured grip position/hand rotation.
Damped least-squares IK checks a Cartesian path before physical execution. This
is a kinematic feasibility check, not collision checking or CuRobo joint planning.
A failed numerical solve means no solution was found, not proven unreachability.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

KINEMATICS_VERSION = "robodojo-direct-panda-ik-1"


def vector(value, size, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return a


def rotation_matrix(v):
    v = vector(v, 3, "rotation vector")
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = v / theta
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def matrix_quaternion(R):
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError("rotation matrix must be finite 3x3")
    if np.trace(R) > 0:
        s = math.sqrt(1 + float(np.trace(R))) * 2
        q = np.array([(R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s,
                      (R[1, 0]-R[0, 1])/s, s/4])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(max(0., 1+R[i, i]-R[j, j]-R[k, k]))*2
        if s < 1e-12:
            raise ValueError("invalid rotation matrix")
        q = np.zeros(4)
        q[i], q[j], q[k], q[3] = s/4, (R[j, i]+R[i, j])/s, (R[k, i]+R[i, k])/s, (R[k, j]-R[j, k])/s
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def rotation_vector(R):
    q = matrix_quaternion(R)
    sine = float(np.linalg.norm(q[:3]))
    return np.zeros(3) if sine < 1e-12 else q[:3] * (2 * math.atan2(sine, float(q[3])) / sine)


def quaternion_matrix(wxyz):
    w, x, y, z = vector(wxyz, 4, "XML quaternion")
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    if norm < 1e-12:
        raise ValueError("zero XML quaternion")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _transform(body):
    if any(key in body.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("unsupported robot XML orientation representation")
    T = np.eye(4)
    T[:3, 3] = vector(body.get("pos", "0 0 0").split(), 3, "XML position")
    T[:3, :3] = quaternion_matrix(body.get("quat", "1 0 0 0").split())
    return T


def discover_robot_xml(python_executable=None):
    """Resolve the installed static robot asset without importing robosuite."""
    candidates = []
    if python_executable:
        root = Path(python_executable).expanduser().absolute().parent.parent
        candidates.extend(sorted(root.glob("lib/python*/site-packages/robosuite/models/assets/robots/panda/robot.xml")))
        # Editable installs may place assets outside site-packages. Ask the
        # configured simulator interpreter to locate the top-level package;
        # find_spec does not import robosuite or initialize a simulator.
        located = subprocess.run([str(python_executable), "-c",
            "import importlib.util; s=importlib.util.find_spec('robosuite'); print(s.origin if s else '')"],
            check=True, capture_output=True, text=True, timeout=20)
        origin = located.stdout.strip()
        if origin:
            candidates.append(Path(origin).parent / "models/assets/robots/panda/robot.xml")
    spec = importlib.util.find_spec("robosuite")
    if spec and spec.origin:
        candidates.append(Path(spec.origin).parent / "models/assets/robots/panda/robot.xml")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Panda robot.xml not found; provide robot_xml or the simulator Python executable")


@dataclass(frozen=True)
class IKResult:
    ok: bool
    joints: tuple[float, ...]
    position_error_m: float
    orientation_error_deg: float
    iterations: int


class PandaKinematics:
    """Static robot-only chain. ``bind`` returns a separate measured-world model."""

    def __init__(self, fixed, axes, anchors, low, high, terminal, metadata):
        self.fixed = tuple(np.asarray(v, dtype=float).copy() for v in fixed)
        self.axes = tuple(np.asarray(v, dtype=float).copy() for v in axes)
        self.anchors = tuple(np.asarray(v, dtype=float).copy() for v in anchors)
        self.low, self.high = np.array(low, dtype=float), np.array(high, dtype=float)
        self.terminal = np.asarray(terminal, dtype=float).copy()
        self.metadata = dict(metadata)
        self.world_from_base = np.eye(4)

    @classmethod
    def from_xml(cls, robot_xml=None, gripper_xml=None, *, python_executable=None):
        robot_xml = Path(robot_xml or discover_robot_xml(python_executable)).resolve()
        gripper_xml = Path(gripper_xml or (robot_xml.parents[2] / "grippers/panda_gripper.xml")).resolve()
        root = ET.parse(robot_xml).getroot()
        world = root.find("worldbody")
        if world is None:
            raise ValueError("robot XML has no worldbody")
        path = None

        def visit(body, ancestors):
            nonlocal path
            chain = ancestors + [body]
            if body.get("name") == "right_hand":
                path = chain
            for child in body.findall("body"):
                visit(child, chain)
        for body in world.findall("body"):
            visit(body, [])
        if path is None:
            raise ValueError("robot XML must contain right_hand")
        fixed, axes, anchors, low, high, names = [], [], [], [], [], []
        pending = np.eye(4)
        for body in path:
            pending = pending @ _transform(body)
            joints = body.findall("joint")
            if len(joints) > 1:
                raise ValueError("one revolute joint per robot body is required")
            if joints:
                joint = joints[0]
                if joint.get("type", "hinge") != "hinge":
                    raise ValueError("Panda arm joint must be a hinge")
                limits = vector(joint.get("range", "").split(), 2, "joint range")
                axis = vector(joint.get("axis", "0 0 1").split(), 3, "joint axis")
                if np.linalg.norm(axis) < 1e-12 or limits[0] >= limits[1]:
                    raise ValueError("invalid joint axis or limits")
                fixed.append(pending)
                axes.append(axis / np.linalg.norm(axis))
                anchors.append(vector(joint.get("pos", "0 0 0").split(), 3, "joint anchor"))
                low.append(limits[0]); high.append(limits[1]); names.append(joint.get("name"))
                pending = np.eye(4)
        if names != [f"joint{i}" for i in range(1, 8)]:
            raise ValueError("expected seven ordered Panda arm joints")
        # Only the gripper fixed root -> eef -> grip_site positional offset is
        # used. Orientation state remains right_hand, as LIBERO specifies.
        groot = ET.parse(gripper_xml).getroot()
        gbody = groot.find("./worldbody/body[@name='right_gripper']")
        eef = None if gbody is None else gbody.find("body[@name='eef']")
        site = None if eef is None else eef.find("site[@name='grip_site']")
        if site is None:
            raise ValueError("gripper XML must contain right_gripper/eef/grip_site")
        Tsite = _transform(gbody) @ _transform(eef) @ _transform(site)
        Tgrip = np.eye(4)
        Tgrip[:3, 3] = Tsite[:3, 3]
        terminal = pending @ Tgrip
        metadata = {"version": KINEMATICS_VERSION, "robot_xml": str(robot_xml), "gripper_xml": str(gripper_xml),
                    "robot_xml_sha256": hashlib.sha256(robot_xml.read_bytes()).hexdigest(),
                    "gripper_xml_sha256": hashlib.sha256(gripper_xml.read_bytes()).hexdigest(),
                    "joint_names": names, "grip_offset_in_right_hand_m": Tsite[:3, 3].tolist(),
                    "scene_geometry": False, "collision_checking": False,
                    "mount_source": "measured state8 pose and measured arm joint positions"}
        return cls(fixed, axes, anchors, low, high, terminal, metadata)

    def bind(self, state8, joint_positions):
        state, q = vector(state8, 8, "state8"), vector(joint_positions, 7, "joint_positions")
        if np.any(q < self.low-0.03) or np.any(q > self.high+0.03):
            raise ValueError("measured arm joints outside robot limits")
        bound = PandaKinematics(self.fixed, self.axes, self.anchors, self.low, self.high, self.terminal, self.metadata)
        local, _ = bound.forward(q)
        measured = np.eye(4)
        measured[:3, :3] = rotation_matrix(state[3:6])
        measured[:3, 3] = state[:3]
        bound.world_from_base = measured @ np.linalg.inv(local)
        return bound

    def forward(self, joint_positions):
        q = vector(joint_positions, 7, "joint_positions")
        T = self.world_from_base.copy()
        origins, directions = [], []
        for index in range(7):
            T = T @ self.fixed[index]
            origins.append(T[:3, 3] + T[:3, :3] @ self.anchors[index])
            directions.append(T[:3, :3] @ self.axes[index])
            R = rotation_matrix(self.axes[index] * q[index])
            J = np.eye(4)
            J[:3, :3] = R
            J[:3, 3] = self.anchors[index] - R @ self.anchors[index]
            T = T @ J
        T = T @ self.terminal
        jacobian = np.column_stack([np.r_[np.cross(axis, T[:3, 3]-origin), axis]
                                    for origin, axis in zip(origins, directions)])
        return T, jacobian

    def solve(self, position, orientation, seed, *, max_iterations=140):
        target = vector(position, 3, "target position")
        Rtarget = np.asarray(orientation, dtype=float)
        if Rtarget.shape != (3, 3) or not np.isfinite(Rtarget).all():
            raise ValueError("target orientation must be finite3x3")
        q = np.clip(vector(seed, 7, "IK seed"), self.low+1e-7, self.high-1e-7)
        # Rotation scaling balances radians and metres; all calculations remain
        # on this isolated robot chain and never step a physical simulator.
        weights = np.r_[np.ones(3), np.full(3, 0.25)]
        for iteration in range(max_iterations+1):
            T, J = self.forward(q)
            dp, dr = target-T[:3, 3], rotation_vector(Rtarget @ T[:3, :3].T)
            pe, re = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
            if pe <= 0.0015 and re <= math.radians(1.0):
                return IKResult(True, tuple(q), pe, math.degrees(re), iteration)
            if iteration == max_iterations:
                break
            A, e = weights[:, None]*J, weights*np.r_[dp, dr]
            dq = A.T @ np.linalg.solve(A @ A.T + (0.008**2)*np.eye(6), e)
            norm = float(np.max(np.abs(dq)))
            if norm > 0.16:
                dq *= 0.16/norm
            # Deterministic short line search avoids accepting worsening steps
            # caused by joint-limit clamping near a singularity.
            current_cost = float(e @ e)
            candidate, best_cost = q, current_cost
            for scale in (1., .5, .25, .125):
                trial = np.clip(q+scale*dq, self.low+1e-7, self.high-1e-7)
                pose, _ = self.forward(trial)
                residual = weights*np.r_[target-pose[:3, 3], rotation_vector(Rtarget @ pose[:3, :3].T)]
                cost = float(residual @ residual)
                if cost < best_cost:
                    candidate, best_cost = trial, cost
            if best_cost >= current_cost-1e-14:
                break
            q = candidate
        return IKResult(False, tuple(q), pe, math.degrees(re), iteration)


# Public concise alias for runner construction.
Kinematics = PandaKinematics
