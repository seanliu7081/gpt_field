"""numpy-only runtime prototype: batched FK, sphere placement, sphere-vs-OBB signed distance.

Runs in the controller .venv (numpy 2.x, no scipy). Loads the npz/json built by build_body_model.py.
"""
import json, math
from pathlib import Path
import numpy as np


def rotvec_mat(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, v / np.maximum(th, 1e-12), 0.)
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    Z = np.zeros_like(x)
    K = np.stack([np.stack([Z, -z, y], -1), np.stack([z, Z, -x], -1), np.stack([-y, x, Z], -1)], -2)
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    I = np.broadcast_to(np.eye(3), K.shape)
    return I + s * K + (1 - c) * (K @ K)


def axis_angle_batch(axis, ang):
    """(K,) angles about a fixed unit axis -> (K,3,3)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    return rotvec_mat(a[None, :] * np.asarray(ang, float)[:, None])


class BodyModel:
    def __init__(self, prefix):
        prefix = Path(prefix)
        self.manifest = json.loads(prefix.with_suffix(".json").read_text())
        arrays = np.load(prefix.with_suffix(".npz"))
        self.bodies = self.manifest["bodies"]
        self.parts = []
        for p in self.manifest["parts"]:
            i = p["index"]
            C, R = arrays[f"p{i}_centers"], arrays[f"p{i}_radii"]
            # part bounding sphere (local) for the broad phase
            bc = (C.max(0) + C.min(0)) / 2
            br = float(np.max(np.linalg.norm(C - bc, axis=1) + R))
            self.parts.append({**p, "centers": C, "radii": R, "bc": bc, "br": br,
                               "eq": arrays[f"p{i}_hull_equations"], "hv": arrays[f"p{i}_hull_vertices"]})
        self.index = {b["name"]: b["index"] for b in self.bodies}
        # sphere arrays concatenated by part
        self.S_centers = np.concatenate([p["centers"] for p in self.parts])
        self.S_radii = np.concatenate([p["radii"] for p in self.parts])
        self.S_part = np.concatenate([np.full(len(p["radii"]), j) for j, p in enumerate(self.parts)])
        self.S_body = np.array([self.parts[j]["body"] for j in self.S_part])
        self.grip_offset = np.array([0., 0., 0.097])

    # ---------------- FK -----------------
    def body_poses(self, q_arm, q_fingers, world_from_base):
        """q_arm (K,7), q_fingers (K,2), world_from_base (4,4) -> dict body_index -> (K,4,4)."""
        K = len(q_arm)
        poses = {}
        for b in self.bodies:
            T = np.broadcast_to(np.asarray(b["T"]), (K, 4, 4)).copy()
            if b["parent"] < 0:
                if b["xml"] == "arm":
                    parent = np.broadcast_to(world_from_base, (K, 4, 4))
                else:
                    parent = poses[self.index["right_hand"]]
            else:
                parent = poses[b["parent"]]
            P = parent @ T
            j = b["joint"]
            if j is not None:
                axis = np.asarray((j.get("axis") or "0 0 1").split(), float)
                anchor = np.asarray((j.get("pos") or "0 0 0").split(), float)
                name = j["name"]
                if name.startswith("finger_joint"):
                    val = q_fingers[:, int(name[-1]) - 1]
                    M = np.broadcast_to(np.eye(4), (K, 4, 4)).copy()
                    M[:, :3, 3] = axis[None, :] * val[:, None]
                else:
                    val = q_arm[:, int(name[5:]) - 1]
                    Rj = axis_angle_batch(axis, val)
                    M = np.broadcast_to(np.eye(4), (K, 4, 4)).copy()
                    M[:, :3, :3] = Rj
                    M[:, :3, 3] = anchor[None] - (Rj @ anchor)
                P = P @ M
            poses[b["index"]] = P
        return poses

    def hand_from_state(self, states):
        """Gripper-only placement from state8 batch (K,8): right_hand poses (K,4,4)."""
        states = np.asarray(states, float)
        R = rotvec_mat(states[:, 3:6])
        T = np.broadcast_to(np.eye(4), (len(states), 4, 4)).copy()
        T[:, :3, :3] = R
        T[:, :3, 3] = states[:, :3] - R @ self.grip_offset
        return T

    def gripper_poses(self, hand, q_fingers):
        K = len(hand)
        poses = {self.index["right_hand"]: hand}
        for b in self.bodies:
            if b["xml"] != "tool":
                continue
            T = np.broadcast_to(np.asarray(b["T"]), (K, 4, 4)).copy()
            parent = hand if b["parent"] < 0 else poses[b["parent"]]
            P = parent @ T
            j = b["joint"]
            if j is not None:
                axis = np.asarray((j.get("axis") or "0 0 1").split(), float)
                val = q_fingers[:, int(j["name"][-1]) - 1]
                M = np.broadcast_to(np.eye(4), (K, 4, 4)).copy()
                M[:, :3, 3] = axis[None, :] * val[:, None]
                P = P @ M
            poses[b["index"]] = P
        return poses

    def place(self, poses, part_ids):
        """World sphere centres for selected parts -> (K,S,3), radii (S,), part index (S,)."""
        sel = np.isin(self.S_part, part_ids)
        bodies = self.S_body[sel]
        local = self.S_centers[sel]
        K = next(iter(poses.values())).shape[0]
        out = np.empty((K, int(sel.sum()), 3))
        for b in np.unique(bodies):
            m = bodies == b
            T = poses[b]
            out[:, m] = local[m] @ T[:, :3, :3].transpose(0, 2, 1) + T[:, None, :3, 3]
        return out, self.S_radii[sel], self.S_part[sel]


def sd_spheres_obb(centers, radii, box_c, box_R, box_h):
    """centers (K,S,3); boxes M: c (M,3), R (M,3,3) world_from_box, h (M,3) -> signed distance (K,S,M)."""
    d = centers[:, :, None, :] - box_c[None, None]
    local = np.einsum("ksmi,mij->ksmj", d, box_R)
    q = np.abs(local) - box_h[None, None]
    outside = np.linalg.norm(np.maximum(q, 0.), axis=-1)
    inside = np.minimum(q.max(-1), 0.)
    return outside + inside - radii[None, :, None]
