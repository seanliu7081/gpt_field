"""Stove-twist EFFECT PROBE (spec §8.2 G0; DIAGNOSIS ONLY; zero model calls; evaluator-positioned).

A scripted knob twist on libero_10 task 2 (episode 0, seed 7, native 512 px, the v2 worker) that records, per
control step, the knob joint angle (truth), the burner region's change in the agentview and the wrist images, and
the gripper / hand state. It measures what EffectWatch (O1, spec §5.2) will see; it is not EffectWatch and it is
never a policy input.

Motion (the Phase A oracle recipe, ``SURVEY/tasks/oracle_t2_t7.py``, evaluator knob position): static hold at the
reset; move above the knob axis (z 1.02) and descend to grip z 0.935 with the hand open; close 20 steps (pinch the
lever at its axis); hold closed; yaw in +0.15 rad increments about world z (the knob's free sense,
``knob_sense_probe.py``), 40 steps max each, holding the grip on the axis; hold; open; retreat up; static.

Effect measurement (the spec's change-based O1, no absolute colour threshold):
- region = the burner plate top (the stove's collision_burner box, +- its half-size, at its top face), found per
  pixel by intersecting the camera ray with that plane (evaluator geometry; the executor registers the region
  from pointing + HeightField). Two variants: ``plate`` (the whole 19 x 19 cm top, which includes the stove's
  reflective metal box face around the grate) and ``disk`` (the dark ring grate, r <= 6.25 cm around the burner
  site);
- masks = robot pixels (segmentation, dilated) and the turned part's pixels (the knob, segmentation, dilated: O7
  requires the effect region to be disjoint from the turned part);
- reference = the pre-twist frame (the last closed hold frame before the first yaw increment). The agentview is
  static, so the change is per pixel; the wrist camera moves with the hand, so each current region pixel is mapped
  onto the reference wrist frame through the burner plane (plane-induced warp from the two camera poses);
- a pixel is changed when |Delta Lab| (CIE76) > max(12, 4 sigma_px), sigma_px from the static pre-twist hold;
  the changed fraction is over unmasked region pixels. ``red_shift_fraction`` counts changed pixels whose Lab a*
  rose (toward red). The legacy absolute red count (R - max(G, B) > 80, ``stove_vis_check.py``) is reported for
  comparison with the spec's 20 % figure only.
- O1 detection = changed fraction >= max(2 %, 10 x the false-change rate) for 3 consecutive frames; the
  false-change rate is measured on the pre-onset frames (q < 0.5) of the same reference (static hold + twist
  before onset), and also against the reset frame during the approach.

Outputs (fresh directory): effect_probe.json (per-step records + summary), effect_probe.png (time series),
effect_vs_angle.png, frames/*.png (agentview / wrist samples, region overlays, change maps).

usage: MUJOCO_EGL_DEVICE_ID=2 /workspace/venvs/libero-plus/bin/python -B scripts/agent_speedup_v2_effect_probe.py \
           --out runs/agent_speedup_v2_m0close_effect_probe_t2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TASK_KEY = "libero_10_t2"
KNOB_JOINT = "flat_stove_1_button"
KNOB_BODY = "flat_stove_1_button"
PLATE_GEOM = "flat_stove_1_collision_burner"
BURNER_SITE = "flat_stove_1_burner"
CAMERAS = {"agentview": ("agentview", "observation/image"), "wrist": ("robot0_eye_in_hand", "observation/wrist_image")}
TURNON_Q = 0.5                   # LIBERO TurnOn threshold (articulated_objects.py:238), diagnosis only
O1_PIXEL_FLOOR = 12.0            # spec §5.2 O1: |Delta Lab| > max(12, 4 sigma_px)
O1_SIGMA_K = 4.0
O1_FRACTION_FLOOR = 0.02         # ... fraction >= max(2 %, 10 x false-change rate) for 3 frames
O1_FALSE_RATE_K = 10.0
O1_FRAMES = 3
MASK_DILATE_PX = 3
YAW_INCREMENT_RAD = 0.15
YAW_INCREMENTS = 8
GRIP_Z_AT_KNOB = 0.935


def decode(url):
    from PIL import Image
    import numpy as np
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))


class Probe:
    def __init__(self):
        from dataclasses import replace
        import mujoco
        import numpy as np
        from scipy.spatial.transform import Rotation
        from gpt_field.agent_speedup_v2_sim import V2LiberoWorker
        from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config, resolve_v2_task
        self.np, self.mj, self.Rot = np, mujoco, Rotation
        spec = resolve_v2_task(TASK_KEY)
        config, monitor = build_v2_sim_config(TASK_KEY)
        self.config = replace(config, stop_on_success=False)
        self.w = V2LiberoWorker(self.config, {**monitor, "truth_window": False})
        self.resp = self.w.reset(spec.task_id, 0)
        self.env = self.w.env
        self.sim = self.env.sim
        self.M, self.D = self.sim.model._model, self.sim.data._data
        self.size = self.config.image_size
        M = self.M
        self.cam_id = {k: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_CAMERA, c) for k, (c, _) in CAMERAS.items()}
        self.plate_geom = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_GEOM, PLATE_GEOM)
        self.burner_site = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_SITE, BURNER_SITE)
        knob_body = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, KNOB_BODY)
        bodyid = np.asarray(M.geom_bodyid)
        names = [mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "" for b in range(M.nbody)]
        self.robot_geoms = np.array([names[int(bodyid[g])].startswith(("robot0_", "gripper0_"))
                                     for g in range(M.ngeom)])
        self.knob_geoms = bodyid == knob_body
        self.knob_contact_geoms = set(np.nonzero(self.knob_geoms)[0].tolist())
        self.flip, self.mirrored = None, None
        self.records, self.frames = [], {}
        self.reference = {}
        self.onset_frames = None
        self.last_frame = None
        self.tau, self.sigma = {}, {}

    # ---------------- camera geometry ----------------
    def camera(self, key):
        cid = self.cam_id[key]
        fovy = float(self.M.cam_fovy[cid])
        f = (self.size / 2) / math.tan(math.radians(fovy) / 2)
        return {"pos": self.np.asarray(self.D.cam_xpos[cid], float).copy(),
                "R": self.np.asarray(self.D.cam_xmat[cid], float).reshape(3, 3).copy(), "f": f}

    def plate(self):
        g = self.plate_geom
        R = self.np.asarray(self.D.geom_xmat[g], float).reshape(3, 3)
        c = self.np.asarray(self.D.geom_xpos[g], float)
        half = self.np.asarray(self.M.geom_size[g], float)
        top = c + R[:, 2] * half[2]
        return {"centre": top, "R": R, "half": half[:2], "site": self.np.asarray(self.D.site_xpos[self.burner_site],
                                                                                    float).copy()}

    def pixel_rays(self, cam):
        """World ray directions of every pixel centre in the POLICY image frame (upright, columns mirrored; the
        mapping is checked against the renderer in ``check_flip``)."""
        np = self.np
        n = self.size
        rows, cols = np.mgrid[0:n, 0:n].astype(float)
        u_right = (n - 1 - cols) + 0.5 if self.mirrored else cols + 0.5
        v_down = rows + 0.5
        x = (u_right - n / 2) / cam["f"]
        y = -(v_down - n / 2) / cam["f"]
        d = np.stack([x, y, -np.ones_like(x)], axis=-1) @ cam["R"].T
        return d

    def plane_hits(self, cam, plate):
        np = self.np
        d = self.pixel_rays(cam)
        nrm = plate["R"][:, 2]
        denom = d @ nrm
        t = ((plate["centre"] - cam["pos"]) @ nrm) / np.where(np.abs(denom) < 1e-9, np.nan, denom)
        pts = cam["pos"] + d * t[..., None]
        local = (pts - plate["centre"]) @ plate["R"]
        inside = (t > 0) & (np.abs(local[..., 0]) <= plate["half"][0]) & (np.abs(local[..., 1]) <= plate["half"][1])
        disk = (t > 0) & (np.hypot(*(((pts - plate["site"]) @ plate["R"])[..., :2].transpose(2, 0, 1))) <= 0.0625)
        return pts, inside, disk

    def project(self, cam, pts):
        """Policy-frame (row, col) continuous pixel coordinates of world points (and depth)."""
        np = self.np
        c = (pts - cam["pos"]) @ cam["R"]
        depth = -c[..., 2]
        u_right = self.size / 2 + cam["f"] * c[..., 0] / depth
        v_down = self.size / 2 - cam["f"] * c[..., 1] / depth
        col = (self.size - u_right) if self.mirrored else u_right
        return v_down - 0.5, col - 0.5, depth

    # ---------------- rendering ----------------
    def raw_render(self, key, segmentation=False):
        return self.sim.render(camera_name=CAMERAS[key][0], width=self.size, height=self.size,
                               segmentation=segmentation)

    def to_policy(self, raw):
        return raw[::-1, ::-1] if self.flip == "rows_cols" else raw[::-1] if self.flip == "rows" else raw

    def check_flip(self, images):
        """Which flip of the raw render equals the policy image (bitwise), and whether columns are mirrored."""
        np = self.np
        raw = self.raw_render("agentview")
        cands = {"rows_cols": raw[::-1, ::-1], "rows": raw[::-1], "none": raw, "cols": raw[:, ::-1]}
        errors = {k: float(np.abs(v.astype(int) - images["agentview"].astype(int)).mean()) for k, v in cands.items()}
        self.flip = min(errors, key=errors.get)
        self.mirrored = self.flip in ("rows_cols", "cols")
        return {"flip": self.flip, "mean_abs_error": errors, "mirrored_columns": self.mirrored}

    def masks(self, key):
        from scipy.ndimage import binary_dilation
        np = self.np
        seg = self.to_policy(self.raw_render(key, segmentation=True))
        objtype, objid = seg[..., 0], seg[..., 1]
        geom = objtype == int(self.mj.mjtObj.mjOBJ_GEOM)
        ids = np.where(geom, objid, -1)
        valid = ids >= 0
        robot = np.zeros(ids.shape, bool)
        knob = np.zeros(ids.shape, bool)
        robot[valid] = self.robot_geoms[ids[valid]]
        knob[valid] = self.knob_geoms[ids[valid]]
        return (binary_dilation(robot, iterations=MASK_DILATE_PX), binary_dilation(knob, iterations=MASK_DILATE_PX),
                robot, ids)

    # ---------------- state ----------------
    def state8(self):
        return self.np.asarray(self.resp["observation"]["observation/state"], float)

    def yaw(self, s):
        R = self.Rot.from_rotvec(s[3:6]).as_matrix()
        return math.atan2(R[1, 0], R[0, 0])

    def knob_q(self):
        return float(self.sim.data.get_joint_qpos(KNOB_JOINT))

    def knob_contacts(self):
        n = 0
        for i in range(int(self.D.ncon)):
            c = self.D.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if c.dist <= c.includemargin and ((g1 in self.knob_contact_geoms and self.robot_geoms[g2])
                                              or (g2 in self.knob_contact_geoms and self.robot_geoms[g1])):
                n += 1
        return n

    # ---------------- motion ----------------
    def step(self, action, phase):
        self.resp = self.w.step([float(a) for a in action])
        self.record(phase, action)

    def goto(self, target, Rt, grip, phase, n=200, tol=0.003, gain=1.5):
        np = self.np
        for _ in range(n):
            s = self.state8()
            p, R = s[:3], self.Rot.from_rotvec(s[3:6])
            dp = np.clip((np.asarray(target) - p) / 0.05 * gain, -1, 1)
            rv = (Rt * R.inv()).as_rotvec()
            self.step([*dp, *np.clip(rv / 0.5 * gain, -1, 1), grip], phase)
            if np.linalg.norm(np.asarray(target) - p) < tol and np.linalg.norm(rv) < 0.02:
                break

    def hold(self, n, grip, phase):
        for _ in range(n):
            self.step([0, 0, 0, 0, 0, 0, grip], phase)

    # ---------------- per-step record ----------------
    def capture(self):
        obs = self.resp["observation"]
        images = {k: decode(obs[field]) for k, (_, field) in CAMERAS.items()}
        out = {}
        plate = self.plate()
        for key in CAMERAS:
            cam = self.camera(key)
            pts, region, disk = self.plane_hits(cam, plate)
            hand, knob, robot_raw, ids = self.masks(key)
            out[key] = {"rgb": images[key], "lab": self.lab(images[key]), "cam": cam, "pts": pts, "region": region,
                        "disk": disk, "hand": hand, "knob": knob}
        return out

    def lab(self, rgb):
        from skimage.color import rgb2lab
        return rgb2lab(rgb.astype(float) / 255.0)

    def set_reference(self, name, frame):
        self.reference[name] = {k: {"lab": v["lab"], "region": v["region"], "hand": v["hand"],
                                    "knob": v["knob"], "cam": v["cam"], "rgb": v["rgb"]}
                                for k, v in frame.items()}

    def change(self, key, now, ref, tau):
        """|Delta Lab| of the current region against the reference (per pixel for the static agentview; through
        the burner plane for the moving wrist camera). Statistics for two region variants: ``plate`` (the whole
        19 x 19 cm plate top, i.e. the stove's metal box top around the grate) and ``disk`` (the dark ring grate,
        r <= 6.25 cm around the burner site)."""
        np = self.np
        lab_now = now["lab"]
        region = now["region"] & ~now["hand"] & ~now["knob"]
        n_region = int(now["region"].sum())
        if key == "agentview":
            ok = region & ref["region"] & ~ref["hand"] & ~ref["knob"]
            d_lab = lab_now - ref["lab"]
        else:
            r, c, depth = self.project(ref["cam"], now["pts"])
            ri, ci = np.rint(r).astype(int), np.rint(c).astype(int)
            inb = (ri >= 0) & (ri < self.size) & (ci >= 0) & (ci < self.size) & (depth > 0)
            ri_c, ci_c = np.clip(ri, 0, self.size - 1), np.clip(ci, 0, self.size - 1)
            ref_ok = ref["region"][ri_c, ci_c] & ~ref["hand"][ri_c, ci_c] & ~ref["knob"][ri_c, ci_c]
            ok = region & inb & ref_ok
            d_lab = lab_now - ref["lab"][ri_c, ci_c]
        de = np.linalg.norm(d_lab, axis=-1)
        changed = ok & (de > tau)
        red = ok & (d_lab[..., 1] > tau)
        rgb = now["rgb"].astype(int)
        legacy = (rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2])) > 80
        stats = {"region_px": n_region, "valid_px": int(ok.sum()),
                 "valid_fraction_of_region": round(int(ok.sum()) / n_region, 4) if n_region else None,
                 "hand_occluded_fraction": round(float((now["region"] & now["hand"]).sum()) / n_region, 4)
                 if n_region else None}
        for name, mask in (("plate", ok), ("disk", ok & now["disk"])):
            n = int(mask.sum())
            stats[name] = {"valid_px": n,
                           "changed_fraction": round(float((changed & mask).sum()) / n, 5) if n else None,
                           "red_shift_fraction": round(float((red & mask).sum()) / n, 5) if n else None,
                           "mean_delta_a": round(float(d_lab[..., 1][mask].mean()), 4) if n else None,
                           "mean_delta_e": round(float(de[mask].mean()), 4) if n else None,
                           "legacy_abs_red_fraction": round(float((legacy & mask).sum()) / n, 5) if n else None}
        return stats, de, ok, changed

    def record(self, phase, action):
        s = self.state8()
        frame = self.capture()
        rec = {"i": len(self.records), "step": self.w.status["steps"], "phase": phase,
               "action": [round(float(a), 4) for a in action],
               "knob_q": round(self.knob_q(), 5), "knob_qd": round(float(self.sim.data.get_joint_qvel(KNOB_JOINT)), 5),
               "burner_site_alpha": float(self.M.site_rgba[self.burner_site][3]),
               "libero_success": bool(self.w.status.get("libero_success")),
               "grip_xyz": [round(float(x), 5) for x in s[:3]], "hand_yaw_deg": round(math.degrees(self.yaw(s)), 3),
               "gap_mm": round((s[6] - s[7]) * 1000, 3), "finger_qpos": [round(float(s[6]), 5), round(float(s[7]), 5)],
               "gripper_command": float(action[6]), "robot_knob_contacts": self.knob_contacts(),
               "forbidden_contact_points": (self.w.status.get("collision") or {}).get("counts", {}).get(
                   "forbidden_contact_points")}
        rec["cams"] = {}
        for name, ref in self.reference.items():
            for key in CAMERAS:
                stats, de, ok, changed = self.change(key, frame[key], ref[key], self.tau.get(key, O1_PIXEL_FLOOR))
                rec["cams"].setdefault(key, {})[name] = stats
        if self.onset_frames is None and rec["knob_q"] >= TURNON_Q:
            self.onset_frames = {"pre_cross": self.last_frame, "cross": frame, "cross_index": rec["i"]}
        elif self.onset_frames is not None and "cross_plus1" not in self.onset_frames:
            self.onset_frames["cross_plus1"] = frame
        self.records.append(rec)
        self.last_frame = frame
        return frame


def first_run(values, threshold, frames, start=0):
    run = 0
    for i, v in enumerate(values):
        if i < start:
            continue
        run = run + 1 if (v is not None and v >= threshold) else 0
        if run >= frames:
            return i - frames + 1, i
    return None


STATS = ("changed_fraction", "red_shift_fraction", "mean_delta_a")
MEAN_A_RESOLUTION = 1e-4         # the recorded rounding of mean Delta a* (a sigma floor for the ratio only)
VARIANTS = ("plate", "disk")


def summarize(P):
    """Noise, effect, separation, O1 detection and monotonicity per camera x region variant x statistic.

    Pre-onset = the static pre-twist hold frames (against the pre-twist reference) plus every twist frame whose
    image was rendered before the burner turned red (q < 0.5 at the end of the PREVIOUS step: LIBERO toggles the
    burner site in the step after the TurnOn predicate first holds). Post-onset = every later frame."""
    np = P.np
    recs = P.records
    out = {}
    first_twist = next(i for i, r in enumerate(recs) if r["phase"] == "twist")
    yaw0 = recs[first_twist - 1]["hand_yaw_deg"]
    cross = next((i for i, r in enumerate(recs) if r["knob_q"] >= TURNON_Q), None)
    visible = None if cross is None else cross + 1
    out["truth"] = {"q_crossing": None if cross is None else _pose(recs[cross], yaw0),
                    "first_red_frame": None if visible is None or visible >= len(recs) else _pose(recs[visible],
                                                                                                    yaw0),
                    "note": "the burner site is toggled in the step AFTER the TurnOn predicate first holds, so the "
                            "first red frame is one control step (50 ms) after the q = 0.5 crossing"}
    pre_static_idx = [i for i, r in enumerate(recs) if r["phase"] == "hold_pre_twist"
                      and "pretwist" in r["cams"]["agentview"]]
    pre_twist_idx = [i for i in range(first_twist, visible if visible is not None else len(recs))
                     if recs[i]["phase"] == "twist"]
    post_idx = [] if visible is None else list(range(visible, len(recs)))
    post_twist_idx = [i for i in post_idx if recs[i]["phase"] == "twist"]
    approach_idx = [i for i, r in enumerate(recs)
                    if r["phase"] in ("static_pre", "approach_above", "descend", "close", "hold_pre_twist")]
    for key in CAMERAS:
        cam = {}
        for variant in VARIANTS:
            block = {}
            for stat in STATS:
                series = [((r["cams"][key].get("pretwist") or {}).get(variant) or {}).get(stat) for r in recs]
                reset_series = [((r["cams"][key].get("reset") or {}).get(variant) or {}).get(stat) for r in recs]
                pre_s = [series[i] for i in pre_static_idx if series[i] is not None]
                pre_t = [series[i] for i in pre_twist_idx if series[i] is not None]
                pre = pre_s + pre_t
                post = [series[i] for i in post_idx if series[i] is not None]
                post_tw_idx = [i for i in post_twist_idx if series[i] is not None]
                post_tw = [series[i] for i in post_tw_idx]
                appr = [reset_series[i] for i in approach_idx if reset_series[i] is not None]
                b = {"pre_static": _stats(pre_s), "pre_onset_twist": _stats(pre_t), "approach_vs_reset": _stats(appr),
                     "post_onset": _stats(post)}
                if stat == "mean_delta_a":
                    # a magnitude statistic (Lab a* units), not an O1 fraction: reported, never thresholded here
                    mu = float(np.mean(pre)) if pre else 0.0
                    sd = float(np.std(pre)) if pre else 0.0
                    b["pre_mean"], b["pre_std"] = _r(mu), _r(sd)
                    b["pre_max_abs"] = _r(max(abs(v) for v in pre)) if pre else None
                    if post:
                        b["post_min_over_pre_std_floored"] = _r((min(post) - mu) / max(sd, MEAN_A_RESOLUTION))
                else:
                    f_static = float(np.mean(pre_s)) if pre_s else 0.0
                    f_pre = float(np.mean(pre)) if pre else 0.0
                    for name, rate in (("o1_calibrate_static_rate", f_static), ("o1_pre_onset_rate", f_pre)):
                        thr = max(O1_FRACTION_FLOOR, O1_FALSE_RATE_K * rate)
                        hit = first_run(series, thr, O1_FRAMES, start=first_twist)
                        b[name] = {"false_change_rate": _r(rate), "threshold": _r(thr),
                                   "detection": _hit(recs, hit, visible, yaw0)}
                    if post and pre:
                        b["separation"] = {"min_post_over_max_pre": _r(min(post) / max(pre)) if max(pre) > 0
                                           else "inf",
                                           "min_post_over_mean_pre": _r(min(post) / f_pre) if f_pre > 0 else "inf",
                                           "min_post_minus_max_pre": _r(min(post) - max(pre))}
                if post_tw:
                    d = np.diff(np.asarray(post_tw))
                    b["monotonicity_during_twist_after_onset"] = {
                        "frames": len(post_tw), "decreases": int((d < -1e-4).sum()),
                        "largest_decrease": _r(float(-d.min())) if d.size else None,
                        "range": [_r(min(post_tw)), _r(max(post_tw))],
                        "spearman_vs_q": _spearman(np.array([recs[i]["knob_q"] for i in post_tw_idx]),
                                                   np.asarray(post_tw))}
                if pre_t:
                    b["pre_onset_spearman_vs_q"] = _spearman(np.array([recs[i]["knob_q"] for i in pre_twist_idx
                                                                       if series[i] is not None]), np.asarray(pre_t))
                if visible is not None and visible < len(recs) and series[visible] is not None \
                        and series[visible - 1] is not None:
                    b["jump_at_first_red_frame"] = _r(series[visible] - series[visible - 1])
                block[stat] = b
            cam[variant] = block
        cam["wrist_region_coverage" if key == "wrist" else "region_coverage"] = {
            "valid_fraction_of_region_during_twist": _stats(
                [recs[i]["cams"][key]["pretwist"]["valid_fraction_of_region"] for i in pre_twist_idx + post_twist_idx
                 if recs[i]["cams"][key]["pretwist"]["valid_fraction_of_region"] is not None]),
            "region_px_during_twist": _stats([recs[i]["cams"][key]["pretwist"]["region_px"]
                                              for i in pre_twist_idx + post_twist_idx]),
            "hand_occluded_fraction_during_twist": _stats(
                [recs[i]["cams"][key]["pretwist"]["hand_occluded_fraction"] for i in pre_twist_idx + post_twist_idx
                 if recs[i]["cams"][key]["pretwist"]["hand_occluded_fraction"] is not None])}
        disk_px = [(recs[i]["hand_yaw_deg"], recs[i]["cams"][key]["pretwist"]["disk"]["valid_px"])
                   for i in pre_twist_idx + post_twist_idx]
        first_px = disk_px[0][1] if disk_px else 0
        lost = next((v for v in disk_px if v[1] < 0.1 * first_px), None)
        cam["disk_visibility_during_twist"] = {
            "valid_disk_px_at_twist_start": first_px,
            "valid_disk_px_at_first_red_frame": (recs[visible]["cams"][key]["pretwist"]["disk"]["valid_px"]
                                                 if visible is not None and visible < len(recs) else None),
            "hand_yaw_from_twist_start_deg_when_below_10pct": None if lost is None else _dyaw(lost[0], yaw0)}
        cam["sigma_px_deltaE_static"] = P.sigma.get(key)
        cam["pixel_threshold_deltaE"] = P.tau.get(key)
        out[key] = cam
    twist = [r for r in recs if r["phase"] == "twist"]
    out["knob"] = {"q_at_twist_start": recs[first_twist - 1]["knob_q"], "q_max": max(r["knob_q"] for r in recs),
                   "q_end": recs[-1]["knob_q"],
                   "spring_back_after_release": round(max(r["knob_q"] for r in recs) - recs[-1]["knob_q"], 5),
                   "hand_yaw_total_deg": _dyaw(twist[-1]["hand_yaw_deg"], yaw0),
                   "knob_over_hand_yaw": _r(math.degrees(twist[-1]["knob_q"] - recs[first_twist - 1]["knob_q"])
                                            / _dyaw(twist[-1]["hand_yaw_deg"], yaw0), 4),
                   "gap_mm_at_close": recs[first_twist - 1]["gap_mm"],
                   "gap_mm_during_twist": [min(r["gap_mm"] for r in twist), max(r["gap_mm"] for r in twist)],
                   "robot_knob_contact_frames_during_twist": f"{sum(1 for r in twist if r['robot_knob_contacts'])}"
                                                             f"/{len(twist)}",
                   "turnon_at_end": recs[-1]["knob_q"] >= TURNON_Q,
                   "libero_task_success_end": recs[-1]["libero_success"],
                   "forbidden_contact_points_end": recs[-1]["forbidden_contact_points"]}
    return out


def _pose(r, yaw0):
    return {"i": r["i"], "step": r["step"], "phase": r["phase"], "knob_q": r["knob_q"],
            "knob_deg": round(math.degrees(r["knob_q"]), 2), "hand_yaw_from_twist_start_deg": _dyaw(r["hand_yaw_deg"],
                                                                                                   yaw0)}


def _hit(recs, hit, visible, yaw0):
    if hit is None:
        return None
    a, b = recs[hit[0]], recs[hit[1]]
    return {"first_frame": _pose(a, yaw0), "detected_frame": _pose(b, yaw0),
            "before_first_red_frame": visible is not None and hit[0] < visible,
            "frames_after_first_red_frame": None if visible is None else b["i"] - visible}


def _stats(values):
    import numpy as np
    values = [v for v in values if v is not None]
    if not values:
        return None
    a = np.asarray(values, float)
    return {"n": int(a.size), "mean": _r(a.mean()), "std": _r(a.std()), "min": _r(a.min()),
            "p95": _r(np.percentile(a, 95)), "max": _r(a.max())}


def _dyaw(y, y0):
    return round((y - y0 + 180.0) % 360.0 - 180.0, 2)


def _r(x, n=5):
    return None if x is None else round(float(x), n)


def _spearman(a, b):
    import numpy as np
    from scipy.stats import spearmanr
    if len(a) < 3 or np.all(b == b[0]):
        return None
    return _r(spearmanr(a, b).correlation, 4)


def save_frames(P, out_dir, tags):
    from PIL import Image
    np = P.np
    frames = out_dir / "frames"
    frames.mkdir(exist_ok=True)
    paths = {}
    for tag, frame in tags.items():
        for key, f in frame.items():
            rgb = f["rgb"]
            Image.fromarray(rgb).save(frames / f"{tag}_{key}.png")
            over = rgb.astype(float).copy()
            edge = f["region"] & ~np.roll(f["region"], 1, 0) | f["region"] & ~np.roll(f["region"], 1, 1) \
                | f["region"] & ~np.roll(f["region"], -1, 0) | f["region"] & ~np.roll(f["region"], -1, 1)
            over[f["hand"]] = 0.5 * over[f["hand"]] + 0.5 * np.array([60, 60, 220])
            over[f["knob"]] = 0.5 * over[f["knob"]] + 0.5 * np.array([230, 160, 0])
            over[edge] = [0, 255, 255]
            Image.fromarray(np.uint8(over)).save(frames / f"{tag}_{key}_regions.png")
            ref = P.reference.get("pretwist")
            if ref is not None:
                stats, de, ok, changed = P.change(key, f, ref[key], P.tau.get(key, O1_PIXEL_FLOOR))
                vis = np.zeros(rgb.shape, np.uint8)
                vis[ok] = [40, 40, 40]
                vis[changed] = [90, 140, 255]                 # changed (|Delta Lab| > tau), not a red rise
                red = ok & ((f["lab"] - ref[key]["lab"])[..., 1] > P.tau.get(key, O1_PIXEL_FLOOR)) \
                    if key == "agentview" else changed & False
                vis[red] = [255, 60, 60]                      # agentview: a* rise > tau
                Image.fromarray(vis).save(frames / f"{tag}_{key}_change.png")
            paths[f"{tag}_{key}"] = str(frames / f"{tag}_{key}.png")
    return paths


def plots(P, out_dir, summary):
    """Small multiples sharing the step axis (one y scale per panel), plus the change against the knob angle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np = P.np
    recs = P.records
    x = np.array([r["i"] for r in recs])
    s1, s2, s3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (dataviz reference palette)
    ink, muted, grid, band = "#0b0b0b", "#52514e", "#e4e3df", "#f3f2ee"
    phases = []
    for r in recs:
        if not phases or phases[-1][0] != r["phase"]:
            phases.append([r["phase"], r["i"], r["i"]])
        phases[-1][2] = r["i"]
    cross = summary["truth"]["q_crossing"]

    def style(ax, ylabel, bands=True):
        ax.set_ylabel(ylabel, color=muted, fontsize=8)
        ax.grid(True, color=grid, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted, labelsize=7)
        if bands:
            for k, (_, a, b) in enumerate(phases):
                if k % 2:
                    ax.axvspan(a - 0.5, b + 0.5, color=band, zorder=0, linewidth=0)
            if cross is not None:
                ax.axvline(cross["i"], color=muted, linewidth=0.8, linestyle="--")

    def series(key, variant, stat, ref="pretwist"):
        return np.array([np.nan if ((r["cams"][key].get(ref) or {}).get(variant) or {}).get(stat) is None
                         else r["cams"][key][ref][variant][stat] for r in recs], float)
    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    q = np.array([r["knob_q"] for r in recs])
    axes[0].plot(x, q, color=ink, linewidth=1.5)
    axes[0].axhline(TURNON_Q, color=muted, linewidth=0.8, linestyle=":")
    axes[0].text(x[0], TURNON_Q, " TurnOn q = 0.5", color=muted, fontsize=7, va="bottom")
    style(axes[0], "knob q (rad, truth)")
    top = max(1.0, float(q.max()))
    for name, a, b in phases:
        axes[0].text((a + b) / 2, top, name.replace("_", " "), ha="center", va="top", fontsize=6, color=muted,
                     rotation=90)
    for ax, key in ((axes[1], "agentview"), (axes[2], "wrist")):
        ax.plot(x, series(key, "plate", "changed_fraction"), color=s1, linewidth=1.5, label="plate: changed")
        ax.plot(x, series(key, "disk", "changed_fraction"), color=s2, linewidth=1.5, label="ring disk: changed")
        ax.plot(x, series(key, "plate", "red_shift_fraction"), color=s3, linewidth=1.5, label="plate: a* rise")
        thr = summary[key]["plate"]["changed_fraction"]["o1_calibrate_static_rate"]["threshold"]
        ax.axhline(thr, color=muted, linewidth=0.8, linestyle=":")
        ax.text(x[-1], thr, f"O1 floor {thr:.2f} ", color=muted, fontsize=7, ha="right", va="bottom")
        ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
        style(ax, f"{key}\nfraction vs pre-twist")
    axes[3].plot(x, series("agentview", "plate", "changed_fraction", "reset"), color=s1, linewidth=1.5,
                 label="agentview plate")
    axes[3].plot(x, series("wrist", "plate", "changed_fraction", "reset"), color=s2, linewidth=1.5,
                 label="wrist plate")
    axes[3].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    style(axes[3], "changed fraction\nvs reset frame")
    axes[4].plot(x, series("agentview", "disk", "mean_delta_a"), color=s1, linewidth=1.5, label="agentview disk")
    axes[4].plot(x, series("wrist", "disk", "mean_delta_a"), color=s2, linewidth=1.5, label="wrist disk")
    axes[4].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    style(axes[4], "mean Delta a*\nvs pre-twist")
    axes[5].plot(x, [r["gap_mm"] for r in recs], color=ink, linewidth=1.5)
    style(axes[5], "finger gap (mm)")
    axes[5].set_xlabel("probe frame (one control step each; dashed = q crosses 0.5)", color=muted, fontsize=8)
    fig.suptitle("libero_10 task 2 stove-twist effect probe (diagnosis only)", fontsize=10, color=ink)
    p1 = out_dir / "effect_probe.png"
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    tw = [r for r in recs if r["phase"] == "twist"]
    qs = [r["knob_q"] for r in tw]
    for ax, key in zip(axes, ("agentview", "wrist")):
        for variant, stat, color, label in (("plate", "changed_fraction", s1, "plate: changed"),
                                            ("disk", "changed_fraction", s2, "ring disk: changed"),
                                            ("plate", "red_shift_fraction", s3, "plate: a* rise")):
            ys = [np.nan if r["cams"][key]["pretwist"][variant][stat] is None else
                  r["cams"][key]["pretwist"][variant][stat] for r in tw]
            ax.plot(qs, ys, color=color, linewidth=1.2, marker="o", markersize=2.5, label=label)
        ax.axvline(TURNON_Q, color=muted, linewidth=0.8, linestyle="--")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
        style(ax, f"{key}: fraction vs pre-twist", bands=False)
        ax.set_xlabel("knob q (rad, truth) during the twist", color=muted, fontsize=8)
    p2 = out_dir / "effect_vs_angle.png"
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    return [str(p1), str(p2)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("out exists; choose a fresh directory")
    args.out.mkdir(parents=True)
    began = time.time()
    P = Probe()
    np = P.np
    try:
        obs = P.resp["observation"]
        images = {k: decode(obs[f]) for k, (_, f) in CAMERAS.items()}
        flip = P.check_flip(images)
        frame0 = P.capture()
        # the analytic plate region against the renderer's segmentation at reset (what the region pixels show)
        labels = {}
        for key in CAMERAS:
            seg = P.to_policy(P.raw_render(key, segmentation=True))
            reg = frame0[key]["region"]
            counts = {}
            for t, i in zip(seg[..., 0][reg].tolist(), seg[..., 1][reg].tolist()):
                if t == int(P.mj.mjtObj.mjOBJ_GEOM):
                    name = "geom:" + (P.mj.mj_id2name(P.M, P.mj.mjtObj.mjOBJ_GEOM, i) or
                                      "body:" + (P.mj.mj_id2name(P.M, P.mj.mjtObj.mjOBJ_BODY,
                                                                 int(P.M.geom_bodyid[i])) or str(i)))
                elif t == int(P.mj.mjtObj.mjOBJ_SITE):
                    name = "site:" + (P.mj.mj_id2name(P.M, P.mj.mjtObj.mjOBJ_SITE, i) or str(i))
                else:
                    name = f"type{t}"
                counts[name] = counts.get(name, 0) + 1
            total = max(1, int(reg.sum()))
            labels[key] = {"region_px": int(reg.sum()), "top_labels": {k: round(v / total, 4) for k, v in
                                                                       sorted(counts.items(), key=lambda kv: -kv[1])[:6]}}
        P.set_reference("reset", frame0)
        tags = {"reset": frame0}
        P.record("reset", [0, 0, 0, 0, 0, 0, -1.0])     # the reset frame itself (no step)
        P.hold(10, -1.0, "static_pre")
        knob = np.asarray(P.D.xpos[P.mj.mj_name2id(P.M, P.mj.mjtObj.mjOBJ_BODY, KNOB_BODY)], float).copy()
        s0 = P.state8()
        R0 = P.Rot.from_rotvec(s0[3:6])
        P.goto([knob[0], knob[1], 1.02], R0, -1.0, "approach_above")
        P.goto([knob[0], knob[1], GRIP_Z_AT_KNOB], R0, -1.0, "descend", n=150, tol=0.003)
        P.hold(20, 1.0, "close")
        P.hold(10, 1.0, "hold_pre_twist")
        # pre-twist reference = the last hold frame. sigma_px from 5 further static hold frames against it (their
        # records use the 12 floor; every later frame uses tau = max(12, 4 sigma_px))
        P.set_reference("pretwist", P.last_frame)
        tags["pretwist"] = P.last_frame
        extra = []
        for _ in range(5):
            P.step([0, 0, 0, 0, 0, 0, 1.0], "hold_pre_twist")
            extra.append(P.last_frame)
        for key in CAMERAS:
            des = [P.change(key, fr[key], P.reference["pretwist"][key], O1_PIXEL_FLOOR) for fr in extra]
            allde = np.concatenate([de[ok] for _, de, ok, _ in des])
            sigma = float(np.sqrt(np.mean(allde ** 2))) if allde.size else 0.0
            P.sigma[key] = round(sigma, 5)
            P.tau[key] = round(max(O1_PIXEL_FLOOR, O1_SIGMA_K * sigma), 5)
        for k in range(1, YAW_INCREMENTS + 1):
            Rt = P.Rot.from_euler("z", YAW_INCREMENT_RAD * k) * R0
            P.goto([knob[0], knob[1], GRIP_Z_AT_KNOB], Rt, 1.0, "twist", n=40, tol=0.004)
        if P.onset_frames is not None:
            for tag in ("pre_cross", "cross", "cross_plus1"):
                if tag in P.onset_frames:
                    tags[tag] = P.onset_frames[tag]
        tags["twist_end"] = P.last_frame
        P.hold(15, 1.0, "hold_post")
        P.hold(15, -1.0, "release")
        P.goto([knob[0], knob[1], 1.06], P.Rot.from_euler("z", YAW_INCREMENT_RAD * YAW_INCREMENTS) * R0, -1.0,
               "retreat", n=100)
        P.hold(10, -1.0, "static_post")
        tags["after_retreat"] = P.last_frame
        (args.out / "records.partial.json").write_text(json.dumps(P.records))
        summary = summarize(P)
        (args.out / "records.partial.json").unlink()
        frame_paths = save_frames(P, args.out, tags)
        plot_paths = plots(P, args.out, summary)
        out = {"diagnosis_only": True, "model_calls": 0, "evaluator_positioned": True, "task_key": TASK_KEY,
               "episode": 0, "config": {"seed": P.config.seed, "image_size": P.size,
                                        "settle_steps": P.config.settle_steps, "suite": P.config.suite},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "recipe": {"grip_z_at_knob": GRIP_Z_AT_KNOB, "yaw_increment_rad": YAW_INCREMENT_RAD,
                          "yaw_increments": YAW_INCREMENTS, "close_steps": 20, "source": "SURVEY oracle_t2_t7.py"},
               "o1_rule": {"pixel": f"|Delta Lab| > max({O1_PIXEL_FLOOR}, {O1_SIGMA_K} sigma_px)",
                           "fraction": f">= max({O1_FRACTION_FLOOR}, {O1_FALSE_RATE_K} x false-change rate) for "
                                       f"{O1_FRAMES} consecutive frames", "mask_dilate_px": MASK_DILATE_PX,
                           "red_shift": "Delta a* > the same pixel threshold (a rise toward red)"},
               "frame_mapping": flip, "region_labels_at_reset": labels, "frame_tags": sorted(tags),
               "summary": summary, "frames": frame_paths, "plots": plot_paths,
               "records": P.records, "wall_s": round(time.time() - began, 1)}
        (args.out / "effect_probe.json").write_text(json.dumps(out, indent=1, allow_nan=False))
        print(json.dumps({"frame_mapping": flip, "region_labels_at_reset": labels, "summary": summary}, indent=1))
    finally:
        P.w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
