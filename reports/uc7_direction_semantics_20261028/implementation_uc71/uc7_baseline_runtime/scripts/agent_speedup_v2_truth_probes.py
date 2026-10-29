"""Strict-truth probe cases for the D1 orientation / perch decision (DIAGNOSIS ONLY; zero model calls; spec §6.5).

D1 (decided 2026-10-18, lead default; the user may override): strict 'inside' does not require orientation_kept
(an object resting in the cavity with floor or contents support counts even when it leans on a wall or the rim,
as long as it is not perched); strict 'on' (object, fixture region, table zone) does (a tipped pot on the burner
fails). The cases reproduce the M0 truth reviewers' probes with evaluator teleports into the real V2LiberoWorker,
then run the worker's own ``truth`` -> ``settle_truth`` ops and AC-2's ``physical_summary``:

- drawer (libero_goal 3, drawer opened by qpos): bowl on the floor centre; the run-003 corner perch (origin at the
  run's terminal xy, foot 1.5 mm above the rim); the front-wall lean (bowl centre on the front inner wall, foot
  1.5 mm above the rim: it tips onto the wall top with a floor contact); the bowl upside down on the floor;
- t2 (libero_10 2, knob q = 0.6 by qpos): the pot upright on the burner centre; on its side (90 deg about y);
  upside down; dropped on the knob side (dx -0.09, it tips over);
- bowl (libero_spatial 0): the bowl upright on the plate; upside down on the plate.

Each task runs in its own worker process (one worker per process, one reset). Teleports and joint writes are
evaluator-only; nothing here is a policy input.

usage: MUJOCO_EGL_DEVICE_ID=2 /workspace/venvs/libero-plus/bin/python -B scripts/agent_speedup_v2_truth_probes.py \
           --out runs/agent_speedup_v2_m0close_truth_probes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TRUTH_SOURCE = ROOT / "src" / "gpt_field" / "agent_speedup_v2_truth.py"
TASKS = {"drawer": ("drawer_bowl", "libero_goal", 3), "t2": ("libero_10_t2", "libero_10", 2),
         "bowl": ("bowl_plate", "libero_spatial", 0)}
ZERO = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
# expected AC-2 outcome per case under D1: (physical_success, verdict)
EXPECTED = {
    "drawer:bowl_floor_centre": (True, "physical_success"),
    "drawer:bowl_corner_perch_run003": (False, "lenient_success"),
    "drawer:bowl_front_wall_lean": (True, "physical_success"),
    "drawer:bowl_upside_down_in_drawer": (True, "physical_success"),
    "t2:pot_upright_centred": (True, "physical_success"),
    "t2:pot_side_y90_centred": (False, None),
    "t2:pot_upside_down_centred": (False, None),
    "t2:pot_knob_side_dx-0.09": (False, None),
    "bowl:bowl_upright_on_plate": (True, "physical_success"),
    "bowl:bowl_upside_down_on_plate": (False, None),
}


class Probe:
    def __init__(self, which):
        from dataclasses import replace
        import numpy as np
        from gpt_field.agent_speedup_v2_sim import V2LiberoWorker
        from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config
        self.np = np
        key, _suite, task_id = TASKS[which]
        config, monitor = build_v2_sim_config(key)
        config = replace(config, stop_on_success=False, render_size=128, image_size=128)
        self.w = V2LiberoWorker(config, monitor)
        self.resp = self.w.reset(task_id, 0)
        self.env, self.sc = self.w.env, self.w.scene
        self.D, self.M = self.sc.D, self.sc.M
        self.gripper = self.env.robots[0].gripper
        self.results = []
        self.base = self.save()

    def save(self):
        D, np = self.D, self.np
        return {"qpos": D.qpos.copy(), "qvel": D.qvel.copy(), "act": D.act.copy(), "time": float(D.time),
                "warm": D.qacc_warmstart.copy(), "ctrl": D.ctrl.copy(),
                "grip": np.array(self.gripper.current_action).copy(), "qfrc_applied": D.qfrc_applied.copy(),
                "xfrc_applied": D.xfrc_applied.copy()}

    def restore(self, s):
        D = self.D
        D.qpos[:] = s["qpos"]; D.qvel[:] = s["qvel"]; D.act[:] = s["act"]; D.time = s["time"]
        D.ctrl[:] = s["ctrl"]; D.qfrc_applied[:] = s["qfrc_applied"]; D.xfrc_applied[:] = s["xfrc_applied"]
        self.gripper.current_action = s["grip"].copy()
        self.sc.mj.mj_forward(self.M, D)
        D.qacc_warmstart[:] = s["warm"]
        self.w.counterfactual_applied = False
        self.w.status["terminated"] = False
        if self.w.window is not None:
            self.w.window.reset()

    def settle(self, n):
        for _ in range(n):
            self.resp = self.w.step(list(ZERO))

    def set_joint(self, name, value):
        data = self.env.sim.data
        data.set_joint_qpos(name, value)
        v = data.get_joint_qvel(name)
        data.set_joint_qvel(name, self.np.zeros_like(v) if self.np.ndim(v) else 0.0)
        self.env.sim.forward()

    def teleport(self, obj, pos, quat_wxyz):
        joint = self.sc.core.get_object(obj).joints[-1]
        self.set_joint(joint, self.np.r_[self.np.asarray(pos, float), self.np.asarray(quat_wxyz, float)])

    def quat(self, obj):
        return self.np.array(self.D.xquat[self.sc.core.obj_body_id[obj]], float).copy()

    def tilted(self, quat_wxyz, axis, deg):
        from scipy.spatial.transform import Rotation
        q = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        x, y, z, w = (Rotation.from_rotvec(self.np.asarray(axis, float) * math.radians(deg)) * q).as_quat()
        return self.np.array([w, x, y, z])

    def low_offset(self, obj):
        pts = self.sc.pts_of(self.sc.geoms_of(self.sc.object_bodies(obj)))
        return float(self.D.xpos[self.sc.core.obj_body_id[obj]][2] - pts[:, 2].min())

    def evaluate(self, which, tag, note):
        from gpt_field import agent_speedup_v2_truth as T
        raw = bool(self.w.status.get("libero_success"))
        terminal = self.w.truth()
        settled = self.w.settle_truth()
        summary = T.physical_summary(terminal, settled, raw_libero_success=raw)

        def atoms(ev):
            keep = ("tilt_deg", "tilt_change_from_reset_deg", "orientation_required", "orientation_kept",
                    "rim_height_normal_force_share", "support_contacts", "support_contacts_below_rim",
                    "deepest_support_contact_below_rim_mm", "low_below_rim_mm", "low_above_floor_mm", "high_z",
                    "rim_z", "container_contacts", "com_margin_region_mm", "com_margin_target_mm", "env_contacts")
            return [{"atom": a["atom"], "kind": a["kind"], "raw": a["raw"], "strict": a["strict"],
                     "why_not": a["why_not"], "numbers": {k: a["numbers"].get(k) for k in keep
                                                          if k in a.get("numbers", {})}}
                    for a in ev["atoms"]]
        key = f"{which}:{tag}"
        expected = EXPECTED.get(key)
        record = {"case": key, "note": note, "raw_status": raw, "terminal": atoms(terminal),
                  "after_settle": atoms(settled),
                  "summary": {k: summary[k] for k in ("verdict", "physical_success", "physical_success_at_terminal",
                                                      "physical_success_after_settle",
                                                      "terminal_failure_is_hold_only", "terminal_failing_checks",
                                                      "raw_libero_success_after_settle", "v2_success")},
                  "expected_physical_success": None if expected is None else expected[0],
                  "expected_verdict": None if expected is None else expected[1]}
        record["matches_expectation"] = (expected is None or (summary["physical_success"] == expected[0] and (
            expected[1] is None or summary["verdict"] == expected[1])))
        self.results.append(T.jsonable(record))
        print(f"[{key}] raw={raw} verdict={summary['verdict']} physical={summary['physical_success']} "
              f"failing={summary['terminal_failing_checks']} ok={record['matches_expectation']}", file=sys.stderr,
              flush=True)

    def close(self):
        self.w.close()


def run_cases(which):
    P = Probe(which)
    sc, D, np = P.sc, P.D, P.np
    try:
        if which == "drawer":
            bowl, joint = "akita_black_bowl_1", "wooden_cabinet_1_top_level"
            P.set_joint(joint, -0.1177)
            P.settle(10)
            opened = P.save()
            cv = sc.cavity(sc.site("wooden_cabinet_1_top_region"))
            bx, floor, rim = cv["cavity_xy_box"], cv["floor_z"], cv["rim_z"]
            q0, om = P.quat(bowl), P.low_offset(bowl)
            ccx, ccy = (bx[0] + bx[1]) / 2, (bx[2] + bx[3]) / 2
            cases = [("bowl_floor_centre", [ccx, ccy, floor + 0.002 + om], q0, 40,
                      "bowl teleported upright onto the open drawer floor centre; 40 zero steps"),
                     ("bowl_corner_perch_run003", [0.118, -0.0737, rim + 0.0015 + om], q0, 40,
                      "bowl dropped at drawer-003's terminal origin xy (0.118, -0.0737), foot 1.5 mm above the "
                      "rim; 40 zero steps (no floor contact)"),
                     ("bowl_front_wall_lean", [ccx, bx[3], rim + 0.0015 + om], q0, 40,
                      "bowl centre on the front inner wall, foot 1.5 mm above the rim: it tips onto the wall "
                      "top with a floor contact; 40 zero steps"),
                     ("bowl_upside_down_in_drawer", [ccx, ccy, floor + 0.055], P.tilted(q0, [1, 0, 0], 180.0), 60,
                      "bowl teleported upside down onto the drawer floor; 60 zero steps")]
            for tag, pos, quat, steps, note in cases:
                P.restore(opened)
                P.teleport(bowl, pos, quat)
                P.settle(steps)
                P.evaluate(which, tag, note)
        elif which == "t2":
            pot, knob = "moka_pot_1", "flat_stove_1_button"
            site = sc.site("flat_stove_1_cook_region")
            cx, cy = float(site["pos"][0]), float(site["pos"][1])
            q0, om = P.quat(pot), P.low_offset(pot)
            burner_top = 0.9305
            lever_top = max(sc.aabb(g)[1][2] for g in sc.geoms_of(sc.subtree(sc.joint_info(knob)["body"])))
            cases = [("pot_upright_centred", [cx, cy, burner_top + om + 0.001], q0, 60,
                      "knob q 0.6; pot teleported upright, centred, 1 mm above rest; 60 zero steps"),
                     ("pot_side_y90_centred", [cx, cy, burner_top + 0.07], P.tilted(q0, [0, 1, 0], 90.0), 80,
                      "knob q 0.6; pot teleported on its side (90 deg about y) above the burner centre; 80 steps"),
                     ("pot_upside_down_centred", [cx, cy, burner_top + 0.07], P.tilted(q0, [1, 0, 0], 180.0), 80,
                      "knob q 0.6; pot teleported upside down above the burner centre; 80 steps"),
                     ("pot_knob_side_dx-0.09", [cx - 0.09, cy, lever_top + 0.002 + om], q0, 60,
                      "knob q 0.6; pot dropped 9 cm toward the knob side onto the lever top; it tips over; 60 "
                      "steps")]
            for tag, pos, quat, steps, note in cases:
                P.restore(P.base)
                P.set_joint(knob, 0.6)
                P.teleport(pot, pos, quat)
                P.settle(steps)
                P.evaluate(which, tag, note)
        else:
            bowl, plate = "akita_black_bowl_1", "plate_1"
            pl = np.array(D.xpos[sc.core.obj_body_id[plate]], float)
            top = float(sc.pts_of(sc.geoms_of(sc.object_bodies(plate)))[:, 2].max())
            q0, om = P.quat(bowl), P.low_offset(bowl)
            cases = [("bowl_upright_on_plate", [pl[0], pl[1], top + om + 0.004], q0, 80,
                      "bowl upright, dropped onto the plate centre; 80 zero steps"),
                     ("bowl_upside_down_on_plate", [pl[0], pl[1], top + 0.062], P.tilted(q0, [1, 0, 0], 180.0), 80,
                      "bowl upside down, dropped onto the plate centre; 80 zero steps")]
            for tag, pos, quat, steps, note in cases:
                P.restore(P.base)
                P.teleport(bowl, pos, quat)
                P.settle(steps)
                P.evaluate(which, tag, note)
        return {"which": which, "task_key": TASKS[which][0], "results": P.results}
    finally:
        P.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--which", default="drawer,t2,bowl")
    parser.add_argument("--single")
    args = parser.parse_args(argv)
    if args.single:
        print("ROW " + json.dumps(run_cases(args.single)), flush=True)
        return 0
    if args.out.exists():
        parser.error("out exists; choose a fresh directory")
    args.out.mkdir(parents=True)
    began = time.time()
    rows = []
    for which in args.which.split(","):
        proc = subprocess.run([sys.executable, "-B", __file__, "--single", which], capture_output=True, text=True)
        (args.out / f"{which}.log").write_text(proc.stderr[-20000:])
        if proc.returncode != 0:
            raise SystemExit(proc.stderr[-3000:])
        rows.append(json.loads([line for line in proc.stdout.splitlines() if line.startswith("ROW ")][-1][4:]))
    from gpt_field import agent_speedup_v2_truth as T
    cases = [case for row in rows for case in row["results"]]
    out = {"diagnosis_only": True, "decision": "D1 (decided 2026-10-18, lead default; the user may override)",
           "truth_module_sha256": hashlib.sha256(TRUTH_SOURCE.read_bytes()).hexdigest(),
           "truth_version": T.TRUTH_VERSION, "strict_orientation_kinds": list(T.STRICT_ORIENTATION_KINDS),
           "tolerances": T.TOL, "cases": cases, "all_match": all(c["matches_expectation"] for c in cases),
           "wall_s": round(time.time() - began, 1)}
    (args.out / "truth_probes.json").write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps({c["case"]: [c["summary"]["verdict"], c["matches_expectation"]] for c in cases}, indent=1))
    print("all_match", out["all_match"])
    return 0 if out["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
