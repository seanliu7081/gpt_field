#!/usr/bin/env python3
"""Model-free official cuRobo FK/IK roundtrip; never starts Isaac or an episode.

Uses one explicitly selected GPU. This checks the real native planner import,
initialization and a zero-joint robot-only target; it cannot qualify loaded
contact, simulator integration, inactive-arm tracking, or task performance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.official_control_setup import activate_setup_record, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, default=BASE / "runtime/native/RoboDojo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    start = time.monotonic()
    result = {"schema": "robodojo-official-ik-check-1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "setup": {"path": str(args.setup.resolve()), "sha256": digest(args.setup)},
              "script_sha256": digest(Path(__file__)), "gpu_uuid": args.gpu_uuid,
              "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
              "native_motion_qualified": False, "native_planner_constructed": False,
              "fk_ik_roundtrip_success": False}
    try:
        activate_setup_record(args.setup, output / "cache")
        sys.path.insert(0, str(args.native_root.resolve()))
        import torch
        from env.planner_manager.curobo_planner import CuroboPlanner
        result["cuda_device"] = str(torch.cuda.get_device_name(0))
        planner = CuroboPlanner(robot_origin_pose=[0., 0., 0., 1., 0., 0., 0.],
            active_joints_name=[f"joint{i}" for i in range(1, 7)],
            all_joints=[f"joint{i}" for i in range(1, 7)], dt=.004,
            yml_path=str(args.native_root.resolve() / "Assets/Robots/x5/curobo.yml"),
            table_height=.74 - .765)
        result["native_planner_constructed"] = True
        q = [0.] * 6
        state = planner.motion_planner.compute_kinematics(planner._build_joint_state(q))
        pose = state.tool_poses.to_dict()[planner.ee_link]
        target = pose.position.detach().cpu().reshape(-1).tolist() + pose.quaternion.detach().cpu().reshape(-1).tolist()
        solved = planner.solve_ik_to_joint(q, target, real_robot_pose=[0., 0., 0., 1., 0., 0., 0.])
        result["target_link6_pose"] = target
        result["ik_status"] = solved["status"]
        result["fk_ik_roundtrip_success"] = solved["status"] == "Success"
        if solved["status"] == "Success":
            result["joint_value"] = solved["joint_value"].tolist()
        result["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated(0)
        planner.__del__()
        del planner
        torch.cuda.empty_cache()
    except BaseException as error:
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    result["elapsed_s"] = time.monotonic() - start
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("native_planner_constructed", "fk_ik_roundtrip_success", "elapsed_s")}))
    return 0 if result["fk_ik_roundtrip_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
