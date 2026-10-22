"""Fresh RGB+joint visual-field / grounded Direct task evaluation, OAuth only."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

from PIL import Image, ImageDraw

from gpt_field.astra import AstraConfig, APIError, MissingCredentials, SchemaError, BudgetExceeded
from gpt_field.chatgpt_auth import SubscriptionCredentials
from gpt_field.demonstration import load_demonstration
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.experiment import source_manifest, write_json
from gpt_field.libero import LiberoConfig, SimProcess
from gpt_field.observation import decode_image
from gpt_field.video import MP4Stream, WIDTH, HEIGHT
from gpt_field.visual_field_policy import VisualFieldPolicy
from gpt_field.visual_field_subscription import VisualFieldSubscriptionClient, VERSION, shared_guidance

ROOT = Path(__file__).resolve().parents[1]


def method_components(args):
    version = getattr(args, "method_version", "v1")
    if version == "v1":
        return VisualFieldSubscriptionClient, VisualFieldPolicy, VERSION, shared_guidance
    if version not in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        raise ValueError("Unsupported visual-field method version")
    if args.controller != "hybrid" or args.demo is not None:
        raise ValueError(f"Visual Field {version} evaluation is zero-shot Hybrid only")
    if version == "v36":
        from gpt_field.visual_field_policy_v36 import VisualFieldPolicyV36
        from gpt_field.visual_field_subscription_v36 import (
            VisualFieldSubscriptionClientV36, VERSION as VERSION_V36, shared_guidance as guidance_v36)
        return VisualFieldSubscriptionClientV36, VisualFieldPolicyV36, VERSION_V36, guidance_v36
    if version == "v35":
        from gpt_field.visual_field_policy_v35 import VisualFieldPolicyV35
        from gpt_field.visual_field_subscription_v35 import (
            VisualFieldSubscriptionClientV35, VERSION as VERSION_V35, shared_guidance as guidance_v35)
        return VisualFieldSubscriptionClientV35, VisualFieldPolicyV35, VERSION_V35, guidance_v35
    if version == "v34":
        from gpt_field.visual_field_policy_v34 import VisualFieldPolicyV34
        from gpt_field.visual_field_subscription_v34 import (
            VisualFieldSubscriptionClientV34, VERSION as VERSION_V34, shared_guidance as guidance_v34)
        return VisualFieldSubscriptionClientV34, VisualFieldPolicyV34, VERSION_V34, guidance_v34
    if version == "v33":
        from gpt_field.visual_field_policy_v33 import VisualFieldPolicyV33
        from gpt_field.visual_field_subscription_v33 import (
            VisualFieldSubscriptionClientV33, VERSION as VERSION_V33, shared_guidance as guidance_v33)
        return VisualFieldSubscriptionClientV33, VisualFieldPolicyV33, VERSION_V33, guidance_v33
    if version == "v32":
        from gpt_field.visual_field_policy_v32 import VisualFieldPolicyV32
        from gpt_field.visual_field_subscription_v32 import (
            VisualFieldSubscriptionClientV32, VERSION as VERSION_V32, shared_guidance as guidance_v32)
        return VisualFieldSubscriptionClientV32, VisualFieldPolicyV32, VERSION_V32, guidance_v32
    if version == "v31":
        from gpt_field.visual_field_policy_v31 import VisualFieldPolicyV31
        from gpt_field.visual_field_subscription_v31 import (
            VisualFieldSubscriptionClientV31, VERSION as VERSION_V31, shared_guidance as guidance_v31)
        return VisualFieldSubscriptionClientV31, VisualFieldPolicyV31, VERSION_V31, guidance_v31
    if version == "v30":
        from gpt_field.visual_field_policy_v30 import VisualFieldPolicyV30
        from gpt_field.visual_field_subscription_v30 import (
            VisualFieldSubscriptionClientV30, VERSION as VERSION_V30, shared_guidance as guidance_v30)
        return VisualFieldSubscriptionClientV30, VisualFieldPolicyV30, VERSION_V30, guidance_v30
    if version == "v29":
        from gpt_field.visual_field_policy_v29 import VisualFieldPolicyV29
        from gpt_field.visual_field_subscription_v29 import (
            VisualFieldSubscriptionClientV29, VERSION as VERSION_V29, shared_guidance as guidance_v29)
        return VisualFieldSubscriptionClientV29, VisualFieldPolicyV29, VERSION_V29, guidance_v29
    if version == "v28":
        from gpt_field.visual_field_policy_v28 import VisualFieldPolicyV28
        from gpt_field.visual_field_subscription_v28 import (
            VisualFieldSubscriptionClientV28, VERSION as VERSION_V28, shared_guidance as guidance_v28)
        return VisualFieldSubscriptionClientV28, VisualFieldPolicyV28, VERSION_V28, guidance_v28
    if version == "v27":
        from gpt_field.visual_field_policy_v27 import VisualFieldPolicyV27
        from gpt_field.visual_field_subscription_v27 import (
            VisualFieldSubscriptionClientV27, VERSION as VERSION_V27, shared_guidance as guidance_v27)
        return VisualFieldSubscriptionClientV27, VisualFieldPolicyV27, VERSION_V27, guidance_v27
    if version == "v26":
        from gpt_field.visual_field_policy_v26 import VisualFieldPolicyV26
        from gpt_field.visual_field_subscription_v26 import (
            VisualFieldSubscriptionClientV26, VERSION as VERSION_V26, shared_guidance as guidance_v26)
        return VisualFieldSubscriptionClientV26, VisualFieldPolicyV26, VERSION_V26, guidance_v26
    if version == "v25":
        from gpt_field.visual_field_policy_v25 import VisualFieldPolicyV25
        from gpt_field.visual_field_subscription_v25 import (
            VisualFieldSubscriptionClientV25, VERSION as VERSION_V25, shared_guidance as guidance_v25)
        return VisualFieldSubscriptionClientV25, VisualFieldPolicyV25, VERSION_V25, guidance_v25
    if version == "v24":
        from gpt_field.visual_field_policy_v24 import VisualFieldPolicyV24
        from gpt_field.visual_field_subscription_v24 import (
            VisualFieldSubscriptionClientV24, VERSION as VERSION_V24, shared_guidance as guidance_v24)
        return VisualFieldSubscriptionClientV24, VisualFieldPolicyV24, VERSION_V24, guidance_v24
    if version == "v23":
        from gpt_field.visual_field_policy_v23 import VisualFieldPolicyV23
        from gpt_field.visual_field_subscription_v23 import (
            VisualFieldSubscriptionClientV23, VERSION as VERSION_V23, shared_guidance as guidance_v23)
        return VisualFieldSubscriptionClientV23, VisualFieldPolicyV23, VERSION_V23, guidance_v23
    if version == "v22":
        from gpt_field.visual_field_policy_v22 import VisualFieldPolicyV22
        from gpt_field.visual_field_subscription_v22 import (
            VisualFieldSubscriptionClientV22, VERSION as VERSION_V22, shared_guidance as guidance_v22)
        return VisualFieldSubscriptionClientV22, VisualFieldPolicyV22, VERSION_V22, guidance_v22
    if version == "v21":
        from gpt_field.visual_field_policy_v21 import VisualFieldPolicyV21
        from gpt_field.visual_field_subscription_v21 import (
            VisualFieldSubscriptionClientV21, VERSION as VERSION_V21, shared_guidance as guidance_v21)
        return VisualFieldSubscriptionClientV21, VisualFieldPolicyV21, VERSION_V21, guidance_v21
    if version == "v20":
        from gpt_field.visual_field_policy_v20 import VisualFieldPolicyV20
        from gpt_field.visual_field_subscription_v20 import (
            VisualFieldSubscriptionClientV20, VERSION as VERSION_V20, shared_guidance as guidance_v20)
        return VisualFieldSubscriptionClientV20, VisualFieldPolicyV20, VERSION_V20, guidance_v20
    if version == "v19":
        from gpt_field.visual_field_policy_v19 import VisualFieldPolicyV19
        from gpt_field.visual_field_subscription_v19 import (
            VisualFieldSubscriptionClientV19, VERSION as VERSION_V19, shared_guidance as guidance_v19)
        return VisualFieldSubscriptionClientV19, VisualFieldPolicyV19, VERSION_V19, guidance_v19
    if version == "v18":
        from gpt_field.visual_field_policy_v18 import VisualFieldPolicyV18
        from gpt_field.visual_field_subscription_v18 import (
            VisualFieldSubscriptionClientV18, VERSION as VERSION_V18, shared_guidance as guidance_v18)
        return VisualFieldSubscriptionClientV18, VisualFieldPolicyV18, VERSION_V18, guidance_v18
    if version == "v17":
        from gpt_field.visual_field_policy_v17 import VisualFieldPolicyV17
        from gpt_field.visual_field_subscription_v17 import (
            VisualFieldSubscriptionClientV17, VERSION as VERSION_V17, shared_guidance as guidance_v17)
        return VisualFieldSubscriptionClientV17, VisualFieldPolicyV17, VERSION_V17, guidance_v17
    if version == "v16":
        from gpt_field.visual_field_policy_v16 import VisualFieldPolicyV16
        from gpt_field.visual_field_subscription_v16 import (
            VisualFieldSubscriptionClientV16, VERSION as VERSION_V16, shared_guidance as guidance_v16)
        return VisualFieldSubscriptionClientV16, VisualFieldPolicyV16, VERSION_V16, guidance_v16
    if version == "v15":
        from gpt_field.visual_field_policy_v15 import VisualFieldPolicyV15
        from gpt_field.visual_field_subscription_v15 import (
            VisualFieldSubscriptionClientV15, VERSION as VERSION_V15, shared_guidance as guidance_v15)
        return VisualFieldSubscriptionClientV15, VisualFieldPolicyV15, VERSION_V15, guidance_v15
    if version == "v14":
        from gpt_field.visual_field_policy_v14 import VisualFieldPolicyV14
        from gpt_field.visual_field_subscription_v14 import (
            VisualFieldSubscriptionClientV14, VERSION as VERSION_V14, shared_guidance as guidance_v14)
        return VisualFieldSubscriptionClientV14, VisualFieldPolicyV14, VERSION_V14, guidance_v14
    if version == "v13":
        from gpt_field.visual_field_policy_v13 import VisualFieldPolicyV13
        from gpt_field.visual_field_subscription_v13 import (
            VisualFieldSubscriptionClientV13, VERSION as VERSION_V13, shared_guidance as guidance_v13)
        return VisualFieldSubscriptionClientV13, VisualFieldPolicyV13, VERSION_V13, guidance_v13
    if version == "v12":
        from gpt_field.visual_field_policy_v12 import VisualFieldPolicyV12
        from gpt_field.visual_field_subscription_v12 import (
            VisualFieldSubscriptionClientV12, VERSION as VERSION_V12, shared_guidance as guidance_v12)
        return VisualFieldSubscriptionClientV12, VisualFieldPolicyV12, VERSION_V12, guidance_v12
    if version == "v11":
        from gpt_field.visual_field_policy_v11 import VisualFieldPolicyV11
        from gpt_field.visual_field_subscription_v11 import (
            VisualFieldSubscriptionClientV11, VERSION as VERSION_V11, shared_guidance as guidance_v11)
        return VisualFieldSubscriptionClientV11, VisualFieldPolicyV11, VERSION_V11, guidance_v11
    if version == "v10":
        from gpt_field.visual_field_policy_v10 import VisualFieldPolicyV10
        from gpt_field.visual_field_subscription_v10 import (
            VisualFieldSubscriptionClientV10, VERSION as VERSION_V10, shared_guidance as guidance_v10)
        return VisualFieldSubscriptionClientV10, VisualFieldPolicyV10, VERSION_V10, guidance_v10
    if version == "v9":
        from gpt_field.visual_field_policy_v9 import VisualFieldPolicyV9
        from gpt_field.visual_field_subscription_v9 import (
            VisualFieldSubscriptionClientV9, VERSION as VERSION_V9, shared_guidance as guidance_v9)
        return VisualFieldSubscriptionClientV9, VisualFieldPolicyV9, VERSION_V9, guidance_v9
    if version == "v8":
        from gpt_field.visual_field_policy_v8 import VisualFieldPolicyV8
        from gpt_field.visual_field_subscription_v8 import (
            VisualFieldSubscriptionClientV8, VERSION as VERSION_V8, shared_guidance as guidance_v8)
        return VisualFieldSubscriptionClientV8, VisualFieldPolicyV8, VERSION_V8, guidance_v8
    if version == "v7":
        from gpt_field.visual_field_policy_v7 import VisualFieldPolicyV7
        from gpt_field.visual_field_subscription_v7 import (
            VisualFieldSubscriptionClientV7, VERSION as VERSION_V7, shared_guidance as guidance_v7)
        return VisualFieldSubscriptionClientV7, VisualFieldPolicyV7, VERSION_V7, guidance_v7
    if version == "v6":
        from gpt_field.visual_field_policy_v6 import VisualFieldPolicyV6
        from gpt_field.visual_field_subscription_v6 import (
            VisualFieldSubscriptionClientV6, VERSION as VERSION_V6, shared_guidance as guidance_v6)
        return VisualFieldSubscriptionClientV6, VisualFieldPolicyV6, VERSION_V6, guidance_v6
    if version == "v5":
        from gpt_field.visual_field_policy_v5 import VisualFieldPolicyV5
        from gpt_field.visual_field_subscription_v5 import (
            VisualFieldSubscriptionClientV5, VERSION as VERSION_V5, shared_guidance as guidance_v5)
        return VisualFieldSubscriptionClientV5, VisualFieldPolicyV5, VERSION_V5, guidance_v5
    if version == "v4":
        from gpt_field.visual_field_policy_v4 import VisualFieldPolicyV4
        from gpt_field.visual_field_subscription_v4 import (
            VisualFieldSubscriptionClientV4, VERSION as VERSION_V4, shared_guidance as guidance_v4)
        return VisualFieldSubscriptionClientV4, VisualFieldPolicyV4, VERSION_V4, guidance_v4
    if version == "v3":
        from gpt_field.visual_field_policy_v3 import VisualFieldPolicyV3
        from gpt_field.visual_field_subscription_v3 import (
            VisualFieldSubscriptionClientV3, VERSION as VERSION_V3, shared_guidance as guidance_v3)
        return VisualFieldSubscriptionClientV3, VisualFieldPolicyV3, VERSION_V3, guidance_v3
    from gpt_field.visual_field_policy_v2 import VisualFieldPolicyV2
    from gpt_field.visual_field_subscription_v2 import (
        VisualFieldSubscriptionClientV2, VERSION as VERSION_V2, shared_guidance as guidance_v2)
    return VisualFieldSubscriptionClientV2, VisualFieldPolicyV2, VERSION_V2, guidance_v2


@contextmanager
def interrupted_signals():
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def simulator_without_credentials():
    names = ("OPENAI_API_KEY", "ACCESS_TOKEN", "OPENAI_ACCESS_TOKEN", "CHATGPT_ACCESS_TOKEN")
    previous = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(previous)


class VisualRecorder:
    def __init__(self, output):
        self.stream = MP4Stream(output / "episode.mp4", fps=20)
        self.output = output
        self.index = []

    def frame(self, obs, step, info):
        frame = Image.new("RGB", (WIDTH, HEIGHT), (16, 20, 28))
        for name, url, offset in (("external", obs.image, 30), ("wrist", obs.wrist_image, 820)):
            frame.paste(decode_image(url).resize((740, 740)), (offset, 100))
        draw = ImageDraw.Draw(frame)
        draw.text((30, 20), f"RGB + measured joints | action {step} | {info}", fill="white")
        draw.text((30, 60), "External RGB", fill="white")
        draw.text((820, 60), "Wrist RGB", fill="white")
        draw.text((30, 870), "Playback shows executed control frames at 20 Hz, excluding model waiting time.", fill="white")
        self.stream.write(frame)
        self.index.append({"frame": len(self.index), "step": step, "info": info})

    def close(self):
        self.stream.close()
        write_json(self.output / "video_index.json", {"fps": 20, "frames": self.index})


def execute(args, *, token_provider, transport=None, sim_factory=SimProcess, recorder_factory=VisualRecorder):
    client_type, policy_type, method_version, guidance = method_components(args)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    task = resolve_distance_task(args.task)
    demo = load_demonstration(args.demo, task.key) if args.demo else None
    config = AstraConfig(model="gpt-6-astra", timeout_s=180,
                         max_api_calls=None, max_wall_time_s=None, max_cost_usd=None)
    # Validate all input paths before creating a trial. Freeze exact bytes before reset.
    sources = {}
    for key, path in (("initial_state", args.initial_state), ("bddl", args.bddl)):
        if path:
            path = Path(path).resolve()
            data = path.read_bytes()
            sources[key] = {"source": str(path), "sha256": hashlib.sha256(data).hexdigest(), "data": data}
    output.mkdir(parents=True)
    frozen_inputs = {}
    for key, value in sources.items():
        filename = "initial_state.npy" if key == "initial_state" else "task.bddl"
        (output / filename).write_bytes(value.pop("data"))
        frozen_inputs[key] = str(output / filename)
    if demo:
        demo.write(output / "demonstration.json")
    sim_config = LiberoConfig(suite=task.suite, seed=args.seed, max_steps=None,
        python_executable=args.sim_python, libero_root=args.libero_root,
        include_arm_joints=True, collision_check_mode="task_manipulation_v1",
        initial_state_override=frozen_inputs.get("initial_state"),
        task_bddl_override=frozen_inputs.get("bddl"))
    client = client_type(config, controller=args.controller, task_key=task.key,
        demonstration=demo, token_provider=token_provider, transport=transport, log_dir=output / "api")
    policy = policy_type(client)
    software = source_manifest()
    frozen = output / "frozen_source"
    frozen.mkdir()
    for filename in software["source_sha256"]:
        shutil.copyfile(ROOT / "src/gpt_field" / filename, frozen / filename)
    shutil.copyfile(__file__, frozen / Path(__file__).name)
    common = guidance(task, demo is not None)
    manifest = {"version": method_version, "method_version": getattr(args, "method_version", "v1"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": asdict(task), "controller": args.controller,
        "baseline_label": "grounded Direct5 with shared local visual feedback" if args.controller == "direct" else None,
        "shots": int(demo is not None), "demonstration": demo.metadata() if demo else None,
        "episode_index": args.episode_index, "sources": sources, "simulator": asdict(sim_config),
        "model": "gpt-6-astra", "reasoning_effort": "medium", "billing": "chatgpt_subscription",
        "api_key_fallback": False, "model_fallback": False, "max_policy_steps": None,
        "max_api_calls": None, "max_wall_time_s": None, "software": software,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_guidance_sha256": hashlib.sha256(common.encode()).hexdigest(),
        "inputs": ["external_rgb", "wrist_rgb", "eef_gripper_state8", "arm_joint_position7", "arm_joint_velocity7"],
        "excluded_inputs": ["depth", "camera_calibration", "object_poses", "contacts", "rewards", "success"],
        "stopping_rule": ["LIBERO success and no forbidden collision", "first forbidden collision",
                          "user interruption", "subscription or infrastructure failure"],
        "completion_metric": "control_elapsed_s minus video_render_s",
        "field_units": "image pixels / weighted image distance, not metric SDF",
        "limitations": ["patch tracking is not semantic segmentation", "no complete 3D collision certificate",
                        "local OSC response extrapolation requires measured support",
                        "grasp/recovery counters are shared observable proxies"]}
    if getattr(args, "method_version", "v1") in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        manifest.update(
            field_units="weighted graph path cost in meters on a hypothesized 3D metric grid; not a measured SDF",
            field_geometry_source="RGB-based model hypotheses plus measured robot state; no depth or supplied calibration",
            field_grid={"shape": [33, 33, 33], "spacing_m": .005, "coverage_m": [.16, .16, .16]},
            field_refresh="persistent model intent, default 30 actions and explicit bound 10..60; earlier on observable contradiction or local boundary",
            camera_roles={"external": "moving tool/payload identity and measured response",
                          "wrist": "optional target bearing; no complete tool footprint requirement"},
            limitations=["hypothesized 3D boxes are not measured scene geometry",
                         "no full-arm or complete 3D collision certificate",
                         "patch tracking cannot certify semantic identity, attachment or completion",
                         "bounded cold-start translations validate response online",
                         "grasp/recovery counters are observable proxies"],
        )
        if args.method_version in ("v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
            manifest.update(
                field_refresh="persistent model intent, default 80 actions and explicit bound 10..120; early on arrival, tracking contradiction or spatial budget",
                visual_tracking="bounded multi-feature ROI consensus; no supplied calibration or semantic identity certificate",
                orientation_hold="computed locally from current measured rotation vector, never model quaternion arithmetic",
                controller_handoff="model-directed Direct5 for near-contact alignment and interaction; no automatic grasp or retreat",
            )
        if args.method_version in ("v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
            manifest.update(
                controller_handoff="model-directed Field with xyz/xy/z constraints, bounded measured gripper settling, and unchanged Direct5 for complex interaction/reorientation",
                visual_reference_updates="explicit seed/retain; reliable existing correspondences are retained without model pixel re-estimation",
                attachment_evidence="windowed independent payload/tool motion; finite uncertainty allowance, no grasp certificate",
                gripper_settle="explicit model open/close; 10..30 action budget, measured finger settling and pose checks; not automatic task contact",
            )
        if args.method_version in ("v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
            manifest.update(
                reference_health="recent controlled-feature motion plus correspondence-persistent unvalidated travel; same rejected feature cannot renew allowance by retaining or renaming",
                reference_diagnostic="on repair only, one derived external RGB image marks tracker claims; both original RGB views remain unchanged",
                transport_retry="at most one stream_missing_completion retry at the same observation before any action; counted and timed, no API/model fallback",
                contact_guidance="navigation centering and descent stall do not establish graspable-feature engagement; model may choose a targeted Direct5 contact correction",
            )
        if args.method_version in ("v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
            manifest.update(
                orientation_hold="bounded shortest world-frame rotation correction to the measured intent-start orientation, with explicit persistent-OSC-target refresh and unchanged hard drift guard",
                reference_health="separate predictive contradiction from weak observable motion; retain finite correspondence-persistent uncertainty travel",
                direct_rotation_screen="model-declared whole-tool RGB envelope and neighboring obstacle rectangles; reference-only rotation is rejected; projected risk is a heuristic, never 3D clearance certification",
                scene_assessment="explicit nearby-object review in every response; every listed Field neighbor maps to a metric obstacle or unknown hypothesis; unobservable Field scene caps measured travel at 0.015m and each command at 0.005m while preserving goal and regions",
                controller_handoff="Field with measured orientation hold, model-commanded gripper settling, and Direct5; accepted Direct commands execute unchanged, rejected rotation blocks are discarded before execution with one explicitly logged current-pose OSC-target refresh",
                rejection_hold="all rejected selections use a logged 1e-9-rad current-pose orientation-target refresh to cancel an unfinished previous OSC rotation goal; no requested rejected motion executes",
            )
    if getattr(args, "method_version", "v1") in ("v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v9 import geometry_semantics
        manifest.update(field_geometry_semantics=geometry_semantics(),
            operator_stop_file=str(output / "operator_stop.request.json"),
            operator_stop_boundary="after completed transition and evaluator outcome checks, before next decision; no automatic timeout")
    if getattr(args, "method_version", "v1") in ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        manifest.update(
            subgoal_reuse="explicit persistent world goal across Field, Direct and gripper repair; measured local arrival is never task success",
            attachment_evidence="bounded fresh displacement-adaptive independent tool/payload evidence; 5..15mm finite uncertainty allowance estimated only from fresh observed response, with explicit useful-motion handoff",
            timing_checkpoint="atomic run_state at each completed transition and decision boundary; partial timing is not final task time",
        )
    if getattr(args, "method_version", "v1") == "v11":
        from gpt_field.robot_tool_geometry import geometry_contract
        from gpt_field.visual_feedback_compact_v11 import VERSION as FEEDBACK_VERSION
        runtime_sources = ("visual_field_subscription_v11.py", "visual_field_policy_v11.py",
            "visual_field_v11.py", "visual_translation_guard_v11.py", "visual_feedback_compact_v11.py",
            "robot_tool_geometry.py", "visual_monitor_v10.py")
        manifest.update(
            projected_translation_screen="current RGB whole-tool and named-neighbor tracks; explicit transit/separate/interaction roles, persistent neighbor identities and finite separating-motion allowance; no 3D clearance certificate",
            robot_tool_geometry=geometry_contract(),
            controller_handoff="Field bulk translation with projected admission, model-directed gripper settling, and Direct5; accepted Direct commands execute unchanged; a rejected translation discards the remaining block and issues one logged current-pose OSC-target refresh",
            feedback_representation={"version": FEEDBACK_VERSION,
                "scope": "presentation-only current/recent feedback compaction; original RGB, measured robot observations and raw model outputs are unchanged; full diagnostics remain in local traces"},
            v11_runtime_source_sha256={name: software["source_sha256"][name] for name in runtime_sources},
        )
    if getattr(args, "method_version", "v1") == "v12":
        from gpt_field.robot_tool_geometry import geometry_contract
        from gpt_field.visual_feedback_compact_v12 import VERSION as FEEDBACK_VERSION
        runtime_sources = ("visual_field_subscription_v12.py", "visual_field_policy_v12.py",
            "visual_field_v12.py", "visual_translation_guard_v12.py", "visual_feedback_compact_v12.py",
            "robot_translation_response.py", "robot_tool_geometry.py", "visual_monitor_v10.py", "visual_monitor_v12.py")
        manifest.update(
            projected_translation_screen="projected overlap is ambiguity; bounded metric-path progress uses current RGB neighbor risk trend, persistent observation allowance and measured robot response; no 3D clearance certificate",
            robot_tool_geometry=geometry_contract(),
            robot_translation_response="observed commanded-versus-measured robot translation response; no camera calibration or environment object truth",
            controller_handoff="Field bulk translation with bounded observed response, model-directed gripper settling, and Direct5; accepted Direct commands execute unchanged; refusal discards remaining commands and logs a current-pose OSC-target refresh",
            feedback_representation={"version": FEEDBACK_VERSION,
                "scope": "presentation-only V11 geometry compaction plus exact duplicated execution-block feedback references within one request; all current feedback, robot observations, RGB and raw model outputs remain available"},
            prompt_composition="single concise task-generic V12 guidance; unchanged strict V11 wire schema, recent two observation turns",
            v12_runtime_source_sha256={name: software["source_sha256"][name] for name in runtime_sources},
        )
    if getattr(args, "method_version", "v1") == "v13":
        from gpt_field.robot_tool_geometry import geometry_contract
        from gpt_field.visual_feedback_compact_v12 import VERSION as FEEDBACK_VERSION
        runtime_sources = ("visual_field_subscription_v13.py", "visual_field_policy_v13.py",
            "visual_field_v13.py", "contact_geometry_v13.py", "visual_contact_alignment_v13.py",
            "visual_translation_guard_v12.py", "visual_feedback_compact_v12.py",
            "robot_translation_response.py", "robot_tool_geometry.py", "visual_monitor_v10.py", "visual_monitor_v12.py")
        manifest.update(
            projected_translation_screen="inherited bounded current-RGB neighbor risk and observed metric response; primary target geometry independently constrains robot parts in every controller; no 3D clearance certificate",
            robot_tool_geometry=geometry_contract(),
            robot_translation_response="observed commanded-versus-measured robot translation response; no camera calibration or environment object truth",
            controller_handoff="Field bulk translation and bounded two-view grasp correction, explicit gripper settling and Direct5; accepted Direct commands unchanged; refusal discards remaining commands and logs a current-pose OSC-target refresh",
            target_contact_contract="required primary target extent independent of nearby obstacle arrays; fingers permission never exempts hand/palm; payload_support exempts no robot part; unavailable geometry cannot authorize approach or closure",
            contact_alignment="optional current two-view physical contact-to-pinch descriptors; bounded correction fitted from measured robot motion; no automatic close or task-success assertion",
            feedback_representation={"version": FEEDBACK_VERSION,
                "scope": "presentation-only geometry compaction and exact duplicated block feedback references; current hazards, original observations and raw model outputs remain available"},
            prompt_composition="concise task-generic V13 target/part-contact and two-view alignment guidance; strict V12 wire plus two required nullable keys, recent two observation turns",
            v13_runtime_source_sha256={name: software["source_sha256"][name] for name in runtime_sources},
        )
    if getattr(args, "method_version", "v1") == "v14":
        from gpt_field.robot_tool_geometry import geometry_contract
        from gpt_field.visual_feedback_compact_v14 import VERSION as FEEDBACK_VERSION
        runtime_sources = ("visual_field_subscription_v14.py", "visual_field_policy_v14.py",
            "visual_field_v13.py", "contact_geometry_v13.py", "visual_contact_alignment_v13.py",
            "visual_translation_guard_v14.py", "visual_reference_diagnostic_v14.py", "visual_feedback_compact_v14.py",
            "robot_translation_response.py", "robot_tool_geometry.py", "visual_monitor_v10.py", "visual_monitor_v12.py")
        manifest.update(
            recovery_contract="explicit model-selected Field recovery proposals are not permission; every executed segment is rechecked, finite shared measured debt cannot reset through new IDs or ROIs",
            projected_translation_screen="inherited bounded current-RGB neighbor risk and observed metric response; primary target geometry independently constrains robot parts in every controller; no 3D clearance certificate",
            robot_tool_geometry=geometry_contract(),
            robot_translation_response="observed commanded-versus-measured robot translation response; no camera calibration or environment object truth",
            controller_handoff="Field bulk translation and bounded two-view grasp correction, explicit gripper settling and Direct5; accepted Direct commands unchanged; refusal discards remaining commands and logs a current-pose OSC-target refresh",
            target_contact_contract="required primary target extent independent of nearby obstacle arrays; fingers permission never exempts hand/palm; payload_support exempts no robot part; unavailable geometry cannot authorize approach or closure",
            contact_alignment="optional current two-view physical contact-to-pinch descriptors; bounded correction fitted from measured robot motion; no automatic close or task-success assertion",
            feedback_representation={"version": FEEDBACK_VERSION,
                "scope": "presentation-only duplicate decision snapshots and known robot detail compaction; authoritative current guard, recovery options, target, contact, attachment, ledger, original observations and raw outputs preserved"},
            prompt_composition="concise V14 evidence-producing recovery and unchanged strict V13 wire; recent two observation turns",
            v14_runtime_source_sha256={name: software["source_sha256"][name] for name in runtime_sources},
        )
    if getattr(args, "method_version", "v1") == "v15":
        from gpt_field.robot_tool_geometry import geometry_contract
        from gpt_field.visual_feedback_compact_v14 import VERSION as FEEDBACK_VERSION
        runtime_sources = ("visual_field_subscription_v15.py", "visual_field_policy_v15.py",
            "visual_field_v13.py", "contact_geometry_v13.py", "visual_contact_alignment_v13.py",
            "visual_translation_guard_v15.py", "visual_reference_diagnostic_v14.py", "visual_motion_diagnostic_v15.py", "visual_feedback_compact_v14.py",
            "robot_translation_response.py", "robot_tool_geometry.py", "visual_monitor_v10.py", "visual_monitor_v12.py")
        manifest.update(
            recovery_contract="side-effect-free current guard preview filters nominal recovery proposals; executed segments are rechecked; finite shared and same-direction measured debt persist across reseeding, IDs and ROIs",
            motion_diagnostic="current acknowledged RGB changes and descriptor bounds, conditionally shown at reference repair handoff; no segmentation, identity or clearance certificate",
            projected_translation_screen="inherited bounded current-RGB neighbor risk and observed metric response; primary target geometry independently constrains robot parts in every controller; no 3D clearance certificate",
            robot_tool_geometry=geometry_contract(),
            robot_translation_response="observed commanded-versus-measured robot translation response; no camera calibration or environment object truth",
            controller_handoff="Field bulk translation and bounded two-view grasp correction, explicit gripper settling and Direct5; accepted Direct commands unchanged; refusal discards remaining commands and logs a current-pose OSC-target refresh",
            target_contact_contract="required primary target extent independent of nearby obstacle arrays; fingers permission never exempts hand/palm; payload_support exempts no robot part; unavailable geometry cannot authorize approach or closure",
            contact_alignment="optional current two-view physical contact-to-pinch descriptors; bounded correction fitted from measured robot motion; no automatic close or task-success assertion",
            feedback_representation={"version": FEEDBACK_VERSION,
                "scope": "presentation-only duplicate decision snapshots and known robot detail compaction; authoritative current guard, recovery options, target, contact, attachment, ledger, original observations and raw outputs preserved"},
            prompt_composition="V15 persistent same-direction recovery debt, admission-previewed options and current RGB motion evidence and unchanged strict V13 wire; recent two observation turns",
            v15_runtime_source_sha256={name: software["source_sha256"][name] for name in runtime_sources},
        )
    if getattr(args, "method_version", "v1") == "v16":
        from gpt_field.robot_tool_geometry import geometry_contract
        from gpt_field.visual_feedback_compact_v14 import VERSION as FEEDBACK_VERSION
        runtime_sources = ("visual_field_subscription_v16.py", "visual_field_policy_v16.py", "visual_translation_guard_v16.py",
            "visual_field_v16.py", "visual_scene_inventory_v16.py", "visual_scene_guard_v16.py", "visual_field_v13.py", "contact_geometry_v13.py", "visual_contact_alignment_v13.py",
            "visual_translation_guard_v15.py", "visual_reference_diagnostic_v14.py", "visual_motion_diagnostic_v15.py", "visual_feedback_compact_v14.py",
            "robot_translation_response.py", "robot_tool_geometry.py", "visual_monitor_v10.py", "visual_monitor_v12.py")
        manifest.update(
            scene_inventory_contract="one current visible-object geometry source, stable IDs, omitted/occluded objects retained in world coordinates; no legacy array-cap truncation; unlocalized image coverage remains explicit uncertainty",
            recovery_contract="side-effect-free current guard preview filters nominal recovery proposals; executed segments are rechecked; finite shared and same-direction measured debt persist across reseeding, IDs and ROIs",
            motion_diagnostic="current acknowledged RGB changes and descriptor bounds, conditionally shown at reference repair handoff; no segmentation, identity or clearance certificate",
            projected_translation_screen="inherited bounded current-RGB neighbor risk and observed metric response; primary target geometry independently constrains robot parts in every controller; no 3D clearance certificate",
            robot_tool_geometry=geometry_contract(),
            robot_translation_response="observed commanded-versus-measured robot translation response; no camera calibration or environment object truth",
            controller_handoff="Field bulk translation and bounded two-view grasp correction, explicit gripper settling and Direct5; accepted Direct commands unchanged; refusal discards remaining commands and logs a current-pose OSC-target refresh",
            target_contact_contract="required primary target extent independent of nearby obstacle arrays; fingers permission never exempts hand/palm; payload_support exempts no robot part; unavailable geometry cannot authorize approach or closure",
            contact_alignment="optional current two-view physical contact-to-pinch descriptors; bounded correction fitted from measured robot motion; no automatic close or task-success assertion",
            feedback_representation={"version": FEEDBACK_VERSION,
                "scope": "presentation-only duplicate decision snapshots and known robot detail compaction; authoritative current guard, recovery options, target, contact, attachment, ledger, original observations and raw outputs preserved"},
            prompt_composition="V16 persistent single-source scene inventory with V15 bounded recovery and current RGB motion evidence; recent two observation turns",
            v16_runtime_source_sha256={name: software["source_sha256"][name] for name in runtime_sources},
        )
    if getattr(args, "method_version", "v1") in ("v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.robot_tool_geometry import geometry_contract
        from gpt_field.visual_feedback_compact_v17 import VERSION as FEEDBACK_VERSION
        runtime_sources = ("visual_field_subscription_v17.py", "visual_field_policy_v17.py", "visual_translation_guard_v17.py",
            "visual_field_v17.py", "visual_scene_inventory_v17.py", "visual_scene_guard_v17.py", "visual_field_v13.py", "contact_geometry_v13.py", "visual_contact_alignment_v13.py",
            "visual_translation_guard_v15.py", "visual_reference_diagnostic_v14.py", "visual_motion_diagnostic_v15.py", "visual_feedback_compact_v14.py", "visual_feedback_compact_v17.py",
            "visual_field_v16.py", "visual_field_policy_v16.py", "visual_field_subscription_v16.py", "visual_scene_inventory_v16.py", "visual_translation_guard_v16.py",
            "robot_translation_response.py", "robot_tool_geometry.py", "visual_monitor_v10.py", "visual_monitor_v12.py")
        manifest.update(
            scene_inventory_contract="one current visible-object geometry source, stable IDs, omitted/occluded objects retained in world coordinates; no legacy array-cap truncation; unlocalized image coverage remains explicit uncertainty; currently tracked partial texture never claims full visibility; full projected/world extents retained; coverage checks actual candidate sweep",
            recovery_contract="side-effect-free current guard preview filters nominal recovery proposals; executed segments are rechecked; finite shared and same-direction measured debt persist across reseeding, IDs and ROIs",
            motion_diagnostic="current acknowledged RGB changes and descriptor bounds, conditionally shown at reference repair handoff; no segmentation, identity or clearance certificate",
            projected_translation_screen="inherited bounded current-RGB neighbor risk and observed metric response; primary target geometry independently constrains robot parts in every controller; no 3D clearance certificate",
            robot_tool_geometry=geometry_contract(),
            robot_translation_response="observed commanded-versus-measured robot translation response; no camera calibration or environment object truth",
            controller_handoff="Field bulk translation and bounded two-view grasp correction, explicit gripper settling and Direct5; accepted Direct commands unchanged; refusal discards remaining commands and logs a current-pose OSC-target refresh",
            target_contact_contract="required primary target extent independent of nearby obstacle arrays; fingers permission never exempts hand/palm; payload_support exempts no robot part; unavailable geometry cannot authorize approach or closure",
            contact_alignment="optional current two-view physical contact-to-pinch descriptors; bounded correction fitted from measured robot motion; no automatic close or task-success assertion",
            feedback_representation={"version": FEEDBACK_VERSION,
                "scope": "presentation-only old revision/world-duplicate removal, exact same-feedback references and reversible same-request older-feedback delta; latest feedback fully self-contained; all inventory IDs/current_geometry/current-or-partial support, current guard, unknown keys, budgets, contact, attachment, ledger, original observations and raw outputs preserved"},
            prompt_composition="V17 actual candidate-sweep scene coverage, current partial texture versus full retained geometry, persistent Field reuse and compact actionable inventory feedback; unchanged V16 native wire; recent two observation turns",
            v17_runtime_source_sha256={name: software["source_sha256"][name] for name in runtime_sources},
        )
    if getattr(args, "method_version", "v1") in ("v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        v18_sources = runtime_sources + ("visual_field_policy_v18.py", "visual_field_subscription_v18.py",
            "visual_reference_candidates_v18.py", "visual_reference_replay_v18.py")
        manifest.update(
            reference_candidate_contract="bounded candidates from acknowledged RGB change; GPT explicitly selects a visible distal-tool feature by an exact-current image-bound ID; no automatic semantic identity or clearance promotion; early review preserves physical travel and scene constraints",
            reference_response_replay="retrospective candidate-specific predictor training after configuration, preserving physical accounting and requiring fresh acknowledged held-out evidence; training replay never executes or counts simulated actions",
            prompt_composition="V18 evidence-bound controlled-feature review and exact candidate grounding, with inherited persistent inventory/candidate-sweep Field planning and compact feedback",
            v18_runtime_source_sha256={name: software["source_sha256"][name] for name in v18_sources},
        )
    if getattr(args, "method_version", "v1") in ("v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v19 import geometry_semantics
        v19_sources = v18_sources + ("visual_field_policy_v19.py", "visual_field_subscription_v19.py",
            "visual_field_v19.py", "visual_translation_guard_v19.py")
        manifest.update(
            model_geometry_admission=geometry_semantics(),
            projected_translation_screen="Exact current full-uncertainty known-tool/payload sweep can establish model-remote separation for that proposal; genuine near/unknown geometry, observed worsening and all contact checks remain. no 3D clearance certificate or uncertain-debt reset.",
            prompt_composition="V19 consistent exact full-margin model-geometry admission, inherited V18 reference repair and prospective validation, persistent inventory and Field planning; unchanged V18 wire",
            v19_runtime_source_sha256={name: software["source_sha256"][name] for name in v19_sources},
        )
    if getattr(args, "method_version", "v1") in ("v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v20 import geometry_semantics as semantics_v20
        v20_sources = v19_sources + ("visual_field_policy_v20.py", "visual_field_subscription_v20.py",
            "visual_field_v20.py")
        manifest.update(
            model_geometry_admission=semantics_v20(),
            prompt_composition="V20 fixed-world-goal robot-state tracking with candidate-bound whole-scene admission only for exhausted weak-image allowance; unchanged V18 wire and all contact/reference checks",
            proprioceptive_transit_contract="Only reliable tool-only noncontact fixed goals with no image retarget; every <=5mm actual candidate must independently pass current full-margin all-scene and target sweeps; no visual validation or physical debt reset",
            v20_runtime_source_sha256={name: software["source_sha256"][name] for name in v20_sources},
        )
    if getattr(args, "method_version", "v1") in ("v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v21 import geometry_semantics as semantics_v21
        from gpt_field.visual_feedback_compact_v21 import VERSION as feedback_v21
        v21_sources = v20_sources + ("visual_field_policy_v21.py", "visual_field_subscription_v21.py",
            "visual_field_v21.py", "visual_translation_guard_v21.py", "visual_reference_track_v21.py",
            "visual_monitor_v21.py", "visual_task_progress_v21.py", "visual_feedback_compact_v21.py")
        manifest.update(
            model_geometry_admission=semantics_v21(),
            feedback_presentation={"version": feedback_v21,
                "base": manifest["feedback_representation"]["version"],
                "scope": "Lossless named column tables and same-request history deltas; raw observations, assistant outputs and execution logs unchanged"},
            prompt_composition="V21 concise general control contract with bounded seed-anchored appearance maintenance, checked direction-dependent Field retreat and persistent declared-objective evidence; unchanged wire",
            appearance_maintenance="Bounded keyframes directly linked to immutable seed descriptors; inactive originals require original appearance; no automatic new features, identities, visual validation or budget reset",
            task_progress_contract="Same-reference image objective and metric waypoint progress measured separately across compatible refreshes; incompatible features archived separately; conflicting objectives require explicit review",
            v21_runtime_source_sha256={name: software["source_sha256"][name] for name in v21_sources},
        )
    if getattr(args, "method_version", "v1") in ("v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v22 import geometry_semantics as semantics_v22
        v22_sources = v21_sources + ("visual_field_v22.py", "visual_field_policy_v22.py", "visual_field_subscription_v22.py")
        manifest.update(
            model_geometry_admission=semantics_v22(),
            prompt_composition="V22 model-selected short reverse goals use continuous checked Field routes; offers preview the actual planner/tracker, unchanged wire and normal distance planning",
            heading_recovery_contract="Explicit separate empty-tool reverse, fixed short goal, actual-direction cone and fresh full-scene/contact checks, unchanged measured debts; current planner/tracker preview is not reusable permission",
            v22_runtime_source_sha256={name: software["source_sha256"][name] for name in v22_sources},
        )
    if getattr(args, "method_version", "v1") in ("v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v23 import geometry_semantics as semantics_v23
        v23_sources = v22_sources + (
            "visual_field_v23.py", "visual_field_policy_v23.py", "visual_field_subscription_v23.py",
            "uncertainty_egress_v23.py", "visual_egress_evidence_v23.py",
            "visual_translation_guard_v23.py", "visual_scene_guard_v23.py")
        manifest.update(
            model_geometry_admission=semantics_v23(),
            prompt_composition="V23 explicit bounded uncertainty egress with fresh observed progress; unchanged wire and normal weighted Field planning",
            uncertainty_egress_contract="Empty-tool model-selected full-exit goal; nominal cores/unknowns/contact band remain hard; only initial uncertainty overlaps may monotonically recede, no new halo entry; measured bootstrap and persistent contextual debt require fresh RGB evidence for continuation",
            v23_runtime_source_sha256={name: software["source_sha256"][name] for name in v23_sources},
        )
    if getattr(args, "method_version", "v1") in ("v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v24 import geometry_semantics as semantics_v24, FEEDBACK_VERSION as feedback_v24
        v24_sources = v23_sources + (
            "visual_field_v24.py", "visual_field_policy_v24.py", "visual_field_subscription_v24.py",
            "visual_route_exposure_v24.py", "visual_translation_guard_v24.py",
            "visual_reference_candidates_v24.py")
        manifest.update(
            model_geometry_admission=semantics_v24(),
            prompt_composition="V24 paid-history reference repair and finite model-conditioned Field route; unchanged wire and ordinary distance planning",
            uncertainty_egress_contract="Fixed original geometry and goal; all nominal cores/unknowns/target band/no-new-halo constraints retained, nominal separation nondecreasing. Persistent total40mm includes bootstrap and one reference confirmation. Qualified causal reference and current contradiction checks replace positive image-gap grants; conditional exposure is not observed physical clearance.",
            reference_repair_contract="Current acknowledged observation-bound board from paid RGB/state history, explicit model candidate selection, raw support witness, no semantic identity or clearance certificate and no renewed travel budget",
            older_feedback_context="Historical action/result summaries only; current actionable context, recent RGB/state/joints, assistant output and raw local traces retained",
            v24_runtime_source_sha256={name: software["source_sha256"][name] for name in v24_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v24)
    if getattr(args, "method_version", "v1") in ("v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v25 import geometry_semantics as semantics_v25, FEEDBACK_VERSION as feedback_v25
        v25_sources = v24_sources + (
            "visual_field_v25.py", "visual_field_policy_v25.py", "visual_field_subscription_v25.py",
            "robot_translation_response_v25.py", "visual_translation_guard_v25.py")
        manifest.update(
            model_geometry_admission=semantics_v25(),
            prompt_composition="V25 explicit persistent recovery route family, cone-constrained checked reverse commands and recent ordered response evidence; inherited whole-scene Field planning",
            recovery_route_contract="Preview and actual family match; reference-only retain cannot change family; no geometry or physical allowance reset",
            response_regime_contract="Two contiguous positive nondeclining responses with aggregate displacement above the unchanged two-sample noise allowance can update pooled responsiveness; full historical empirical bounds, requested command and measured allowance unchanged",
            finite_reverse_exposure_contract="An explicit checked fixed reverse uses at most its original8mm actual travel, existing8mm heading and20mm shared allowances;4mm image checkpoint is diagnostic and cannot replenish debt; all further-worsening, reference, geometry, unknown and contact vetoes remain",
            v25_runtime_source_sha256={name: software["source_sha256"][name] for name in v25_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v25)
    if getattr(args, "method_version", "v1") in ("v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v26 import geometry_semantics as semantics_v26, FEEDBACK_VERSION as feedback_v26
        v26_sources = v25_sources + (
            "visual_field_policy_v26.py", "visual_field_subscription_v26.py", "visual_translation_guard_v26.py",
            "visual_neighbor_track_v26.py")
        manifest.update(
            model_geometry_admission=semantics_v26(),
            prompt_composition="V26 comparable temporal risk epochs; current revised geometry, prior valid risk and physical debts retained; inherited V25 explicit recovery routes and Field planning",
            measurement_epoch_contract="Accepted reference or extent revision is not robot motion. Temporal trend requires a comparable representation; current revised extent and independent current-risk checks remain. No permission, risk erasure or allowance from a new epoch.",
            neighbor_descriptor_contract="Opt-in projected-neighbor matching retains all legacy rejection checks. Only an exact same-coordinate bilinear-sampled grayscale descriptor and an already vetted local match can remove subpixel fitting bias. This is not whole-RGB equality; no static object/camera assumption, risk-threshold change or allowance reset.",
            v26_runtime_source_sha256={name: software["source_sha256"][name] for name in v26_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v26)
    if getattr(args, "method_version", "v1") in ("v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v27 import geometry_semantics as semantics_v27, FEEDBACK_VERSION as feedback_v27
        v27_sources = v26_sources + (
            "visual_field_policy_v27.py", "visual_field_subscription_v27.py", "visual_field_v27.py",
            "visual_translation_guard_v27.py", "visual_boundary_evidence_v27.py", "visual_route_exposure_v27.py")
        manifest.update(
            model_geometry_admission=semantics_v27(),
            prompt_composition="V27 bounded normal empty-tool Field transit reconciles ambiguous projected warnings with current candidate geometry and heading-specific empirical image evidence; warnings and physical debt remain",
            projected_warning_transit_contract="Normal empty-tool no-contact fixed-goal Field only: exact current full-scene margin sweep, nondecreasing modeled gap for every warned object, ambiguous boundary sensitivity and prospectively validated per-object heading response. Command at most3mm; existing8mm direction/20mm shared and intent remaining bound measured travel. No new allowance, calibrated confidence or clearance certificate; resolved closing/stale/lost/contradictory evidence remains a veto.",
            retained_route_goal_contract="The first acknowledged raw goal remains immutable. A maximum absolute2e-12m serialization difference can match it only with identical geometry, origin and initial parts; relative tolerance0, no cumulative drift or physical-budget reset.",
            v27_runtime_source_sha256={name: software["source_sha256"][name] for name in v27_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v27)
    if getattr(args, "method_version", "v1") in ("v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v28 import geometry_semantics as semantics_v28, FEEDBACK_VERSION as feedback_v28
        v28_sources = v27_sources + (
            "visual_field_policy_v28.py", "visual_field_subscription_v28.py",
            "visual_reference_candidates_v28.py", "visual_translation_guard_v28.py",
            "visual_heading_evidence_v28.py", "visual_reverse_evidence_v28.py")
        manifest.update(
            model_geometry_admission=semantics_v28(),
            paid_handoff_reference_contract="Explicit risk/response handoffs can display current observation-bound candidates from already acknowledged RGB/state history; inherited32-frame cap, explicit model selection only, no new motion or physical-debt reset.",
            heading_evidence_contract="Finite current-geometry-checked normal empty-tool acquisition uses existing heading/shared/intent allowances; not validated response or certified clearance, warnings retained.",
            v28_runtime_source_sha256={name: software["source_sha256"][name] for name in v28_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v28)

    if getattr(args, "method_version", "v1") in ("v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v29 import geometry_semantics as semantics_v29, FEEDBACK_VERSION as feedback_v29
        v29_sources = v28_sources + (
            "visual_field_policy_v29.py", "visual_field_subscription_v29.py", "visual_field_v29.py", "visual_translation_guard_v29.py")
        manifest.update(
            model_geometry_admission=semantics_v29(),
            prompt_composition="V29 bounded constraint-aware tracking preserves the model-selected Field route, fixed world goal and current geometry/reference/contact/physical-budget admission; unchanged wire and explicit control roles",
            projected_warning_transit_contract="Normal empty-tool no-contact fixed-goal Field: exact current full-scene margin sweep, warned model-gap nondecrease, nonclosing empirical boundary and current per-object image/robot response. Unvalidated acquisition spends existing8mm heading/20mm shared allowances; currently validated transit is counted separately under fixed intent limits. No new allowance from renaming/reseed and no clearance certificate; lost/stale/contradictory evidence still stops or falls back to old remaining acquisition debt.",
            validated_transit_contract="Fresh current per-object held-out image response, responsive robot and exact model constraints qualify supported normal Field travel under fixed intent limits; old acquisition debts retained and separate supported travel tracked; loss falls back only to remaining old allowance",
            constrained_tracking_contract="Useful along-route progress with bounded cross-track correction; every exact selected candidate retains scene and warned-object constraints, measured arrival and intent limits; no physical budget refund or clearance certificate",
            v29_runtime_source_sha256={name: software["source_sha256"][name] for name in v29_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v29)

    if getattr(args, "method_version", "v1") in ("v30", "v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v30 import geometry_semantics as semantics_v30, FEEDBACK_VERSION as feedback_v30
        v30_sources = v29_sources + (
            "visual_field_policy_v30.py", "visual_field_subscription_v30.py", "visual_translation_guard_v30.py",
            "visual_reference_candidates_v30.py", "visual_reference_review_v30.py")
        manifest.update(
            model_geometry_admission=semantics_v30(),
            prompt_composition="V30 one current concise contract, unchanged wire and raw history: typed paid-reference events plus explicit target-only noncontact approach semantics; all non-target constraints retained",
            paid_handoff_reference_contract="Typed actual reference/control events classify extraction as not requested, ineligible, none extracted or available. Observation-bound paid-history candidates require explicit model selection, maximum32 frames; no automatic identity promotion, extra motion or physical-debt reset.",
            target_approach_contract="Normal empty-tool Field approach/align, transit/task_translation, target_contact none: only intended target gap may decrease under current positive full-margin model clearance, outside contact band and causal same-heading qualification. Every non-target gate remains; target is not contact-exempt or certified safe.",
            projected_warning_transit_contract="Current exact model geometry and causal response remain. Explicit grounded target approach can close only its target gap under target_approach_execution; non-target warnings retain nondecrease/contrary-evidence checks. Acquisition8/20mm debt and separate supported intent accounting persist; no rename/reseed credit or clearance certificate.",
            v30_runtime_source_sha256={name: software["source_sha256"][name] for name in v30_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v30)

    if getattr(args, "method_version", "v1") in ("v31", "v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v31 import geometry_semantics as semantics_v31, FEEDBACK_VERSION as feedback_v31
        v31_sources = v30_sources + (
            "visual_field_policy_v31.py", "visual_field_subscription_v31.py",
            "visual_field_proposals_v31.py", "visual_goal_field_v31.py", "visual_scene_inventory_v31.py", "visual_field_v31.py")
        manifest.update(
            model_geometry_admission=semantics_v31(),
            prompt_composition="V31 measured visual-goal Field offers and explicit current scene-estimate revision with separate metric-estimate freshness",
            visual_goal_field_contract="Paid measured EEF/image residual response defines a local objective only in its observed motion subspace. Current observation-bound offers pass the actual fixed-goal planner/tracker preview and require explicit model selection; all executed commands are rechecked. No inferred calibrated pose, automatic geometry update, motion or physical-debt refund.",
            scene_revision_contract="Current visible geometry estimates may be explicitly corrected with a per-object reason/evidence declaration. Metric estimate age is separate from image ROI freshness. Omission/null retains remembered geometry; changes are unverified hypotheses, not clearance certificates.",
            v31_runtime_source_sha256={name: software["source_sha256"][name] for name in v31_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v31)

    if getattr(args, "method_version", "v1") in ("v32", "v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v32 import geometry_semantics as semantics_v32, FEEDBACK_VERSION as feedback_v32
        v32_sources = v31_sources + (
            "visual_field_policy_v32.py", "visual_field_subscription_v32.py", "visual_objective_v32.py",
            "visual_monitor_v32.py", "visual_reference_candidates_v32.py", "visual_translation_guard_v32.py",
            "visual_reference_track_v32.py", "visual_goal_replay_v32.py")
        manifest.update(
            model_geometry_admission=semantics_v32(),
            prompt_composition="V32 separate rigid motion witness and actual-hand outline with exact-descriptor point registration and compact current debug presentation",
            rigid_proxy_contract="Explicit fixed-orientation empty-tool noncontact proxy role requires a separately observed actual-hand point and outline; proxy affine transforms cannot move the hand outline or become a pinch/attachment/image-goal certificate. Metric Field planning remains available.",
            point_registration_contract="Exact unchanged descriptors are zero-motion candidates only after inherited appearance/matching checks; no stationary-object assumption, threshold relaxation or physical allowance reset.",
            v32_runtime_source_sha256={name: software["source_sha256"][name] for name in v32_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v32)

    if getattr(args, "method_version", "v1") in ("v33", "v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v33 import geometry_semantics as semantics_v33
        v33_sources = v32_sources + (
            "visual_field_policy_v33.py", "visual_field_subscription_v33.py",
            "visual_field_v33.py", "visual_translation_guard_v33.py")
        manifest.update(
            model_geometry_admission=semantics_v33(),
            prompt_composition="V33 explicitly reviewed bounded reversal of adverse measured motion with preserved V32 perception and Field planning",
            adverse_heading_abort_contract="Only a current explicit fixed reverse route may abort a precheckpoint heading with acknowledged adverse motion. Current whole-scene, reference, posture and boundary checks remain; all measured debt and direction/shared limits persist. Prospective preview cannot authorize execution or refund budget.",
            v33_runtime_source_sha256={name: software["source_sha256"][name] for name in v33_sources},
        )

    if getattr(args, "method_version", "v1") in ("v34", "v35", "v36"):
        from gpt_field.visual_field_subscription_v34 import geometry_semantics as semantics_v34, FEEDBACK_VERSION as feedback_v34
        v34_sources = v33_sources + (
            "visual_field_policy_v34.py", "visual_field_subscription_v34.py",
            "visual_field_proposals_v34.py", "visual_reference_review_v34.py",
            "visual_feedback_transaction_v34.py", "visual_field_v34.py",
            "visual_translation_guard_v34.py")
        manifest.update(
            model_geometry_admission=semantics_v34(),
            prompt_composition="V34 explicit checked recovery-to-approach offers, current paid-reference review and one final feedback materialization",
            task_resumption_contract="Qualified current visual goals can be offered as an explicit noncontact approach transition from recovery. Every offer retains the full route and first-command preview; model selection is required, with no automatic stage change or physical-debt refund.",
            current_reference_choice_contract="Movement offers and current observation-bound reference candidates are distinct explicit model choices. Null references remove support, not task completion; existing anatomy/proxy and semantic-identity checks remain.",
            prospective_recovery_role_contract="After a reference-only Direct handoff, noncontact recovery previews bind the proposed Field controller, hold gripper and current measured orientation transactionally. Reverse and ordinary recovery previews retain current target/reference, geometry and physical-debt checks. Live admission still requires an explicit model response; the original role context is restored after every preview.",
            feedback_transaction_contract="All observer/control updates retain their original order. Only intermediate diagnostic materializations inside observe_transition are suppressed; the complete current row is rendered once after all observers. No across-observation cache or excluded control work.",
            hand_outline_runtime_change=False,
            v34_runtime_source_sha256={name: software["source_sha256"][name] for name in v34_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v34)

    if getattr(args, "method_version", "v1") in ("v35", "v36"):
        from gpt_field.visual_field_subscription_v35 import geometry_semantics as semantics_v35, FEEDBACK_VERSION as feedback_v35
        v35_sources = v34_sources + (
            "visual_field_policy_v35.py", "visual_field_subscription_v35.py",
            "visual_field_proposals_v35.py", "visual_navigation_fields_v35.py",
            "visual_wrist_goal_v35.py", "visual_translation_guard_v35.py",
            "visual_feedback_compact_v35.py")
        manifest.update(
            model_geometry_admission=semantics_v35(),
            prompt_composition="V35 separate causal wrist-bearing navigation, same-observation hand-source continuity, exact shared current feedback and explicit effective-limit provenance",
            wrist_navigation_contract="Independent actual wrist target and declared desired bearing define a squared-pixel local objective. Only acknowledged measured motion, prospective prediction evidence and supported same-heading trust produce proposals. Current checked noncontact offers may coexist with an external rigid proxy. No estimated depth, contact alignment, automatic action, scene change or physical-debt refund.",
            hand_source_continuity_contract="Only identical current robot/image/track and unchanged declared or transported hand extent retain original source corners before inherited boundary snapshots. Changed identity, extent, state, pixels or unreliable evidence uses inherited revision checks. No adverse warning or ledger is removed.",
            current_feedback_compaction_contract="Only latest-observation request presentation factors exact shared values and named-column tables. Expansion restores every numerical/state/geometry/rejection/budget/offer fact. Raw trace, model response and controller inputs remain unchanged; processing remains in control timing.",
            hand_outline_runtime_change="same-source-coordinate-continuity-only; scalar/admission checks retained",
            v35_runtime_source_sha256={name: software["source_sha256"][name] for name in v35_sources},
        )
        manifest["feedback_presentation"].update(version=feedback_v35)

    if getattr(args, "method_version", "v1") == "v36":
        from gpt_field.visual_field_subscription_v36 import geometry_semantics as semantics_v36
        v36_sources = v35_sources + (
            "visual_field_policy_v36.py", "visual_field_subscription_v36.py",
            "visual_intent_watch_v36.py", "visual_intent_replan_v36.py")
        manifest.update(
            model_geometry_admission=semantics_v36(),
            prompt_composition="V36 ongoing causal wrist visibility review with ordinary Field replan alternatives",
            ongoing_visual_review_contract="After all acknowledged observers, current same-heading prospective response and measured target visibility may request an early model decision. No permanent motion lock, automatic reverse, outcome declaration or physical-debt refund. The trigger evidence reaches the request before the next intent acknowledges it.",
            ordinary_replan_preview_contract="At a current visual-intent review, cloned noncontact axis alternatives undergo existing full-route and first-command checks. They are ordinary model-choice hypotheses, not validated visual corrections or executable proposal IDs. Actual current admission is unchanged.",
            v36_runtime_source_sha256={name: software["source_sha256"][name] for name in v36_sources},
        )

    write_json(output / "manifest.json", manifest)
    sim = recorder = None
    steps = field_actions = direct_actions = gripper_actions = 0
    field_selection_statuses = Counter()
    field_rejected_actions = local_servo_actions = local_information_probe_actions = 0
    status, error = "starting", None
    control_started = control_stopped = None
    wall_started = time.monotonic()
    timings = {k: 0. for k in ("model_pipeline_s", "local_planning_s", "simulator_step_s", "video_render_s", "decision_s", "feedback_s", "failed_decision_s")}
    last_sim_status = {}
    cleanup_errors = []

    def state(stage):
        end = control_stopped if control_stopped is not None else time.monotonic()
        elapsed = None if control_started is None else end-control_started
        checkpoint = {"status": status, "stage": stage, "step": steps,
            "pid": os.getpid(), "controller": args.controller,
            "elapsed_control_s": elapsed,
            "budget": client.budget_snapshot(), "error": error}
        if getattr(args, "method_version", "v1") in ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"):
            checkpoint.update(
                timing_components=dict(timings),
                control_without_video_s=None if elapsed is None else elapsed-timings["video_render_s"],
                timing_scope="observed prefix at checkpoint; pending work after checkpoint is unknown unless finalized",
                finalized=stage == "finished",
                field_actions=field_actions, direct_actions=direct_actions, gripper_actions=gripper_actions,
                policy_diagnostic_counters=dict(getattr(policy, "diagnostic_counters", {})),
            )
            temporary = output / "run_state.json.tmp"
            write_json(temporary, checkpoint)
            temporary.replace(output / "run_state.json")
        else:
            write_json(output / "run_state.json", checkpoint)

    def append(log, value):
        log.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        log.flush()

    with interrupted_signals():
        try:
            state("starting_simulator")
            with simulator_without_credentials():
                sim = sim_factory(sim_config)
                sim.start()
            obs = sim.reset(task.task_id, args.episode_index)
            last_sim_status = copy_status(sim.status)
            if obs.prompt != task.prompt or obs.joint_position is None:
                raise ValueError("Simulator observable interface does not match the experiment")
            expected_initial = getattr(args, "expected_initial", None)
            if expected_initial and obs.to_dict() != json.loads(Path(expected_initial).read_text()):
                raise ValueError("Paired initial RGB/robot observation differs; no policy inference executed")
            if sim.status.get("collision", {}).get("detected"):
                raise ValueError("Initial state has a forbidden contact; no policy trial started")
            write_json(output / "simulator.json", {"metadata": sim.metadata, "initial_status": last_sim_status})
            write_json(output / "initial_observation.json", obs.to_dict())
            recorder = recorder_factory(output)
            recorder.frame(obs, 0, "initial observation")
            status = "running"
            control_started = time.monotonic()
            with (output / "trace.jsonl").open("w") as log:
                append(log, {"event": "reset", "observation": obs.to_dict(), "sim_status": last_sim_status})
                while True:
                    collision = sim.status.get("collision")
                    if not isinstance(collision, dict) or collision.get("mode") != "task_manipulation_v1":
                        raise ValueError("Required collision evaluator unavailable")
                    if collision["detected"]:
                        status = "collision_failure"
                        break
                    if sim.status.get("success"):
                        status = "success"
                        break
                    if sim.status.get("terminated"):
                        raise RuntimeError("Unexpected simulator termination")
                    stop_file = output / "operator_stop.request.json"
                    if getattr(args, "method_version", "v1") in ("v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and stop_file.is_file():
                        status, error = "user_stop", {"type": "OperatorStopFile",
                            "sha256": hashlib.sha256(stop_file.read_bytes()).hexdigest()}
                        append(log, {"event": "operator_stop", "completed_actions": steps,
                                     "request_sha256": error["sha256"]})
                        break
                    state("deciding")
                    start = time.monotonic()
                    try:
                        decision = policy.act(obs)
                    except BaseException:
                        # No Decision exists from which to read its component
                        # timers. Retain the entire failed call, including any
                        # model repair request, as a subset of decision_s.
                        failed_s = time.monotonic() - start
                        timings["decision_s"] += failed_s
                        timings["failed_decision_s"] += failed_s
                        raise
                    timings["decision_s"] += time.monotonic() - start
                    for name in ("model_pipeline_s", "local_planning_s"):
                        timings[name] += decision.diagnostics[name]
                    append(log, {"event": "action", "step": steps, "action": decision.action,
                                 "diagnostics": decision.diagnostics})
                    state("executing")
                    before = obs
                    start = time.monotonic()
                    obs, success = sim.step(decision.action)
                    timings["simulator_step_s"] += time.monotonic() - start
                    steps += 1
                    field_actions += int(decision.diagnostics["controller"] == "visual_field")
                    direct_actions += int(decision.diagnostics["controller"] == "direct")
                    gripper_actions += int(decision.diagnostics["controller"] == "gripper")
                    selection = decision.diagnostics.get("selection")
                    if decision.diagnostics["controller"] == "visual_field" and selection:
                        field_selection_statuses[selection["status"]] += 1
                        field_rejected_actions += int(not selection.get("accepted", False))
                        local_servo_actions += int(
                            selection["status"] in ("visual_distance_descent", "metric_field_tracking")
                            and selection.get("accepted") is True
                            and not decision.diagnostics.get("attachment_gate")
                        )
                        local_information_probe_actions += int(
                            selection["status"] == "local_information_probe"
                            and selection.get("accepted") is True
                            and not decision.diagnostics.get("attachment_gate")
                        )
                    last_sim_status = copy_status(sim.status)
                    append(log, {"event": "executed", "step": steps - 1, "observation": obs.to_dict(),
                                 "action": decision.action, "sim_status": last_sim_status})
                    start = time.monotonic()
                    feedback = policy.observe_transition(before, decision, obs)
                    timings["feedback_s"] += time.monotonic() - start
                    append(log, {"event": "feedback", "step": steps - 1, "feedback": feedback})
                    start = time.monotonic()
                    recorder.frame(obs, steps, decision.diagnostics["controller"] + " / " + decision.diagnostics["phase"])
                    timings["video_render_s"] += time.monotonic() - start
                    state("observed")
                    if decision.diagnostics["refreshed"] or success or sim.status["collision"]["detected"]:
                        print(json.dumps({"step": steps, "controller": decision.diagnostics["controller"],
                            "phase": decision.diagnostics["phase"], "requests": client.budget_snapshot()["api_calls"],
                            "success": success, "collision": sim.status["collision"]["detected"]}), flush=True)
        except KeyboardInterrupt:
            status, error = "user_stop", {"type": "KeyboardInterrupt"}
        except (APIError, MissingCredentials, BudgetExceeded, SchemaError) as exc:
            status, error = "subscription_error", {"type": type(exc).__name__,
                "http_status": getattr(exc, "http_status", None), "error_code": getattr(exc, "error_code", None)}
            if isinstance(exc, SchemaError):
                error["category"] = "response_schema_validation"
        except Exception as exc:
            status, error = "infrastructure_error", {"type": type(exc).__name__,
                "message": "Evaluation failed; consult frozen code and last trace event"}
            # Local traceback only, no arbitrary transport bodies or credentials.
            import traceback
            traceback.print_exc()
        finally:
            control_stopped = time.monotonic()
            if sim is not None:
                last_sim_status = copy_status(sim.status)
            for name, cleanup in (("policy", policy.finish), ("video", recorder.close if recorder else None),
                                   ("simulator", sim.close if sim else None)):
                if cleanup:
                    try:
                        cleanup()
                    except Exception as exc:
                        cleanup_errors.append({"component": name, "type": type(exc).__name__})
    elapsed = 0. if control_started is None else control_stopped - control_started
    collision = last_sim_status.get("collision", {})
    qualified = status == "success" and last_sim_status.get("success") is True and not collision.get("detected", True)
    summary = {"status": status, "task_success": qualified, "method_version": getattr(args, "method_version", "v1"),
        "libero_success": last_sim_status.get("libero_success", False), "collision": collision,
        "controller": args.controller, "task": task.key, "shots": int(demo is not None),
        "steps": steps, "field_actions": field_actions, "direct_actions": direct_actions,
        "gripper_actions": gripper_actions,
        "field_selection_status_counts": dict(field_selection_statuses),
        "field_rejected_actions": field_rejected_actions, "local_servo_actions": local_servo_actions,
        "local_information_probe_actions": local_information_probe_actions,
        "policy_diagnostic_counters": dict(getattr(policy, "diagnostic_counters", {})),
        "control_elapsed_s": elapsed, "control_without_video_s": elapsed - timings["video_render_s"],
        "wall_elapsed_s": time.monotonic() - wall_started, "timing_components": timings,
        "timing_accounting": "decision_s includes failed_decision_s; model_pipeline_s/local_planning_s are returned-Decision components. Failed decisions remain separately timed and included in full control time. Components overlap and must not all be added.",
        "local_total_s": policy.local_total_s, "budget": client.budget_snapshot(),
        "attempts": policy.attempts, "recovery_events": policy.events,
        "metric_caveat": "attempts=open-to-close episodes; recovery events are versioned observable controller diagnostics, not certified physical failures; field_actions includes holds/rejections; local_servo_actions and local_information_probe_actions count acknowledged accepted selections without an attachment gate, not certified task progress",
        "completion_time_eligible": qualified, "scored": status in ("success", "collision_failure"),
        "error": error, "cleanup_errors": cleanup_errors}
    write_json(output / "summary.json", summary)
    state("finished")
    print(json.dumps(summary), flush=True)
    return summary


def copy_status(status):
    return json.loads(json.dumps(status))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=("bowl_plate", "drawer_bowl"), required=True)
    parser.add_argument("--controller", choices=("hybrid", "direct"), required=True)
    parser.add_argument("--method-version", choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36"), default="v1")
    parser.add_argument("--demo", type=Path)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--bddl", type=Path)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--expected-initial", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sim-python", default="/workspace/venvs/libero-plus/bin/python")
    parser.add_argument("--libero-root", default="/workspace/heading_flow_pub/third_party/LIBERO")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--store-dir", type=Path, default=None)
    args = parser.parse_args()
    if not 0 <= args.episode_index < 50:
        parser.error("episode-index must be in [0,49]")
    if args.method_version in ("v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v34", "v35", "v36") and (args.controller != "hybrid" or args.demo is not None):
        parser.error(f"{args.method_version} trials require zero-shot Hybrid; use existing Direct records as references")
    credentials = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile)
    result = execute(args, token_provider=credentials.access_token)
    return 0 if result["status"] in ("success", "collision_failure") else 3


if __name__ == "__main__":
    raise SystemExit(main())
