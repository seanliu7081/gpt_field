"""Evaluate model-routed coarse fields and direct contact actions on a registered task.

Defaults to fresh LIBERO-Spatial task 0 / initial state 0 / seed 7; distance
control also supports --task drawer_bowl (LIBERO-Goal task 3). Live execution has no
action, call-count or episode-time horizon and continues through gripper closure
until LIBERO success, user interruption, subscription failure or infrastructure
failure. An opt-in compact protocol also stops immediately when Astra declares
completion, without treating that declaration as LIBERO success. Only
the isolated scripted smoke stops after two thirty-action field blocks and one
five-action direct block. --speed-first restores compact distance navigation and
Direct interaction with collisions recorded as secondary metrics, raw LIBERO
completion, and no automatic episode cap.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from gpt_field.astra import APIError, AstraClient, BudgetExceeded, MissingCredentials, SchemaError
from gpt_field.direct_history import HISTORY_PROTOCOL
from gpt_field.compact_subscription import ChatGPTPlanCompactClient
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.hybrid_policy import HybridFieldDirectPolicy, ModelTaskComplete
from gpt_field.hybrid_subscription import (
    ChatGPTPlanHybridClient, HYBRID_CHUNK_SIZE, HYBRID_FIELD_UPDATE_STEPS, HYBRID_GRID, HYBRID_PROMPT_VERSION,
)
from gpt_field.hybrid_video import HybridVideoRecorder
from gpt_field.distance_subscription import ChatGPTPlanDistanceClient
from gpt_field.distance_policy import DistanceHybridPolicy
from gpt_field.distance_video import DistanceHybridVideoRecorder
from gpt_field.distance_tasks import DISTANCE_TASK_KEYS, resolve_distance_task
from gpt_field.demonstration import load_demonstration
from gpt_field.demo_direct_subscription import ChatGPTPlanDemoDirectClient
from gpt_field.zero_shot_direct_subscription import ChatGPTPlanZeroShotDirectClient
from gpt_field.historical_zero_shot_direct import ChatGPTPlanHistoricalZeroShotDirectClient
from gpt_field.demo_direct_video import DemoDirectVideoRecorder, ZeroShotDirectVideoRecorder, HistoricalZeroShotDirectVideoRecorder
from gpt_field.libero import (
    LiberoConfig, SimProcess, validate_initial_state_override, validate_task_bddl_override,
)
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig
from gpt_field.target_identity_guidance import normalize_target_description, target_identity_metadata


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_frozen_subscription_close_runner",
    ROOT / "scripts/evaluate_bowl_subscription_close.py")
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
TASK = _old.TASK
DEFAULT_SIM_PYTHON = _old.DEFAULT_SIM_PYTHON
DEFAULT_LIBERO_ROOT = _old.DEFAULT_LIBERO_ROOT
configuration = _old.configuration
validate_config = _old.validate_config
SMOKE_STEPS = 2 * HYBRID_FIELD_UPDATE_STEPS + HYBRID_CHUNK_SIZE
STOPPING_RULE = ["LIBERO success", "user stop", "subscription authorization or quota error", "infrastructure error"]


def _demo_direct(controller_mode, task_key, distance_field_control=False, *, demonstration=False,
                 historical_zero_shot_direct=False):
    return (not historical_zero_shot_direct and controller_mode == "direct" and not distance_field_control
            and (task_key == "drawer_bowl" or task_key == "bowl_plate" and demonstration))


def _client_type(field_representation, distance_field_control=False, *,
                 controller_mode="hybrid", task_key="bowl_plate", demonstration=False,
                 zero_shot_collision_free=False, historical_zero_shot_direct=False, speed_first=False, speed_first_v2=False,
                 speed_first_v3=False, operation_control=False, contact_region_mode=None):
    _validate_contact_region_mode(contact_region_mode, operation_control)
    if type(operation_control) is not bool or (operation_control and not speed_first_v3):
        raise ValueError("Operation control requires the speed_first_v3 evaluation protocol")
    if speed_first_v3 and not (speed_first and speed_first_v2):
        raise ValueError("speed_first_v3 requires the speed_first_v2 evaluation protocol")
    if speed_first:
        if (field_representation != "compact" or controller_mode != "hybrid" or demonstration
                or not distance_field_control or zero_shot_collision_free or historical_zero_shot_direct):
            raise ValueError("Speed-first requires zero-shot compact distance Hybrid without other protocols")
        if operation_control:
            if contact_region_mode is not None:
                from gpt_field.contact_region_policy import ChatGPTPlanContactRegionClient
                return ChatGPTPlanContactRegionClient
            from gpt_field.speed_first_v4 import ChatGPTPlanSpeedFirstV4Client
            return ChatGPTPlanSpeedFirstV4Client
        if speed_first_v3:
            from gpt_field.speed_first_v3 import ChatGPTPlanSpeedFirstV3Client
            return ChatGPTPlanSpeedFirstV3Client
        if speed_first_v2:
            from gpt_field.speed_first_v2 import ChatGPTPlanSpeedFirstV2Client
            return ChatGPTPlanSpeedFirstV2Client
        from gpt_field.speed_first_hybrid import ChatGPTPlanSpeedFirstClient
        return ChatGPTPlanSpeedFirstClient
    if historical_zero_shot_direct:
        if (field_representation != "compact" or controller_mode != "direct" or demonstration
                or distance_field_control or zero_shot_collision_free):
            raise ValueError("Historical zero-shot Direct requires compact pure Direct without demonstration or collision protocol")
        return ChatGPTPlanHistoricalZeroShotDirectClient
    if zero_shot_collision_free and controller_mode == "direct":
        if field_representation != "compact" or demonstration or distance_field_control or task_key != "bowl_plate":
            raise ValueError("Zero-shot Direct requires compact bowl control without demonstrations or fields")
        return ChatGPTPlanZeroShotDirectClient
    if _demo_direct(controller_mode, task_key, distance_field_control, demonstration=demonstration):
        if field_representation != "compact":
            raise ValueError("Demonstration Direct requires compact transport")
        return ChatGPTPlanDemoDirectClient
    if distance_field_control:
        if field_representation != "compact":
            raise ValueError("Distance control requires compact representation")
        return ChatGPTPlanDistanceClient
    if field_representation == "compact":
        return ChatGPTPlanCompactClient
    if field_representation == "dense":
        return ChatGPTPlanHybridClient
    raise ValueError("field_representation must be compact or dense")


def _validate_contact_region_mode(mode, operation_control):
    if mode not in (None, "apply", "shadow") or (mode is not None and not operation_control):
        raise ValueError("Contact-region feedback requires operation control and apply/shadow mode")


def _control_options(field_representation, reasoning_effort, stop_on_model_complete, *, offline_smoke=False,
                     controller_mode="hybrid", execution_mode="fixed", feedback_history="full",
                     centered_placement_guidance=False, grasp_settle_guidance=False, field_improvements=False,
                     distance_field_control=False, task_key="bowl_plate", demonstration=False,
                     zero_shot_collision_free=False, historical_zero_shot_direct=False, speed_first=False, speed_first_v2=False,
                     speed_first_v3=False, robot_geometry_guidance=False, target_description=None, attempt_memory=False, pregrasp_feedback=False,
                     operation_control=False, operation_planner="field", contact_region_mode=None):
    _validate_contact_region_mode(contact_region_mode, operation_control)
    task_spec = resolve_distance_task(task_key)
    target_description = normalize_target_description(target_description)
    if type(operation_control) is not bool or (operation_control and not speed_first_v3):
        raise ValueError("Operation control requires the speed_first_v3 evaluation protocol")
    if operation_planner not in ("field", "waypoint") or (not operation_control and operation_planner != "field"):
        raise ValueError("The waypoint comparison requires operation control")
    if operation_control and not (attempt_memory and pregrasp_feedback):
        raise ValueError("Operation control requires its state memory and pregrasp feedback interfaces")
    if type(pregrasp_feedback) is not bool or (pregrasp_feedback and not speed_first_v3):
        raise ValueError("Pregrasp feedback requires the opt-in speed_first_v3 protocol")
    if type(attempt_memory) is not bool or (attempt_memory and not speed_first_v3):
        raise ValueError("Attempt memory requires the opt-in speed_first_v3 protocol")
    if target_description is not None and not speed_first_v3:
        raise ValueError("Target description requires the opt-in speed_first_v3 protocol")
    if type(robot_geometry_guidance) is not bool or (robot_geometry_guidance and not speed_first_v3):
        raise ValueError("Robot geometry guidance requires the opt-in speed_first_v3 protocol")
    if type(speed_first_v3) is not bool or (speed_first_v3 and not (speed_first and speed_first_v2)):
        raise ValueError("speed_first_v3 requires the speed_first_v2 evaluation protocol")
    if type(speed_first_v2) is not bool or (speed_first_v2 and not speed_first):
        raise ValueError("speed_first_v2 requires the speed_first evaluation protocol")
    if type(speed_first) is not bool:
        raise ValueError("speed_first must be a boolean")
    if speed_first and (field_representation != "compact" or controller_mode != "hybrid" or demonstration
            or not distance_field_control or zero_shot_collision_free or historical_zero_shot_direct
            or offline_smoke or reasoning_effort != "medium" or stop_on_model_complete
            or execution_mode != "tracking" or feedback_history != "block_summary"
            or not centered_placement_guidance or not grasp_settle_guidance or not field_improvements):
        raise ValueError("Speed-first requires zero-shot medium compact distance Hybrid, tracking, block summaries, shared guidance and LIBERO-only stopping")
    if historical_zero_shot_direct:
        if (field_representation != "compact" or controller_mode != "direct" or demonstration
                or distance_field_control or zero_shot_collision_free or offline_smoke
                or reasoning_effort != "medium" or stop_on_model_complete or execution_mode != "fixed"
                or feedback_history != "block_summary" or not centered_placement_guidance
                or not grasp_settle_guidance or field_improvements):
            raise ValueError("Historical zero-shot Direct requires medium, no demo, fixed five-action Direct, shared guidance and LIBERO-only stopping")
        return {"reasoning_effort": reasoning_effort, "controller_mode": controller_mode,
                "execution_mode": execution_mode, "feedback_history": feedback_history,
                "centered_placement_guidance": True, "grasp_settle_guidance": True,
                "stop_on_model_complete": False, "task_key": task_spec.key}
    if type(zero_shot_collision_free) is not bool:
        raise ValueError("zero_shot_collision_free must be a boolean")
    if zero_shot_collision_free and (demonstration or task_key != "bowl_plate" or offline_smoke
            or field_representation != "compact" or reasoning_effort != "medium" or stop_on_model_complete
            or feedback_history != "block_summary" or not centered_placement_guidance or not grasp_settle_guidance
            or (controller_mode == "hybrid" and not distance_field_control)
            or (controller_mode == "direct" and (distance_field_control or field_improvements or execution_mode != "fixed"))):
        raise ValueError("Zero-shot collision-free evaluation requires no demo, bowl task, medium, shared guidance, block summaries and qualified LIBERO-only stopping; use distance Hybrid or fixed Direct")
    demo_direct = _demo_direct(controller_mode, task_key, distance_field_control, demonstration=demonstration)
    if task_spec.key != "bowl_plate" and not distance_field_control and not demo_direct:
        raise ValueError("Explicit task selection requires --distance-field-control")
    _client_type(field_representation, distance_field_control, controller_mode=controller_mode,
                 task_key=task_key, demonstration=demonstration, zero_shot_collision_free=zero_shot_collision_free,
                 speed_first=speed_first, speed_first_v2=speed_first_v2, speed_first_v3=speed_first_v3,
                 operation_control=operation_control, contact_region_mode=contact_region_mode)
    if demo_direct and (offline_smoke or reasoning_effort != "medium" or execution_mode != "fixed"
            or feedback_history != "block_summary" or not centered_placement_guidance
            or not grasp_settle_guidance or field_improvements or stop_on_model_complete):
        raise ValueError("Demonstration Direct requires medium, fixed, block summaries, shared guidance and LIBERO-only stopping without field improvements")
    if type(distance_field_control) is not bool:
        raise ValueError("distance_field_control must be a boolean")
    if distance_field_control and (not field_improvements or feedback_history != "block_summary"
            or not centered_placement_guidance or not grasp_settle_guidance):
        raise ValueError("Distance control requires field improvements and shared guidance with block summaries")
    if reasoning_effort not in ("medium", "high"):
        raise ValueError("reasoning_effort must be medium or high")
    if controller_mode not in ("hybrid", "direct"):
        raise ValueError("full-task controller_mode must be hybrid or direct")
    if execution_mode not in ("fixed", "tracking") or feedback_history not in ("full", "block_summary"):
        raise ValueError("Unsupported field execution or feedback history mode")
    if type(centered_placement_guidance) is not bool:
        raise ValueError("centered_placement_guidance must be a boolean")
    if centered_placement_guidance and (offline_smoke or field_representation != "compact"
            or stop_on_model_complete):
        raise ValueError("Centered placement guidance requires live compact Hybrid or Direct with LIBERO-only completion")
    if type(grasp_settle_guidance) is not bool:
        raise ValueError("grasp_settle_guidance must be a boolean")
    if grasp_settle_guidance and not centered_placement_guidance:
        raise ValueError("Grasp settling guidance requires centered placement guidance")
    if type(field_improvements) is not bool:
        raise ValueError("field_improvements must be a boolean")
    if field_improvements and (offline_smoke or field_representation != "compact"
            or controller_mode != "hybrid" or execution_mode != "tracking"
            or reasoning_effort != "medium" or stop_on_model_complete):
        raise ValueError("Field improvements require compact Hybrid, tracking, medium effort and LIBERO-only completion")
    if (offline_smoke or field_representation == "dense") and (
            controller_mode != "hybrid" or execution_mode != "fixed" or feedback_history != "full"):
        raise ValueError("Optimized execution and forced Direct require live compact control")
    if (offline_smoke or field_representation == "dense") and (
            reasoning_effort != "high" or stop_on_model_complete):
        raise ValueError("Medium effort and model-completion stopping require live compact control")
    return ({"reasoning_effort": reasoning_effort, "stop_on_model_complete": stop_on_model_complete,
             "controller_mode": controller_mode, "execution_mode": execution_mode, "feedback_history": feedback_history,
             "centered_placement_guidance": centered_placement_guidance,
             "grasp_settle_guidance": grasp_settle_guidance, "field_improvements": field_improvements,
             **({"zero_shot_collision_free": True} if zero_shot_collision_free else {}),
             **({"robot_geometry_guidance": True} if robot_geometry_guidance else {}),
             **({"target_description": target_description} if target_description is not None else {}),
             **({"attempt_memory": True} if attempt_memory else {}),
             **({"pregrasp_feedback": True} if pregrasp_feedback else {}),
             **({"planner_mode": operation_planner} if operation_control else {}),
             **({"distance_field_control": True} if distance_field_control else {}),
             **({"task_key": task_spec.key} if task_spec.key != "bowl_plate" or demo_direct else {})}
            if field_representation == "compact" else {})


def _stopping_rule(stop_on_model_complete, zero_shot_collision_free=False):
    if zero_shot_collision_free:
        return ["LIBERO success with no forbidden collision", "forbidden collision (task failure)", *STOPPING_RULE[1:]]
    return (["Astra declares task_complete=true"] if stop_on_model_complete else []) + STOPPING_RULE


def _validate_collision_configuration(sim_config, enabled, *, speed_first=False):
    mode = getattr(sim_config, "collision_check_mode", "none")
    if speed_first:
        if enabled or mode != "task_manipulation_v1" or getattr(sim_config, "stop_on_collision", True):
            raise ValueError("Speed-first requires task_manipulation_v1 collision monitoring in record-only mode")
        return
    if enabled and (mode != "bowl_obstacles_v1" or sim_config.task_bddl_override is None
                    or sim_config.initial_state_override is None):
        raise ValueError("Zero-shot collision-free evaluation requires a custom bowl scene/state and bowl_obstacles_v1 monitoring")
    if not enabled and mode != "none":
        raise ValueError("Collision monitoring requires --zero-shot-collision-free in this runner")


def _collision_record(sim_status, *, speed_first=False):
    record = sim_status.get("collision")
    required_mode = "task_manipulation_v1" if speed_first else "bowl_obstacles_v1"
    if not isinstance(record, dict) or record.get("enabled") is not True or record.get("mode") != required_mode:
        raise ValueError("Required collision monitor is missing or disabled")
    if type(record.get("detected")) is not bool or type(sim_status.get("libero_success")) is not bool:
        raise ValueError("Collision monitor and raw LIBERO status must be explicit")
    if speed_first:
        if record.get("stop_on_collision") is not False or record.get("collision_policy") != "record_only":
            raise ValueError("Speed-first collision monitor must be record-only")
        if sim_status.get("success") != sim_status["libero_success"]:
            raise ValueError("Speed-first simulator success must equal raw LIBERO success")
        if record["detected"] and not sim_status["libero_success"] and sim_status.get("terminated"):
            raise ValueError("Recorded collision must not terminate speed-first evaluation")
        return dict(record)
    if sim_status.get("success") != (sim_status["libero_success"] and not record["detected"]):
        raise ValueError("Simulator success must require the LIBERO goal and no forbidden collision")
    if record["detected"] and not sim_status.get("terminated"):
        raise ValueError("A forbidden collision must terminate the trial")
    return dict(record)


def _operation_metadata(enabled, planner_mode="field", contact_region_mode=None):
    _validate_contact_region_mode(contact_region_mode, enabled)
    if not enabled:
        return {}
    version = "operation-field-hybrid-v4"
    if contact_region_mode is not None:
        from gpt_field.contact_region_policy import CONTACT_REGION_POLICY_VERSION
        version = CONTACT_REGION_POLICY_VERSION
    return {"speed_first_protocol": version, "operation_control": {
        "version": version, "planner_mode": planner_mode,
        **({"contact_region_mode": contact_region_mode,
            "contact_correction_enabled": contact_region_mode == "apply"} if contact_region_mode else {}),
        "state_sources": "accepted model claims, RGB tracking and measured robot transitions",
        "model_claims_independently_verified": False,
        "verification_motion": "explicit model-requested offset only",
        "maximum_operation_actions": 60,
        "extra_dedicated_model_calls": 0,
        "depth_or_environment_calibration": False,
        "task_success_certified": False,
    }}


def _pregrasp_metadata(enabled, contact_region_mode=None):
    if not enabled:
        return {}
    if contact_region_mode is not None:
        from gpt_field.contact_region_feedback import contact_region_schema
        from gpt_field.contact_region_policy import CONTACT_REGION_POLICY_VERSION
        return {"pregrasp_feedback": {
            "version": CONTACT_REGION_POLICY_VERSION, "mode": contact_region_mode,
            "scope": "semantic_image_regions_with_held_out_relative_motion_validation",
            "source": "two_RGB_views_and_measured_robot_feedback_only",
            "reference_schema_sha256": hashlib.sha256(json.dumps(contact_region_schema(),
                sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "region_anchors_are_contact_measurements": False,
            "maximum_total_nominal_goal_correction_m": .006,
            "extra_model_calls": 0, "automatic_probe": False, "automatic_close": False,
            "task_success_certified": False, "depth_or_environment_calibration": False,
            "fallback": "normal_model_selected_field_or_direct",
        }}
    from gpt_field.speed_first_pregrasp import (PREGRASP_VERSION,
        pregrasp_instructions, pregrasp_reference_schema)
    return {"pregrasp_feedback": {
        "version": PREGRASP_VERSION,
        "scope": "fixed_orientation_open_gripper_local_xyz_field_goals",
        "source": "two_RGB_views_and_measured_robot_feedback_only",
        "instructions_sha256": hashlib.sha256(pregrasp_instructions().encode()).hexdigest(),
        "reference_schema_sha256": hashlib.sha256(json.dumps(pregrasp_reference_schema(),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "extra_model_calls": 0, "automatic_probe": False, "automatic_close": False,
        "task_success_certified": False, "depth_or_environment_calibration": False,
        "fallback": "normal_model_selected_field_or_direct",
    }}


def _attempt_memory_metadata(enabled):
    if not enabled:
        return {}
    from gpt_field.attempt_memory import (ATTEMPT_MEMORY_VERSION, attempt_memory_instructions,
        attempt_review_schema)
    return {"attempt_memory": {
        "version": ATTEMPT_MEMORY_VERSION, "scope": "model_context_only",
        "source": "accepted_model_statements_and_measured_robot_feedback_only",
        "instructions_sha256": hashlib.sha256(attempt_memory_instructions().encode()).hexdigest(),
        "review_schema_sha256": hashlib.sha256(json.dumps(attempt_review_schema(),
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model_claims_independently_verified": False,
        "additional_inference_calls": 0, "execution_gates_added": False,
    }}


def _target_identity_metadata(description):
    if description is None:
        return {}
    return {"target_identity_guidance": target_identity_metadata(description)}


def _robot_geometry_metadata(enabled):
    if not enabled:
        return {}
    from gpt_field.robot_geometry_guidance import robot_geometry_metadata
    return {"robot_geometry_guidance": robot_geometry_metadata()}


def _speed_first_metadata(enabled, v2=False, v3=False):
    if not enabled:
        return {}
    return {"speed_first": True, "speed_first_protocol": "speed-first-distance-hybrid-v3" if v3 else "speed-first-distance-hybrid-v2" if v2 else "speed-first-distance-hybrid-v1",
            **({"speed_first_v2": True, "local_gripper_operations": True, "bounded_input_history": True} if v2 else {}),
            **({"speed_first_v3": True, "local_object_correction": True,
                "camera_render_size": 512, "policy_image_size": 512,
                "map_spacing_m": .001, "map_size": 161, "map_node_count": 161 ** 3,
                "map_coverage_m": .16, "map_storage": "lazy_compact_field"} if v3 else {}),
            "zero_shot": True, "collision_policy": "record_only",
            "success_qualification": "raw_libero_goal", "primary_timing_metric": "control_without_video_s",
            "pixel_goal_control": v3, "automatic_view_recovery": False,
            "operator_stop_file": "operator_stop.request.json"}


def _validate_task_configuration(config, sim_config, task_spec, *, speed_first_v3=False):
    if sim_config.suite != task_spec.suite:
        raise ValueError("Simulator suite must match the selected task")
    # Reuse the frozen runner's effort-independent transport, image, seed and
    # horizon checks. Only its old hard-coded suite check needs substitution;
    # actual simulator construction and every artifact retain the real suite.
    if speed_first_v3:
        if sim_config.render_size != 512 or sim_config.image_size != 512:
            raise ValueError("Speed-first v3 requires native 512 render and 512 policy images")
        # Only the image dimensions differ from the historical transport/seed/
        # horizon checks. The actual simulator and artifacts retain native512.
        validate_config(config, replace(sim_config, suite="libero_spatial", render_size=256, image_size=224))
    else:
        validate_config(config, replace(sim_config, suite="libero_spatial"))


def _validate_episode_index(episode_index):
    if type(episode_index) is not int or not 0 <= episode_index < 50:
        raise ValueError("episode_index must be an integer in [0, 49]; no modulo wrapping")
    return episode_index


def _initial_state_override(sim_config):
    """Validate and capture exact override bytes before any run side effects."""
    if sim_config.initial_state_override is None:
        return None, None
    source = validate_initial_state_override(sim_config.initial_state_override)
    data = Path(source["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError("Initial-state override changed during validation")
    return source, data


def _initial_state_metadata(source):
    if source is None:
        return {}
    return {"benchmark_variant": "custom_layout", "initial_state_override_source": source,
            "initial_state_override": {"artifact": "initial_state_override.npy",
                **{key: source[key] for key in ("sha256", "shape", "dtype")}}}


def _task_bddl_override(sim_config):
    if sim_config.task_bddl_override is None:
        return None, None
    source = validate_task_bddl_override(sim_config.task_bddl_override)
    data = Path(source["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError("Task BDDL override changed during validation")
    return source, data


def _task_bddl_metadata(source):
    if source is None:
        return {}
    return {"benchmark_variant": "custom_scene", "task_bddl_override_source": source,
            "task_bddl_override": {"artifact": "task_override.bddl",
                **{key: source[key] for key in ("sha256", "byte_count")}}}


def _freeze_original_task_bddl(output, sim_status, override_source):
    """Archive evaluator provenance before the first model request."""
    if override_source is None:
        return
    if (sim_status.get("bddl_sha256") != override_source["sha256"]
            or sim_status.get("task_bddl_override_sha256") != override_source["sha256"]):
        raise ValueError("Simulator did not load the frozen custom task BDDL")
    original = Path(sim_status["original_bddl_file"]).read_bytes()
    if hashlib.sha256(original).hexdigest() != sim_status["original_bddl_sha256"]:
        raise ValueError("Original task BDDL changed during simulator reset")
    (output / "original_task.bddl").write_bytes(original)


def _demonstration(path, *, distance_field_control, task_key, controller_mode="hybrid",
                   historical_zero_shot_direct=False):
    if historical_zero_shot_direct:
        if path is not None:
            raise ValueError("Historical zero-shot Direct cannot receive a demonstration")
        return None
    demo_direct = _demo_direct(controller_mode, task_key, distance_field_control, demonstration=path is not None)
    if path is None:
        if demo_direct:
            raise ValueError("Drawer Direct requires --demonstration for the matched comparison")
        return None
    if task_key not in DISTANCE_TASK_KEYS or not (distance_field_control or demo_direct):
        raise ValueError("--demonstration requires task-matched distance control or Direct")
    return load_demonstration(path, task_key=task_key)


def preflight(config, sim_config, output=None, *, field_representation="compact",
              reasoning_effort="high", stop_on_model_complete=False,
              controller_mode="hybrid", execution_mode="fixed", feedback_history="full",
              centered_placement_guidance=False, grasp_settle_guidance=False, field_improvements=False,
              distance_field_control=False, task_key="bowl_plate", demonstration_path=None, episode_index=0,
                    zero_shot_collision_free=False, historical_zero_shot_direct=False, speed_first=False, speed_first_v2=False, speed_first_v3=False,
                    robot_geometry_guidance=False, target_description=None, attempt_memory=False, pregrasp_feedback=False,
                    operation_control=False, operation_planner="field", contact_region_mode=None):
    episode_index = _validate_episode_index(episode_index)
    task_spec = resolve_distance_task(task_key)
    demo_direct = _demo_direct(controller_mode, task_key, distance_field_control, demonstration=demonstration_path is not None,
                               historical_zero_shot_direct=historical_zero_shot_direct)
    zero_shot_direct = (zero_shot_collision_free or historical_zero_shot_direct) and controller_mode == "direct"
    direct_only = demo_direct or zero_shot_direct
    client_options = _control_options(field_representation, reasoning_effort, stop_on_model_complete,
        controller_mode=controller_mode, execution_mode=execution_mode, feedback_history=feedback_history,
        centered_placement_guidance=centered_placement_guidance, grasp_settle_guidance=grasp_settle_guidance,
        field_improvements=field_improvements, distance_field_control=distance_field_control, task_key=task_key,
        demonstration=demonstration_path is not None, zero_shot_collision_free=zero_shot_collision_free,
        historical_zero_shot_direct=historical_zero_shot_direct, speed_first=speed_first, speed_first_v2=speed_first_v2, speed_first_v3=speed_first_v3,
        robot_geometry_guidance=robot_geometry_guidance, target_description=target_description, attempt_memory=attempt_memory, pregrasp_feedback=pregrasp_feedback,
        operation_control=operation_control, operation_planner=operation_planner, contact_region_mode=contact_region_mode)
    _validate_collision_configuration(sim_config, zero_shot_collision_free, speed_first=speed_first)
    _validate_task_configuration(config, sim_config, task_spec, speed_first_v3=speed_first_v3)
    override_source, _ = _initial_state_override(sim_config)
    task_override_source, _ = _task_bddl_override(sim_config)
    demonstration = _demonstration(demonstration_path, distance_field_control=distance_field_control,
                                   task_key=task_key, controller_mode=controller_mode,
                                   historical_zero_shot_direct=historical_zero_shot_direct)
    if demonstration is not None:
        client_options["demonstration"] = demonstration
    if output is not None and Path(output).exists():
        raise FileExistsError("Fresh hybrid experiment requires a new output directory")
    camera = image_to_data_url(np.zeros((sim_config.image_size, sim_config.image_size, 3), dtype=np.uint8), size=sim_config.image_size, rotate=False)
    obs = PolicyObservation(camera, camera, (0, 0, 1, 0, 0, 0, .04, -.04), task_spec.prompt)
    client_type = _client_type(field_representation, distance_field_control,
                               controller_mode=controller_mode, task_key=task_key, demonstration=demonstration is not None,
                               zero_shot_collision_free=zero_shot_collision_free,
                               historical_zero_shot_direct=historical_zero_shot_direct, speed_first=speed_first, speed_first_v2=speed_first_v2, speed_first_v3=speed_first_v3, operation_control=operation_control, contact_region_mode=contact_region_mode)
    client = client_type(config, token_provider=lambda: None, **client_options)
    payload = client.build_payload(obs.to_dict())
    return {"status": "preflight_only", "task": task_spec.prompt, "task_id": task_spec.task_id, "episode_index": episode_index,
        **_speed_first_metadata(speed_first, speed_first_v2, speed_first_v3),
        **_robot_geometry_metadata(robot_geometry_guidance),
        **_target_identity_metadata(target_description),
        **_attempt_memory_metadata(attempt_memory),
        **_pregrasp_metadata(pregrasp_feedback, contact_region_mode),
        **_operation_metadata(operation_control, operation_planner, contact_region_mode),
        **({"zero_shot_collision_free": True} if zero_shot_collision_free else {}),
        **_initial_state_metadata(override_source),
        **_task_bddl_metadata(task_override_source),
        **({"demonstration": {"artifact": "demonstration.json", **demonstration.metadata()}}
           if demonstration is not None else {}),
        **({"distance_task": task_spec.key, "task_spec": asdict(task_spec)} if task_spec.key != "bowl_plate" or demonstration is not None or zero_shot_collision_free else {}),
        "billing": "chatgpt_subscription", "simulator": asdict(sim_config), "api": asdict(config),
        "synthetic_observation": True, "api_calls": 0, "api_key_calls": 0,
        "credentials_checked": False, "simulator_started": False, "files_written": False,
        "sim_python_exists": Path(sim_config.python_executable or "").is_file(),
        "libero_checkout_exists": (Path(sim_config.libero_root or "") / "libero/libero/benchmark/__init__.py").is_file(),
        "payload": {key: payload[key] for key in ("model", "store", "stream", "instructions", "text", "reasoning")
                    if key in payload},
        "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "stopping_rule": _stopping_rule(stop_on_model_complete, zero_shot_collision_free), "evaluation_scored": True,
        "stop_on_model_complete": stop_on_model_complete,
        "model_completion_is_libero_success": False,
        "policy": {"method": "operation_field_hybrid" if operation_control else "speed_first_distance_hybrid" if speed_first else "zero_shot_direct" if zero_shot_direct else "demonstration_direct" if demo_direct else "hybrid_field_direct", "grid": client.grid.to_dict(),
            **_speed_first_metadata(speed_first, speed_first_v2, speed_first_v3),
            **_robot_geometry_metadata(robot_geometry_guidance),
            **_target_identity_metadata(target_description),
            **_attempt_memory_metadata(attempt_memory),
            **_pregrasp_metadata(pregrasp_feedback, contact_region_mode),
            **_operation_metadata(operation_control, operation_planner, contact_region_mode),
            **({"demo_direct_control": True} if demo_direct else {}),
            **({"zero_shot_direct_control": True} if zero_shot_direct else {}),
            **({"historical_zero_shot_direct_control": True} if historical_zero_shot_direct else {}),
            **({"zero_shot_collision_free": True} if zero_shot_collision_free else {}),
            "field_representation": getattr(client, "field_representation", "dense"),
            "controller_mode": getattr(client, "controller_mode", "hybrid"), "prompt_version": client.prompt_version,
            "planner": asdict(PlannerConfig(horizon=client.planner_horizon)), "controller_update_steps": client.chunk_size if direct_only else client.field_update_steps,
            "field_update_steps": client.field_update_steps, "direct_action_steps": client.chunk_size,
            "field_execution_steps": client.field_execution_steps,
            "reasoning_effort": client.reasoning_effort,
            "execution_mode": execution_mode, "feedback_history": feedback_history,
            "centered_placement_guidance": centered_placement_guidance,
            "grasp_settle_guidance": grasp_settle_guidance,
            "field_improvements": field_improvements,
            "distance_field_control": distance_field_control,
            "field_reuse_limit": 60 if operation_control else 30 if speed_first else 120 if distance_field_control else None,
            "field_refresh_policy": ("events_or_explicit_budget_progress_checked_at_30_max60" if operation_control else "events_or_30_actions" if speed_first else None if direct_only else "visual_progress_guarded_30_to_120" if distance_field_control else
                                     "events_or_30_actions" if field_improvements else "fixed_interval"),
            "local_path_strategy": ("straight_waypoint_same_tracker" if operation_control and operation_planner == "waypoint" else None if direct_only else "analytic_clear_field_else_lazy_26_neighbor_astar_1mm" if speed_first_v3 else "dijkstra_distance_field" if distance_field_control else
                                    "analytic_pure_goal_else_lattice" if field_improvements else "lattice"),
            "goal_tolerance_m": 0.005 if field_improvements else None,
            "goal_orientation_tolerance_rad": 0.05 if field_improvements else None,
            "tracking_lookahead_m": .03 if execution_mode == "tracking" else None,
            "tracking_max_command_m": .05 if execution_mode == "tracking" else None,
            "direct_action_chunks": client.chunk_size,
            "routing_source": ("model RGB/state/history prediction, not simulator contact"
                               if controller_mode == "hybrid" else "fixed Direct comparison arm"),
            "controllers": ["contact_direct"] if direct_only else ["coarse_field", "contact_direct"], "recent_image_observations": 2,
            **({"field_generation": False, "local_field_planning": False} if direct_only else {}),
            "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
            "object_state_feedback_to_policy": False}}


class ScriptedHybridTransport:
    """Three explicit synthetic envelopes, never a live model request."""

    def __init__(self):
        self.calls = 0

    def __call__(self, endpoint, payload, headers, timeout):
        index = self.calls
        self.calls += 1
        if index >= 3:
            raise AssertionError("Smoke must stop before a fourth scripted response")
        if index == 1:
            envelope = {"controller": "contact_direct", "target_region": "black_bowl", "near_target": True,
                "routing_reason": "Scripted smoke: exercise direct close commands and measured feedback.",
                "field": None, "direct": {"actions": [[0, 0, 0, 0, 0, 0, 1] for _ in range(HYBRID_CHUNK_SIZE)]}}
        else:
            grid = HYBRID_GRID
            field = {"stage": "scripted_coarse_approach" if index == 0 else "scripted_closed_gripper_hold",
                "orientation_target_xyzw": [0, 0, 0, 1], "gripper_command": "open" if index == 0 else "hold",
                "motion_mode": "move" if index == 0 else "hold",
                "values": (-grid.coordinates((0, 0, 0))[:, 0]).tolist(),
                "blocked_indices": [], "unknown_indices": []}
            envelope = {"controller": "coarse_field", "target_region": "black_bowl" if index == 0 else "plate",
                "near_target": False, "routing_reason": "Scripted smoke: exercise field planning and mode return.",
                "field": field, "direct": None}
        return {"id": f"scripted-hybrid-{index}", "status": "completed", "offline_mock": True,
            "model": "gpt-6-astra", "output": [{"id": f"scripted-hybrid-message-{index}",
                "type": "message", "role": "assistant", "status": "completed", "phase": "final_answer",
                "content": [{"type": "output_text", "text": json.dumps(envelope)}]}],
            "usage": {"input_tokens": 0, "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


class _SmokeClient(ChatGPTPlanHybridClient):
    def budget_snapshot(self):
        usage = super().budget_snapshot()
        return {**usage, "billing": "offline_mock", "api_calls": 0,
                "subscription_inference_calls": 0, "mock_response_calls": self._api_calls}

    def _write_log(self, path, data):
        if isinstance(data, dict):
            data = {**data, "billing": "offline_mock", "offline_smoke": True,
                    "transport_version": "scripted-hybrid-no-network-fixture"}
            if "payload" in data and "request_ordinal" in data:
                data.update(prompt_version=self.prompt_version, history_protocol=HISTORY_PROTOCOL,
                    image_history=self.image_history, history_stats_before_request=self.history_stats())
                self._history_request_settled = False
        AstraClient._write_log(self, path, data)


def execute_episode(output, *, config, sim_config, token_provider=None, offline_smoke=False,
                    sim_factory=None, recorder_factory=None, transport=None, field_representation="compact",
                    reasoning_effort="high", stop_on_model_complete=False,
                    controller_mode="hybrid", execution_mode="fixed", feedback_history="full",
                    centered_placement_guidance=False, grasp_settle_guidance=False, field_improvements=False,
                    distance_field_control=False, task_key="bowl_plate", demonstration_path=None, episode_index=0,
                    zero_shot_collision_free=False, historical_zero_shot_direct=False,
                    expected_initial_observation_sha256=None, speed_first=False, speed_first_v2=False, speed_first_v3=False,
                    robot_geometry_guidance=False, target_description=None, attempt_memory=False, pregrasp_feedback=False,
                    operation_control=False, operation_planner="field", contact_region_mode=None):
    episode_index = _validate_episode_index(episode_index)
    task_spec = resolve_distance_task(task_key)
    demo_direct = _demo_direct(controller_mode, task_key, distance_field_control, demonstration=demonstration_path is not None,
                               historical_zero_shot_direct=historical_zero_shot_direct)
    zero_shot_direct = (zero_shot_collision_free or historical_zero_shot_direct) and controller_mode == "direct"
    direct_only = demo_direct or zero_shot_direct
    client_options = _control_options(field_representation, reasoning_effort, stop_on_model_complete,
        offline_smoke=offline_smoke, controller_mode=controller_mode,
        execution_mode=execution_mode, feedback_history=feedback_history,
        centered_placement_guidance=centered_placement_guidance, grasp_settle_guidance=grasp_settle_guidance,
        field_improvements=field_improvements, distance_field_control=distance_field_control, task_key=task_key,
        demonstration=demonstration_path is not None, zero_shot_collision_free=zero_shot_collision_free,
        historical_zero_shot_direct=historical_zero_shot_direct, speed_first=speed_first, speed_first_v2=speed_first_v2, speed_first_v3=speed_first_v3,
        robot_geometry_guidance=robot_geometry_guidance, target_description=target_description, attempt_memory=attempt_memory, pregrasp_feedback=pregrasp_feedback,
        operation_control=operation_control, operation_planner=operation_planner, contact_region_mode=contact_region_mode)
    _validate_collision_configuration(sim_config, zero_shot_collision_free, speed_first=speed_first)
    _validate_task_configuration(config, sim_config, task_spec, speed_first_v3=speed_first_v3)
    override_source, override_bytes = _initial_state_override(sim_config)
    task_override_source, task_override_bytes = _task_bddl_override(sim_config)
    demonstration = _demonstration(demonstration_path, distance_field_control=distance_field_control,
                                   task_key=task_key, controller_mode=controller_mode,
                                   historical_zero_shot_direct=historical_zero_shot_direct)
    if demonstration is not None:
        client_options["demonstration"] = demonstration
    output = Path(output)
    if output.exists():
        raise FileExistsError("Fresh hybrid experiment requires a new output directory; no resume")
    if offline_smoke:
        if transport is not None or token_provider is not None:
            raise ValueError("Offline smoke uses only its scripted responses, without credentials")
        client = _SmokeClient(config, token_provider=lambda: "OFFLINE-OAUTH-FIXTURE",
                             transport=ScriptedHybridTransport(), log_dir=output / "requests")
    else:
        if not callable(token_provider):
            raise MissingCredentials("Live hybrid execution requires a ChatGPT subscription token provider")
        client = _client_type(field_representation, distance_field_control,
                             controller_mode=controller_mode, task_key=task_key,
                             demonstration=demonstration is not None, zero_shot_collision_free=zero_shot_collision_free,
                             historical_zero_shot_direct=historical_zero_shot_direct, speed_first=speed_first, speed_first_v2=speed_first_v2, speed_first_v3=speed_first_v3, operation_control=operation_control, contact_region_mode=contact_region_mode)(config, token_provider=token_provider, transport=transport,
                                        log_dir=output / "requests", **client_options)
    if speed_first:
        from gpt_field.speed_first_hybrid import SpeedFirstHybridPolicy
        if contact_region_mode is not None:
            from gpt_field.contact_region_policy import ContactRegionPolicy
            policy = ContactRegionPolicy(client, planner_mode=operation_planner,
                apply_contact_correction=contact_region_mode == "apply")
        elif operation_control:
            from gpt_field.speed_first_v4 import SpeedFirstV4Policy
            policy = SpeedFirstV4Policy(client, planner_mode=operation_planner)
        elif speed_first_v3:
            from gpt_field.speed_first_v3 import SpeedFirstV3Policy
            policy = SpeedFirstV3Policy(client)
        elif speed_first_v2:
            from gpt_field.speed_first_v2 import SpeedFirstV2Policy
            policy = SpeedFirstV2Policy(client)
        else:
            policy = SpeedFirstHybridPolicy(client)
    else:
        policy = (DistanceHybridPolicy if distance_field_control else HybridFieldDirectPolicy)(client)
    sim_factory = sim_factory or SimProcess
    if recorder_factory is None and speed_first_v3:
        from gpt_field.speed_first_v3_video import SpeedFirstV3VideoRecorder
        recorder_factory = SpeedFirstV3VideoRecorder
    recorder_factory = recorder_factory or (HistoricalZeroShotDirectVideoRecorder if historical_zero_shot_direct else ZeroShotDirectVideoRecorder if zero_shot_direct else DemoDirectVideoRecorder if demo_direct else
                                            DistanceHybridVideoRecorder if distance_field_control else HybridVideoRecorder)
    output.mkdir(parents=True, exist_ok=False)
    if override_source is not None:
        override_artifact = output / "initial_state_override.npy"
        override_artifact.write_bytes(override_bytes)
        sim_config = replace(sim_config, initial_state_override=str(override_artifact.resolve()))
    if task_override_source is not None:
        task_artifact = output / "task_override.bddl"
        task_artifact.write_bytes(task_override_bytes)
        sim_config = replace(sim_config, task_bddl_override=str(task_artifact.resolve()))
    demonstration_record = None
    if demonstration is not None:
        demonstration.write(output / "demonstration.json")
        demonstration_record = {"artifact": "demonstration.json", **demonstration.metadata()}
    mode = ("offline_mock_hybrid_smoke" if offline_smoke else "live_chatgpt_subscription_zero_shot_direct"
            if zero_shot_direct else "live_chatgpt_subscription_demo_direct"
            if demo_direct else "live_chatgpt_subscription_hybrid")
    identity = {"kind": "subscription_hybrid_full_task", "mode": mode,
        **_speed_first_metadata(speed_first, speed_first_v2, speed_first_v3),
        **_robot_geometry_metadata(robot_geometry_guidance),
        **_target_identity_metadata(target_description),
        **_attempt_memory_metadata(attempt_memory),
        **_pregrasp_metadata(pregrasp_feedback, contact_region_mode),
        **_operation_metadata(operation_control, operation_planner, contact_region_mode),
        **({"zero_shot_collision_free": True} if zero_shot_collision_free else {}),
        **_initial_state_metadata(override_source),
        **_task_bddl_metadata(task_override_source),
        **({"demonstration": demonstration_record} if demonstration_record is not None else {}),
        "task": task_spec.prompt, "task_id": task_spec.task_id, "episode_index": episode_index, "simulator": asdict(sim_config),
        **({"distance_task": task_spec.key, "task_spec": asdict(task_spec)} if task_spec.key != "bowl_plate" or demonstration is not None or zero_shot_collision_free else {}),
        "api": asdict(config), "billing": "offline_mock" if offline_smoke else "chatgpt_subscription",
        "software": source_manifest(), "runner": {"path": "scripts/evaluate_bowl_hybrid_subscription.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "policy": {"class": ("gpt_field.contact_region_policy.ContactRegionPolicy" if contact_region_mode is not None else "gpt_field.speed_first_v4.SpeedFirstV4Policy" if operation_control else "gpt_field.speed_first_v3.SpeedFirstV3Policy" if speed_first_v3 else "gpt_field.speed_first_v2.SpeedFirstV2Policy" if speed_first_v2 else "gpt_field.speed_first_hybrid.SpeedFirstHybridPolicy" if speed_first else
                             "gpt_field.distance_policy.DistanceHybridPolicy" if distance_field_control else
                             "gpt_field.hybrid_policy.HybridFieldDirectPolicy"),
            "method": "operation_field_hybrid" if operation_control else "speed_first_distance_hybrid" if speed_first else "zero_shot_direct" if zero_shot_direct else "demonstration_direct" if demo_direct else "hybrid_field_direct",
            **_speed_first_metadata(speed_first, speed_first_v2, speed_first_v3),
            **_robot_geometry_metadata(robot_geometry_guidance),
            **_target_identity_metadata(target_description),
            **_attempt_memory_metadata(attempt_memory),
            **_pregrasp_metadata(pregrasp_feedback, contact_region_mode),
            **_operation_metadata(operation_control, operation_planner, contact_region_mode),
            **({"demo_direct_control": True} if demo_direct else {}),
            **({"zero_shot_direct_control": True} if zero_shot_direct else {}),
            **({"historical_zero_shot_direct_control": True} if historical_zero_shot_direct else {}),
            **({"zero_shot_collision_free": True} if zero_shot_collision_free else {}),
            "field_representation": getattr(client, "field_representation", "dense"),
            "controller_mode": getattr(client, "controller_mode", "hybrid"), "prompt_version": client.prompt_version,
            "update_steps": client.chunk_size if direct_only else policy.update_steps, "field_update_steps": client.field_update_steps,
            "field_execution_steps": client.field_execution_steps,
            "direct_action_steps": client.chunk_size, "recent_image_observations": 2, "grid": policy.grid.to_dict(),
            "reasoning_effort": client.reasoning_effort,
            "execution_mode": execution_mode, "feedback_history": feedback_history,
            "centered_placement_guidance": centered_placement_guidance,
            "grasp_settle_guidance": grasp_settle_guidance,
            "field_improvements": field_improvements,
            "distance_field_control": distance_field_control,
            "field_reuse_limit": 60 if operation_control else 30 if speed_first else 120 if distance_field_control else None,
            "field_refresh_policy": ("events_or_explicit_budget_progress_checked_at_30_max60" if operation_control else "events_or_30_actions" if speed_first else None if direct_only else "visual_progress_guarded_30_to_120" if distance_field_control else
                                     "events_or_30_actions" if field_improvements else "fixed_interval"),
            "local_path_strategy": ("straight_waypoint_same_tracker" if operation_control and operation_planner == "waypoint" else None if direct_only else "analytic_clear_field_else_lazy_26_neighbor_astar_1mm" if speed_first_v3 else "dijkstra_distance_field" if distance_field_control else
                                    "analytic_pure_goal_else_lattice" if field_improvements else "lattice"),
            "goal_tolerance_m": 0.005 if field_improvements else None,
            "goal_orientation_tolerance_rad": 0.05 if field_improvements else None,
            "tracking_lookahead_m": .03 if execution_mode == "tracking" else None,
            "tracking_max_command_m": .05 if execution_mode == "tracking" else None,
            "planner": asdict(policy.planner.config), "adapter_max_rotation_rad": policy.adapter.max_rotation_rad,
            "initial_history_settings": client.history_stats(),
            "controllers": ["contact_direct"] if direct_only else ["coarse_field", "contact_direct"],
            "routing_source": ("model RGB/state/history prediction, not simulator contact"
                               if controller_mode == "hybrid" else "fixed Direct comparison arm"),
            "local_plans_per_field_action": None if field_improvements else 1 / client.field_execution_steps,
            "local_plans_per_direct_action": 0,
            "field_generation": controller_mode == "hybrid", "local_field_planning": controller_mode == "hybrid", "direct_action_chunks": client.chunk_size,
            "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
            "object_state_feedback_to_policy": False},
        "stopping_rule": ([f"{SMOKE_STEPS} scripted diagnostic actions", *STOPPING_RULE]
                          if offline_smoke else _stopping_rule(stop_on_model_complete, zero_shot_collision_free)),
        "stop_on_model_complete": stop_on_model_complete,
        "model_completion_is_libero_success": False,
        "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
        "offline_diagnostic_steps": SMOKE_STEPS if offline_smoke else None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "evaluation_scored": not offline_smoke, "prior_budgets_inherited": False}
    write_json(output / "manifest.json", {"identity_hash": digest(identity), "configuration": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(), "simulation_pauses_for_api": True})
    frozen = output / "frozen_source"
    frozen.mkdir()
    for filename in identity["software"]["source_sha256"]:
        shutil.copyfile(ROOT / "src/gpt_field" / filename, frozen / filename)
    shutil.copyfile(Path(__file__), frozen / Path(__file__).name)
    status, stage, error = "running", "starting", None
    steps = refreshes = fields = direct_chunks = plans = feedback_count = close_actions = open_actions = neutral_actions = 0
    coarse_actions = direct_actions = switches = 0
    pending_action, first_close_event, success_seen, path_length = None, None, False, 0.0
    controller = None
    recorder, sim, cleanup_errors = None, None, []
    started = time.monotonic()
    control_started = control_stopped = None
    model_completion = None
    collision = None
    timings = {key: 0.0 for key in ("decision_s", "model_pipeline_s", "field_expansion_s",
        "local_planning_s", "simulator_step_s", "video_render_s")}

    def record_decision_time(latency, details):
        timings["decision_s"] += latency
        for key, detail in (("model_pipeline_s", "api_latency_s"),
                            ("field_expansion_s", "field_expansion_latency_s"),
                            ("local_planning_s", "plan_latency_s")):
            timings[key] += float(details.get(detail) or 0.0)

    def state(new_stage):
        nonlocal stage
        stage = new_stage
        write_json(output / "run_state.json", {"status": status, "stage": stage, "mode": mode,
            **_speed_first_metadata(speed_first, speed_first_v2, speed_first_v3),
            **_robot_geometry_metadata(robot_geometry_guidance),
            **_target_identity_metadata(target_description),
            **_attempt_memory_metadata(attempt_memory),
            **_pregrasp_metadata(pregrasp_feedback, contact_region_mode),
            **_operation_metadata(operation_control, operation_planner, contact_region_mode),
            **({"pregrasp_diagnostics": policy.pregrasp_stats()} if pregrasp_feedback else {}),
            **({"operation_diagnostics": policy.operation_stats()} if operation_control else {}),
            **({"contact_region_diagnostics": policy.contact_region_stats()} if contact_region_mode else {}),
            "task_id": task_spec.task_id, "episode_index": episode_index,
            "step": steps, "controller": controller, "controller_switches": switches,
            "pending_action_index": pending_action, "budget": client.budget_snapshot(),
            **({"zero_shot_collision_free": True, "collision": collision} if zero_shot_collision_free else {}),
            **({"collision": collision} if speed_first else {}),
            "first_close_event": first_close_event, "model_completion": model_completion, "error": error})

    state("starting")
    with _old._base.termination_as_interrupt():
        try:
            with _old._simulator_without_credentials():
                sim = sim_factory(sim_config)
                sim.start()
            tasks = sim.list_tasks()
            selected = [task for task in tasks if task.get("task_id") == task_spec.task_id]
            if len(selected) != 1 or selected[0].get("prompt") != task_spec.prompt:
                raise ValueError("Official task does not match the selected task profile")
            write_json(output / "simulator.json", {"metadata": sim.metadata, "tasks": tasks})
            obs = sim.reset(task_spec.task_id, episode_index)
            _freeze_original_task_bddl(output, sim.status, task_override_source)
            if zero_shot_collision_free:
                collision = _collision_record(sim.status)
                if collision["detected"]:
                    raise ValueError("Custom scene initially violates the collision criterion; no policy trial executed")
            elif speed_first:
                collision = _collision_record(sim.status, speed_first=True)
            if obs.prompt != task_spec.prompt:
                raise ValueError("Reset observation has a different task")
            initial_observation = obs.to_dict()
            initial_sha256 = hashlib.sha256(json.dumps(initial_observation, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            write_json(output / "initial_observation.json", initial_observation)
            write_json(output / "initial_observation_check.json", {
                "sha256": initial_sha256, "expected_sha256": expected_initial_observation_sha256,
                "matched": (initial_sha256 == expected_initial_observation_sha256
                            if expected_initial_observation_sha256 else None)})
            if expected_initial_observation_sha256 and initial_sha256 != expected_initial_observation_sha256:
                raise ValueError("Reset observation differs from the frozen historical comparison; no inference performed")
            policy.reset()
            recorder = recorder_factory(output / "episode.mp4", grid=policy.grid,
                episode_id=(f"CHATGPT SUBSCRIPTION / {task_spec.suite}_t{task_spec.task_id:02d}_i{episode_index:02d} / {controller_mode.upper()} FULL TASK"
                            + (" / SPEED FIRST / COLLISIONS RECORDED" if speed_first else "")
                            + (" / ZERO SHOT / NO OBSTACLE COLLISION" if zero_shot_collision_free else "")
                            + (" / CUSTOM SCENE" if task_override_source is not None else
                               " / CUSTOM LAYOUT" if override_source is not None else "")), offline_smoke=offline_smoke)
            recorder.observe(obs, step=0)
            control_started = time.monotonic()
            with (output / "trace.jsonl").open("w") as log:
                append = lambda row: _old._base._append(log, row)
                append({"event": "reset", "task_id": task_spec.task_id, "episode_index": episode_index,
                        "observation": obs.to_dict(), "sim_status": dict(sim.status), "mode": mode})
                while True:
                    if zero_shot_collision_free:
                        collision = _collision_record(sim.status)
                        success_seen = success_seen or sim.status["libero_success"]
                        if collision["detected"]:
                            status = "collision_failure"
                            break
                    elif speed_first:
                        collision = _collision_record(sim.status, speed_first=True)
                        success_seen = success_seen or sim.status["libero_success"]
                    if sim.status.get("success"):
                        success_seen = True
                        status = "diagnostic_sim_terminated" if offline_smoke else "success"
                        break
                    if speed_first and (output / "operator_stop.request.json").is_file():
                        status = "user_stop"
                        error = {"type": "OperatorStopRequested", "stage": stage,
                                 "message": "Operator stop requested; stopped at a complete-feedback boundary"}
                        append({"event": "operator_stop", "step": steps, "feedback_count": feedback_count,
                                "pending_action_index": pending_action, "request_file": "operator_stop.request.json"})
                        break
                    if offline_smoke and steps == SMOKE_STEPS:
                        status = "diagnostic_complete"
                        break
                    if sim.status.get("terminated"):
                        raise RuntimeError("Simulator terminated without task success or an action horizon")
                    if obs.prompt != task_spec.prompt:
                        raise ValueError("Policy task changed during rollout")
                    state("deciding")
                    began = time.monotonic()
                    try:
                        decision = policy.act(obs)
                    except ModelTaskComplete as completion:
                        if not stop_on_model_complete:
                            raise RuntimeError("Model completion was not enabled for this experiment") from completion
                        control_stopped = time.monotonic()
                        latency = control_stopped - began
                        details = completion.details
                        record_decision_time(latency, details)
                        previous_controller, controller = controller, completion.prediction["controller"]
                        refreshes += 1
                        fields += int(controller == "coarse_field")
                        direct_chunks += int(controller == "contact_direct")
                        model_completion = {
                            "step": steps, "acknowledged_actions": steps, "task_complete": True,
                            "completion_reason": details["completion_reason"], "controller": controller,
                            "previous_controller": previous_controller,
                            "decision_latency_s": latency, "elapsed_s": control_stopped - started,
                            "control_elapsed_s": control_stopped - control_started,
                            "api_latency_s": details.get("api_latency_s"),
                            "api_request_key": details.get("api_request_key"),
                            "api_request_log": details.get("api_request_log"),
                            "reasoning_effort": client.reasoning_effort,
                            "libero_success_observed": success_seen,
                            "action_proposed": False, "action_executed": False,
                        }
                        append({"event": "model_completion", **model_completion,
                                "diagnostics": details, "prediction": completion.prediction, "mode": mode})
                        status = "model_declared_complete"
                        state("model_declared_complete")
                        recorder.model_completion(obs, step=steps,
                            completion_reason=details["completion_reason"],
                            decision_latency_s=latency, diagnostics=details)
                        print(json.dumps({"event": "model_completion", **model_completion}), flush=True)
                        break
                    latency = time.monotonic() - began
                    details = decision.diagnostics
                    record_decision_time(latency, details)
                    previous_controller, controller = controller, details["controller"]
                    if details.get("refreshed"):
                        refreshes += 1
                        fields += int(controller == "coarse_field")
                        direct_chunks += int(controller == "contact_direct")
                        append({"event": "routing_decision", "step": steps, "controller": controller,
                                "routing": details["routing"], "previous_controller": previous_controller,
                                "source": "model_prediction" if controller_mode == "hybrid" else "forced_direct", "mode": mode})
                        if previous_controller is not None and previous_controller != controller:
                            switches += 1
                            append({"event": "controller_switch", "step": steps, "from_controller": previous_controller,
                                    "to_controller": controller, "routing": details["routing"], "source": "model_prediction"})
                    plans += int(details.get("plan_refreshed", "plan" in details))
                    append({"event": "action", "step": steps, "state": list(obs.state),
                            "action": list(decision.action), "diagnostics": details,
                            "decision_latency_s": latency, "mode": mode})
                    video_began = time.monotonic()
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    timings["video_render_s"] += time.monotonic() - video_began
                    pending_action = steps
                    state("executing")
                    before = obs
                    collision_was_detected = bool(collision and collision.get("detected"))
                    sim_began = time.monotonic()
                    obs, success = sim.step(decision.action)
                    timings["simulator_step_s"] += time.monotonic() - sim_began
                    executed_step, steps = steps, steps + 1
                    pending_action = None
                    if zero_shot_collision_free or speed_first:
                        collision = _collision_record(sim.status, speed_first=speed_first)
                        if success != sim.status["success"]:
                            raise ValueError("Step result must match the configured simulator success criterion")
                    success_seen = success_seen or (sim.status["libero_success"] if zero_shot_collision_free or speed_first else success)
                    close_actions += int(decision.action[6] > 0)
                    open_actions += int(decision.action[6] < 0)
                    neutral_actions += int(decision.action[6] == 0)
                    coarse_actions += int(controller == "coarse_field")
                    direct_actions += int(controller == "contact_direct")
                    path_length += float(np.linalg.norm(np.asarray(obs.state[:3]) - np.asarray(before.state[:3])))
                    append({"event": "executed", "step": executed_step, "controller": controller,
                            "next_state": list(obs.state), "observation": obs.to_dict(), "action": list(decision.action),
                            "success": success, "sim_status": dict(sim.status), "mode": mode})
                    if zero_shot_collision_free and collision["detected"]:
                        append({"event": "forbidden_collision", "step": executed_step, "collision": collision,
                                "libero_success": sim.status["libero_success"], "task_success": False})
                    elif speed_first and collision["detected"] and not collision_was_detected:
                        append({"event": "recorded_collision", "step": executed_step, "collision": collision,
                                "collision_policy": "record_only", "libero_success": sim.status["libero_success"],
                                "affects_completion": False})
                    if first_close_event is None and decision.action[6] > 0:
                        first_close_event = {"action_index": executed_step, "action_ordinal": steps,
                            "controller": controller, "gripper_joint_delta": [obs.state[i] - before.state[i] for i in (6, 7)],
                            "full_closure_verified": False, "grasp_verified": False}
                    feedback = policy.observe_transition(before, decision, obs)
                    feedback_count += 1
                    append({"event": "feedback", "step": executed_step, "feedback": feedback})
                    if field_improvements and policy.pending_field_event is not None:
                        event_name = "distance_control_event" if distance_field_control else "field_control_event"
                        append({"event": event_name, "step": steps,
                                event_name: policy.pending_field_event})
                    if distance_field_control and not speed_first:
                        append({"event": "local_visual_feedback", "step": executed_step,
                                "feedback": policy.local_feedback,
                                "local_field_revision": policy.local_revision,
                                "local_recovery_active": policy._recovery_active})
                    elif speed_first_v3:
                        append({"event": "local_visual_feedback", "step": executed_step,
                                "feedback": policy.local_feedback,
                                "local_field_revision": policy.local_revision,
                                "local_goal_updates": policy.local_goal_updates})
                    if operation_control:
                        append({"event": "operation_feedback", "step": executed_step,
                                "diagnostics": policy.operation_stats()})
                    if contact_region_mode is not None:
                        append({"event": "contact_region_feedback", "step": executed_step,
                                "diagnostics": policy.contact_region_stats()})
                    video_began = time.monotonic()
                    recorder.executed(obs, step=executed_step, success=success, sim_status=dict(sim.status),
                        feedback=feedback, history_stats=client.history_stats())
                    timings["video_render_s"] += time.monotonic() - video_began
                    if zero_shot_collision_free and collision["detected"]:
                        status = "collision_failure"
                    elif success:
                        status = "diagnostic_sim_terminated" if offline_smoke else "success"
                    state("observed")
                    print(json.dumps({"event": "executed", "mode": mode, "steps": steps, "controller": controller,
                        "controller_switches": switches, "close_actions": close_actions, "success_observed": success,
                        "api_attempts": client.budget_snapshot()["api_calls"]}), flush=True)
                    if success or (zero_shot_collision_free and collision["detected"]):
                        break
        except KeyboardInterrupt:
            status, error = "user_stop", {"type": "KeyboardInterrupt", "stage": stage, "message": "User interrupted the hybrid episode"}
        except (APIError, MissingCredentials, BudgetExceeded, SchemaError) as exc:
            status, error = "subscription_error", _old._safe_error(exc, stage)
        except Exception as exc:
            status, error = "infrastructure_error", _old._safe_error(exc, stage)
        finally:
            if control_started is not None and control_stopped is None:
                control_stopped = time.monotonic()
            terminal_raw_libero_success = sim.status.get("libero_success") if sim is not None else None
            if sim is not None:
                try:
                    sim.close()
                except Exception as exc:
                    cleanup_errors.append(_old._safe_error(exc, "simulator_close"))
            if cleanup_errors:
                status = "infrastructure_error"
            if recorder is not None:
                try:
                    recorder.close(status=status, error=json.dumps(error or cleanup_errors) if error or cleanup_errors else None)
                except Exception as exc:
                    cleanup_errors.append(_old._safe_error(exc, "video_finalization"))
                    status = "infrastructure_error"
            completed = (not offline_smoke and (status == "success" or
                (status == "model_declared_complete" and success_seen)))
            scored_failure = zero_shot_collision_free and status == "collision_failure"
            if distance_field_control and (not speed_first or speed_first_v3):
                timings["local_visual_s"] = policy.local_visual_total_s
                timings["local_field_update_s"] = policy.local_update_total_s
            summary = {"status": status, "mode": mode, "task": task_spec.prompt,
                **_speed_first_metadata(speed_first, speed_first_v2, speed_first_v3),
                **_robot_geometry_metadata(robot_geometry_guidance),
                **_target_identity_metadata(target_description),
                **_attempt_memory_metadata(attempt_memory),
                **_pregrasp_metadata(pregrasp_feedback, contact_region_mode),
                **_operation_metadata(operation_control, operation_planner, contact_region_mode),
                **({"pregrasp_diagnostics": policy.pregrasp_stats()} if pregrasp_feedback else {}),
                **({"operation_diagnostics": policy.operation_stats()} if operation_control else {}),
                **({"contact_region_diagnostics": policy.contact_region_stats()} if contact_region_mode else {}),
                **({"local_gripper_operations": policy.gripper_operations,
                    "local_gripper_actions": policy.gripper_operation_actions,
                    "ordinary_direct_actions": direct_actions - policy.gripper_operation_actions,
                    "ordinary_direct_chunks": direct_chunks - policy.gripper_operations,
                    "contact_direct_count_includes_local_gripper": True}
                   if speed_first_v2 else {}),
                **({"local_goal_updates": policy.local_goal_updates,
                    "local_field_revision": policy.local_revision,
                    "final_local_visual_feedback": policy.local_feedback}
                   if speed_first_v3 else {}),
                **_initial_state_metadata(override_source),
                **_task_bddl_metadata(task_override_source),
                **({"demo_direct_control": True} if demo_direct else {}),
            **({"zero_shot_direct_control": True} if zero_shot_direct else {}),
            **({"historical_zero_shot_direct_control": True} if historical_zero_shot_direct else {}),
            **({"zero_shot_collision_free": True} if zero_shot_collision_free else {}),
                **({"demonstration": demonstration_record} if demonstration_record is not None else {}),
                "suite": task_spec.suite, "task_id": task_spec.task_id,
                **({"distance_task": task_spec.key} if task_spec.key != "bowl_plate" or demonstration is not None or zero_shot_collision_free else {}),
                "episode_index": episode_index, "seed": 7, "requested_episodes": 1, "steps": steps,
                "validated_hybrid_refreshes": refreshes, "validated_field_refreshes": fields,
                "validated_direct_chunks": direct_chunks, "local_plan_count": plans, "distance_field_control": distance_field_control,
                "coarse_field_actions": coarse_actions, "contact_direct_actions": direct_actions,
                "controller_switches": switches, "final_controller": controller,
                "measured_feedback_count": feedback_count, "gripper_close_actions": close_actions,
                "gripper_open_actions": open_actions, "gripper_neutral_actions": neutral_actions,
                "path_length_m": path_length, "pending_action_index": pending_action,
                "first_close_event": first_close_event,
                "diagnostic_complete": offline_smoke and status == "diagnostic_complete",
                "stop_on_model_complete": stop_on_model_complete,
                "model_declared_complete": status == "model_declared_complete",
                "model_completion": model_completion,
                "benchmark_completed": not offline_smoke and status in ("success", "model_declared_complete", "collision_failure"),
                "reasoning_effort": client.reasoning_effort,
                "controller_mode": controller_mode, "execution_mode": execution_mode,
                "feedback_history": feedback_history, "timing_components": timings,
                "centered_placement_guidance": centered_placement_guidance,
                "grasp_settle_guidance": grasp_settle_guidance,
                "field_improvements": field_improvements,
                "timing_components_note": "field_expansion_s is included in model_pipeline_s; model_pipeline_s and local_planning_s are included in decision_s. "
                    + ("local_visual_s includes both decision-time setup and post-action tracking; local_field_update_s measures post-action updates. These components overlap and must not all be added. " if distance_field_control and (not speed_first or speed_first_v3) else "")
                    + "Terminal video and cleanup are excluded from control_elapsed_s.",
                "libero_success_observed": success_seen,
                "libero_success": terminal_raw_libero_success
                    if zero_shot_collision_free or speed_first else True if completed else None,
                **({"task_success": completed, "collision": collision,
                    "completion_time_eligible": completed,
                    "failure_reason": "forbidden_collision" if scored_failure else None}
                   if zero_shot_collision_free or speed_first else {}),
                "evaluation_completed": completed or scored_failure,
                "success_rate": 1.0 if completed else 0.0 if scored_failure else None,
                "scored_episodes": int(completed or scored_failure), "incomplete_episodes": int(not (completed or scored_failure)),
                "api_calls": client.budget_snapshot()["api_calls"], "mock_response_calls": client._api_calls if offline_smoke else 0,
                "billing": "offline_mock" if offline_smoke else "chatgpt_subscription", "budget": client.budget_snapshot(),
                "api_key_calls": 0, "api_key_fallback": False, "max_policy_steps": None, "max_api_calls": None,
                "max_wall_time_s": None, "elapsed_s": time.monotonic() - started,
                "startup_to_initial_observation_s": (control_started - started if control_started is not None else None),
                "control_elapsed_s": (control_stopped - control_started if control_started is not None else None),
                **({"control_without_video_s": (control_stopped - control_started - timings["video_render_s"]
                        if control_started is not None else None)} if speed_first else {}),
                "video": "episode.mp4" if (output / "episode.mp4").is_file() else None,
                "trace": "trace.jsonl", "error": error, "cleanup_errors": cleanup_errors, "output": str(output),
                "interpretation": ("Scripted field/direct/field smoke with real simulator execution; no model performance score."
                    if offline_smoke else "One zero-shot speed-first distance Hybrid episode. Only raw LIBERO success completes the evaluation. Collisions are evaluator-only secondary metrics and do not stop or disqualify completion. Video-excluded elapsed time is a completion time only when completion_time_eligible is true. Other stops remain incomplete; no automatic episode cap."
                    if speed_first else "One zero-shot obstacle episode. Success requires the LIBERO placement goal with no forbidden contact at any monitored physics substep. A forbidden contact is a scored failure and stops the rollout. Raw LIBERO status remains separate; other stops are incomplete."
                    if zero_shot_collision_free else "One historical-protocol zero-shot Direct episode. Five unchanged model actions per request; no demonstration or local field planner. Only actual LIBERO success completes the evaluation; other stops remain incomplete."
                    if historical_zero_shot_direct else "One demonstration-conditioned Direct episode. Only actual LIBERO success completes and scores the evaluation; model statements and reference completion do not stop the task. Other stops remain incomplete and unscored."
                    if demo_direct else "One distance-field hybrid episode. Only actual LIBERO success completes and scores the evaluation; visual agreement and model statements do not stop the task. Other stops remain incomplete and unscored."
                    if distance_field_control else "One hybrid episode. A model completion declaration completes the requested timing run, but only actual LIBERO success completes and scores the evaluation; other stops remain incomplete and unscored.")}
            write_json(output / "runner_summary.json", summary)
            write_json(output / "summary.json", summary)
            if zero_shot_collision_free or speed_first:
                write_json(output / "collision.json", collision)
                write_json(output / "task_evaluation.json", {key: summary[key] for key in (
                    "status", "task_success", "libero_success", "collision", "failure_reason",
                    "evaluation_completed", "success_rate", "scored_episodes", "incomplete_episodes",
                    "completion_time_eligible")})
            with (output / "episodes.jsonl").open("w") as results:
                _old._base._append(results, summary)
            state("finished")
    print(json.dumps({"event": "finished", **summary}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--field-representation", choices=("compact", "dense"), default="compact")
    parser.add_argument("--reasoning-effort", choices=("medium", "high"))
    parser.add_argument("--stop-on-model-complete", action="store_true")
    parser.add_argument("--controller-mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument("--execution-mode", choices=("fixed", "tracking"))
    parser.add_argument("--feedback-history", choices=("full", "block_summary"))
    parser.add_argument("--historical-zero-shot-direct", action="store_true",
        help="Pure Direct5 with the previous distance method task guidance and no demonstration")
    parser.add_argument("--pregrasp-feedback", action="store_true",
        help="Opt-in two-view pregrasp local XYZ Field-goal feedback; speed-first-v3 only")
    parser.add_argument("--operation-control", action="store_true",
        help="Typed operation goals, persistent evidence, bounded local execution; implies v3 interfaces")
    parser.add_argument("--contact-region-feedback", choices=("apply", "shadow"), dest="contact_region_mode",
        help="Semantic-region contact feedback; apply bounded corrections or only log proposals; implies operation control")
    parser.add_argument("--operation-planner", choices=("field", "waypoint"), default="field",
        help="Operation controller path strategy; waypoint is the matched straight-path ablation")
    parser.add_argument("--attempt-memory", action="store_true",
        help="Add bounded accepted attempt/outcome memory to existing v3 requests; control remains unchanged")
    parser.add_argument("--target-description",
        help="Optional user-provided target appearance from initial RGB; speed-first-v3 only")
    parser.add_argument("--robot-geometry-guidance", action="store_true",
        help="Add known Panda robot grasp-frame guidance to speed-first-v3; all control settings stay unchanged")
    parser.add_argument("--speed-first-v3", action="store_true",
        help="Speed-first v3: native 512 RGB, local object correction and a lazy 1 mm field")
    parser.add_argument("--speed-first-v2", action="store_true",
        help="Speed-first v2: explicit local gripper settle, bounded history and Field recovery")
    parser.add_argument("--speed-first", action="store_true",
        help="Zero-shot medium distance Hybrid: Field30/Direct5, no visual-goal gate, collisions recorded without stopping")
    parser.add_argument("--expected-initial-observation-sha256",
        help="Abort before inference if the RGB/state reset observation differs from the frozen reference")
    parser.add_argument("--distance-field-control", action="store_true",
                        help="Enable v7 Dijkstra distance fields, local RGB feedback and placement recovery; implies field improvements")
    parser.add_argument("--zero-shot-collision-free", action="store_true",
                        help="No demo; shared obstacle-avoidance prompts and physics-substep collision failure scoring")
    parser.add_argument("--task", choices=DISTANCE_TASK_KEYS, default="bowl_plate",
                        help="Select a registered task; both support distance control or Direct with a demonstration")
    parser.add_argument("--episode-index", type=int, choices=range(50), default=0,
                        help="LIBERO initial-state index for this task (0..49); use the same index for paired arms")
    parser.add_argument("--initial-state-override", type=Path,
                        help="Optional validated .npy simulator state for a custom layout; frozen into the run before reset")
    parser.add_argument("--task-bddl-override", type=Path,
                        help="Custom task scene with unchanged language/goal/target assets; requires a matching initial-state override")
    parser.add_argument("--demonstration", type=Path,
                        help="One validated external LIBERO demonstration bundle; supports task-matched Hybrid or Direct")
    parser.add_argument("--field-improvements", action="store_true",
                        help="Enable visual goal checks, grasp-offset targets and variable analytic paths; defaults to medium/tracking/block_summary")
    parser.add_argument("--centered-placement-guidance", action="store_true",
                        help="Ask Hybrid or Direct to visually center the bowl; requires LIBERO-only completion")
    parser.add_argument("--grasp-settle-guidance", action="store_true",
                        help="Explain control timing and measured grasp settling; requires centered placement guidance")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--actions-per-block", type=int, default=30)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline-smoke", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--timing-comparison", action="store_true")
    args = parser.parse_args(argv)
    if args.contact_region_mode is not None:
        args.operation_control = True
    if args.operation_control:
        args.speed_first_v3 = args.attempt_memory = args.pregrasp_feedback = True
    args.speed_first_v2 = args.speed_first_v2 or args.speed_first_v3
    args.speed_first = args.speed_first or args.speed_first_v2
    if args.speed_first:
        if (args.zero_shot_collision_free or args.historical_zero_shot_direct or args.timing_comparison
                or args.offline_smoke or args.demonstration is not None or args.stop_on_model_complete
                or args.field_representation != "compact" or args.controller_mode != "hybrid"
                or args.reasoning_effort not in (None, "medium")
                or args.execution_mode not in (None, "tracking")
                or args.feedback_history not in (None, "block_summary")):
            parser.error("--speed-first requires zero-shot medium compact Hybrid with tracking and block summaries; incompatible protocols and model-completion stops are excluded")
        args.distance_field_control = args.field_improvements = True
        args.reasoning_effort = "medium"
        args.execution_mode = "tracking"
        args.feedback_history = "block_summary"
        args.centered_placement_guidance = args.grasp_settle_guidance = True
    if args.historical_zero_shot_direct and (args.controller_mode != "direct" or args.demonstration is not None
            or args.distance_field_control or args.field_improvements or args.zero_shot_collision_free
            or args.offline_smoke or args.timing_comparison or args.stop_on_model_complete):
        parser.error("--historical-zero-shot-direct requires pure Direct without demonstrations or alternative protocols")
    if args.zero_shot_collision_free and (args.demonstration is not None or args.task != "bowl_plate"
            or args.task_bddl_override is None or args.initial_state_override is None
            or args.offline_smoke or args.timing_comparison or args.stop_on_model_complete
            or (args.controller_mode == "hybrid" and not args.distance_field_control)):
        parser.error("--zero-shot-collision-free requires the bowl custom scene/state, no demonstration, no model stop, and distance Hybrid or Direct")
    if args.timing_comparison and args.episode_index != 0:
        parser.error("Legacy --timing-comparison supports only --episode-index 0")
    if args.task_bddl_override is not None and args.initial_state_override is None:
        parser.error("--task-bddl-override requires --initial-state-override for the custom scene")
    if args.timing_comparison and args.task_bddl_override is not None:
        parser.error("Legacy --timing-comparison does not support custom task scenes")
    if args.timing_comparison and args.initial_state_override is not None:
        parser.error("Legacy --timing-comparison does not support --initial-state-override")
    demo_direct = _demo_direct(args.controller_mode, args.task, args.distance_field_control,
                               demonstration=args.demonstration is not None,
                               historical_zero_shot_direct=args.historical_zero_shot_direct)
    zero_shot_direct = (args.zero_shot_collision_free or args.historical_zero_shot_direct) and args.controller_mode == "direct"
    if args.demonstration is not None and not (args.distance_field_control or demo_direct):
        parser.error("--demonstration requires task-matched distance control or Direct")
    if demo_direct and args.demonstration is None:
        parser.error("Drawer Direct requires --demonstration for the matched comparison")
    args.field_improvements = args.field_improvements or args.distance_field_control
    args.reasoning_effort = args.reasoning_effort or ("medium" if args.field_improvements or demo_direct or zero_shot_direct else "high")
    args.execution_mode = args.execution_mode or ("tracking" if args.field_improvements else "fixed")
    args.feedback_history = args.feedback_history or ("block_summary" if args.field_improvements or demo_direct or zero_shot_direct else "full")
    if args.field_improvements or demo_direct or zero_shot_direct:
        args.centered_placement_guidance = args.grasp_settle_guidance = True
    if args.timing_comparison and args.field_representation != "compact":
        parser.error("--timing-comparison requires --field-representation compact")
    if args.timing_comparison and (args.reasoning_effort != "high" or args.stop_on_model_complete):
        parser.error("--timing-comparison supports high effort and no model-completion stop")
    if args.timing_comparison and (args.controller_mode != "hybrid" or args.execution_mode != "fixed"
                                   or args.feedback_history != "full" or args.centered_placement_guidance
                                   or args.grasp_settle_guidance or args.field_improvements):
        parser.error("Legacy equal-action timing comparison requires its fixed/full defaults")
    try:
        _control_options(args.field_representation, args.reasoning_effort, args.stop_on_model_complete,
            offline_smoke=args.offline_smoke, controller_mode=args.controller_mode,
            execution_mode=args.execution_mode, feedback_history=args.feedback_history,
            centered_placement_guidance=args.centered_placement_guidance,
            grasp_settle_guidance=args.grasp_settle_guidance, field_improvements=args.field_improvements,
            distance_field_control=args.distance_field_control, task_key=args.task,
            demonstration=args.demonstration is not None, zero_shot_collision_free=args.zero_shot_collision_free,
            historical_zero_shot_direct=args.historical_zero_shot_direct, speed_first=args.speed_first, speed_first_v2=args.speed_first_v2, speed_first_v3=args.speed_first_v3,
            robot_geometry_guidance=args.robot_geometry_guidance, target_description=args.target_description, attempt_memory=args.attempt_memory, pregrasp_feedback=args.pregrasp_feedback,
            operation_control=args.operation_control, operation_planner=args.operation_planner, contact_region_mode=args.contact_region_mode)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.preflight and args.output is None:
        parser.error("--output is required for execution")
    config = configuration()
    task_spec = resolve_distance_task(args.task)
    sim_config = LiberoConfig(suite=task_spec.suite, seed=7, max_steps=None,
        libero_root=str(Path(args.libero_root).expanduser().resolve()),
        python_executable=os.path.abspath(os.path.expanduser(args.sim_python)),
        initial_state_override=(str(args.initial_state_override.expanduser().resolve())
                                if args.initial_state_override is not None else None),
        task_bddl_override=(str(args.task_bddl_override.expanduser().resolve())
                            if args.task_bddl_override is not None else None),
        include_arm_joints=False,
        **({"render_size": 512, "image_size": 512} if args.speed_first_v3 else {}),
        collision_check_mode="task_manipulation_v1" if args.speed_first else "bowl_obstacles_v1" if args.zero_shot_collision_free else "none",
        **({"stop_on_collision": False} if args.speed_first else {}))
    if args.initial_state_override is not None or args.task_bddl_override is not None:
        try:
            _initial_state_override(sim_config)
            _task_bddl_override(sim_config)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    if args.preflight:
        print(json.dumps(preflight(config, sim_config, args.output, field_representation=args.field_representation,
            reasoning_effort=args.reasoning_effort, stop_on_model_complete=args.stop_on_model_complete,
            controller_mode=args.controller_mode, execution_mode=args.execution_mode,
            feedback_history=args.feedback_history,
            centered_placement_guidance=args.centered_placement_guidance,
            grasp_settle_guidance=args.grasp_settle_guidance, field_improvements=args.field_improvements,
            distance_field_control=args.distance_field_control, task_key=args.task,
            demonstration_path=args.demonstration, episode_index=args.episode_index,
            zero_shot_collision_free=args.zero_shot_collision_free,
            historical_zero_shot_direct=args.historical_zero_shot_direct, speed_first=args.speed_first, speed_first_v2=args.speed_first_v2, speed_first_v3=args.speed_first_v3,
            robot_geometry_guidance=args.robot_geometry_guidance, target_description=args.target_description, attempt_memory=args.attempt_memory, pregrasp_feedback=args.pregrasp_feedback,
            operation_control=args.operation_control, operation_planner=args.operation_planner, contact_region_mode=args.contact_region_mode), indent=2), flush=True)
        return 0
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    token_provider = None
    if not args.offline_smoke:
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
    if args.timing_comparison:
        spec = importlib.util.spec_from_file_location("_compact_timing_comparison", ROOT / "scripts/benchmark_compact_subscription.py")
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
        result = benchmark.run_benchmark(args.output, config=config, sim_config=sim_config,
            token_provider=token_provider, blocks=args.blocks, actions_per_block=args.actions_per_block)
        print(json.dumps(result), flush=True)
        return 0 if result["comparison"]["complete"] else 3
    result = execute_episode(args.output, config=config, sim_config=sim_config,
        token_provider=token_provider, offline_smoke=args.offline_smoke, field_representation=args.field_representation,
        reasoning_effort=args.reasoning_effort, stop_on_model_complete=args.stop_on_model_complete,
        controller_mode=args.controller_mode, execution_mode=args.execution_mode, feedback_history=args.feedback_history,
        centered_placement_guidance=args.centered_placement_guidance,
        grasp_settle_guidance=args.grasp_settle_guidance, field_improvements=args.field_improvements,
        distance_field_control=args.distance_field_control, task_key=args.task,
        demonstration_path=args.demonstration, episode_index=args.episode_index,
        zero_shot_collision_free=args.zero_shot_collision_free,
        historical_zero_shot_direct=args.historical_zero_shot_direct,
        expected_initial_observation_sha256=args.expected_initial_observation_sha256, speed_first=args.speed_first, speed_first_v2=args.speed_first_v2, speed_first_v3=args.speed_first_v3,
        robot_geometry_guidance=args.robot_geometry_guidance, target_description=args.target_description, attempt_memory=args.attempt_memory, pregrasp_feedback=args.pregrasp_feedback,
            operation_control=args.operation_control, operation_planner=args.operation_planner, contact_region_mode=args.contact_region_mode)
    return 0 if result["evaluation_completed"] or result["diagnostic_complete"] or result["model_declared_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
