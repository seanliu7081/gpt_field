#!/usr/bin/env python3
"""Prepare or request the single planning-only draft; never start a simulator.

Requires the completed, measured runtime bundle. Only the public native task
sentence and three clean/three grid images reach the frozen planner. Calibration,
robot state, diagnostic history, scoring, action budgets and evaluator truth are
not included. The original response and semantic validation are preserved. No
automatic plan edits or another planning attempt are performed by this script.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.approval import file_sha256
from robodojo_uc71.native_worker_client import write_private_record


def public_instruction(native_root):
    source = Path(native_root) / "task/RoboDojo/tasks/make_toast.py"
    tree = ast.parse(source.read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "gen_instruction"]
    if len(methods) != 1 or len(methods[0].body) != 1 or not isinstance(methods[0].body[0], ast.Return):
        raise ValueError("literal_native_public_instruction_required")
    instruction = ast.literal_eval(methods[0].body[0].value)
    if (not isinstance(instruction, list) or len(instruction) != 1 or instruction[0] !=
            "Pick up two slices of bread, place them into the toaster, and press the lever down."):
        raise ValueError("native_public_instruction_changed")
    return instruction[0], {"path": str(source), "sha256": file_sha256(source), "method": "gen_instruction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-bundle", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request", action="store_true")
    args = parser.parse_args()
    from robodojo_uc71.native_runtime_bundle import VerifiedRuntimeBundle
    from robodojo_uc71.client import NativePlanClient
    from robodojo_uc71.planning import plan_validation
    from gpt_field.astra import AstraConfig
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    import numpy as np
    from PIL import Image
    bundle = VerifiedRuntimeBundle.load(args.runtime_bundle, args.native_root)
    bundle.verify()
    instruction, source = public_instruction(args.native_root)
    images, bindings = {}, bundle.reset_reference["image_files"]
    for camera, entry in bindings.items():
        path = Path(entry["path"])
        if file_sha256(path) != entry["sha256"]: raise ValueError("reviewed_preview_changed")
        with Image.open(path) as picture:
            if picture.mode != "RGB" or picture.size != (640,480): raise ValueError("native_rgb_required")
            images[camera] = np.asarray(picture).copy()
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    # Construction is offline. The provider is invoked only by request_plan.
    credentials = SubscriptionCredentials('/workspace/gpt_field/.private/chatgpt_plan')
    client = NativePlanClient(AstraConfig(model="gpt-6-astra", timeout_s=180.), task_prompt=instruction,
        token_provider=credentials.access_token, log_dir=output / "subscription_logs",
        reasoning_effort="high", request_deadline_s=180)
    request = client.planning_request(images)
    payload = client.build_payload(request)
    write_private_record(output / "planning_payload.json", payload)
    preparation = {"schema": "robodojo-planning-only-preparation-1", "task_prompt": instruction,
        "instruction_source": source, "runtime_bundle": {"path": str(args.runtime_bundle.resolve()),
        "sha256": file_sha256(args.runtime_bundle)}, "images": bindings,
        "model": "gpt-6-astra", "reasoning_effort": "high", "request_deadline_s": 180,
        "planning_images": 6, "execution_numeric_channels": False, "evaluator_input": False,
        "simulator_starts": 0, "benchmark_actions": 0, "policy_model_calls": 0,
        "full_test_authorized": False, "automatic_plan_edits": False}
    write_private_record(output / "preparation.json", preparation)
    if not args.request:
        print(json.dumps({"prepared": True, "model_calls": 0, "full_test_started": False}))
        return 0
    result = {"planning_only": True, "full_test_started": False, "benchmark_actions": 0,
              "policy_model_calls": 0, "automatic_plan_edits": False}
    try:
        draft = client.request_plan(images)
        write_private_record(output / "original_draft.json", draft)
        validation = plan_validation(draft)
        write_private_record(output / "validation.json", validation)
        result["draft_received"] = True
        result["semantic_validation_passed"] = validation["valid"]
    except BaseException as error:
        result.update(draft_received=False, error_type=type(error).__name__)
    finally:
        if isinstance(client.last_response, dict):
            write_private_record(output / "original_response.json", client.last_response)
        result["usage"] = client.budget_snapshot()
        write_private_record(output / "result.json", result)
    print(json.dumps(result))
    return 0 if result.get("draft_received") else 1


if __name__ == "__main__":
    raise SystemExit(main())
