#!/usr/bin/env python3
"""Single-RGB Astra implicit-field experiment, subscription only.

Preparation is credential/network free. Live execution consumes one frozen
prepared directory and creates a fresh output directory. No simulator, learned
depth model, evaluator reference, previous response, or API-key path is used.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(HERE))

from gpt_field.astra import AstraClient, ENDPOINT, SchemaError, _strict_loads
from gpt_field.chatgpt_auth import SubscriptionCredentials
from gpt_field.chatgpt_plan import _access_token, _failure_metadata
from gpt_field.ordered_transport import ORDERED_TRANSPORT_VERSION, ordered_dumps, ordered_sse_transport

MODEL = "gpt-6-astra"
EFFORT = "medium"
PROTOCOL = "astra-single-rgb-csg-field-v1"
EXPECTED_NAMES = ("bowl002_initial", "drawer003_closed_step60", "drawer003_open_step552")
DEFAULT_INPUTS = HERE.parent / "rgb_scene/inputs_001/rgb_inputs.json"

INSTRUCTIONS = """You are estimating a three-dimensional scene implicit field from ONE RGB image.
Return only the requested structured JSON. Do not output code or a mesh, and do
not use tools, external depth models, simulator assets, hidden camera metadata,
previous frames, or memorized benchmark scene coordinates. Every geometric
parameter must be your visual estimate from this image and ordinary size priors.

The purpose is geometry reconstruction, not robot actions or a waypoint potential.
Represent the visible scene, including large background/support surfaces, main
objects and visible robot parts if present. Do not spend the whole primitive
budget on a robot at the expense of the surrounding scene. No known
object list, dimensions, metric scale, camera pose, depth, or calibration is
provided. Infer metric scale and focal length yourself and acknowledge that one
image leaves these ambiguous. Do not claim measured or calibrated accuracy.

Coordinates are CAMERA coordinates: origin at optical center, X to image right,
Y down, Z forward. Units are metres. Use a pinhole estimate with fx=fy equal to
estimated_focal_px, image-center principal point and zero skew. The supplied RGB
is upright; do not flip it. Place visible geometry at positive camera Z.

Your output is a compact constructive-solid-geometry (CSG) scalar field. It uses
analytic primitive signed distances, negative inside solids and positive outside;
union=min, intersection=max, difference(A,B)=max(A,-B). This CSG field has a useful
zero set but is not necessarily an exact Euclidean SDF near Boolean seams.
The decoder will sample ONLY this field, save samples, and extract its zero level.

Nodes are indexed by their position in the nodes array, starting at zero. A
compound node's left/right must refer to earlier nodes. root must be the last
node. All nodes must be reachable from root. Use at most 64 nodes, including at
most 32 primitive nodes and 32 Boolean nodes. Aim to use this budget effectively
for the main visible surfaces; approximate unresolvable detail and report it.

Each node has exactly name, op, center_m, rotation_xyzw, size_m, left, right.
Primitives have null left/right. Their center_m is in camera metres. Their
rotation_xyzw is a UNIT quaternion rotating local vectors into camera coordinates.
A sphere uses size_m=[radius,0,0] and identity quaternion [0,0,0,1]. A box uses
size_m=[half_extent_x,half_extent_y,half_extent_z]. A cylinder has its axis along
local Z and uses size_m=[radius,half_height,0]. All used dimensions are positive.
For union/intersection/difference, center_m, rotation_xyzw and size_m are null.
Avoid zero-thickness primitives. Finite boxes can represent large planar surfaces.

For example, if a visibly hollow container appears, preserve its cavity with
differences/intersections or a union of wall/base solids. If furniture has a
visible opening, preserve that empty space instead of filling the whole object.
These are generic representation examples, not a supplied scene inventory.
Use only image-supported orientation and opening state; hidden backs, undersides,
wall thicknesses and occluded surfaces are hypotheses. Describe uncertain scale,
focal length, omitted detail, approximation and hidden geometry in uncertainty_notes.
Do not change the representation or invent precision because evaluation will
compare against withheld ground truth. Ground truth is not available to you.
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _field_module():
    return importlib.import_module("field_csg")


def read_inputs(path: Path) -> list[dict]:
    path = path.resolve()
    data = _strict_loads(path.read_text())
    if not isinstance(data, dict) or set(data) != {"schema_version", "inputs"} or data["schema_version"] != 1:
        raise ValueError("Expected restricted RGB-only input manifest")
    if not isinstance(data["inputs"], list) or len(data["inputs"]) != 3:
        raise ValueError("Exactly three preselected RGB images are required")
    entries, seen = [], set()
    for item in data["inputs"]:
        if not isinstance(item, dict) or set(item) != {"name", "rgb", "sha256"}:
            raise ValueError("Input entries require only name, rgb and sha256")
        name = item["name"]
        if name not in EXPECTED_NAMES or name in seen:
            raise ValueError("Unexpected or duplicate snapshot name")
        seen.add(name)
        rgb = (path.parent / item["rgb"]).resolve()
        if not isinstance(item["sha256"], str) or not re.fullmatch("[a-f0-9]{64}", item["sha256"]):
            raise ValueError("Invalid RGB hash")
        if sha256(rgb) != item["sha256"]:
            raise ValueError("RGB input hash mismatch")
        with Image.open(rgb) as image:
            image.load()
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError("Inputs must be original upright 512x512 RGB images")
        entries.append({"name": name, "original": str(rgb), "file": f"inputs/{name}.png",
                        "sha256": item["sha256"], "width": 512, "height": 512})
    return sorted(entries, key=lambda x: EXPECTED_NAMES.index(x["name"]))


def build_payload(rgb_bytes: bytes, schema: dict) -> dict:
    return {
        "model": MODEL, "service_tier": "default", "store": False,
        "instructions": INSTRUCTIONS,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Estimate the scene field from this single 512x512 RGB image. No metric scale or camera calibration is supplied."},
            {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64," + base64.b64encode(rgb_bytes).decode("ascii")},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "astra_scene_field_v1", "strict": True, "schema": copy.deepcopy(schema)}},
        "stream": True, "reasoning": {"effort": EFFORT}, "include": ["reasoning.encrypted_content"],
    }


def _runtime_sources() -> list[Path]:
    _field_module()
    files = {Path(__file__).resolve(), HERE / "field_csg.py"}
    for name, module in tuple(sys.modules.items()):
        value = getattr(module, "__file__", None)
        if value and (name == "gpt_field" or name.startswith("gpt_field.")):
            path = Path(value).resolve()
            if path.suffix == ".py":
                files.add(path)
    return sorted(files)


def _file_records(directory: Path, exclude=("manifest.json",)) -> list[dict]:
    return [{"file": str(path.relative_to(directory)), "sha256": sha256(path)}
            for path in sorted(directory.rglob("*")) if path.is_file() and str(path.relative_to(directory)) not in exclude]


def prepare(inputs_path: Path, output: Path) -> dict:
    output, inputs_path = output.resolve(), inputs_path.resolve()
    if output.exists():
        raise FileExistsError("Refusing an existing preparation output directory")
    inputs = read_inputs(inputs_path)
    schema = _field_module().field_schema()
    sources = _runtime_sources()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "protocol": PROTOCOL, "status": "preparing",
                "model": MODEL, "reasoning_effort": EFFORT, "endpoint": ENDPOINT,
                "billing": "chatgpt_subscription", "transport_version": ORDERED_TRANSPORT_VERSION,
                "max_inference_calls": 3, "automatic_retries": 0, "history": "none; each image independent",
                "input_manifest": {"path": str(inputs_path), "sha256": sha256(inputs_path)},
                "inputs": inputs, "sources": [], "requests": [], "api_key_calls": 0,
                "subscription_inference_calls": 0, "credentials_accessed": False}
    write_json(output / "schema.json", schema)
    (output / "instructions.txt").write_text(INSTRUCTIONS)
    shutil.copyfile(inputs_path, output / "input_manifest.json")
    for source in sources:
        rel = "source/" + str(source.relative_to(PROJECT))
        destination = output / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest, copied = sha256(source), sha256(destination)
        if digest != copied:
            raise RuntimeError("Runtime source changed while freezing")
        manifest["sources"].append({"path": str(source), "sha256": digest, "copy": rel, "copy_sha256": copied})
    for entry in inputs:
        destination = output / entry["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["original"], destination)
        payload = build_payload(destination.read_bytes(), schema)
        request_file = f"requests/{entry['name']}.json"
        wire_file = f"requests/{entry['name']}.wire.json"
        write_json(output / request_file, payload)
        (output / wire_file).write_bytes(ordered_dumps(payload).encode("utf-8"))
        manifest["requests"].append({"name": entry["name"], "file": request_file,
                                     "sha256": sha256(output / request_file), "wire_file": wire_file,
                                     "wire_sha256": sha256(output / wire_file)})
    manifest["sealed_files"] = _file_records(output)
    manifest.update(status="prepared", prompt_sha256=sha256(output / "instructions.txt"), schema_sha256=sha256(output / "schema.json"))
    write_json(output / "manifest.json", manifest)
    return manifest


def _verified_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes experiment directory")
    return path


def _integrity(prepared: Path, plan: dict, output: Path | None = None) -> dict:
    def rows(items):
        return [{"path": str(path), "unchanged": path.is_file() and sha256(path) == digest} for path, digest in items]
    result = {
        "sources": rows((Path(x["path"]), x["sha256"]) for x in plan["sources"]),
        "inputs": rows([(Path(plan["input_manifest"]["path"]), plan["input_manifest"]["sha256"])] + [(Path(x["original"]), x["sha256"]) for x in plan["inputs"]]),
        "prepared": rows((_verified_file(prepared, x["file"]), x["sha256"]) for x in plan["sealed_files"]),
        "settings": plan["model"] == MODEL and plan["reasoning_effort"] == EFFORT and plan["endpoint"] == ENDPOINT and plan["max_inference_calls"] == 3 and plan["automatic_retries"] == 0,
    }
    if output is not None:
        result["artifacts"] = rows((_verified_file(output, x["file"]), x["sha256"]) for x in plan["sealed_files"])
    return result


def _all_integrity(value: dict) -> bool:
    return all(bool(items) and all(x["unchanged"] for x in items) if isinstance(items, list) else items is True for items in value.values())


def _redact(value, token: str):
    if isinstance(value, str):
        return value.replace(token, "[REDACTED]") if token else value
    if isinstance(value, dict):
        return {_redact(k, token): _redact(v, token) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v, token) for v in value]
    return value


def run_prepared(prepared: Path, output: Path, *, timeout_s: float = 300., token_provider=None, transport=None) -> dict:
    prepared, output = prepared.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Refusing an existing live output directory")
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
        raise ValueError("Request timeout must be positive and at most 300 seconds")
    plan = _strict_loads((prepared / "manifest.json").read_text())
    prepared_manifest_sha256 = sha256(prepared / "manifest.json")
    if plan.get("status") != "prepared" or plan.get("protocol") != PROTOCOL:
        raise ValueError("A complete frozen preparation is required")
    if [x["name"] for x in plan["inputs"]] != list(EXPECTED_NAMES) or [x["name"] for x in plan["requests"]] != list(EXPECTED_NAMES):
        raise ValueError("Preparation must contain exactly the three preselected images")
    if not _all_integrity(_integrity(prepared, plan)):
        raise ValueError("Prepared sources, inputs or artifacts changed")
    # All schema and payload checks happen before even constructing credentials.
    field_module = _field_module()
    schema = field_module.field_schema()
    payloads = []
    for image, request in zip(plan["inputs"], plan["requests"]):
        expected = build_payload((prepared / image["file"]).read_bytes(), schema)
        payload = _strict_loads((prepared / request["file"]).read_text())
        if payload != expected or ordered_dumps(payload).encode("utf-8") != (prepared / request["wire_file"]).read_bytes():
            raise ValueError("Prepared request differs from exact RGB-only protocol")
        payloads.append(payload)
    shutil.copytree(prepared, output)
    (output / "manifest.json").rename(output / "preparation_manifest.json")
    mock = transport is not None
    send = transport or ordered_sse_transport
    manifest = {**plan, "status": "running", "prepared_directory": str(prepared),
                "prepared_manifest_sha256": prepared_manifest_sha256,
                "billing": "offline_mock" if mock else "chatgpt_subscription", "offline_mock": mock,
                "timeout_s": timeout_s, "started_unix": time.time(), "predictions": [],
                "subscription_inference_calls": 0, "inference_attempts": 0, "credentials_accessed": False,
                "integrity": _integrity(prepared, plan, output)}
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        provider = token_provider if token_provider is not None else SubscriptionCredentials().access_token
        for entry, payload in zip(plan["inputs"], payloads):
            if manifest["inference_attempts"] >= 3:
                raise RuntimeError("Experiment call limit reached")
            if not _all_integrity(_integrity(prepared, plan, output)) or sha256(prepared / "manifest.json") != prepared_manifest_sha256:
                raise RuntimeError("Frozen experiment files changed before request")
            token = ""
            request_started = time.monotonic()
            record = {"name": entry["name"], "status": "requesting", "attempt": 1,
                      "rgb": {"file": entry["file"], "sha256": entry["sha256"]}}
            manifest["predictions"].append(record)
            try:
                token = _access_token(provider)
                manifest["credentials_accessed"] = not mock
                manifest["inference_attempts"] += 1
                manifest["subscription_inference_calls"] += int(not mock)
                write_json(output / "manifest.json", manifest)
                response = send(ENDPOINT, payload, {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "text/event-stream"}, timeout_s)
                response_path = f"responses/{entry['name']}.json"
                write_json(output / response_path, _redact(response, token))
                record["raw_response"] = {"file": response_path, "sha256": sha256(output / response_path)}
                if not isinstance(response, dict) or response.get("model") != MODEL:
                    raise SchemaError("Response model does not match required model")
                reported = response.get("reasoning")
                effort = reported.get("effort") if isinstance(reported, dict) else None
                if effort is not None and effort != EFFORT:
                    raise SchemaError("Response effort does not match required effort")
                raw_field = AstraClient._extract(response)
                raw_field_path = f"raw_fields/{entry['name']}.json"
                write_json(output / raw_field_path, _redact(raw_field, token))
                record["raw_field"] = {"file": raw_field_path, "sha256": sha256(output / raw_field_path)}
                field = field_module.validate_field(raw_field)
                field_path = f"fields/{entry['name']}.json"
                write_json(output / field_path, _redact(field, token))
                record.update(status="valid", field={"file": field_path, "sha256": sha256(output / field_path)},
                              response_model=response["model"], response_reasoning_effort=effort,
                              usage=_redact(response.get("usage"), token))
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, **_failure_metadata(error))
                raise
            finally:
                record["elapsed_s"] = time.monotonic() - request_started
                token = ""
                write_json(output / "manifest.json", manifest)
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error_type=type(error).__name__, **_failure_metadata(error))
    finally:
        manifest["elapsed_s"] = time.monotonic() - started
        manifest["integrity"] = _integrity(prepared, plan, output)
        manifest["integrity"]["prepared"].append({"path": str(prepared / "manifest.json"), "unchanged": sha256(prepared / "manifest.json") == prepared_manifest_sha256})
        for record in manifest["predictions"]:
            for key in ("field", "raw_field", "raw_response"):
                if key in record:
                    item = record[key]
                    path = output / item["file"]
                    manifest["integrity"]["artifacts"].append({"path": str(path), "unchanged": path.is_file() and sha256(path) == item["sha256"]})
        if not _all_integrity(manifest["integrity"]):
            manifest.update(status="failed", error_type="IntegrityError")
        manifest["artifacts"] = _file_records(output)
        write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze exact inputs, sources, prompt, schema and requests without credentials/network")
    prep.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    prep.add_argument("--output", type=Path, required=True)
    live = sub.add_parser("run", help="Execute exactly the frozen three stateless requests; no retries")
    live.add_argument("--prepared", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--timeout-s", type=float, default=300.)
    args = parser.parse_args()
    try:
        result = prepare(args.inputs, args.output) if args.command == "prepare" else run_prepared(args.prepared, args.output, timeout_s=args.timeout_s)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, **_failure_metadata(error)}))
        return 1
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "inference_attempts": result.get("inference_attempts", 0), "subscription_inference_calls": result["subscription_inference_calls"], "api_key_calls": result["api_key_calls"]}))
    return 0 if result["status"] in ("prepared", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
