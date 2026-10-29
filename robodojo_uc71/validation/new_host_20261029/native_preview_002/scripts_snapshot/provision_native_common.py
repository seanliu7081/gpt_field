"""Shared fail-closed native setup helpers; no simulator imports."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

BASE = Path(__file__).resolve().parents[1]
PINS = BASE / "configs/source_pins.json"
REVIEWED_CAMERA003_ASSESSMENT = "6cdf56eef1d0242b8cc386b1af65e1e47d1a6f4de54e0c5a50f81a7e3065bbd6"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def asset_path(value: str) -> str:
    """Reject traversal, URL, absolute and non-asset paths before local writes."""
    path = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or path.is_absolute()
            or ".." in path.parts or not path.parts or path.parts[0] != "Assets"
            or len(path.parts) < 2):
        raise ValueError(f"Not a safe dataset asset path: {value!r}")
    return path.as_posix()


def reviewed_exit_correction(result_path: Path, result: dict, review: dict, assessment_path: Path) -> dict:
    """Accept only the preserved, independently reviewed camera003 exit race."""
    assessment_path = assessment_path.resolve(strict=True)
    assessment_sha = digest(assessment_path)
    if assessment_sha != REVIEWED_CAMERA003_ASSESSMENT or review.get("assessment_sha256") != assessment_sha:
        raise ValueError("Unknown or unbound camera assessment; no generic failed-gate bypass")
    assessment = json.loads(assessment_path.read_text())
    if (assessment.get("schema") != "robodojo-new-host-camera-assessment-1"
            or assessment.get("attempt") != "camera_003"
            or assessment.get("assessment_type") != "posthoc_exit_accounting_correction_no_rerun"
            or assessment.get("camera_gate_passed") is not True
            or assessment.get("corrected_runtime_compatibility_pass") is not True
            or not assessment.get("checks") or any(value is not True for value in assessment["checks"].values())
            or assessment["raw_result"]["sha256"] != digest(result_path)):
        raise ValueError("Invalid reviewed camera003 correction")
    base = assessment_path.parent
    if (base / assessment["raw_result"]["path"]).resolve() != result_path:
        raise ValueError("Assessment refers to a different raw result")
    for name, spec in assessment["raw_provenance"].items():
        path = (base / name).resolve(strict=True)
        if not path.is_relative_to(base) or path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError(f"Reviewed camera artifact changed: {name}")
    child = json.loads((result_path.parent / "launcher_pid.json").read_text())["child"]
    cuda = json.loads((result_path.parent / "cuda_identity.json").read_text())
    pid, tick = child["pid"], child["start_ticks"]
    if (result.get("pid") != pid or cuda.get("child_pid") != pid or cuda.get("child_start_ticks") != tick
            or assessment["identity"]["pid"] != pid or assessment["identity"]["start_ticks"] != tick
            or result.get("diagnostic_passed") is not False or result.get("termination") != "RuntimeError"):
        raise ValueError("Camera003 correction identity or failure kind differs")
    prefix = "Unexpected or ambiguous GPU process occupancy: "
    detail = result.get("termination_detail", "")
    if not detail.startswith(prefix):
        raise ValueError("Only the reviewed occupancy accounting error is correctable")
    rows = ast.literal_eval(detail[len(prefix):])
    if not rows or any(row.get("pid") != pid for row in rows):
        raise ValueError("Reported occupancy contains an unknown process")
    samples = [json.loads(line) for line in (result_path.parent / "gpu_monitor.jsonl").read_text().splitlines()]
    if len(samples) != 33 or assessment["identity"]["monitor_samples"] != 33:
        raise ValueError("Camera003 GPU monitor evidence differs")
    for sample in samples:
        if (len(sample["gpus"]) != 1 or sample["gpus"][0]["uuid"] != result["binding"]["gpu_uuid"]
                or any(row["pid"] != pid for row in sample["gpus"][0]["processes"])):
            raise ValueError("GPU monitor contains unreviewed device or process occupancy")
        for owned in sample["owned_processes"]:
            expected_tick = assessment["identity"]["owned_process_start_ticks"].get(str(owned["pid"]))
            if owned["pgrp"] != pid or owned["start_ticks"] != expected_tick:
                raise ValueError("Owned process identity changed during camera003")
    if rows != samples[-1]["gpus"][0]["processes"]:
        raise ValueError("Occupancy correction does not match the exact final monitor sample")
    return {"path": str(assessment_path), "sha256": assessment_sha,
            "type": assessment["assessment_type"], "raw_diagnostic_passed": False}


def require_camera_gate(result_path: Path, review_path: Path, assessment_path: Path | None = None) -> dict:
    """A diagnostic's frame alone cannot authorize downstream provisioning.

    The operator review record must explicitly bind its inspected image and
    the complete successful diagnostic result. This is setup, not test approval.
    """
    result_path, review_path = result_path.resolve(strict=True), review_path.resolve(strict=True)
    result = json.loads(result_path.read_text())
    review = json.loads(review_path.read_text())
    required = ("camera_render_passed", "shutdown_contract_satisfied",
                "renderer_identity_verified", "cuda_identity_verified",
                "gpu_processes_released", "gpu_memory_released")
    if (result.get("schema") != "robodojo-new-host-camera-1"
            or any(result.get(key) is not True for key in required)
            or result.get("returncode") != 0 or result.get("remaining_owned_pids")
            or result.get("cleanup_error")
            or result.get("termination_signals") != []):
        raise ValueError("A complete successful camera diagnostic with cleanup is required")
    correction = None
    if result.get("diagnostic_passed") is True:
        if result.get("termination"):
            raise ValueError("A successful diagnostic cannot contain a termination error")
    elif assessment_path is not None:
        correction = reviewed_exit_correction(result_path, result, review, assessment_path)
    else:
        raise ValueError("A failed raw diagnostic requires the exact reviewed camera003 assessment")
    mode = result.get("shutdown_mode")
    if not ((mode == "stock-fast" and result.get("preclose_ready_verified") is True)
            or (mode == "full" and result.get("full_teardown_returned") is True)):
        raise ValueError("The declared camera shutdown contract was not verified")
    image = result_path.parent / "rgb.png"
    if (review.get("passed") is not True
            or review.get("image_sha256") != digest(image)
            or review.get("launch_result_sha256") != digest(result_path)):
        raise ValueError("A positive visual review bound to this image and result is required")
    uuid = result.get("binding", {}).get("gpu_uuid")
    current = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=20
    ).splitlines()
    if not uuid or uuid not in [item.strip() for item in current]:
        raise ValueError("Camera gate belongs to a different GPU host")
    return {"result": str(result_path), "result_sha256": digest(result_path),
            "visual_review": str(review_path), "visual_review_sha256": digest(review_path),
            "gpu_uuid": uuid, "shutdown_mode": mode, "assessment": correction,
            "full_test_authorized": False}


def pins() -> dict:
    value = json.loads(PINS.read_text())
    if value.get("schema_version") != "robodojo-uc71-source-pins-1":
        raise ValueError("Unknown source pins schema")
    return value


def require_native_checkout(root: Path, pin_data: dict) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=30).strip()
    expected = pin_data["native_repository"]["commit"]
    if git("rev-parse", "HEAD") != expected:
        raise ValueError("Native source HEAD differs from the preserved pin")
    if git("diff", "--name-only", "HEAD"):
        raise ValueError("Tracked native source differs from pinned HEAD")
    verified = {"native_commit": expected, "gitlinks": {}}
    for entry in pin_data["gitlinks"]:
        row = git("ls-tree", "HEAD", entry["path"])
        if row.split("\t", 1)[0] != f"160000 commit {entry['commit']}":
            raise ValueError(f"Native gitlink differs: {entry['path']}")
        child = root / entry["path"]
        actual = subprocess.check_output(["git", "-C", str(child), "rev-parse", "HEAD"], text=True, timeout=30).strip()
        if actual != entry["commit"]:
            raise ValueError(f"Checked out submodule differs: {entry['path']}")
        changed = subprocess.check_output(["git", "-C", str(child), "diff", "--name-only", "HEAD"], text=True, timeout=30).strip()
        if changed:
            raise ValueError(f"Tracked submodule source modified: {entry['path']}")
        verified["gitlinks"][entry["path"]] = actual
    hashes = {}
    for name, spec in pin_data.get("native_source_content_sha256", {}).items():
        if name.startswith("/") or not spec.get("url", "").startswith("https://raw.githubusercontent.com/RoboDojo-Benchmark/RoboDojo/"):
            continue
        actual = digest(root / name)
        if actual != spec["sha256"]:
            raise ValueError(f"Pinned source content hash differs: {name}")
        hashes[name] = actual
    verified["source_sha256"] = hashes
    return verified
