#!/usr/bin/env python3
"""Freeze a concrete review packet and disabled service; never grant approval.

Requires completed measured characterization and a semantically valid original
planning draft. No simulator, service, policy call or authorization consumption
is started. The root and worker UID both run the static full-input preflight.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))
from robodojo_uc71.approval import build_manifest, canonical_sha256, file_sha256
from robodojo_uc71.native_runtime_bundle import VerifiedRuntimeBundle
from robodojo_uc71.native_episode_runner import execution_configuration
from robodojo_uc71.native_runtime import published_asset_policy, runtime_program_files
from robodojo_uc71.planning import direction_frame_from_calibration, validate_plan, plan_validation
from robodojo_uc71.reset_binding import verify_fresh_reset
from prepare_initial_plan import public_instruction
from prepare_native_preview import verify_manifests
from prepare_standard_service import prepare_service
from provision_native_common import pins
from provision_controller_characterization import prepare_worker_git_config
from provision_new_host_camera import child_environment


def shared_json(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(fd, "wb") as stream:
        os.fchown(stream.fileno(), 0, 65534); os.fchmod(stream.fileno(), 0o640)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def make_directory(path, uid, mode):
    path.mkdir(mode=mode, parents=False, exist_ok=False)
    os.chown(path, uid, 65534); os.chmod(path, mode)


def verified_original_plan(planning):
    from gpt_field.astra import AstraClient, _strict_loads
    original = _strict_loads((planning / "original_draft.json").read_text())
    response = _strict_loads((planning / "original_response.json").read_text())
    if canonical_sha256(AstraClient._extract(response)) != canonical_sha256(original):
        raise ValueError("draft_differs_from_original_model_response")
    plan = validate_plan(original)
    validation = _strict_loads((planning / "validation.json").read_text())
    if validation != plan_validation(plan):
        raise ValueError("saved_plan_validation_differs_from_recomputation")
    return plan, validation


def verified_planning_payload(planning):
    """Check sanitized request logs without copying parent transport metadata."""
    payload_path = planning / "planning_payload.json"
    payload = json.loads(payload_path.read_text())
    requests = sorted((planning / "subscription_logs").glob("*.request.json"))
    if not requests:
        raise ValueError("actual_planning_request_log_required")
    records = []
    for path in requests:
        record = json.loads(path.read_text())
        if canonical_sha256(record.get("payload")) != canonical_sha256(payload):
            raise ValueError("sent_planning_payload_differs_from_prepared_payload")
        records.append({"source_request_sha256":file_sha256(path),
            "request_key":record.get("request_key"),"request_ordinal":record.get("request_ordinal")})
    return {"prepared_payload_sha256":file_sha256(payload_path),
        "sent_structured_payloads_match_prepared":True,"transport_request_count":len(records),
        "requests":records}


class FileBindings:
    def __init__(self):
        self.values = {}
        self.historical_aliases = {}
        self.historical_bindings = {}

    def add(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        if not path.is_file() or ".private" in path.parts:
            raise ValueError("only_nonsecret_regular_inputs_may_be_bound")
        actual = file_sha256(path)
        if expected is not None and actual != expected: raise ValueError("source_binding_changed: " + str(path))
        if str(path) in self.values and self.values[str(path)] != actual:
            raise ValueError("conflicting_source_binding")
        self.values[str(path)] = actual

    @staticmethod
    def _references(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                yield value
            for child in value.values(): yield from FileBindings._references(child)
        elif isinstance(value, list):
            for child in value: yield from FileBindings._references(child)

    def _historical_sources(self, documents):
        """Resolve only explicitly bound companions, never discover side files.

        This format describes historical Python sources of the comparison in
        the same directory. Its aliases apply only to that exact report's bytes
        and source_files pairs. They never replace the current program binding.
        """
        aliases = {}
        for companion, (record, _) in documents.items():
            if not isinstance(record, dict) or record.get("schema") != "robodojo-offline-analysis-source-snapshots-1":
                continue
            if (companion.name != "archived_sources.json"
                    or set(record) - {"schema", "comparison_sha256", "sources", "note"}
                    or not isinstance(record.get("sources"), list) or not record["sources"]):
                raise ValueError("invalid_historical_source_companion")
            comparison = companion.parent / "comparison.json"
            if comparison not in documents or documents[comparison][1] != record.get("comparison_sha256"):
                raise ValueError("historical_comparison_binding_mismatch")
            report, report_sha = documents[comparison]
            aliases.update(self._source_aliases(comparison, report, report_sha,
                record["sources"], archive_root=companion.parent / "source_snapshot"))
        return aliases

    @staticmethod
    def _digest(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

    def _source_aliases(self, owner, report, report_sha, rows, *, archive_root=None):
        sources = report.get("source_files") if isinstance(report, dict) else None
        if not isinstance(sources, list) or not isinstance(rows, list) or not rows:
            raise ValueError("historical_comparison_sources_missing")
        source_pairs = {(row.get("path"), row.get("sha256")) for row in sources
                        if isinstance(row, dict) and isinstance(row.get("path"), str)
                        and isinstance(row.get("sha256"), str)}
        aliases = {}
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"original_path", "archived_path", "sha256"}
                    or any(not isinstance(row.get(key), str) for key in row)
                    or not self._digest(row["sha256"])
                    or (row["original_path"], row["sha256"]) not in source_pairs):
                raise ValueError("invalid_historical_source_alias")
            original, archive = Path(row["original_path"]), Path(row["archived_path"])
            source_roots = (BASE / "scripts", BASE / "src/robodojo_uc71")
            if (not original.is_absolute() or original != original.resolve(strict=False)
                    or original.suffix != ".py" or not any(original.is_relative_to(root) for root in source_roots)
                    or not archive.is_absolute() or archive != archive.resolve(strict=True)
                    or not archive.is_relative_to(BASE) or archive.name != original.name
                    or archive.parent.name not in ("source_snapshot", "scripts_snapshot")
                    or (archive_root is not None and archive.parent != archive_root)):
                raise ValueError("invalid_historical_source_archive_path")
            key = (owner, report_sha, str(original), row["sha256"])
            if key in aliases:
                raise ValueError("duplicate_historical_source_alias")
            self.add(archive, row["sha256"])
            aliases[key] = archive
        return aliases

    def load_historical_aliases(self, path, expected_sha):
        """Explicitly freeze a reviewed companion before traversing any report."""
        path = Path(path)
        if (not self._digest(expected_sha) or not path.is_absolute()
                or path != path.resolve(strict=True) or not path.is_relative_to(BASE)):
            raise ValueError("invalid_historical_companion_binding")
        self.add(path, expected_sha)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError("source_binding_changed: " + str(path))
        record = json.loads(raw)
        if (not isinstance(record, dict) or set(record) != {"schema", "documents"}
                or record["schema"] != "robodojo-historical-evidence-aliases-1"
                or not isinstance(record["documents"], list) or not record["documents"]):
            raise ValueError("invalid_historical_source_companion")
        aliases, bindings = {}, {str(path): expected_sha}
        for entry in record["documents"]:
            if (not isinstance(entry, dict) or set(entry) != {"document", "sources"}
                    or not isinstance(entry["document"], dict) or set(entry["document"]) != {"path", "sha256"}
                    or not isinstance(entry["document"]["path"], str)
                    or not self._digest(entry["document"]["sha256"])):
                raise ValueError("invalid_historical_document_binding")
            owner = Path(entry["document"]["path"]); digest = entry["document"]["sha256"]
            if (not owner.is_absolute() or owner != owner.resolve(strict=True)
                    or not owner.is_relative_to(BASE) or owner.suffix != ".json"):
                raise ValueError("invalid_historical_document_binding")
            self.add(owner, digest)
            raw = owner.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("source_binding_changed: " + str(owner))
            rows = self._source_aliases(owner, json.loads(raw), digest, entry["sources"])
            if set(rows) & (set(aliases) | set(self.historical_aliases)):
                raise ValueError("duplicate_historical_source_alias")
            aliases.update(rows); bindings[str(owner)] = digest
            bindings.update({str(archive):key[3] for key,archive in rows.items()})
        self.historical_aliases.update(aliases)
        self.historical_bindings.update(bindings)

    def document_closure(self, path):
        # JSON discovery and source verification are separate so the ordering of
        # comparison/companion references cannot change the result. Every JSON
        # reference is checked against its parent-provided hash before parsing.
        for bound, digest in self.historical_bindings.items(): self.add(bound, digest)
        queue, documents, deferred = [(Path(path), None)], {}, []
        while queue:
            current, expected = queue.pop()
            current = current.resolve(strict=True)
            self.add(current, expected)
            if current in documents: continue
            if current.suffix != ".json": continue
            raw = current.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != self.values[str(current)]:
                raise ValueError("source_binding_changed: " + str(current))
            document = json.loads(raw)
            documents[current] = (document, digest)
            if isinstance(document, dict) and document.get("schema") == "robodojo-new-host-camera-assessment-1":
                for name, row in document["raw_provenance"].items():
                    target = (current.parent / name).resolve(strict=True)
                    if not target.is_relative_to(current.parent) or target.stat().st_size != row["bytes"]:
                        raise ValueError("camera_assessment_relative_artifact_invalid")
                    self.add(target,row["sha256"])
                repair = document["source_repair"]
                self.add(BASE / repair["path"],repair["sha256"])
            gate = document.get("gate", {}) if isinstance(document, dict) else {}
            if isinstance(gate, dict):
                for name in ("result", "visual_review"):
                    if isinstance(gate.get(name), str) and isinstance(gate.get(name + "_sha256"), str):
                        self.add(gate[name],gate[name + "_sha256"])
            for value in self._references(document):
                target = Path(value["path"])
                # Explicit absolute evidence references only; asset-relative
                # entries are bound against the pinned native root below.
                if target.is_absolute():
                    if not target.resolve(strict=False).is_relative_to(BASE):
                        raise ValueError("unexpected_external_evidence_reference")
                    if target.suffix == ".json": queue.append((target, value["sha256"]))
                    else: deferred.append((current, digest, target, value["sha256"]))
        aliases = self._historical_sources(documents)
        if set(aliases) & set(self.historical_aliases):
            raise ValueError("duplicate_historical_source_alias")
        aliases.update(self.historical_aliases)
        for current, digest, target, expected in deferred:
            archived = aliases.get((current, digest, str(target), expected))
            self.add(archived if archived is not None else target, expected)

    def mapping(self):
        # Freeze the verified versions, not newer bytes that appeared after
        # traversal but before build_manifest reads its final binding paths.
        for path, expected in self.values.items(): self.add(path, expected)
        return {f"input_{index:05d}":path for index,path in enumerate(sorted(self.values))}


def prepare(args):
    if os.getuid() != 0: raise ValueError("root_preparation_required")
    v = BASE / "validation/new_host_20261029"
    native = BASE / "runtime/native/RoboDojo"
    bundle = VerifiedRuntimeBundle.load(args.runtime_bundle, native); bundle.verify()
    source, assets, assessment = (v / "native_source_001/source_manifest.json",
        v / "native_assets_002/asset_manifest.json", v / "native_assets002_diagnostic_assessment.json")
    preparation = verify_manifests(native, source, assets, pins(), assessment)
    instruction, instruction_source = public_instruction(native)
    planning = args.planning_dir.resolve(strict=True)
    result = json.loads((planning / "result.json").read_text())
    if (result.get("draft_received") is not True or result.get("semantic_validation_passed") is not True
            or result.get("planning_only") is not True or result.get("full_test_started") is not False):
        raise ValueError("valid_original_planning_only_draft_required")
    plan, plan_check = verified_original_plan(planning)
    payload_check = verified_planning_payload(planning)
    planned = json.loads((planning / "preparation.json").read_text())
    if (planned["runtime_bundle"]["sha256"] != file_sha256(args.runtime_bundle)
            or planned["task_prompt"] != instruction or planned["images"] != bundle.reset_reference["image_files"]):
        raise ValueError("planning_draft_uses_other_prepared_inputs")
    reference = bundle.reset_reference
    robot = json.loads(Path(reference["robot_file"]["path"]).read_text())
    cameras = json.loads(Path(reference["camera_file"]["path"]).read_text())
    check = verify_fresh_reset(reference, robot, cameras,
        {name:Path(row["path"]) for name,row in reference["image_files"].items()})
    calibration = bundle.public_calibration(cameras)
    packet, attempt, service = args.output.absolute(), args.attempt.absolute(), args.service.absolute()
    for path in (packet, attempt, service):
        if path.exists() or path.parent.resolve(strict=True) != path.parent:
            raise ValueError("fresh_canonical_packet_attempt_service_required")
    make_directory(packet,0,0o750); make_directory(attempt,0,0o750)
    make_directory(attempt / "worker",65534,0o700)
    runtime = attempt / "runtime"; make_directory(runtime,65534,0o700)
    for name in ("home","cache","config","data","tmp"): make_directory(runtime / name,65534,0o700)
    git_config = prepare_worker_git_config(runtime,native,pins())
    log = attempt / "native_stdout_stderr.log"
    fd = os.open(log,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o640)
    os.fchown(fd,0,65534); os.fchmod(fd,0o640); os.close(fd)
    preview_root = v / "native_preview_002"
    preview_launch = json.loads((preview_root / "launch_result.json").read_text())
    cuda_path = preview_root / "preview/cuda_identity.json"
    cuda = json.loads(cuda_path.read_text())
    if (preview_launch.get("returncode") != 0 or preview_launch.get("gpu_processes_released") is not True
            or preview_launch.get("remaining_owned_pids") != [] or preview_launch["pid"] != cuda["child_pid"]):
        raise ValueError("actual_archived_preparation_identity_required")
    fingerprint = {"pid":cuda["child_pid"],"start_ticks":cuda["child_start_ticks"],
        "preview_result_sha256":file_sha256(preview_root / "preview/preview_result.json")}
    identity = {"pid":fingerprint["pid"],"boot_nonce":"archived-preview-sha256:"+canonical_sha256(fingerprint)}
    shared_json(packet / "preparation_identity.json", {"schema":"robodojo-archived-preparation-identity-1",
        "nonce_kind":"derived_archive_fingerprint_not_live_handshake","fingerprint":fingerprint,
        "preparation_worker":identity,"live_ready_claimed":False,"worker_mode":"fresh",
        "source_files":[{"path":str(p),"sha256":file_sha256(p)} for p in
            (cuda_path,preview_root/"launch_result.json",preview_root/"preview/preview_result.json")]})
    shared_json(packet / "reviewed_plan.json",plan)
    shared_json(packet / "plan_validation.json",plan_check)
    # Preserve only the planning artifacts in this read-only review closure.
    # The parent-only request/transport logs and credential directory stay private.
    originals = packet / "planning_originals"; make_directory(originals,0,0o750)
    provenance = {}
    for name in ("original_draft.json","original_response.json","result.json","preparation.json","validation.json","planning_payload.json"):
        target = originals / name
        fd = os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o640)
        with os.fdopen(fd,"wb") as stream:
            os.fchown(stream.fileno(),0,65534); os.fchmod(stream.fileno(),0o640)
            stream.write((planning/name).read_bytes()); stream.flush(); os.fsync(stream.fileno())
        provenance[name] = {"path":str(target),"sha256":file_sha256(target)}
        if provenance[name]["sha256"] != file_sha256(planning/name):
            raise ValueError("original_planning_artifact_changed_during_copy")
    shared_json(packet / "planning_provenance.json",{"originals":provenance,
        "original_response_exactly_matches_draft":True,"validation_recomputed":True,"edits":[],
        "request_payload_audit":payload_check,
        "human_approval_pending":True,"policy_model_calls":0})
    shared_json(packet / "calibration.json",calibration)
    shared_json(packet / "preview.json",{"schema":"robodojo-standard-review-preview-1",
        "images":reference["image_files"],"task_prompt":instruction,"instruction_source":instruction_source,
        "plan":{"path":str(packet/"reviewed_plan.json"),"sha256":file_sha256(packet/"reviewed_plan.json")},
        "runtime_bundle":{"path":str(args.runtime_bundle.resolve()),"sha256":file_sha256(args.runtime_bundle)},
        "full_test_authorized":False,"full_test_started":False})
    binding = {"gpu_uuid":"GPU-a9e034f8-a424-d7fe-6eaa-45667dd2c6b2","pci_bus_id":"0000:01:00.0",
               "kit_index":0,"cuda_ordinal":0}
    runtime_config = {"python":str(BASE/"runtime/new_host_env/bin/python"),
        "library_path":str(BASE/"runtime/new_host_sysroot_002/usr/lib/x86_64-linux-gnu"),
        "icd":"/etc/vulkan/icd.d/nvidia_icd.json","overlay_target":str(BASE/"runtime/native_overlay"),
        "compat_target":str(BASE/"runtime/native_compat"),"worker_script":str(BASE/"scripts/run_native_worker.py"),
        "maximum_wall_time_s":10800,"ready_timeout_s":600,"minimum_free_mib":16384}
    manifest_path, spec_path = packet/"manifest.json", packet/"worker_spec.json"
    spec = {"schema":"robodojo-native-worker-spec-1","manifest_path":str(manifest_path),
        "runtime_bundle_path":str(args.runtime_bundle.resolve()),"native_root":str(native),
        "source_manifest_path":str(source),"asset_manifest_path":str(assets),"asset_assessment_path":str(assessment),
        "runtime_dir":str(runtime),"parent_log":str(log),"gpu_binding":binding,"boot_nonce":uuid.uuid4().hex}
    shared_json(spec_path,spec)
    managed = prepare_service(service,python=runtime_config["python"],spec=spec_path,
        overlay=runtime_config["overlay_target"],compatibility=runtime_config["compat_target"])
    files = FileBindings()
    historical = getattr(args, "historical_aliases", [])
    for path, digest in historical: files.load_historical_aliases(path, digest)
    for path, _ in historical: files.document_closure(path)
    files.document_closure(args.runtime_bundle)
    files.document_closure(source); files.document_closure(assets); files.document_closure(assessment)
    for path in sorted(packet.rglob("*")):
        if path.is_file(): files.add(path)
    files.add(managed["configuration"]); files.add(git_config["path"],git_config["sha256"])
    for directory in (BASE/"scripts",BASE/"src/robodojo_uc71"):
        for path in sorted(directory.rglob("*.py")): files.add(path)
    for path in runtime_program_files(): files.add(path)
    files.add(BASE / "configs/source_pins.json")
    core = BASE/"vendor/uc71"; release = core/"release_manifest.json"
    files.add(release)
    for name,digest in json.loads(release.read_text())["files"].items(): files.add(core/name,digest)
    for checkout in [native,*(native/row["path"] for row in pins()["gitlinks"])]:
        names = subprocess.check_output(["git","-C",str(checkout),"ls-files","-z"],timeout=30).decode().split("\0")
        for name in names:
            if name and (checkout/name).is_file(): files.add(checkout/name)
    for row in json.loads(assets.read_text())["files"]: files.add(native/row["path"],row["sha256"])
    for path in Path(runtime_config["library_path"]).iterdir():
        if path.is_file(): files.add(path)
    for name in ("python","icd"): files.add(runtime_config[name])
    for relative in ("install_001/install_status.json","native_dependencies_001/install_status.json",
                     "native_compatibility_001/install_status.json"):
        files.document_closure(v/relative)
    configuration = {"task_prompt":instruction,"approved_plan":plan,
        "direction_frame":direction_frame_from_calibration(calibration),
        "model":{"name":"gpt-6-astra","reasoning_effort":"high","request_deadline_s":180},
        "execution":execution_configuration(controller_config=bundle.controller_config,
            monitoring=bundle.monitoring_settings,limits=bundle.execution_limits,
            initial_gripper_commands={"left":"open","right":"open"},controller_capabilities=bundle.controller_capabilities),
        "runtime":runtime_config,"asset_policy":published_asset_policy(preparation),
        "native_standard":{"action_limit":1400,"substeps_per_action":10,"physics_dt_s":.004,"layout":0},
        "scope":{"qualitative_pilot":True,"matched_reference_reproduction":False,
            "loaded_contact_validated":False,"workspace_wide_tracking_validated":False}}
    manifest = build_manifest(task_key="make_toast_standard_layout0",attempt_dir=attempt,bindings=files.mapping(),
        configuration=configuration,sensor_sha256=canonical_sha256(bundle.sensor_signature(cameras)),
        start_state_sha256=check["start_state_sha256"],preview_sha256=file_sha256(packet/"preview.json"),
        preparation_worker=identity)
    shared_json(manifest_path,manifest)
    command = [runtime_config["python"],"-B",str(BASE/"scripts/run_standard_attempt.py"),"--spec",str(spec_path)]
    env = child_environment(Path(runtime_config["python"]),runtime,binding,Path(runtime_config["icd"]),
                            Path(runtime_config["library_path"]),"stock-fast")
    env = {k:v for k,v in env.items() if not k.startswith("UC71_CAMERA_")}
    env.update(PYTHONPATH=runtime_config["compat_target"]+os.pathsep+runtime_config["overlay_target"],
               TMPDIR=str(runtime/"tmp"),GIT_OPTIONAL_LOCKS="0")
    checks = []
    for identity_args in ({},{"user":65534,"group":65534,"extra_groups":[]}):
        checked = subprocess.run(command,env=env,cwd=attempt,capture_output=True,text=True,timeout=180,**identity_args)
        if checked.returncode:
            shared_json(packet/f"static_preflight_failure_{len(checks)}.json",{"returncode":checked.returncode,
                "stdout":checked.stdout,"stderr":checked.stderr,"worker_uid":identity_args.get("user",0)})
            raise RuntimeError("static_review_packet_preflight_failed")
        checks.append({"uid":identity_args.get("user",0),"result":json.loads(checked.stdout)})
    summary = {"schema":"robodojo-standard-review-packet-1","manifest_path":str(manifest_path),
        "manifest_sha256":manifest["manifest_sha256"],"preview_sha256":manifest["preview_sha256"],
        "bound_files":len(manifest["files"]),"static_preflight":checks,"service":managed,
        "historical_aliases":[{"path":str(path),"sha256":digest} for path,digest in historical],
        "approval_present":False,"approval_consumed":False,"simulator_starts":0,"policy_model_calls":0,
        "full_test_started":False,"full_test_authorized":False,"single_attempt":True}
    shared_json(packet/"preparation_result.json",summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime-bundle","planning-dir","output","attempt","service"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--historical-aliases", action="append", nargs=2, default=[], metavar=("PATH", "SHA256"),
        help="Explicit reviewed historical-source companion; repeat for multiple hash-bound companions.")
    print(json.dumps(prepare(parser.parse_args())))
