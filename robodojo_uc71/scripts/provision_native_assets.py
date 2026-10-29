#!/usr/bin/env python3
"""Fetch a bounded pinned Make Toast layout-0 asset subset, never training data.

Requires the reviewed camera gate and verified native checkout. USD dependency
inspection uses a separate plain OpenUSD Python interpreter, never SimulationApp.
Unresolved or unparsed references keep the manifest incomplete and exit nonzero.
"""
from __future__ import annotations

import argparse
from collections import deque
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile

from provision_native_common import BASE, PINS, asset_path, digest, exclusive_json, pins, require_camera_gate, require_native_checkout

SUFFIXES = frozenset({
    ".usd", ".usda", ".usdc", ".usdz", ".mdl", ".png", ".jpg", ".jpeg", ".exr", ".hdr",
    ".json", ".yml", ".yaml", ".urdf", ".srdf", ".obj", ".stl", ".mtl", ".dae",
    ".ply", ".bmp", ".tif", ".tiff", ".bin", ".glb", ".gltf", ".drc", ".ktx", ".ktx2", ".webp",
})
TEXT_SUFFIXES = frozenset({".mdl", ".json", ".yaml", ".yml", ".urdf", ".srdf", ".mtl", ".dae", ".gltf"})
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc"})
USD_SCANNER = r"""
import json,re,sys
from pxr import Usd, Sdf
layer=Sdf.Layer.FindOrOpen(sys.argv[1])
if layer is None:
    raise RuntimeError("Could not inspect USD layer")
text=layer.ExportToString()
paths=re.findall(r"@([^@\r\n]+)@", text)
print(json.dumps({"references":sorted(set(paths)),"parser":"Sdf.Layer.ExportToString"}))
"""


def allowed_file(name: str) -> bool:
    path = PurePosixPath(asset_path(name))
    if path.suffix.lower() not in SUFFIXES:
        return False
    forbidden = {"training", "training_data", "demonstrations", "demos", "checkpoints", "weights", "trajectories"}
    return not any(part.lower() in forbidden for part in path.parts)


def seed_roots(pin_data: dict) -> tuple[list[str], list[str]]:
    layout = BASE / "docs/source_evidence/make_toast_layout0.evaluator_only.json"
    if digest(layout) != pin_data["selected_layout"]["sha256"]:
        raise ValueError("Cached selected layout differs from the pinned hash")
    data = json.loads(layout.read_text())
    roots = {"Assets/Robots/x5"}
    categories = set()
    for kind in ("Rigid", "Geometry", "Articulation"):
        for category, instances in data.get(kind, {}).items():
            base = asset_path(f"Assets/Object/RoboDojo/{kind}/{category}")
            categories.add(base)
            for instance in instances:
                index = instance["category_idx"]
                if type(index) is not int or index < 0:
                    raise ValueError("Invalid selected asset index")
                roots.add(f"{base}/{index:05d}")
    roots.update({
        asset_path("Assets/Room/" + data["Room"]["default"]),
        asset_path("Assets/Material/" + data["Table"]["default"]),
        asset_path("Assets/Material/" + data["Ground"]["materials"]["default"]),
        asset_path("Assets/Background/" + data["Background"]["category_name"]),
        asset_path(pin_data["selected_layout"]["path"]),
    })
    return sorted(roots), sorted(categories)


class Dataset:
    def __init__(self, pin_data: dict):
        spec = pin_data["assets"]
        self.repo, self.revision = spec["repository"], spec["revision"]
        if self.repo != "RoboDojo-Benchmark/RoboDojo" or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Unexpected dataset identity")
        self.api = f"https://huggingface.co/api/datasets/{self.repo}/tree/{self.revision}"
        self.entries: dict[str, dict] = {}
        self.requests = []

    def tree(self, directory: str, *, recursive: bool = True) -> list[dict]:
        directory = asset_path(directory)
        url = self.api + "/" + urllib.parse.quote(directory, safe="/") + "?recursive=" + str(recursive).lower() + "&expand=false"
        found = []
        while url:
            if not url.startswith(self.api + "/"):
                raise ValueError("Unexpected metadata pagination URL")
            self.requests.append(url)
            request = urllib.request.Request(url, headers={"User-Agent": "robodojo-uc71-pinned-setup/1"})
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
                if len(raw) > 32 * 1024 * 1024:
                    raise ValueError("Dataset metadata page is unexpectedly large")
                rows = json.loads(raw)
                if not isinstance(rows, list):
                    raise ValueError("Unexpected dataset tree response")
                link = response.headers.get("Link", "")
            for row in rows:
                name = asset_path(row["path"])
                if row.get("type") == "file":
                    row = dict(row, path=name)
                    self.entries[name] = row
                    found.append(row)
            match = re.search(r'<([^>]+)>;\s*rel="?next"?', link)
            url = match.group(1) if match else None
        return found

    def lookup(self, name: str) -> dict | None:
        name = asset_path(name)
        if name not in self.entries:
            try:
                self.tree(str(PurePosixPath(name).parent), recursive=False)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
        return self.entries.get(name)

    def root_files(self, root: str) -> list[dict]:
        if PurePosixPath(root).suffix.lower() in SUFFIXES:
            entry = self.lookup(root)
            if entry is None:
                raise ValueError(f"Pinned asset missing: {root}")
            return [entry]
        return self.tree(root)


def verify_file(path: Path, entry: dict) -> dict:
    size = path.stat().st_size
    expected_size = entry.get("lfs", {}).get("size", entry["size"])
    if size != expected_size:
        raise ValueError(f"Asset size mismatch: {entry['path']}")
    sha256 = digest(path)
    lfs_oid = entry.get("lfs", {}).get("oid")
    if lfs_oid is not None:
        if sha256 != lfs_oid.removeprefix("sha256:"):
            raise ValueError(f"LFS hash mismatch: {entry['path']}")
    else:
        git_hash = hashlib.sha1()
        git_hash.update(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                git_hash.update(block)
        if git_hash.hexdigest() != entry["oid"]:
            raise ValueError(f"Git blob hash mismatch: {entry['path']}")
    return {"path": entry["path"], "bytes": size, "sha256": sha256,
            "git_oid": entry["oid"], "lfs_oid": lfs_oid}


def download(dataset: Dataset, source: Path, entry: dict, *, remaining_bytes: int) -> dict:
    name = asset_path(entry["path"])
    if not allowed_file(name):
        raise ValueError(f"Refusing non-asset file type or excluded data: {name}")
    target = source / name
    if not target.resolve().is_relative_to(source.resolve()):
        raise ValueError("Asset destination escapes native checkout")
    size = entry.get("lfs", {}).get("size", entry["size"])
    if type(size) is not int or size < 0 or size > remaining_bytes:
        raise ValueError(f"Asset would exceed the explicit byte bound: {name}")
    if target.exists():
        return dict(verify_file(target, entry), reused=True)
    if shutil.disk_usage(source).free < size + 10 * 1024**3:
        raise ValueError("Asset download would leave less than 10 GiB free")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".uc71-partial")
    url = f"https://huggingface.co/datasets/{dataset.repo}/resolve/{dataset.revision}/" + urllib.parse.quote(name, safe="/")
    request = urllib.request.Request(url, headers={"User-Agent": "robodojo-uc71-pinned-setup/1"})
    # urllib follows the public signed CDN redirect; no credentials are supplied.
    with partial.open("xb") as stream, urllib.request.urlopen(request, timeout=180) as response:
        count = 0
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > size:
                raise ValueError(f"Download exceeds pinned asset size: {name}")
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
    result = verify_file(partial, entry)
    if target.exists():
        raise ValueError("Asset appeared concurrently; refusing replacement")
    partial.rename(target)
    return dict(result, reused=False)


def text_references(text: str) -> list[str]:
    # Resource strings in MDL/XML/YAML/JSON; scalar numbers and prose are ignored.
    result = set(re.findall(r'["\']([^"\'\r\n]+)["\']', text))
    result.update(re.findall(r"(?m):\s*([^\s#\"']+)\s*(?:#.*)?$", text))
    return sorted(value for value in result if PurePosixPath(value).suffix.lower() in SUFFIXES)


def structured_references(value) -> list[str]:
    """Read YAML/JSON values, excluding comments and non-resource scalar values."""
    if isinstance(value, dict):
        return sorted({ref for child in value.values() for ref in structured_references(child)})
    if isinstance(value, list):
        return sorted({ref for child in value for ref in structured_references(child)})
    if isinstance(value, str) and PurePosixPath(value).suffix.lower() in SUFFIXES:
        return [value]
    return []


def material_thumbnail_references(path: Path) -> set[str]:
    """MDL anno::thumbnail is material-browser metadata, not a shader texture."""
    if path.suffix.lower() != ".mdl":
        return set()
    text = path.read_text()
    pattern = r'::anno::thumbnail\s*\(\s*"([^"\r\n]+)"\s*\)'
    annotated = set(re.findall(pattern, text))
    # Keep a resource required if the shader also references that same image.
    return annotated - set(text_references(re.sub(pattern, "", text)))


def inspect_references(path: Path, usd_python: Path) -> list[tuple[str, str]]:
    """Return (anchor, authored reference), including layers inside USDZ packages."""
    suffix = path.suffix.lower()
    if suffix in USD_SUFFIXES:
        raw = subprocess.check_output([str(usd_python), "-c", USD_SCANNER, str(path)], text=True, timeout=180)
        result = json.loads(raw)
        return [("", ref) for ref in result["references"]]
    if suffix == ".usdz":
        refs = []
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            for name in names:
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise ValueError("Unsafe USDZ package member")
                if pure.suffix.lower() in USD_SUFFIXES:
                    raw = subprocess.check_output([str(usd_python), "-c", USD_SCANNER, f"{path}[{name}]"], text=True, timeout=180)
                    authored = json.loads(raw)["references"]
                elif pure.suffix.lower() in TEXT_SUFFIXES:
                    if archive.getinfo(name).file_size > 16 * 1024**2:
                        raise ValueError("Unexpectedly large text member")
                    authored = text_references(archive.read(name).decode("utf-8"))
                else:
                    continue
                for ref in authored:
                    internal = posixpath.normpath(posixpath.join(str(pure.parent), ref))
                    if internal in names:
                        continue
                    refs.append((name, ref))
        return refs
    if suffix in TEXT_SUFFIXES:
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError("Unexpectedly large asset text file")
        if suffix == ".json":
            return [("", ref) for ref in structured_references(json.loads(path.read_text()))]
        if suffix in {".yml", ".yaml"}:
            import yaml
            return [("", ref) for ref in structured_references(yaml.safe_load(path.read_text()))]
        return [("", ref) for ref in text_references(path.read_text())]
    return []


def candidate_path(owner: str, reference: str) -> str | None:
    reference = reference.replace("\\", "/")
    if "://" in reference or reference.startswith("::") or "[" in reference or "<" in reference:
        return None
    # Do not silently rewrite absolute authored paths: they need explicit review.
    if reference.startswith("/"):
        return None
    if reference.startswith("${ASSETS_PATH}/Assets/"):
        return asset_path(reference[len("${ASSETS_PATH}/"):])
    for marker in ("$RoboDojo_ASSETS/", "$Robodojo_ASSETS/"):
        if reference.startswith(marker):
            return asset_path("Assets/" + reference[len(marker):])
    for marker in ("$RoboDojo_HOME/",):
        if reference.startswith(marker):
            reference = reference[len(marker):]
    if reference.startswith("Assets/"):
        return asset_path(reference)
    normalized = posixpath.normpath(posixpath.join(str(PurePosixPath(owner).parent), reference))
    try:
        return asset_path(normalized)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-result", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--camera-assessment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--usd-python", type=Path, required=True,
                        help="Plain OpenUSD interpreter providing pxr.Sdf; no Isaac Sim startup")
    parser.add_argument("--max-gib", type=float, default=20.0)
    parser.add_argument("--max-files", type=int, default=5000)
    args = parser.parse_args()
    if not 0 < args.max_gib <= 60 or not 0 < args.max_files <= 20000:
        raise SystemExit("Choose finite asset limits: <=60 GiB and <=20000 files")
    gate = require_camera_gate(args.camera_result, args.visual_review, args.camera_assessment)
    pin_data = pins()
    source = BASE / "runtime/native/RoboDojo"
    source_verification = require_native_checkout(source, pin_data)
    # Preserve a venv's executable symlink so Python retains its environment.
    usd_python = args.usd_python.absolute()
    if not usd_python.is_file() or not os.access(usd_python, os.X_OK):
        raise SystemExit("USD inspector Python is not an executable file")
    usd_metadata = json.loads(subprocess.check_output(
        [str(usd_python), "-c", "from pxr import Sdf,Usd; import importlib.metadata as m,json,sys; "
         "print(json.dumps({'usd_core_version':m.version('usd-core'),'usd_version':list(Usd.GetVersion()),'python':sys.version}))"],
        text=True, timeout=30))
    if usd_metadata["usd_core_version"] != "25.5.1":
        raise SystemExit("Use the reviewed plain usd-core==25.5.1 inspector")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = Dataset(pin_data)
    roots, categories = seed_roots(pin_data)
    record = {"schema": "robodojo-native-assets-1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "gate": gate, "pins_sha256": digest(PINS), "source": source_verification,
              "dataset": dataset.repo, "revision": dataset.revision, "seed_roots": roots,
              "scope": "Selected layout component folders plus recursively referenced assets; evaluator/setup only",
              "max_bytes": int(args.max_gib * 1024**3), "max_files": args.max_files,
              "files": [], "dependencies": [], "unresolved": [], "scan_errors": [],
              "metadata_only_references": [],
              "complete": False, "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
              "full_test_authorized": False, "usd_parser": str(usd_python),
              "usd_parser_metadata": usd_metadata}
    try:
        queue = deque()
        for root in roots:
            queue.extend(row["path"] for row in dataset.root_files(root) if allowed_file(row["path"]))
        for category in categories:
            # Category map/metadata support is small; no sibling model directories.
            queue.extend(row["path"] for row in dataset.tree(category, recursive=False)
                         if PurePosixPath(row["path"]).suffix.lower() in {".json", ".yml", ".yaml"})
        done, total = set(), 0
        while queue:
            name = queue.popleft()
            if name in done:
                continue
            if len(done) >= args.max_files:
                raise ValueError("Asset dependency closure exceeds explicit file bound")
            entry = dataset.lookup(name)
            if entry is None:
                raise ValueError(f"Queued asset missing from pinned metadata: {name}")
            item = download(dataset, source, entry, remaining_bytes=record["max_bytes"] - total)
            record["files"].append(item)
            done.add(name)
            total += item["bytes"]
            print(json.dumps({"asset": name, "bytes": item["bytes"], "total_bytes": total}), flush=True)
            try:
                references = inspect_references(source / name, usd_python)
                thumbnails = material_thumbnail_references(source / name)
            except Exception as error:
                record["scan_errors"].append({"path": name, "error": f"{type(error).__name__}: {error}"})
                continue
            for member, ref in references:
                if ref in thumbnails:
                    record["metadata_only_references"].append({"from": name, "authored": ref,
                        "reason": "MDL anno::thumbnail material-browser annotation, not a shader texture"})
                    continue
                # Missing package members are unresolved; do not guess a disk
                # path for a reference anchored inside a USDZ archive.
                candidate = None if member else candidate_path(name, ref)
                if name == pin_data["selected_layout"]["path"]:
                    selected_layout = json.loads((source / name).read_text())
                    if ref == selected_layout["Background"]["category_name"]:
                        candidate = asset_path("Assets/Background/" + ref)
                dep = {"from": name, "package_member": member or None, "authored": ref, "candidate": candidate}
                record["dependencies"].append(dep)
                if candidate is None or not allowed_file(candidate) or dataset.lookup(candidate) is None:
                    record["unresolved"].append(dep)
                else:
                    queue.append(candidate)
        layout = source / pin_data["selected_layout"]["path"]
        if digest(layout) != pin_data["selected_layout"]["sha256"]:
            raise ValueError("Downloaded selected layout hash differs")
        record["total_bytes"] = total
        record["complete"] = not record["unresolved"] and not record["scan_errors"]
        record["limitations"] = ("No simulation/rendering validation. MDL resource strings are scanned, but dynamic "
                                "shader module imports and engine assets are not certified by this manifest. "
                                "Native scene loading must also report no unresolved resources.")
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["metadata_requests"] = dataset.requests
        record["metadata_entries"] = list(dataset.entries.values())
        record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        exclusive_json(output / "asset_manifest.json", record)
    print(json.dumps({"complete": record["complete"], "files": len(record["files"]),
                      "unresolved": len(record["unresolved"]), "scan_errors": len(record["scan_errors"])}))
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
