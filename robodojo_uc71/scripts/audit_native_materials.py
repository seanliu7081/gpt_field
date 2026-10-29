#!/usr/bin/env python3
"""Bounded MDL provenance audit; no simulator imports or authored asset changes.

Only the three known NVIDIA CDN material seeds and their active literal
texture_2d dependencies may be fetched. Comments are lexed out while quoted
strings remain intact. Engine search modules are read from the installed wheel,
hashed, and matched to distribution RECORD ownership. This is provenance and
resource availability evidence, not a renderer compilation or asset-closure pass.
"""
from __future__ import annotations

import argparse
import base64
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import posixpath
import re
import time
import urllib.parse
import urllib.request


BASE = Path(__file__).resolve().parents[1]
PREFIX = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Materials/2023_1/Base/"
SEEDS = tuple(PREFIX + name for name in (
    "Metals/Aluminum_Anodized.mdl", "Metals/Aluminum_Cast.mdl", "Plastics/Plastic_ABS.mdl"))
ENGINE = {"OmniPBR.mdl": "Base/OmniPBR.mdl", "OmniGlass.mdl": "Base/OmniGlass.mdl",
          "gltf/pbr.mdl": "mdl/gltf/pbr.mdl"}
MAX_BYTES, MAX_FILES, MAX_SECONDS = 200 * 1024 * 1024, 50, 300
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|[^\s]')


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def strip_comments(text: str) -> tuple[str, list[str]]:
    """Keep quoted/escaped strings and line numbering; blank both comment forms."""
    result, comments = [], []
    i = 0
    while i < len(text):
        start = i
        if text[i] == '"':
            i += 1
            while i < len(text):
                if text[i] == "\\":
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            else:
                raise ValueError("Unterminated MDL string")
            result.append(text[start:i])
        elif text.startswith("//", i):
            end = text.find("\n", i + 2)
            i = len(text) if end < 0 else end
            comments.append(text[start + 2:i])
            result.append(" " * (i - start))
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("Unterminated MDL comment")
            i = end + 2
            comments.append(text[start + 2:end])
            result.append("".join("\n" if c == "\n" else " " for c in text[start:i]))
        else:
            result.append(text[i])
            i += 1
    return "".join(result), comments


def texture_literals(text: str) -> list[dict]:
    tokens = list(TOKEN.finditer(text))
    references = []
    for i, token in enumerate(tokens[:-2]):
        if token.group() != "texture_2d" or tokens[i + 1].group() != "(":
            continue
        at = i + 2
        if (tokens[at].group() == "name" and at + 2 < len(tokens)
                and tokens[at + 1].group() == ":"):
            at += 2
        literal = tokens[at].group()
        if literal.startswith('"'):
            path = json.loads(literal)
            if path:
                references.append({"authored": path, "line": text.count("\n", 0, token.start()) + 1})
    return references


def inspect_mdl(raw: bytes) -> dict:
    text = raw.decode("utf-8-sig")
    active, comments = strip_comments(text)
    return {"active_texture_2d_references": texture_literals(active),
            "commented_texture_2d_references_ignored": sorted({row["authored"]
                for comment in comments for row in texture_literals(comment)}),
            "imports": re.findall(r"\bimport\s+([^;]+);", active),
            "method": "lexical comment removal, then tokenized literal texture_2d constructors"}


def allowed_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    expected = urllib.parse.urlsplit(PREFIX)
    decoded = urllib.parse.unquote(parsed.path)
    if (parsed.scheme != expected.scheme or parsed.netloc != expected.netloc
            or parsed.query or parsed.fragment or "\\" in decoded
            or not decoded.startswith(expected.path) or posixpath.normpath(decoded) != decoded
            or not url.startswith(PREFIX)):
        raise ValueError("Dependency or redirect leaves the exact approved NVIDIA CDN prefix")
    return url


class StrictRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        allowed_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def owners(site: Path, paths: list[Path]) -> dict:
    wanted = {p.relative_to(site).as_posix(): p for p in paths}
    found = {name: [] for name in wanted}
    for info in sorted(site.glob("*.dist-info")):
        record = info / "RECORD"
        if not record.is_file():
            continue
        with record.open(newline="") as stream:
            matches = [row for row in csv.reader(stream) if row and row[0] in wanted]
        if not matches:
            continue
        metadata = (info / "METADATA").read_text()
        package = re.search(r"^Name: (.+)$", metadata, re.M).group(1)
        version = re.search(r"^Version: (.+)$", metadata, re.M).group(1)
        for name, record_hash, record_size in matches:
            actual = sha256(wanted[name])
            encoded = base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=")
            verified = record_hash == "sha256=" + encoded and record_size == str(wanted[name].stat().st_size)
            if not verified:
                raise ValueError("Installed engine module does not match its distribution RECORD")
            found[name].append({"distribution": package, "version": version,
                "record": str(record.resolve()), "record_sha256": sha256(record),
                "record_file_hash": record_hash, "record_verified": True})
    if any(not entries for entries in found.values()):
        raise ValueError("An engine module has no verified installed distribution owner")
    return found


def audit(manifest_path: Path, site: Path, output: Path) -> dict:
    source_hash = sha256(manifest_path)
    source = json.loads(manifest_path.read_text())
    if source.get("schema") != "robodojo-native-assets-1" or source.get("complete") is not False:
        raise ValueError("Expected the incomplete pinned asset manifest")
    unresolved = source["unresolved"]
    if len(unresolved) != 15 or sum(row["authored"] in ENGINE for row in unresolved) != 11:
        raise ValueError("Unresolved set differs from the reviewed 15-reference material audit")
    if {row["authored"] for row in unresolved if row["authored"].startswith("https:")} != set(SEEDS):
        raise ValueError("Unexpected remote material seeds")
    output.mkdir(parents=False, exist_ok=False)
    result = {"schema": "robodojo-native-material-audit-1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {"path": str(manifest_path), "sha256": source_hash},
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "dataset_revision": source["revision"], "engine_modules": [], "remote_files": [],
        "unresolved_classification": [], "fetch_errors": [], "total_remote_bytes": 0,
        "limits": {"bytes": MAX_BYTES, "files": MAX_FILES, "seconds": MAX_SECONDS, "url_prefix": PREFIX},
        "simulator_starts": 0, "model_calls": 0, "benchmark_actions": 0,
        "full_test_authorized": False, "dependency_closure_complete": False,
        "authored_assets_modified": False, "limitations": [
            "Read-only material provenance; MDL imports are recorded but not recursively compiled or audited.",
            "Remote CDN resources are observed hashes, not immutable upstream revision pins.",
            "Engine search-path resolution still requires the separately gated native diagnostic.",
            "The upstream bound absolute normal-map reference remains unavailable; no substitution or URL rewrite."]}
    started = time.monotonic()
    try:
        root = site / "isaacsim/kit/mdl/core"
        engine_paths = [root / name for name in ENGINE.values()]
        ownership = owners(site, engine_paths)
        for authored, relative in ENGINE.items():
            path = root / relative
            entries = ownership[path.relative_to(site).as_posix()]
            if not any(row["distribution"] == "isaacsim-kernel" and row["version"] == "5.1.0.0" for row in entries):
                raise ValueError("Expected the installed IsaacSim 5.1.0.0 kernel owner")
            result["engine_modules"].append({"authored": authored, "path": str(path),
                "bytes": path.stat().st_size, "sha256": sha256(path), "owners": entries,
                **inspect_mdl(path.read_bytes())})
        pending = deque(SEEDS)
        known = set()
        opener = urllib.request.build_opener(StrictRedirect())
        while pending:
            url = allowed_url(pending.popleft())
            if url in known:
                continue
            if len(known) >= MAX_FILES or time.monotonic() - started > MAX_SECONDS:
                raise ValueError("Material fetch file or elapsed-time bound reached")
            known.add(url)
            path = output / "remote" / urllib.parse.unquote(url.removeprefix(PREFIX))
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "robodojo-uc71-material-audit/1"})
                with opener.open(request, timeout=30) as response, path.open("xb") as stream:
                    allowed_url(response.url)
                    declared = response.headers.get("Content-Length")
                    if declared and int(declared) > MAX_BYTES - result["total_remote_bytes"]:
                        raise ValueError("Material fetch byte bound would be exceeded")
                    count = 0
                    while True:
                        remaining = MAX_BYTES - result["total_remote_bytes"]
                        if remaining <= 0 or time.monotonic() - started > MAX_SECONDS:
                            raise ValueError("Material fetch byte or elapsed-time bound reached")
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        count += len(block)
                        result["total_remote_bytes"] += len(block)
                        stream.write(block)
                    if declared and count != int(declared):
                        raise ValueError("Remote material size differs from Content-Length")
                    row = {"url": url, "final_url": response.url, "http_status": response.status,
                        "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
                        "content_type": response.headers.get("Content-Type"), "path": str(path),
                        "bytes": count}
                row["sha256"] = sha256(path)
                if path.suffix.lower() == ".mdl":
                    row.update(inspect_mdl(path.read_bytes()))
                    for dependency in row["active_texture_2d_references"]:
                        dependency["url"] = allowed_url(urllib.parse.urljoin(url, dependency["authored"]))
                        pending.append(dependency["url"])
                result["remote_files"].append(row)
            except Exception as error:
                result["fetch_errors"].append({"url": url, "error_type": type(error).__name__, "message": str(error)})
        for reference in unresolved:
            authored = reference["authored"]
            if authored in ENGINE:
                classification = "engine_search_module"
                evidence_index = list(ENGINE).index(authored)
            elif authored in SEEDS:
                classification = "remote_material"
                evidence_index = next((i for i, row in enumerate(result["remote_files"]) if row["url"] == authored), None)
            else:
                classification, evidence_index = "unresolved_external_asset", None
            result["unresolved_classification"].append({"raw_reference": reference,
                "classification": classification, "evidence_index": evidence_index})
        result["classification_complete"] = not result["fetch_errors"] and len(result["unresolved_classification"]) == 15
    except Exception as error:
        result["audit_error"] = {"error_type": type(error).__name__, "message": str(error)}
        result["classification_complete"] = False
    result["source_manifest_unchanged"] = sha256(manifest_path) == source_hash
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["elapsed_seconds"] = time.monotonic() - started
    with (output / "audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def self_test():
    fixture = r'''// texture_2d("ignored-line.png")
/* texture_2d("ignored-block.png") */
texture_2d("https://example.test/path//keep.png", gamma: 1.0);
texture_2d(name: "../texture.png");
string label = "texture_2d(\"not-a-constructor.png\")";
texture_2d(dynamic_name); texture_2d();'''
    parsed = inspect_mdl(fixture.encode())
    assert [row["authored"] for row in parsed["active_texture_2d_references"]] == [
        "https://example.test/path//keep.png", "../texture.png"]
    assert parsed["commented_texture_2d_references_ignored"] == ["ignored-block.png", "ignored-line.png"]
    assert allowed_url(SEEDS[0]) == SEEDS[0]
    for url in ("https://evil.test/a", PREFIX + "../x", PREFIX + "%2e%2e/x", PREFIX + "x?redirect=1"):
        try:
            allowed_url(url)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid URL was allowed")
    print("Comment/string tokenizer and exact-CDN-boundary self-tests passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--asset-manifest", type=Path, default=BASE / "validation/new_host_20261029/native_assets_002/asset_manifest.json")
    parser.add_argument("--site-packages", type=Path, default=BASE / "runtime/new_host_env/lib/python3.11/site-packages")
    parser.add_argument("--output", type=Path, default=BASE / "validation/new_host_20261029/native_material_audit_001")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    result = audit(args.asset_manifest.resolve(strict=True), args.site_packages.resolve(strict=True), args.output.absolute())
    print(json.dumps({"audit": str(args.output.absolute() / "audit.json"),
        "classification_complete": result["classification_complete"],
        "engine_modules": len(result["engine_modules"]), "remote_files": len(result["remote_files"]),
        "remote_bytes": result["total_remote_bytes"], "errors": result["fetch_errors"],
        "audit_error": result.get("audit_error")}))
    if not result["classification_complete"] or not result["source_manifest_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
