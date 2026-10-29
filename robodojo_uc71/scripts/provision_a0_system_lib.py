#!/usr/bin/env python3
"""Unpack one pinned Ubuntu user-space library into runtime; never apt-install."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

BASE = Path(__file__).resolve().parents[1]
URL = "https://archive.ubuntu.com/ubuntu/pool/main/libx/libxt/libxt6t64_1.2.1-1.2build1_amd64.deb"
SHA256 = "2e4317d5ae1a45e517771ea0b8d93519b5f915ab08983facecd480d64dbd1489"


def main():
    artifacts = BASE / "runtime/artifacts"
    artifacts.mkdir(exist_ok=True)
    destination = artifacts / URL.rsplit("/", 1)[1]
    if not destination.exists():
        payload = urllib.request.urlopen(URL, timeout=45).read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise RuntimeError("libXt archive hash does not match Ubuntu package metadata")
        destination.write_bytes(payload)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Existing libXt archive hash mismatch")
    subprocess.run(["dpkg-deb", "--extract", str(destination), str(BASE / "runtime/sysroot")], check=True)
    report = {"url": URL, "sha256": SHA256, "version": "1:1.2.1-1.2build1",
              "scope": "task-local extraction only; no dpkg registration or system library modification"}
    (BASE / "validation/provisioning/local_libxt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
