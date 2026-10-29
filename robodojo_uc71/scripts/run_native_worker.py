#!/usr/bin/env python3
"""Launch one separately approved restricted worker; no model code or auto-retry."""
import argparse
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--ipc-fd",type=int,required=True)
    parser.add_argument("--parent-pid",type=int,required=True)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if not args.launch or args.ipc_fd < 3: parser.error("Explicit --launch and an inherited dedicated socket fd are required")
    # This proof precedes numpy, native runtime, bundle, Kit and native imports.
    from robodojo_uc71.worker_isolation import enforce_child_boundary
    enforce_child_boundary(credential_store="/workspace/gpt_field/.private/chatgpt_plan",parent_pid=args.parent_pid)
    from robodojo_uc71.native_runtime import WorkerRuntimeSpec, run_native_worker
    from robodojo_uc71.native_runtime_bundle import VerifiedRuntimeBundle
    spec = WorkerRuntimeSpec.load(args.spec)
    return run_native_worker(spec,ipc_fd=args.ipc_fd,parent_pid=args.parent_pid,bundle_loader=VerifiedRuntimeBundle.load)


if __name__ == "__main__":
    raise SystemExit(main())
