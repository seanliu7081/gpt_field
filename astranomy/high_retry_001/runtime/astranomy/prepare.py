"""Write a reviewable preflight bundle without authentication or simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .policy import review_plan, policy_instructions, dynamic_tools
from . import run


def prepare(output):
    output = Path(output).resolve()
    if not output.is_relative_to(run.ROOT / "astranomy"):
        raise ValueError("Preparation must stay under gpt_field/astranomy")
    output.mkdir(parents=True, exist_ok=False)
    plan = review_plan()
    catalog = Path(__file__).with_name("model_catalog.json")
    plan["local_preflight"] = {
        "codex_executable": shutil.which("codex"),
        "supervisord_executable": shutil.which("supervisord"),
        "simulator_python": run.life.DEFAULT_SIM_PYTHON,
        "simulator_python_exists": Path(run.life.DEFAULT_SIM_PYTHON).exists(),
        "libero_root": run.life.DEFAULT_LIBERO_ROOT,
        "libero_root_exists": Path(run.life.DEFAULT_LIBERO_ROOT).exists(),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "credentials_read": False, "model_calls": 0, "simulator_started": False,
    }
    codex = plan["local_preflight"]["codex_executable"]
    if codex:
        plan["local_preflight"]["codex_version"] = subprocess.check_output([codex, "--version"], text=True).strip()
    plan["launch_after_approval"] = (
        "cd /workspace/gpt_field && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. "
        ".venv/bin/python -m astranomy.launch --approved-full-test --name drawer_bowl_001")
    (output / "review_plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n")
    (output / "policy_prompt.txt").write_text(policy_instructions() + "\n")
    (output / "robot_tool.json").write_text(json.dumps(dynamic_tools(), indent=2) + "\n")
    sources = {}
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file() and path.suffix in (".py", ".json", ".md"):
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "new_source_sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().output), indent=2, default=str))
