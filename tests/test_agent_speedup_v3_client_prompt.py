"""Prompt contract checks; AST only, no transport or simulator execution."""
import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "src/gpt_field/agent_speedup_v3_client.py"


def constants():
    values = {}
    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "strip":
                return literal(node.func.value).strip()
            if node.func.attr == "join" and len(node.args) == 1 and isinstance(node.args[0], ast.Tuple):
                return literal(node.func.value).join(literal(item) for item in node.args[0].elts)
        raise ValueError("not a prompt constant")
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = literal(node.value)
            except (ValueError, KeyError):
                pass
    return values


def test_lean_contract_preserves_persistence_verification_and_unknown_admission():
    values = constants()
    text = values["AGENT_INSTRUCTIONS"]
    for phrase in ("program={operations:[...]}", "program={stages:[...]}", "source_inventory",
                   "destination_inventory", "accepted visual release", "operation_state",
                   "after geometry confirmation is accepted", "ALL missing_evidence_requests",
                   "execution_allowed=true", "status=unknown", "execution_status=admissible",
                   "physical_clearance_verified remains false", "blocking_reasons", "advisories",
                   "Normal phase transitions do not require another model call",
                   "consumed watchdog and recovery counts", "posture, not a grasp"):
        assert phrase in text
    assert len(text) < 12000
    assert "Resolve ALL missing_evidence_requests" not in text
    assert "Supply relevant destination interior and bounded free-region" not in text
    assert values["CLIENT_VERSION"] == "agent-speedup-v3-client-10"
    assert values["PROMPT_VERSION"] == "agent-speedup-v3-prompt-10"
    assert text.endswith(values["COMPACT_OUTPUT_INSTRUCTIONS"])


def test_sensor_and_goal_boundary_survives_prompt_reduction():
    text = constants()["AGENT_INSTRUCTIONS"]
    for phrase in ("No simulator object poses", "model_visual_estimate", "uncertainty_m",
                   "anchor_local_m", "local_pose_revision", "u=cx+fx*X/(-Z)",
                   "do not add that offset", "accepted outputs", "opaque reasoning",
                   "acquire_evidence", "evidence_acquisition_failed", "independent evaluator",
                   "scene_update and motion_context in this SAME response"):
        assert phrase.lower() in " ".join(text.lower().split())
    assert "libero" not in text.lower()
    assert "robocasa" not in text.lower()
