"""TH-EXT (contract §C4.9): planned control modules in CONTROL_MODULES, per-module LABELS merged by AST, the
model-interface lint class, orphan labels, the function-body literal audit, the spec §5.2 threshold-id register, and
TH-EXT-6 (lead decision L9): the firewall rules of ``firewall_violations`` inside ``lint_module``.
Runs in every wave against whatever v2 modules exist (``require_all=False``); BI's Wave-3 file adds require_all."""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path

import pytest

from gpt_field import agent_speedup_v2_thresholds as th

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
PACKAGE = th.PACKAGE_DIR
# Contract §C16: every spec §5.2 / §4.4 id the slice names (in the slice, recorded, reserved or test-only).
C16_IDS = (
    [f"R{i}" for i in range(1, 15)]
    + ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G12", "G13"]
    + [f"D{i}" for i in range(1, 16)]
    + [f"P{i}" for i in range(1, 18)]
    + [f"M{i}" for i in range(1, 19)]
    + ["K1w", "K2", "K3", "K4w", "K5", "K6", "K7", "K8", "K9", "K10", "K11"]
    + [f"O{i}" for i in range(1, 9)] + [f"V{i}" for i in range(1, 8)] + [f"L{i}" for i in range(1, 6)]
    + [f"B{i}" for i in range(1, 6)])
RESERVED = {"G10", "D6", "D12", "D13", "M9", "M10", "M15", "M18", "K8", "K10", "O6", "L2", "L3", "L4", "L5"}


@pytest.fixture(scope="module")
def vocab():
    if not Path(LIBERO, "libero/libero/bddl_files").is_dir():
        pytest.fail("LIBERO checkout required for AC-1(c) (the T0 lint vocabulary)")
    return th.lint_vocabulary(LIBERO)["vocabulary"]


def plant(tmp_path, source, name="planted.py"):
    path = tmp_path / name
    path.write_text(source)
    return path


def rules(found):
    return {(v["rule"], v.get("name") or v.get("text")) for v in found}


# ------------------------------------------------------------------------------------------------ TH-EXT-1
def test_version_and_module_lists():
    assert th.THRESHOLDS_VERSION == "agent-speedup-v2-thresholds-v8"          # v8: decision J §4 + gap round §6
    assert th.M0_CONTROL_MODULES == ("agent_speedup_v2_robot.py", "agent_speedup_v2_thresholds.py")
    assert th.PLANNED_CONTROL_MODULES == (
        "agent_speedup_v2_types.py", "agent_speedup_v2_geometry.py", "agent_speedup_v2_scene.py",
        "agent_speedup_v2_planner.py", "agent_speedup_v2_primitives.py", "agent_speedup_v2_observers.py",
        "agent_speedup_v2_relations.py", "agent_speedup_v2_plan.py", "agent_speedup_v2.py")
    assert th.MODEL_INTERFACE_MODULES == ("agent_speedup_v2_client.py",)
    assert "agent_speedup_v2_video.py" not in th.PLANNED_CONTROL_MODULES + th.MODEL_INTERFACE_MODULES
    present = tuple(name for name in th.PLANNED_CONTROL_MODULES if (PACKAGE / name).is_file())
    assert th.CONTROL_MODULES == th.M0_CONTROL_MODULES + present
    assert "agent_speedup_v2_types.py" in th.CONTROL_MODULES              # the kernel exists from Wave 0
    assert th.PRESENT_INTERFACE_MODULES == tuple(n for n in th.MODEL_INTERFACE_MODULES if (PACKAGE / n).is_file())


def test_lint_control_modules_passes_in_every_wave():
    report = th.lint_control_modules(LIBERO)
    assert report["passed"], (report["modules"], report["interface_modules"])
    assert set(report["modules"]) == set(th.CONTROL_MODULES)
    assert set(report["interface_modules"]) == set(th.PRESENT_INTERFACE_MODULES)
    assert set(report["missing"]) == set(th.PLANNED_CONTROL_MODULES + th.MODEL_INTERFACE_MODULES) - set(
        th.CONTROL_MODULES) - set(th.PRESENT_INTERFACE_MODULES)
    assert th.LABEL_MERGE_ERRORS == {}
    strict = th.lint_control_modules(LIBERO, require_all=True)
    assert strict["passed"] is (not strict["missing"]) and strict["require_all"] is True


# ------------------------------------------------------------------------------------------------ TH-EXT-2
LABELLED = '''import math
LABELS = {"GAP_FLOOR_MM": ("DD", "planted floor"),
          "Cfg.SIZE_PX": ("ROBOT", "planted image size"),
          "guard.limit": ("DD", "planted default"
                                " (adjacent literals are one string)"),
          "RATE": ("ONLINE", "planted rate")}
GAP_FLOOR_MM = 0.5
RATE = math.radians(3.0)
NAME = "a string constant needs no label"


class Cfg:
    SIZE_PX = 512


def guard(x, limit=0.25):
    return x > limit
'''


def test_module_labels_three_key_forms(tmp_path):
    path = plant(tmp_path, LABELLED)
    labels = th.module_labels(path)
    assert set(labels) == {"planted.GAP_FLOOR_MM", "planted.Cfg.SIZE_PX", "planted.guard.limit", "planted.RATE"}
    assert labels["planted.guard.limit"] == {"label": "DD", "derivation": "planted default (adjacent literals are "
                                                                          "one string)"}
    assert th.module_labels(path, module_name="other")["other.RATE"]["label"] == "ONLINE"
    assert th.local_labels(ast.parse(LABELLED))["Cfg.SIZE_PX"] == ("ROBOT", "planted image size")
    assert th.module_labels(plant(tmp_path, "X = 'no labels'\n", "bare.py")) == {}


def test_lint_module_uses_own_labels(tmp_path, vocab):
    path = plant(tmp_path, LABELLED)
    assert th.lint_module(path, vocab) == []                     # labels=None: CONSTANT_LABELS + the module's own
    found = th.lint_module(path, vocab, labels={})                # explicit labels are used as given (M0 semantics)
    assert rules(found) >= {("unlabelled_constant", "planted.GAP_FLOOR_MM"), ("unlabelled_constant", "planted.RATE"),
                            ("unlabelled_constant", "planted.Cfg.SIZE_PX"), ("numeric_default", "planted.guard.limit")}


@pytest.mark.parametrize("source", [
    'X = 1.0\nLABELS = {"X": ("DD", f"{X}")}\n',                         # f-string
    'D = "d"\nX = 1.0\nLABELS = {"X": ("DD", D)}\n',                      # a name
    'X = 1.0\nLABELS = {"X": ("DD", "a" + "b")}\n',                       # an operator
    'X = 1.0\nOTHER = {}\nLABELS = {**OTHER, "X": ("DD", "d")}\n',          # unpacking
    'X = 1.0\nLABELS = dict(X=("DD", "d"))\n',                            # a call
    'X = 1.0\nLABELS = {"X": ("DD", "d"), "X": ("DD", "e")}\n',            # a key twice
    'X = 1.0\nLABELS = {"X": ("TUNED", "d")}\n',                          # a label class that does not exist
    'X = 1.0\nLABELS = {"X": ("DD", "  ")}\n',                            # an empty derivation
    'X = 1.0\nLABELS = {"X": "DD"}\n',                                    # not a pair
    'X = 1.0\nLABELS = {"X Y": ("DD", "d")}\n',                           # not a key form
    'X = 1.0\nLABELS = {"<lambda>": ("DD", "d")}\n',                      # a lambda is not a constant
    'X = 1.0\nLABELS = {"f.<lambda>": ("DD", "d")}\n',                    # the last component names the argument
])
def test_non_literal_or_bad_labels_raise(tmp_path, vocab, source):
    path = plant(tmp_path, source)
    with pytest.raises(ValueError):
        th.module_labels(path)
    found = th.lint_module(path, vocab)
    assert any(v["rule"] == "invalid_labels" for v in found)
    assert ("unlabelled_constant", "planted.X") in rules(found)          # an invalid register labels nothing


def test_lambda_default_key_form(tmp_path, vocab):
    """A lambda default is keyed "<lambda>.arg", the form numeric_defaults reports; labelling it keeps the register
    valid and the module clean, while a nested-path key for it is an orphan (M0 keys lambdas by their bare name)."""
    path = plant(tmp_path, 'LABELS = {"<lambda>.z": ("DD", "planted lambda default"), "F": ("DD", "planted")}\n'
                           'F = (lambda y, z=0.75: y + z)\n')
    assert th.module_labels(path) == {"planted.<lambda>.z": {"label": "DD", "derivation": "planted lambda default"},
                                      "planted.F": {"label": "DD", "derivation": "planted"}}
    assert th.lint_module(path, vocab) == []
    nested = plant(tmp_path, 'LABELS = {"f.<lambda>.z": ("DD", "x")}\n\n\ndef f():\n    return (lambda z=0.5: z)\n',
                   "nested.py")
    found = rules(th.lint_module(nested, vocab))
    assert ("orphan_label", "nested.f.<lambda>.z") in found and ("numeric_default", "nested.<lambda>.z") in found
    assert not any(rule == "invalid_labels" for rule, _ in found)


def test_merge_into_constant_labels(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    for name in th.M0_CONTROL_MODULES:
        shutil.copy(PACKAGE / name, package / name)
    (package / "agent_speedup_v2_types.py").write_text('LABELS = {"A_M": ("DD", "a")}\nA_M = 1.0\n')
    (package / "agent_speedup_v2_client.py").write_text('LABELS = {"B_PX": ("DD", "b")}\nB_PX = 2\n')
    merged = th.merged_labels(package)
    assert merged["agent_speedup_v2_types.A_M"] == {"label": "DD", "derivation": "a"}
    assert merged["agent_speedup_v2_client.B_PX"]["derivation"] == "b"
    assert set(th.CENTRAL_LABELS) <= set(merged)
    (package / "agent_speedup_v2_scene.py").write_text('LABELS = {"A_M": ("DD", f"x")}\nA_M = 1.0\n')
    with pytest.raises(ValueError):
        th.merged_labels(package)
    labels, errors = th._collect_labels(package)
    assert set(errors) == {"agent_speedup_v2_scene.py"} and "agent_speedup_v2_types.A_M" in labels
    assert not any(key.startswith("agent_speedup_v2_scene.") for key in labels)


def test_incomplete_register_never_reaches_a_manifest(monkeypatch):
    """A module whose LABELS could not be merged makes the register and its hash raise (the run manifest reads both),
    instead of silently hashing a register without that module."""
    assert th.constants_register() and len(th.constants_sha256()) == 64
    monkeypatch.setitem(th.LABEL_MERGE_ERRORS, "agent_speedup_v2_scene.py", "ValueError: planted")
    with pytest.raises(ValueError, match="agent_speedup_v2_scene.py"):
        th.constants_register()
    with pytest.raises(ValueError, match="incomplete"):
        th.constants_sha256()
    monkeypatch.delitem(th.LABEL_MERGE_ERRORS, "agent_speedup_v2_scene.py")
    assert len(th.constants_sha256()) == 64


def test_import_time_merge_covers_existing_modules():
    for name in th.PLANNED_CONTROL_MODULES + th.MODEL_INTERFACE_MODULES:
        path = PACKAGE / name
        if path.is_file():
            for key, entry in th.module_labels(path).items():
                assert th.CONSTANT_LABELS[key] == entry, key
    register = th.constants_register()
    assert any(key.startswith("agent_speedup_v2_types.") for key in register)
    assert set(th.CENTRAL_LABELS) <= set(th.CONSTANT_LABELS)


def test_central_labels_have_no_orphans():
    """M0's central entries each name a numeric constant of their module (the TH-EXT-4 rule for the register)."""
    for name in th.M0_CONTROL_MODULES:
        stem = Path(name).stem
        tree = ast.parse((PACKAGE / name).read_text())
        numerics = set(th.module_numeric_constants(tree)) | set(th.numeric_defaults(tree))
        for key in th.CENTRAL_LABELS:
            if key.startswith(stem + "."):
                assert key[len(stem) + 1:] in numerics, key


# ------------------------------------------------------------------------------------------------ TH-EXT-3
TASK_TEXT = '''from gpt_field.agent_speedup_v2_truth import Scene
LABELS = {"LIMIT": ("DD", "planted")}
LIMIT = 3


def build(task_prompt):
    return task_prompt.strip()
'''


def test_allow_task_text_skips_only_that_rule(tmp_path, vocab):
    path = plant(tmp_path, TASK_TEXT)
    strict = rules(th.lint_module(path, vocab))
    assert ("task_text_identifier", "task_prompt") in strict and any(r == "forbidden_import" for r, _ in strict)
    relaxed = rules(th.lint_module(path, vocab, allow_task_text=True))
    assert not any(r == "task_text_identifier" for r, _ in relaxed)
    assert any(r == "forbidden_import" for r, _ in relaxed)               # every other rule still applies
    path.write_text('def build(task_prompt):\n    return "the moka pot"\n')
    assert any(v["rule"] == "lint_vocabulary" for v in th.lint_module(path, vocab, allow_task_text=True))


def test_lint_control_modules_on_a_planted_package(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    for name in th.M0_CONTROL_MODULES:
        shutil.copy(PACKAGE / name, package / name)
    (package / "agent_speedup_v2_types.py").write_text('LABELS = {"A_M": ("DD", "a")}\nA_M = 1.0\n')
    (package / "agent_speedup_v2_client.py").write_text(
        'LABELS = {"B_PX": ("DD", "b")}\nB_PX = 2\n\n\ndef make(task_prompt):\n    return task_prompt\n')
    report = th.lint_control_modules(LIBERO, package_dir=package)
    assert report["passed"], report
    assert set(report["modules"]) == set(th.M0_CONTROL_MODULES) | {"agent_speedup_v2_types.py"}
    assert set(report["interface_modules"]) == {"agent_speedup_v2_client.py"}
    assert "agent_speedup_v2_geometry.py" in report["missing"] and "agent_speedup_v2_client.py" not in report["missing"]
    assert not th.lint_control_modules(LIBERO, package_dir=package, require_all=True)["passed"]
    # the client is still linted for vocabulary and imports
    (package / "agent_speedup_v2_client.py").write_text('import gpt_field.agent_speedup_v1_client as old\n')
    report = th.lint_control_modules(LIBERO, package_dir=package)
    assert not report["passed"] and report["interface_modules"]["agent_speedup_v2_client.py"]
    # a planted control module with an unlabelled constant and a task-text identifier fails
    (package / "agent_speedup_v2_client.py").unlink()
    (package / "agent_speedup_v2_plan.py").write_text('K = 4\n\n\ndef f(task_prompt):\n    return 1\n')
    report = th.lint_control_modules(LIBERO, package_dir=package)
    found = rules(report["modules"]["agent_speedup_v2_plan.py"])
    assert ("unlabelled_constant", "agent_speedup_v2_plan.K") in found
    assert ("task_text_identifier", "task_prompt") in found
    # an invalid LABELS register is a label-merge failure, reported under its module
    (package / "agent_speedup_v2_plan.py").write_text('LABELS = {"K": ("DD", f"x")}\nK = 4\n')
    report = th.lint_control_modules(LIBERO, package_dir=package)
    assert not report["passed"]
    assert {"invalid_labels", "label_merge"} <= {v["rule"] for v in report["modules"]["agent_speedup_v2_plan.py"]}
    # full set present: require_all passes
    for name in th.PLANNED_CONTROL_MODULES + th.MODEL_INTERFACE_MODULES:
        if not (package / name).exists():
            (package / name).write_text('"""planted"""\n')
    (package / "agent_speedup_v2_plan.py").write_text('"""planted"""\n')
    report = th.lint_control_modules(LIBERO, package_dir=package, require_all=True)
    assert report["passed"] and report["missing"] == []


# ------------------------------------------------------------------------------------------------ TH-EXT-4
def test_orphan_labels(tmp_path, vocab):
    path = plant(tmp_path, 'LABELS = {"GONE_MM": ("DD", "x"), "REASON": ("DD", "a string is never labelled"),\n'
                           '          "Cfg.N": ("DD", "n"), "f.k": ("DD", "k"), "f.missing": ("DD", "m")}\n'
                           'REASON = "free_sense"\n\n\nclass Cfg:\n    N = 2\n\n\ndef f(k=3):\n    return k\n')
    found = rules(th.lint_module(path, vocab))
    assert ("orphan_label", "planted.GONE_MM") in found and ("orphan_label", "planted.REASON") in found
    assert ("orphan_label", "planted.f.missing") in found
    assert ("orphan_label", "planted.Cfg.N") not in found and ("orphan_label", "planted.f.k") not in found


# ------------------------------------------------------------------------------------------------ TH-EXT-5
AUDIT = '''import math
LABELS = {"SIZE_PX": ("ROBOT", "s"), "Cfg.SIZE_PX": ("ROBOT", "s"), "f.margin": ("DD", "m")}
SIZE_PX = 512
TOP = 0.123


class Cfg:
    SIZE_PX = 512
    NAMED = SIZE_PX


def f(x, margin=0.0, *, scale=None):
    a = x * 0.5 + 1.0 - 1e-9 + 1000.0 * 2.0 - 1.0
    b = x * 0.25
    c = -0.5 * x - (0.5 * x) + x * -2.0 - 1e-9 + -1000.0 - x ** -1e-12
    d = -0.25 * x + -(0.25 * x)
    n = 3 + int(x)
    g = lambda y, z=0.75: y + 0.0
    return math.sqrt(a + b + c + d + n) + g(2.0)
'''


def test_literal_audit_planted(tmp_path, vocab):
    path = plant(tmp_path, AUDIT)
    found = th.literal_audit(path)
    floats = sorted(v["value"] for v in found if v["rule"] == "float_literal")
    # the magnitude decides: negated structural values pass, -0.25 and -(0.25 x) are both flagged; 0.123 is module
    # level (the label rule's business); 3 is an int
    assert floats == [-0.25, 0.25, 0.25]
    named = {(v["where"], v["name"], v["value"]) for v in found if v["rule"] == "use_named_constant"}
    assert ("default", "f.margin", 0.0) in named and ("class_body", "Cfg", 512) in named
    assert ("default", "<lambda>.z", 0.75) in named
    assert all(v["line"] > 0 for v in found)
    # the M0 rules accept labelled defaults and class-body literals (only TOP and the lambda default are unlabelled)
    assert {v["name"] for v in th.lint_module(path, vocab) if v["rule"] in ("unlabelled_constant", "numeric_default")
            } == {"planted.TOP", "planted.<lambda>.z"}
    # ... but the audit still asks for named constants there (contract §C1.5 item 2)
    clean = plant(tmp_path, 'LABELS = {"W_PX": ("ROBOT", "w")}\nW_PX = 512\n\n\n'
                            'class Cam:\n    size_px: int = W_PX\n\n\n'
                            'def g(x, k=None, *, n=W_PX):\n    return x * 2.0 + (0.0 if k is None else k)\n',
                  "clean.py")
    assert th.literal_audit(clean) == [] and th.lint_module(clean, vocab) == []


@pytest.mark.parametrize("source, values", [
    ("try:\n    import cv2\n    GAIN = 0.37\nexcept ImportError:\n    GAIN = 0.41\n", [0.37, 0.41]),
    ("import os\nif os.name:\n    TOL_M = 0.013\n", [0.013]),
    ("TABLE = {}\nfor i in range(3):\n    TABLE[i] = 0.77 * i\n", [0.77, 3]),
    ("with open('x') as f:\n    LIMIT = 4\n", [4]),
    ("while False:\n    STEP = 0.2\n", [0.2]),
    ("X = None\nX += 0.1\n", [0.1]),                                     # augmented assignment
    ("TABLE = {}\nTABLE['k'] = 0.5\n", [0.5]),                            # subscript target
    ("A, *B = (1.5, 2.5, 3.5)\n", [1.5, 2.5, 3.5]),                       # starred unpacking
    ("print(0.25)\n", [0.25]),                                            # expression statement
])
def test_literal_audit_flags_module_compound_statements(tmp_path, source, values):
    """Numbers the label rule cannot see (it reads plain top-level name assignments only) are audit findings."""
    path = plant(tmp_path, source)
    found = th.literal_audit(path)
    assert sorted(v["value"] for v in found if v["rule"] == "module_compound_literal") == sorted(values)
    assert all(v["where"] == "module_statement" and v["line"] > 0 for v in found)


def test_literal_audit_nested_definitions_in_compound_statements(tmp_path):
    source = ("try:\n    import cv2\n\n    def f(x, k=None):\n        return x * 0.3 + 0.5\n\n"
              "    class C:\n        SIZE = 64\nexcept ImportError:\n    cv2 = None\n")
    found = {(v["rule"], v["value"]) for v in th.literal_audit(plant(tmp_path, source))}
    assert found == {("float_literal", 0.3), ("use_named_constant", 64)}
    clean = 'LABELS = {"A_M": ("DD", "a"), "B": ("DD", "b")}\nA_M = 0.25\nB, C = 1, "c"\n\n\ndef g():\n    return A_M\n'
    assert th.literal_audit(plant(tmp_path, clean, "clean.py")) == []


def test_structural_floats_constant():
    assert th.STRUCTURAL_FLOATS == (0.0, 1.0, -1.0, 0.5, 2.0, 1000.0, 1e-6, 1e-9, 1e-12)
    assert "agent_speedup_v2_thresholds.STRUCTURAL_FLOATS" in th.CONSTANT_LABELS


def test_literal_audit_empty_for_existing_control_and_interface_modules():
    for name in th.CONTROL_MODULES + th.PRESENT_INTERFACE_MODULES:
        assert th.literal_audit(PACKAGE / name) == [], name


# ------------------------------------------------------------------------------------------------ TH-EXT-6 (L9)
FIREWALL_PLANTED = [
    ("import importlib\nM = importlib.import_module('gpt_field.x')\n", "dynamic_import"),
    ("import importlib.util\ndef f(p):\n    return importlib.util.spec_from_file_location('m', p)\n",
     "dynamic_import"),
    ("def f():\n    return __import__('x')\n", "dynamic_import"),
    ("import sys\ndef f():\n    return sys.modules\n", "dynamic_import"),
    ("import builtins\n", "dynamic_import"),
    ("def f(s):\n    return eval(s)\n", "dynamic_code"),
    ("def f(s):\n    exec(s)\n", "dynamic_code"),
    ("def f(s):\n    return compile(s, 'f', 'exec')\n", "dynamic_code"),
    ("def f(sim):\n    return getattr(sim, 'truth')\n", "evaluator_name"),
    ("def f(sim):\n    return sim.settle_truth\n", "evaluator_name"),
    ("def f(world):\n    return world.goal_satisfied()\n", "evaluator_name"),
    ("def f(sim):\n    return sim.answer_key()\n", "evaluator_name"),
    ("def f(a):\n    return a.answer('plan', {})\n", "scripted_answer_call"),
    ("PATH = 'x/bddl_files/y.bddl'\n", "bddl_access"),
    ("from . import agent_speedup_v2_thresholds as th\ndef f(r):\n    return th.lint_vocabulary(r)\n", "lint_reader"),
    ("from .agent_speedup_v2_thresholds import bddl_vocabulary\n", "lint_reader"),
    ("def f():\n    from .agent_speedup_v2_geometry import structure_axis\n    return structure_axis\n", "lazy_import"),
]


@pytest.mark.parametrize("source, rule", FIREWALL_PLANTED)
def test_lint_module_reports_the_firewall_rules(tmp_path, vocab, source, rule):
    found = th.lint_module(plant(tmp_path, source), vocab, labels={})
    assert rule in {v["rule"] for v in found}, found
    assert all(v["line"] > 0 for v in found if v["rule"] == rule)


def test_firewall_rules_fail_lint_control_modules(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    for name in th.M0_CONTROL_MODULES:
        shutil.copy(PACKAGE / name, package / name)
    scene = package / "agent_speedup_v2_scene.py"
    scene.write_text("def f(sim):\n    return getattr(sim, 'oracle')\n")
    report = th.lint_control_modules(LIBERO, package_dir=package)
    assert not report["passed"]
    assert ("evaluator_name", "oracle") in rules(report["modules"]["agent_speedup_v2_scene.py"])
    scene.write_text("def f():\n    from . import agent_speedup_v2_geometry as geometry\n    return geometry\n")
    assert th.lint_control_modules(LIBERO, package_dir=package)["passed"]
    transport = ("class ScriptedV2Transport:\n    def __init__(self, answers):\n        self.answers = answers\n\n"
                 "    def send(self, kind, meta):\n        return self.answers.answer(kind, meta)\n")
    (package / "agent_speedup_v2_client.py").write_text(transport)          # the MIM's scripted transport
    assert th.lint_control_modules(LIBERO, package_dir=package)["passed"]
    scene.write_text(transport)                                            # ... is the ORACLE protocol in a CM
    report = th.lint_control_modules(LIBERO, package_dir=package)
    assert ("scripted_answer_call", "answer") in rules(report["modules"]["agent_speedup_v2_scene.py"])


def test_firewall_rule_exemptions():
    """The lint's own module defines the banned names and reads BDDL; M0's robot keeps its function-level import."""
    own = (PACKAGE / "agent_speedup_v2_thresholds.py").read_text()
    assert th.firewall_violations(ast.parse(own), "agent_speedup_v2_thresholds") == []
    found = {v["rule"] for v in th.firewall_violations(ast.parse(own), "agent_speedup_v2_scene")}
    assert {"lint_reader", "evaluator_name", "dynamic_import", "bddl_access"} <= found    # the same text elsewhere
    lazy = ast.parse("def f():\n    from .agent_speedup_v2_thresholds import CONSTANT_LABELS\n")
    assert th.firewall_violations(lazy, "agent_speedup_v2_robot") == []
    assert {v["rule"] for v in th.firewall_violations(lazy, "agent_speedup_v2_scene")} == {"lazy_import"}
    exact = ast.parse(f"def f():\n    {th.LAZY_IMPORT_STATEMENT}\n    return geometry\n")
    assert th.firewall_violations(exact, "agent_speedup_v2_planner") == []
    assert th.firewall_violations(ast.parse(th.LAZY_IMPORT_STATEMENT), "agent_speedup_v2_relations") == []


def test_firewall_rule_has_no_false_positives_on_generic_code(tmp_path, vocab):
    path = plant(tmp_path, (
        '"""Reads no evaluator truth, no ORACLE and no BDDL name; never runs text as code."""\n'
        "import ast\nimport re\nimport sys\n\nfrom . import agent_speedup_v2_thresholds as th\n"
        "PATTERN = re.compile('x')\nEPS = sys.float_info.epsilon\n\n\n"
        "def summary(o, s, ctx):\n    for m in ('to_dict', 'summary'):\n        call = getattr(o, m, None)\n"
        "    object.__setattr__(o, 'value', call)\n"
        "    return ast.literal_eval(s), th.tokens(s), th.constants_sha256(), ctx.answer, o.__dict__\n"))
    assert th.lint_module(path, vocab, labels={}) == []


def test_firewall_definition_strings_pass_the_vocabulary_lint(vocab):
    """The new lists are themselves linted (strings scanned), in the lint's own module."""
    found = th.lint_module(PACKAGE / "agent_speedup_v2_thresholds.py", vocab)
    assert found == [], found
    assert set(th.FIREWALL_DEFINITIONS) >= {"DYNAMIC_IMPORT_NAMES", "EVALUATOR_ONLY_NAMES", "LINT_READERS"}


# ------------------------------------------------------------------------------------------------ threshold ids
def test_threshold_ids_cover_contract_c16():
    assert set(th.THRESHOLD_IDS) == set(C16_IDS) and len(C16_IDS) == len(set(C16_IDS))
    modules = {Path(n).stem for n in th.M0_CONTROL_MODULES + th.PLANNED_CONTROL_MODULES + th.MODEL_INTERFACE_MODULES}
    for key, entry in th.THRESHOLD_IDS.items():
        assert set(entry) == {"owners", "names", "labels", "slice", "rule"}, key
        assert entry["slice"] in th.SLICE_STATUSES and entry["rule"].strip(), key
        assert all(label in th.LABEL_CLASSES for label in entry["labels"]), key
        if key in RESERVED:
            assert entry["slice"] == "reserved" and not entry["owners"] and not entry["names"], key
            continue
        assert entry["owners"], key
        assert all(owner in modules or owner in ("evaluate_agent_speedup_v2", "test") for owner in entry["owners"]), key
        if entry["slice"] in ("yes", "record") and key != "P13":
            assert entry["labels"], key
        if not key.startswith("R"):
            for name in entry["names"]:
                assert name.startswith(key + "_") and th.threshold_id(name) == key, (key, name)
    assert th.THRESHOLD_IDS["V7"]["slice"] == "test" and th.THRESHOLD_IDS["G9"]["slice"] == "record"
    robot_names = {t.name for t in __import__("gpt_field.agent_speedup_v2_robot", fromlist=["x"]).robot_thresholds()}
    for key in (f"R{i}" for i in range(1, 15)):
        assert set(th.THRESHOLD_IDS[key]["names"]) <= robot_names, key
    assert len(th.threshold_ids_sha256()) == 64


def test_threshold_id_parser():
    assert th.threshold_id("K1W_DECIDE_STEP") == "K1w" and th.threshold_id("K4w_stop_rate_mm") == "K4w"
    assert th.threshold_id("G5_FLOOR_MM") == "G5" and th.threshold_id("G13_width_band_mm") == "G13"
    assert th.threshold_id("P15_LATERAL_CAP_M") == "P15" and th.threshold_id("M14_IOU_MIN") == "M14"
    for name in ("HF_CELL_M", "DEST_SEPARATION_FRACTION", "MARK_COUNT", "G11_X", "K1_X", "YAW_GRID_RAD", "B1X"):
        assert th.threshold_id(name) is None, name


# A spec-id-shaped prefix: letter, a number 1-99 (no §5.2 id is numbered 0), optional "w", underscore.
ID_PREFIX = re.compile(r"[RGDPMKOVLB][1-9]\d?W?_", re.IGNORECASE)


def test_labelled_constants_with_an_id_prefix_name_a_known_id():
    """Catches id typos (G11_..., K1_... for K1w). Names that start with a 0-numbered token (R0_ for the hand axis
    R0, M0_, B0_) or spell an axis out (JAW_, APPROACH_) are never read as ids; R1 / R2 prefixes parse as the robot
    facts R1 / R2 whatever they mean in the name."""
    for key in th.CONSTANT_LABELS:
        leaf = key.split(".")[-1]
        if ID_PREFIX.match(leaf):
            assert th.threshold_id(leaf) is not None, key


def test_id_prefix_rule_scope():
    for name in ("R0_TWIN_SCORE", "M0_COMPAT", "B0_X", "JAW_MARGIN_M", "HF_CELL_M", "R12X"):
        assert not ID_PREFIX.match(name), name
    for name in ("G11_X", "K1_DECIDE_STEP", "G5_FLOOR_MM", "K1W_DECIDE_STEP", "R1_MARGIN_M"):
        assert ID_PREFIX.match(name), name
    assert th.threshold_id("R1_MARGIN_M") == "R1" and th.threshold_id("G11_X") is None


# ------------------------------------------------------------- THRESHOLDS_VERSION v6 (design review 1 §3, K-2)
# Decision §3 table: name -> (owner module, label classes the decision gives it).
DECISION_V6 = {
    "FT_PROBE_RADIUS_PX": ("geometry", {"DD"}), "FT_DENSE_STEP_PX": ("geometry", {"DD"}),
    "FT_DENSE_WINDOW_PX": ("geometry", {"DD"}), "TOP_WINDOW_PX": ("scene", {"DD"}),
    "FT_BANDPASS_SIGMAS": ("geometry", {"DD"}), "D15_PROBE_MIN_STD": ("geometry", {"DD"}),
    "V1_PROBE_MIN_PEAK": ("geometry", {"DD"}), "FT_TRACKED_SEARCH_PX": ("geometry", {"DD"}),
    "TOP_GATE_M": ("scene", {"DD"}), "TOP_BAND_K": ("scene", {"DD"}), "TOP_BAND_SIGMA_K": ("scene", {"DD"}),
    "TOP_BAND_SUPPORT_N": ("scene", {"DD"}), "TOP_MODEL_M": ("scene", {"DD"}), "TOP_MODEL_THIN_M": ("scene", {"DD"}),
    "D10_MIN_REGION_CELLS": ("scene", {"DD"}), "PB_REACH_M": ("scene", {"ROBOT", "DD"}),
    "D7_PARENT_SIGMA_CAP_M": ("scene", {"DD"}), "P11_CAP_M": ("planner", {"DD"}),
    "PROBE_STALL_MIN_STEPS": ("primitives", {"DD", "ONLINE"}), "PROBE_FREE_RATIO_FLOOR": ("primitives", {"DD"}),
    "HOVER_LOCATE_M": ("relations", {"DD"}), "SIGMA_VC_M": ("planner", {"DD"}), "D4_MODEL_SIGMA_M": ("geometry", {"DD"}),
}


def test_v6_register_transcribes_decision_section_3(vocab):
    assert set(th.V6_CONSTANTS) == set(DECISION_V6)
    stems = {Path(n).stem for n in th.PLANNED_CONTROL_MODULES}
    for name, entry in th.V6_CONSTANTS.items():
        owner, labels = DECISION_V6[name]
        assert set(entry) == {"owner", "ids", "labels", "decision", "derivation"}, name
        assert entry["owner"] == f"agent_speedup_v2_{owner}" and entry["owner"] in stems, name
        assert set(entry["labels"]) == labels, name
        assert set(entry["ids"]) <= set(th.THRESHOLD_IDS), name
        assert th.threshold_id(name) is None or th.threshold_id(name) in entry["ids"], name    # D15_ -> D15, ...
        assert not set().union(*(th.token_hits(t, set(vocab), set(th.LINT_ALLOWLIST)) for t in th.tokens(name)))
    assert th.V6_CONSTANTS["P11_CAP_M"]["ids"] == ("P11",) and "PROBE_FREE_RATIO_FLOOR" in th.V6_CONSTANTS
    # informational only: no number in the register, and the constants register / its hash ignore it
    tree = ast.parse((PACKAGE / "agent_speedup_v2_thresholds.py").read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) == "V6_CONSTANTS" for t in n.targets))
    assert not any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) and not isinstance(c.value, bool)
                   for c in ast.walk(node))
    assert not any(key.endswith(tuple(th.V6_CONSTANTS)) and key.startswith(th.T) for key in th.constants_register())


def test_v6_amended_rules_name_the_decision():
    rules = {key: entry["rule"] for key, entry in th.THRESHOLD_IDS.items()}
    for key, words in {"D10": ("S-4'", "axis_inconsistent"), "D7": ("RC-9a", "DSW", "PB_REACH_M"),
                       "D15": ("D15_PROBE_MIN_STD",), "V1": ("V1_PROBE_MIN_PEAK",), "P11": ("P11_CAP_M", "first_stage"),
                       "G12": ("PROBE_FREE_RATIO_FLOOR",), "G13": ("G13+",), "L1": ("12.7 px",),
                       "P8": ("landing_measured",), "M1": ("top_upper_z",), "P12": ("top_upper_z",),
                       "O3": ("grip_unstable",), "D4": ("TOP_MODEL_M",)}.items():
        assert "v6" in rules[key] and all(word in rules[key] for word in words), key
    assert "v6" not in rules["P6"]                                       # P-G: P6 unchanged, RC-2 rejected
    assert "DD" in th.THRESHOLD_IDS["G13"]["labels"]                     # G13+ adds the DD floor term


def plant_owner(package, owner, source):
    (package / f"agent_speedup_v2_{owner}.py").write_text(source)


def test_v6_constant_status_reads_the_owner_ast(tmp_path):
    plant_owner(tmp_path, "geometry", (
        "FT_PROBE_RADIUS_PX = 5\nD15_PROBE_MIN_STD = 1.0\nV1_PROBE_MIN_PEAK = 0.9\n"
        "LABELS = {'FT_PROBE_RADIUS_PX': ('DD', 'x'), 'D15_PROBE_MIN_STD': ('ROBOT', 'x')}\n"))
    plant_owner(tmp_path, "scene", "from .x import D8_FLOOR_M\nTOP_MODEL_M = D8_FLOOR_M\nPB_REACH_M = 0.05\n"
                                   "LABELS = {'PB_REACH_M': ('ROBOT', 'R7 + D7')}\n")
    plant_owner(tmp_path, "primitives", "class Leg:\n    PROBE_STALL_MIN_STEPS = 5\n")     # class body: not it
    status = th.v6_constant_status(tmp_path)
    assert set(status) == set(th.V6_CONSTANTS)
    assert {name: status[name]["state"] for name in ("FT_PROBE_RADIUS_PX", "D15_PROBE_MIN_STD", "V1_PROBE_MIN_PEAK",
                                                     "TOP_MODEL_M", "PB_REACH_M", "PROBE_STALL_MIN_STEPS",
                                                     "P11_CAP_M", "HOVER_LOCATE_M", "FT_DENSE_STEP_PX")} == {
        "FT_PROBE_RADIUS_PX": "labelled", "D15_PROBE_MIN_STD": "mislabelled", "V1_PROBE_MIN_PEAK": "unlabelled",
        "TOP_MODEL_M": "alias", "PB_REACH_M": "labelled", "PROBE_STALL_MIN_STEPS": "pending",
        "P11_CAP_M": "module_absent", "HOVER_LOCATE_M": "module_absent", "FT_DENSE_STEP_PX": "pending"}
    assert status["D15_PROBE_MIN_STD"]["label"] == "ROBOT" and status["D15_PROBE_MIN_STD"]["allowed"] == ["DD"]
    assert all(entry["state"] in th.V6_STATES for entry in status.values())
    plant_owner(tmp_path, "geometry", "LABELS = {'FT_PROBE_RADIUS_PX': 3}\nFT_PROBE_RADIUS_PX = 5\n")
    assert th.v6_constant_status(tmp_path)["FT_PROBE_RADIUS_PX"]["state"] == "unlabelled"    # invalid LABELS


def test_v6_constants_present_in_the_owner_modules_carry_the_decided_labels():
    """Wave-by-wave: a §3 name that its owner has defined is labelled with one of the decision's classes (or is an
    alias of another named constant); names not yet defined, or whose owner does not exist yet, are pending."""
    status = th.v6_constant_status()
    wrong = {name: entry for name, entry in status.items() if entry["state"] in ("mislabelled", "unlabelled")}
    assert not wrong, wrong
    assert status["SIGMA_VC_M"]["state"] == "labelled" and status["D4_MODEL_SIGMA_M"]["state"] == "labelled"


# ------------------------------------------------------------- THRESHOLDS_VERSION v7 (design review 2 §3, K-6)
# Decision §3 table: name -> (owner module, label classes, the spec ids whose rule it amends).
DECISION_V7 = {
    "ORIENT_SMOOTH_DEG": ("geometry", {"DD"}, {"D10"}), "OUTLINE_TANGENT_PTS": ("geometry", {"DD"}, {"D10"}),
    "D10_OUTLINE_SHARE_MIN": ("scene", {"DD"}, {"D10"}), "D10_OUTLINE_SECOND_MAX": ("scene", {"DD"}, {"D10"}),
    "D10_EDGE_SHARE_MIN": ("scene", {"DD"}, {"D10"}), "D10_EDGE_TOP_FRACTION": ("geometry", {"DD"}, {"D10"}),
    "D10_EDGE_BLUR_PX": ("geometry", {"DD"}, {"D10"}), "D10_EDGE_REGION_DILATE_PX": ("scene", {"DD"}, {"D10"}),
    "D10_EDGE_HULL_DILATE_PX": ("scene", {"DD"}, {"D10"}), "REGION_COLLAPSE_RATIO": ("scene", {"DD"}, {"D10", "P2"}),
    "COLLAPSE_HULL_DILATE_PX": ("scene", {"DD"}, {"D10", "P2"}), "FT_TRACKED_SEARCH_PX": ("geometry", {"DD"}, set()),
}
RETIRED_V7 = {"D10_MIN_COHERENCE", "D10_MIN_ASPECT", "D10_MIN_REGION_CELLS"}


def test_v7_register_transcribes_decision_section_3(vocab):
    assert set(th.V7_CONSTANTS) == set(DECISION_V7)
    stems = {Path(n).stem for n in th.PLANNED_CONTROL_MODULES}
    for name, entry in th.V7_CONSTANTS.items():
        owner, labels, ids = DECISION_V7[name]
        assert set(entry) == {"owner", "ids", "labels", "decision", "derivation"}, name
        assert entry["owner"] == f"agent_speedup_v2_{owner}" and entry["owner"] in stems, name
        assert set(entry["labels"]) == labels and set(entry["ids"]) == ids, name
        assert set(entry["ids"]) <= set(th.THRESHOLD_IDS), name
        assert th.threshold_id(name) is None or th.threshold_id(name) in entry["ids"], name    # D10_ -> D10
        assert not set().union(*(th.token_hits(t, set(vocab), set(th.LINT_ALLOWLIST)) for t in th.tokens(name)))
    # the opt-in tracked search is restated, not adopted; the v6 register is history and stays as it was
    assert "not adopted" in th.V7_CONSTANTS["FT_TRACKED_SEARCH_PX"]["decision"]
    assert set(th.V6_CONSTANTS) == set(DECISION_V6) and th.V7_STATES == th.V6_STATES
    # retired deciders: informational, scene-owned, never re-registered as v7 constants
    assert set(th.V7_RETIRED_DECIDERS) == RETIRED_V7 and not RETIRED_V7 & set(th.V7_CONSTANTS)
    assert all(entry["owner"] == "agent_speedup_v2_scene" and entry["ids"] == ("D10",) and entry["ruling"].strip()
               for entry in th.V7_RETIRED_DECIDERS.values())
    # informational only: no number in either register, and the constants register / its hash ignore them
    tree = ast.parse((PACKAGE / "agent_speedup_v2_thresholds.py").read_text())
    for register in ("V7_CONSTANTS", "V7_RETIRED_DECIDERS"):
        node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == register for t in n.targets))
        assert not any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float))
                       and not isinstance(c.value, bool) for c in ast.walk(node)), register
    assert not any(key.endswith(tuple(th.V7_CONSTANTS)) and key.startswith(th.T) for key in th.constants_register())
    assert "v7_constant_status" in th.LINT_READERS


def test_v7_amended_rules_name_the_decision():
    rules = {key: entry["rule"] for key, entry in th.THRESHOLD_IDS.items()}
    for key, words in {"D10": ("S-11", "D10-OE", "outline_edges", "D10_OUTLINE_SHARE_MIN", "D10_OUTLINE_SECOND_MAX",
                               "D10_EDGE_SHARE_MIN", "D10_EDGE_REGION_DILATE_PX", "D10_EDGE_HULL_DILATE_PX",
                               "AXIS_AGREE_MAX_RAD", "not_elongated", "outline_gate", "no_edge_agreement",
                               "S-12", "views_disagree", "kept_earlier", "part_moved", "S-13",
                               "REGION_COLLAPSE_RATIO", "region_collapsed"),
                       "D8": ("S-14", "D8_FLOOR_M", "top_inconsistent"),
                       "L1": ("G-8", "V2v", "HL_MIN_BLOB_PX", "PROJECTOR_AFFINE_FRACTION", "L1_FLOOR_PX",
                              "finger_pair_too_few", "finger_pair_one_piece", "finger_pair_inconsistent"),
                       "P2": ("P-I", "width_by_yaw", "geometry_unmeasured")}.items():
        assert "v7" in rules[key] and all(word in rules[key] for word in words), key
    assert "v6" in rules["D10"] and "S-4'" in rules["D10"]                 # the v6 history is kept
    amended = {key for key, rule in rules.items() if "v7" in rule}
    assert amended == {"D8", "D10", "L1", "P2"}, amended


def test_v7_constant_status_reads_the_owner_ast(tmp_path):
    plant_owner(tmp_path, "geometry", (
        "ORIENT_SMOOTH_DEG = 3.0\nOUTLINE_TANGENT_PTS = 4\nD10_EDGE_BLUR_PX = 1.0\n"
        "LABELS = {'ORIENT_SMOOTH_DEG': ('DD', 'x'), 'OUTLINE_TANGENT_PTS': ('ONLINE', 'x')}\n"))
    plant_owner(tmp_path, "scene", "from .x import D10_OUTLINE_SHARE_MIN as S\nD10_EDGE_SHARE_MIN = S\n"
                                   "REGION_COLLAPSE_RATIO = 2.0\nLABELS = {'REGION_COLLAPSE_RATIO': ('DD', 'lens')}\n")
    status = th.v7_constant_status(tmp_path)
    assert set(status) == set(th.V7_CONSTANTS)
    assert {name: status[name]["state"] for name in ("ORIENT_SMOOTH_DEG", "OUTLINE_TANGENT_PTS", "D10_EDGE_BLUR_PX",
                                                     "D10_EDGE_SHARE_MIN", "REGION_COLLAPSE_RATIO",
                                                     "D10_OUTLINE_SHARE_MIN", "COLLAPSE_HULL_DILATE_PX")} == {
        "ORIENT_SMOOTH_DEG": "labelled", "OUTLINE_TANGENT_PTS": "mislabelled", "D10_EDGE_BLUR_PX": "unlabelled",
        "D10_EDGE_SHARE_MIN": "alias", "REGION_COLLAPSE_RATIO": "labelled", "D10_OUTLINE_SHARE_MIN": "pending",
        "COLLAPSE_HULL_DILATE_PX": "pending"}
    assert status["OUTLINE_TANGENT_PTS"]["label"] == "ONLINE" and status["OUTLINE_TANGENT_PTS"]["allowed"] == ["DD"]
    assert all(entry["state"] in th.V7_STATES for entry in status.values())
    empty = tmp_path / "empty"
    empty.mkdir()
    assert {entry["state"] for entry in th.v7_constant_status(empty).values()} == {"module_absent"}
    # the v6 report is unchanged by the shared helper
    assert set(th.v6_constant_status(tmp_path)) == set(th.V6_CONSTANTS)


def test_v7_constants_present_in_the_owner_modules_carry_the_decided_labels():
    """Wave-by-wave (the owners implement decision 2 concurrently): a §3 name that its owner has defined is labelled
    with one of the decision's classes (or is an alias of another named constant); names not yet defined are
    pending. The opt-in FT_TRACKED_SEARCH_PX stays labelled from v6."""
    status = th.v7_constant_status()
    wrong = {name: entry for name, entry in status.items() if entry["state"] in ("mislabelled", "unlabelled")}
    assert not wrong, wrong
    assert status["FT_TRACKED_SEARCH_PX"]["state"] == "labelled"


# ------------------------------- THRESHOLDS_VERSION v8 (grasp-refusal decision J §4 + gap round judge decision §6)
# Decision tables: name -> (owner module, label classes, the spec ids whose rule it amends, alternates).
DECISION_V8 = {
    "OWN_RING_CELLS": ("scene", {"DD"}, {"P5"}, ()), "P5_SIGMA_RESOLUTION_M": ("planner", {"DD"}, {"P5"}, ()),
    "GC_CONTACT_TOL_MM": ("planner", {"DD"}, {"G13"}, ()), "GC_TOUCH_FLOOR_MM": ("primitives", {"DD"}, {"G13"}, ()),
    "GC_TOUCH_SIGMA_K": ("primitives", {"DD"}, {"G13"}, ()), "GC_FIRST_JUDGED_K": ("primitives", {"DD"}, {"G13"}, ()),
    "GC_FREE_MIN_MM": ("primitives", {"DD"}, {"G13"}, ()), "TOUCH_KEEPOUT_DEPTH_M": ("relations", {"DD"}, {"O5"}, ()),
    "HS_P3_GUARD_SIGMA_K": ("planner", {"DD"}, {"P3", "P6"}, ()),
    "PV_PARENT_SIGMA_K": ("planner", {"DD"}, {"P7", "M16"}, ()),
    "CANDIDATE_YAW_TOL_RAD": ("planner", {"DD"}, set(), ()), "V5U_CONE_LINES": ("planner", {"DD"}, {"V5"}, ()),
    "ANNULUS_MIN_ARC_DEG": ("scene", {"DD"}, {"D11"}, ()), "SUBPART_REGION_MAX_RATIO": ("scene", {"RULING"}, set(), ()),
    "REPAIR_IDENT_K": ("plan", {"DD"}, {"B5"}, ("agent_speedup_v2",)),
    "REPAIR_STALL_S": ("plan", {"DD"}, {"B5"}, ("agent_speedup_v2",)),
    "REPAIR_NOCHANGE_R": ("plan", {"DD"}, {"B5"}, ("agent_speedup_v2",)),
    "REPAIR_HISTORY_H": ("plan", {"DD"}, {"B5"}, ("agent_speedup_v2",)),
    "RIM_WALL_RADIUS_FRACTION": ("types", {"DD"}, {"D11"}, ("agent_speedup_v2_planner",)),
    "SIGMA_VC_M": ("planner", {"DD"}, {"P8", "P9"}, ()),
}
AMENDED_V8 = {"P1", "P3", "P4", "P5", "P6", "P7", "G13", "D7", "D8", "D10", "D11", "M8", "M11", "M16", "M17", "O3",
              "O4", "O5", "V5", "B2", "B5"}


def test_v8_register_transcribes_decisions_j4_and_gap_round_6(vocab):
    assert set(th.V8_CONSTANTS) == set(DECISION_V8)
    stems = {Path(n).stem for n in th.PLANNED_CONTROL_MODULES}
    for name, entry in th.V8_CONSTANTS.items():
        owner, labels, ids, alternates = DECISION_V8[name]
        assert set(entry) <= {"owner", "ids", "labels", "decision", "derivation", "alternates"}, name
        assert entry["owner"] == f"agent_speedup_v2_{owner}" and entry["owner"] in stems, name
        assert entry.get("alternates", ()) == alternates and set(alternates) <= stems, name
        assert set(entry["labels"]) == labels and set(entry["ids"]) == ids, name
        assert set(entry["ids"]) <= set(th.THRESHOLD_IDS), name
        assert th.threshold_id(name) is None or th.threshold_id(name) in entry["ids"], name    # P5_ -> P5
        assert entry["decision"].strip() and entry["derivation"].strip(), name
        assert not set().union(*(th.token_hits(t, set(vocab), set(th.LINT_ALLOWLIST)) for t in th.tokens(name)))
    # not adopted (ruling L7) and re-used constants add no name; the earlier registers are history, unchanged
    assert not {"P_BR1", "W_COMP", "M17_FRACTION", "B2_LOCAL_RETRIES"} & set(th.V8_CONSTANTS)
    assert set(th.V6_CONSTANTS) == set(DECISION_V6) and set(th.V7_CONSTANTS) == set(DECISION_V7)
    assert "unchanged" in th.V8_CONSTANTS["SIGMA_VC_M"]["decision"] and "L12" in th.V8_CONSTANTS["SIGMA_VC_M"]["decision"]
    # decision J's code tuple: registered, codes only
    assert set(th.V8_REGISTRIES) == {"GC_REASONS"}
    assert th.V8_REGISTRIES["GC_REASONS"]["codes"] == ("early_finger_contact", "span_inconsistent")
    assert th.V8_STATES == th.V6_STATES + ("registry",)
    # informational only: no number in the registers, and the constants register / its hash ignore them
    tree = ast.parse((PACKAGE / "agent_speedup_v2_thresholds.py").read_text())
    for register in ("V8_CONSTANTS", "V8_REGISTRIES", "V8_RULE_AMENDMENTS"):
        node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == register for t in n.targets))
        assert not any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float))
                       and not isinstance(c.value, bool) for c in ast.walk(node)), register
    assert not any(key.endswith(tuple(th.V8_CONSTANTS)) and key.startswith(th.T) for key in th.constants_register())
    assert "v8_constant_status" in th.LINT_READERS


def test_v8_registry_codes_are_kernel_codes():
    from gpt_field import agent_speedup_v2_primitives as primitives
    from gpt_field import agent_speedup_v2_types as k
    codes = th.V8_REGISTRIES["GC_REASONS"]["codes"]
    assert tuple(primitives.GC_REASONS) == codes
    assert set(codes) <= set(k.PRIMITIVE_REASONS) and set(codes) <= set(k.RELATION_REASONS["held"])
    assert "pose_out_of_envelope" in k.REASONS and "pose_out_of_envelope" not in codes        # PC-1: no primitive


def test_v8_amended_rules_name_the_decision():
    rules = {key: entry["rule"] for key, entry in th.THRESHOLD_IDS.items()}
    assert set(th.V8_RULE_AMENDMENTS) == AMENDED_V8
    amended = {key for key, rule in rules.items() if "; v8 " in rule}
    assert amended == AMENDED_V8, amended
    for key in AMENDED_V8:
        assert rules[key].count("; v8 ") == 1 and rules[key].endswith(th.V8_RULE_AMENDMENTS[key]), key
    for key, words in {"P1": ("PC-1", "pose_out_of_envelope"), "P3": ("HS_P3_GUARD_SIGMA_K",),
                       "P4": ("F6-B", "D7a", "marginal", "P14"), "P5": ("RC-1e", "P5_SIGMA_RESOLUTION_M", "P-J",
                                                                       "OWN_RING_CELLS"),
                       "P6": ("IF-HS", "deficit"), "P7": ("IF-PV", "PV_PARENT_SIGMA_K", "parent_sigma"),
                       "G13": ("GC-1", "GC_TOUCH_FLOOR_MM", "GC_TOUCH_SIGMA_K", "GC_FIRST_JUDGED_K", "GC_FREE_MIN_MM",
                               "GC_CONTACT_TOL_MM", "early_finger_contact", "span_inconsistent"),
                       "D7": ("SAL-R1", "SAL-R4"), "D8": ("PO6", "top_inconsistent", "MC-R2a", "kept_whole"),
                       "D10": ("MC-R2a", "S-12", "window_start"),
                       "D11": ("MC-R2b", "ANNULUS_MIN_ARC_DEG", "RIM_WALL_RADIUS_FRACTION", "S-13"),
                       "M8": ("C-7'", "lateral_unverified"), "M11": ("PR-1", "object_moved_after_release",
                                                                     "post_release_unverified"),
                       "M16": ("IF-R3'", "PV_PARENT_SIGMA_K", "tight_fit", "never a refusal"),
                       "M17": ("HS-H1", "M17_FRACTION", "OWN_RING_CELLS"),
                       "O3": ("HS-H3", "external_unverified", "HS-H2", "external_template_on_robot",
                              "template_on_robot"),
                       "O4": ("HS-H3",), "O5": ("TOUCH_KEEPOUT_DEPTH_M", "F6-A"),
                       "V5": ("V5-U", "V5U_CONE_LINES"),
                       "B2": ("F6-A", "blocked_above_part", "CI-4", "marginal_refuted"),
                       "B5": ("RL-C3", "REPAIR_IDENT_K", "REPAIR_STALL_S", "REPAIR_NOCHANGE_R", "REPAIR_HISTORY_H",
                              "recovery_exhausted", "automatic_cap", "retry_unchanged")}.items():
        assert all(word in rules[key] for word in words), (key, [w for w in words if w not in rules[key]])
    # the v6 / v7 history is kept and not re-labelled
    assert "v6" in rules["P5"] and "v6" not in rules["P6"] and "v7" in rules["D10"]
    assert {key for key, rule in rules.items() if "v7" in rule} == {"D8", "D10", "L1", "P2"}
    # negative: an unknown id or an empty text is refused, and the helper appends once per call
    register = {"P1": {"rule": "base"}}
    th._amend_rules(register, "v9", {"P1": "x"})
    assert register["P1"]["rule"] == "base; v9 x"
    for bad in ({"Z9": "x"}, {"P1": " "}):
        with pytest.raises(ValueError):
            th._amend_rules({"P1": {"rule": "base"}}, "v9", bad)


def test_v8_constant_status_reads_the_owner_ast(tmp_path):
    plant_owner(tmp_path, "planner", (
        "from .x import HF_CELL_M\nP5_SIGMA_RESOLUTION_M = HF_CELL_M\nHS_P3_GUARD_SIGMA_K = 2.0\n"
        "PV_PARENT_SIGMA_K = 1.0\nGC_CONTACT_TOL_MM = 1.0\n"
        "LABELS = {'HS_P3_GUARD_SIGMA_K': ('DD', 'x'), 'PV_PARENT_SIGMA_K': ('ONLINE', 'x')}\n"))
    plant_owner(tmp_path, "scene", "SUBPART_REGION_MAX_RATIO = 1.0\nANNULUS_MIN_ARC_DEG = 90.0\n"
                                   "LABELS = {'SUBPART_REGION_MAX_RATIO': ('DD', 'x'), "
                                   "'ANNULUS_MIN_ARC_DEG': ('DD', 'x')}\n")
    # REPAIR_* sit in the policy (an alternate) in this planted tree, not in the plan module
    (tmp_path / "agent_speedup_v2_plan.py").write_text("X = 'none'\n")
    (tmp_path / "agent_speedup_v2.py").write_text("REPAIR_IDENT_K = 3\nLABELS = {'REPAIR_IDENT_K': ('DD', 'x')}\n")
    plant_owner(tmp_path, "primitives", "GC_REASONS = ('early_finger_contact', 'span_inconsistent')\n")
    status = th.v8_constant_status(tmp_path)
    assert set(status) == set(th.V8_CONSTANTS) | set(th.V8_REGISTRIES)
    assert {name: status[name]["state"] for name in (
        "P5_SIGMA_RESOLUTION_M", "HS_P3_GUARD_SIGMA_K", "PV_PARENT_SIGMA_K", "GC_CONTACT_TOL_MM",
        "SUBPART_REGION_MAX_RATIO", "ANNULUS_MIN_ARC_DEG", "REPAIR_IDENT_K", "REPAIR_STALL_S", "GC_REASONS",
        "GC_TOUCH_FLOOR_MM", "OWN_RING_CELLS", "TOUCH_KEEPOUT_DEPTH_M")} == {
        "P5_SIGMA_RESOLUTION_M": "alias", "HS_P3_GUARD_SIGMA_K": "labelled", "PV_PARENT_SIGMA_K": "mislabelled",
        "GC_CONTACT_TOL_MM": "unlabelled", "SUBPART_REGION_MAX_RATIO": "mislabelled",
        "ANNULUS_MIN_ARC_DEG": "labelled", "REPAIR_IDENT_K": "labelled", "REPAIR_STALL_S": "pending",
        "GC_REASONS": "registry", "GC_TOUCH_FLOOR_MM": "pending", "OWN_RING_CELLS": "pending",
        "TOUCH_KEEPOUT_DEPTH_M": "module_absent"}
    assert status["REPAIR_IDENT_K"]["owner"] == "agent_speedup_v2"                  # found in the alternate
    assert status["REPAIR_STALL_S"]["owner"] == "agent_speedup_v2_plan"              # pending in the owner
    assert status["SUBPART_REGION_MAX_RATIO"]["allowed"] == ["RULING"]
    assert all(entry["state"] in th.V8_STATES for entry in status.values())
    plant_owner(tmp_path, "primitives", "GC_REASONS = 2\n")
    assert th.v8_constant_status(tmp_path)["GC_REASONS"]["state"] == "unlabelled"   # a number is not a registry
    empty = tmp_path / "empty"
    empty.mkdir()
    assert {entry["state"] for entry in th.v8_constant_status(empty).values()} == {"module_absent"}
    # the v6 / v7 reports are unchanged by the shared helper
    assert set(th.v6_constant_status(tmp_path)) == set(th.V6_CONSTANTS)
    assert set(th.v7_constant_status(tmp_path)) == set(th.V7_CONSTANTS)


def test_v8_constants_present_in_the_owner_modules_carry_the_decided_labels():
    """Wave-by-wave (the owners implement the gap round concurrently): a v8 name its owner (or an alternate) has
    defined is labelled with one of the decision's classes or is an alias; J's constants landed with J."""
    status = th.v8_constant_status()
    wrong = {name: entry for name, entry in status.items() if entry["state"] in ("mislabelled", "unlabelled")}
    assert not wrong, wrong
    for name in ("OWN_RING_CELLS", "GC_CONTACT_TOL_MM", "GC_TOUCH_FLOOR_MM", "GC_TOUCH_SIGMA_K", "GC_FIRST_JUDGED_K",
                 "GC_FREE_MIN_MM", "TOUCH_KEEPOUT_DEPTH_M", "SIGMA_VC_M", "RIM_WALL_RADIUS_FRACTION"):
        assert status[name]["state"] == "labelled", (name, status[name])
    assert status["P5_SIGMA_RESOLUTION_M"]["state"] == "alias" and status["GC_REASONS"]["state"] == "registry"
