"""Cross-module firewall (contract §C20.2, §C1.8, §C2.2; spec §1.1, §5.1 item 7). Written in Wave 0 by the kernel
builder; green from Wave 0 on, skipping modules that do not exist yet.

1. Import firewall (AST, function-level imports included): control (CM) and model-interface (MIM) modules never
   reach evaluator truth, the simulator, LIBERO / BDDL, v1's agent modules, scripts, tests or the ORACLE; the v2
   import graph follows §C2.2 and is acyclic; robot facts are read only through ROBOT_FACTS["asset"].
2. AC-1(a) one constants register for every registry key: the run manifest is built for every key (one-step
   plumbing smoke, fake simulator) and carries one constants hash, register, instruction and lint identity;
   3. AC-1(b) constructor signatures and the official sentence; 4. AC-1(c) the T0 lint and the literal audit;
   5. AC-5 schema scan, instruction scan and the client payload scan (preflight payloads of every key: no
   vocabulary outside the official sentence and the echoed part texts; last_outcome = reason, evidence_text,
   thresholds); 6. AC-6 outcome records of the fake end-to-end run; 7. no model call inside a primitive, Align
   without pointing.
8. Firewall hardening (lead decision L9, Wave 0 gate W1), checked here independently of M0's lint rule
   (``th.firewall_violations``, TH-EXT-6) and through it: no CM / MIM loads code by name or runs text as code
   (importlib and relatives, ``__import__``, ``sys.modules``, exec, eval, compile); none names an evaluator-only
   accessor (a call, an attribute read, a getattr string); only the MIM's scripted transport calls the ScriptedAnswers
   (ORACLE) protocol; only the thresholds module names its BDDL / lint readers; a function-level v2 import is exactly
   ``from . import agent_speedup_v2_geometry as geometry``; importing a CM / MIM loads no evaluator module at run
   time. The gate's probes (impl/v2/wave0/gate/probes/firewall_probe.py) are planted negative tests.
ORACLE: unavailable in live mode (worker option, process client, ORACLE constructor, runner import).
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field import agent_speedup_v2_types as kernel

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = th.PACKAGE_DIR
LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
V2 = "agent_speedup_v2"
TYPES, GEOMETRY, SCENE, PLANNER = V2 + "_types", V2 + "_geometry", V2 + "_scene", V2 + "_planner"
PRIMITIVES, OBSERVERS, RELATIONS, PLAN = V2 + "_primitives", V2 + "_observers", V2 + "_relations", V2 + "_plan"
POLICY, CLIENT, ROBOT, THRESHOLDS = V2, V2 + "_client", V2 + "_robot", V2 + "_thresholds"
M0_ALWAYS = {ROBOT, THRESHOLDS}
# §C2.2 allowed v2 imports per module (M0 robot / thresholds are allowed for every planned module).
V2_ALLOWED = {
    TYPES: set(), GEOMETRY: {TYPES}, PRIMITIVES: {TYPES}, PLAN: {TYPES},
    SCENE: {TYPES, GEOMETRY}, OBSERVERS: {TYPES, GEOMETRY}, PLANNER: {TYPES, GEOMETRY},
    RELATIONS: {TYPES, GEOMETRY, SCENE, PLANNER, PRIMITIVES, OBSERVERS},
    POLICY: {TYPES, GEOMETRY, SCENE, OBSERVERS, PLANNER, RELATIONS, PLAN},
    CLIENT: {TYPES, PLAN},
    ROBOT: {THRESHOLDS}, THRESHOLDS: set(),
}
LAZY_ONLY = {(SCENE, GEOMETRY), (OBSERVERS, GEOMETRY), (PLANNER, GEOMETRY)}     # Wave-1 parallelism (§C2.2)
# §C2.2 allowed v1 (pre-v2 package) modules per module, read-only.
V1_ALLOWED = {
    TYPES: {"contact_skills", "action", "wrist_servo"},
    GEOMETRY: {"wrist_servo", "external_axes", "mark_prompting", "contact_skills"},
    PRIMITIVES: {"contact_skills", "speed_first_gripper", "action"},
    PLAN: set(),
    SCENE: {"wrist_servo", "mark_prompting", "contact_skills"},
    OBSERVERS: {"wrist_servo", "contact_skills"},
    PLANNER: set(),
    RELATIONS: {"wrist_servo", "contact_skills", "speed_first_gripper"},
    POLICY: {"contact_skills", "visual_field_v8", "policy"},
    CLIENT: {"hybrid_subscription", "astra", "chatgpt_plan", "ordered_transport", "speed_first_hybrid",
             "direct_history", "mark_prompting"},
    ROBOT: set(), THRESHOLDS: set(),
}
# Never reachable from a CM / MIM: evaluator truth and infrastructure, the simulator stack, BDDL, v1's agent
# modules, scripts (runner, ORACLE, smoke harness, verdict report), tests and fakes, the runner-side video tool.
FORBIDDEN_COMPONENTS = set(th.FORBIDDEN_IMPORT_COMPONENTS) | {
    "robosuite", "mujoco", "bddl", "scripts", "tests", "agent_speedup_v2_oracle", "agent_speedup_v2_relation_smoke",
    "agent_speedup_v2_smoke_scenarios", "agent_speedup_v2_verdict_report", "evaluate_agent_speedup_v2",
    "agent_speedup_v2_fakes", "agent_speedup_fakes", "agent_speedup_v2_video", "agent_speedup_v2_replay",
    "agent_speedup_v2_truth_check"}
FORBIDDEN_PREFIXES = ("agent_speedup_v1", "evaluate_", "test_")
# Attribute calls that only an evaluator-side object answers (simulator / fake truth accessors, truth checker).
TRUTH_CALLS = {"truth", "settle_truth", "answer_key", "oracle", "physical_truth", "goal_satisfied", "body_pose",
               "colour_region_changed", "evaluate_goal", "parse_check", "physical_summary", "heights_truth"}
CLIENT_CALLS = {"request_plan", "request_point", "request_repair", "predict", "build_payload"}
BDDL_STRINGS = (".bddl", "bddl_files", "get_libero_path", "libero/libero")
# 8. L9 hardening (test-side lists; M0's lint keeps its own, compared in test_lint_and_firewall_lists_agree).
DYNAMIC_MODULES = {"importlib", "imp", "runpy", "pkgutil", "zipimport", "builtins"}
DYNAMIC_NAMES = {"__import__", "__builtins__", "__loader__", "__spec__", "__globals__", "__subclasses__", "importlib",
                 "import_module", "spec_from_file_location", "spec_from_loader", "module_from_spec", "exec_module",
                 "find_spec", "find_module", "load_module", "run_path", "run_module"}
CODE_BUILTINS = {"exec", "eval", "compile"}
SYS_LOADER_ATTRIBUTES = {"modules", "path", "meta_path", "path_hooks", "path_importer_cache"}
EVALUATOR_NAMES = TRUTH_CALLS | {"contact_log", "release_and_settle", "failing_checks"}
SIM_EVALUATOR_CALLS = {"geometry"}                  # sim.geometry(); the bare name is the lazy module alias
EVALUATOR_TOKENS = {"truth", "oracle"}
SCRIPTED_ANSWER_CALLS = {"answer"}                  # ScriptedAnswers.answer(kind, meta): the ORACLE protocol
LINT_READERS = {"bddl_vocabulary", "lint_vocabulary", "lint_module", "lint_control_modules", "literal_audit",
                "module_labels", "merged_labels", "local_labels", "firewall_violations", "PART_NOUNS",
                "LINT_ALLOWLIST", "v6_constant_status",          # v6: the K-2 report reads module sources too
                "v7_constant_status",                            # v7: the K-6 report, the same reader
                "v8_constant_status"}                            # v8: decision J §4 + gap round §6, the same reader
LAZY_GEOMETRY = "from . import agent_speedup_v2_geometry as geometry"
DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def existing(names):
    return [name for name in names if (PACKAGE / name).is_file()]


CM_FILES = existing(th.M0_CONTROL_MODULES + th.PLANNED_CONTROL_MODULES)
MIM_FILES = existing(th.MODEL_INTERFACE_MODULES)
ALL_FILES = CM_FILES + MIM_FILES
PLANNED_STEMS = {Path(n).stem for n in th.PLANNED_CONTROL_MODULES + th.MODEL_INTERFACE_MODULES}


def tree_of(name):
    return ast.parse((PACKAGE / name).read_text(), filename=name)


def module_exists(stem):
    return (PACKAGE / f"{stem}.py").is_file()


def imports(tree, nodes=False):
    """[(target, internal, lazy, line)] for every import, function-level ones included. Internal targets are module
    stems of the gpt_field package (relative or absolute imports); external targets are dotted paths. With
    ``nodes`` each row also carries its import statement node."""
    out = []

    def visit(node, lazy):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            lazy = True
        rows = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "gpt_field":
                    if len(parts) > 1:
                        rows.append((parts[1], True, lazy, node.lineno))
                else:
                    rows.append((alias.name, False, lazy, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = module.split(".") if module else []
            if node.level or (parts and parts[0] == "gpt_field"):
                inner = parts if node.level else parts[1:]
                if inner:                                   # from .x import y / from gpt_field.x import y
                    rows.append((inner[0], True, lazy, node.lineno))
                else:                                       # from . import x / from gpt_field import x
                    rows.extend((alias.name, True, lazy, node.lineno) for alias in node.names)
            else:
                rows.append((module, False, lazy, node.lineno))
                rows.extend((f"{module}.{alias.name}", False, lazy, node.lineno) for alias in node.names)
        out.extend(row + (node,) if nodes else row for row in rows)
        for child in ast.iter_child_nodes(node):
            visit(child, lazy)
    visit(tree, False)
    return out


def forbidden(target):
    parts = target.split(".")
    return any(part in FORBIDDEN_COMPONENTS or part.startswith(FORBIDDEN_PREFIXES) or "oracle" in part.lower()
               or "bddl" in part.lower() for part in parts)


# ------------------------------------------------------------------------------------------------ the checks
def forbidden_import_problems(tree):
    return [(target, line) for target, _, _, line in imports(tree) if forbidden(target)]


def graph_problems(stem, tree):
    """§C2.2: allowed v2 edges (geometry lazily for scene / observers / planner) and allowed v1 modules."""
    allowed_v2 = V2_ALLOWED[stem] | (M0_ALWAYS - {stem} if stem in PLANNED_STEMS else set())
    problems = []
    for target, internal, lazy, line in imports(tree):
        if not internal:
            continue
        if target.startswith(V2):
            if target not in allowed_v2:
                problems.append(("v2 edge not allowed", target, line))
            elif (stem, target) in LAZY_ONLY and not lazy:
                problems.append(("geometry must be imported inside a function", target, line))
        elif target not in V1_ALLOWED[stem]:
            problems.append(("v1 module not allowed", target, line))
    return problems


def cycle_in(edges):
    state = {}

    def cyclic(node):
        if state.get(node) == "open":
            return True
        if state.get(node) == "done":
            return False
        state[node] = "open"
        found = any(cyclic(nxt) for nxt in edges.get(node, ()))
        state[node] = "done"
        return found
    return any(cyclic(node) for node in list(edges))


def robot_facts_problems(tree):
    """Every ROBOT_FACTS read is ROBOT_FACTS["asset"] [C-24]."""
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    problems = []
    for node in ast.walk(tree):
        if not ((isinstance(node, ast.Name) and node.id == "ROBOT_FACTS")
                or (isinstance(node, ast.Attribute) and node.attr == "ROBOT_FACTS")):
            continue
        parent = parents.get(id(node))
        if not (isinstance(parent, ast.Subscript) and parent.value is node and isinstance(parent.slice, ast.Constant)
                and parent.slice.value == "asset"):
            problems.append(node.lineno)
    problems += [n.lineno for n in ast.walk(tree) if isinstance(n, ast.alias) and n.name == "ROBOT_FACTS"]
    return problems


def evaluator_problems(tree):
    """BDDL file access, evaluator-only accessor calls and (L9) every other read of an evaluator-only name."""
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    problems = [("bddl", s[:60]) for s in strings if any(marker in s for marker in BDDL_STRINGS)]
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    return (problems + [("truth call", c) for c in sorted(calls & (TRUTH_CALLS | SIM_EVALUATOR_CALLS))]
            + evaluator_name_problems(tree))


def names_in(tree, strings=True):
    """(name, line, kind) for every identifier, attribute, imported path component and, with ``strings``, every
    component of a dotted identifier-shaped string constant (``getattr(sim, "truth")``, ``"gpt_field.x_truth"``)."""
    out = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            out.append((node.id, line, "name"))
        elif isinstance(node, ast.Attribute):
            out.append((node.attr, line, "attribute"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append((node.name, line, "name"))
        elif isinstance(node, ast.arg):
            out.append((node.arg, line, "name"))
        elif isinstance(node, ast.keyword) and node.arg:
            out.append((node.arg, line, "name"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.extend((part, line, "import") for part in node.module.split("."))
        elif isinstance(node, ast.alias):
            out.extend((part, line, "import") for part in node.name.split("."))
            if node.asname:
                out.append((node.asname, line, "name"))
        elif strings and isinstance(node, ast.Constant) and isinstance(node.value, str) and DOTTED_NAME.fullmatch(
                node.value):
            out.extend((part, line, "string") for part in node.value.split("."))
    return out


def evaluator_name_problems(tree, strings=True):
    """L9: an evaluator-only name read in any form (call, attribute read, getattr / hasattr / attrgetter string)."""
    return [("evaluator name", name, line) for name, line, _ in names_in(tree, strings)
            if name in EVALUATOR_NAMES or set(th.tokens(name)) & EVALUATOR_TOKENS]


def dynamic_problems(tree, strings=True):
    """L9: loading code by name (importlib and relatives, __import__, the builtins table, sys.modules / sys.path)
    and running text as code (exec, eval, compile as names; re.compile and ast.literal_eval are attributes)."""
    problems, sys_names = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DYNAMIC_MODULES:
                    problems.append(("dynamic module", alias.name, node.lineno))
                if alias.name == "sys":
                    sys_names.add(alias.asname or "sys")
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if node.module.split(".")[0] in DYNAMIC_MODULES:
                problems.append(("dynamic module", node.module, node.lineno))
            if node.module == "sys" and {a.name for a in node.names} & (SYS_LOADER_ATTRIBUTES | {"*"}):
                problems.append(("sys loader state", node.module, node.lineno))
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in sys_names:
            parent = parents.get(id(node))
            if not isinstance(parent, ast.Attribute) or parent.attr in SYS_LOADER_ATTRIBUTES:
                problems.append(("sys loader state", getattr(parent, "attr", "sys"), node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr == "sys":
            problems.append(("sys loader state", "sys", node.lineno))
    for name, line, kind in names_in(tree, strings):
        if name in DYNAMIC_NAMES:
            problems.append(("dynamic name", name, line))
        if kind == "name" and name in CODE_BUILTINS:
            problems.append(("dynamic code", name, line))
    return problems


def lint_reader_problems(stem, tree, strings=True):
    """L9: only the thresholds module itself names its BDDL / lint readers (calls, references, imports, getattr)."""
    if stem == THRESHOLDS:
        return []
    return [("lint reader", name, line) for name, line, _ in names_in(tree, strings) if name in LINT_READERS]


def lazy_import_problems(stem, tree):
    """L9 / §C2.2: a function-level import of a v2 module is exactly LAZY_GEOMETRY (M0's modules keep theirs)."""
    if stem in M0_ALWAYS:
        return []
    return [("lazy import form", ast.unparse(node), line) for target, internal, lazy, line, node in imports(
        tree, nodes=True) if internal and lazy and target.startswith(V2) and ast.unparse(node) != LAZY_GEOMETRY]


def firewall_problems(stem, tree):
    """Every firewall check of this file on one (planted) module named ``stem``."""
    return (forbidden_import_problems(tree) + graph_problems(stem, tree) + robot_facts_problems(tree)
            + evaluator_problems(tree) + dynamic_problems(tree) + lint_reader_problems(stem, tree)
            + lazy_import_problems(stem, tree) + call_problems(stem, tree))


def lint_firewall_rules(tree, module_name="planted"):
    """M0's lint side of the same rules (forbidden_import + TH-EXT-6)."""
    return {v["rule"] for v in th.forbidden_imports(tree) + th.firewall_violations(tree, module_name)}


def call_problems(stem, tree):
    """Align always gets allow_pointing=False; only the policy (and the client itself) calls client methods."""
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called == "Align":
            flags = [kw for kw in node.keywords if kw.arg == "allow_pointing"]
            if not (len(flags) == 1 and isinstance(flags[0].value, ast.Constant) and flags[0].value.value is False):
                problems.append(("Align may point", node.lineno))
        if called in CLIENT_CALLS and stem not in (POLICY, CLIENT):
            problems.append(("client call outside the policy", called, node.lineno))
        if isinstance(func, ast.Attribute) and called in SCRIPTED_ANSWER_CALLS and stem != CLIENT:
            problems.append(("ScriptedAnswers call outside the client's transport", called, node.lineno))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs = [(t, node.value) for t in node.targets]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs = [(node.target, node.value)]
        else:
            continue
        for target, value in pairs:
            label = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
            if label == "pointing_request" and not (isinstance(value, ast.Constant) and value.value is None):
                problems.append(("pointing_request set", node.lineno))
    return problems


# ------------------------------------------------------------------------------------------------ 1. imports
@pytest.mark.parametrize("name", ALL_FILES)
def test_no_cm_or_mim_reaches_truth_sim_libero_scripts_or_oracle(name):
    assert forbidden_import_problems(tree_of(name)) == [], name


@pytest.mark.parametrize("name", ALL_FILES)
def test_v2_import_graph_follows_c22(name):
    assert graph_problems(Path(name).stem, tree_of(name)) == [], name


def test_v2_import_graph_is_acyclic_and_layered():
    edges = {Path(name).stem: {t for t, internal, _, _ in imports(tree_of(name)) if internal and t.startswith(V2)}
             for name in ALL_FILES}
    assert not cycle_in(edges), edges
    assert edges.get(TYPES, set()) <= M0_ALWAYS                         # the kernel imports no v2 module
    assert not edges.get(PRIMITIVES, set()) & {GEOMETRY, SCENE, PLANNER, OBSERVERS}
    assert CLIENT not in edges.get(POLICY, set())                      # the policy never imports the client
    assert "TYPE_CHECKING" not in (PACKAGE / f"{TYPES}.py").read_text()


@pytest.mark.parametrize("name", ALL_FILES)
def test_task_keyed_robot_facts_never_appear(name):
    source = (PACKAGE / name).read_text()
    assert "start_poses" not in source and "measured_tasks" not in source, name


@pytest.mark.parametrize("name", [n for n in ALL_FILES if n != f"{ROBOT}.py"])
def test_robot_facts_read_only_through_asset(name):
    assert robot_facts_problems(tree_of(name)) == [], name


@pytest.mark.parametrize("name", [n for n in ALL_FILES if n != f"{THRESHOLDS}.py"])
def test_no_bddl_and_no_truth_accessors(name):
    """The lint module itself reads BDDL for the vocabulary (lint infrastructure); no other CM / MIM does, and none
    calls an evaluator-only accessor."""
    assert evaluator_problems(tree_of(name)) == [], name


@pytest.mark.parametrize("source, check", [
    ("from gpt_field.agent_speedup_v2_truth import parse_check\n", "forbidden"),
    ("def f():\n    from .agent_speedup_v2_sim import V2SimProcess\n", "forbidden"),
    ("import libero.libero.benchmark as b\n", "forbidden"),
    ("from gpt_field import agent_speedup_v1 as v1\n", "forbidden"),
    ("import scripts.agent_speedup_v2_oracle\n", "forbidden"),
    ("from agent_speedup_v2_fakes import FakeV2Sim\n", "forbidden"),
    ("import robosuite\n", "forbidden"),
    ("from . import agent_speedup_v2_geometry as geometry\n", "graph_scene"),
    ("from .agent_speedup_v2_observers import X\n", "graph_primitives"),
    ("from .agent_speedup_v2_client import AgentSpeedUpV2Client\n", "graph_policy"),
    ("from .agent_speedup_v2_scene import SceneModel\n", "graph_types"),
    ("from .observation import decode_image\n", "graph_geometry"),
    ("from gpt_field.contact_skills import StepView\n", "graph_plan"),
    ("from .agent_speedup_v2_robot import ROBOT_FACTS\nX = ROBOT_FACTS['start_poses']\n", "facts"),
    ("from . import agent_speedup_v2_robot as r\nX = r.ROBOT_FACTS.get('asset')\n", "facts"),
    ("PATH = 'libero/libero/bddl_files/x.bddl'\n", "evaluator"),
    ("def f(sim):\n    return sim.truth()\n", "evaluator"),
    ("def f(world):\n    return world.goal_satisfied()\n", "evaluator"),
    ("def f(w):\n    return w.Align(1, 2)\n", "calls_relations"),
    ("def f(w):\n    return Align(1, allow_pointing=True)\n", "calls_relations"),
    ("def f(client):\n    return client.request_point(1)\n", "calls_relations"),
    ("class P:\n    pointing_request = 'need'\n", "calls_primitives"),
])
def test_firewall_checks_fire_on_planted_sources(source, check):
    tree = ast.parse(source)
    kind, _, stem_key = check.partition("_")
    stem = {"scene": SCENE, "primitives": PRIMITIVES, "policy": POLICY, "types": TYPES, "geometry": GEOMETRY,
            "plan": PLAN, "relations": RELATIONS}.get(stem_key)
    found = {"forbidden": lambda: forbidden_import_problems(tree), "graph": lambda: graph_problems(stem, tree),
             "facts": lambda: robot_facts_problems(tree), "evaluator": lambda: evaluator_problems(tree),
             "calls": lambda: call_problems(stem, tree)}[kind]()
    assert found, (source, check)


def test_firewall_checks_accept_allowed_patterns():
    lazy = ast.parse("def f():\n    from . import agent_speedup_v2_geometry as geometry\n    return geometry\n")
    assert graph_problems(SCENE, lazy) == []
    assert graph_problems(RELATIONS, ast.parse("from . import agent_speedup_v2_geometry as g\n"
                                               "from .wrist_servo import Align\n")) == []
    assert graph_problems(TYPES, ast.parse("from . import agent_speedup_v2_robot as r\n"
                                           "from .agent_speedup_v2_thresholds import Threshold\n")) == []
    assert robot_facts_problems(ast.parse("from . import agent_speedup_v2_robot as r\n"
                                          "X = r.ROBOT_FACTS['asset']['palm_profile_mm']\n")) == []
    assert call_problems(RELATIONS, ast.parse("def f(w):\n    return w.Align(1, allow_pointing=False)\n")) == []
    assert call_problems(POLICY, ast.parse("def f(c):\n    return c.request_plan(1)\n")) == []
    assert not cycle_in({"a": {"b"}, "b": {"c"}}) and cycle_in({"a": {"b"}, "b": {"a"}})


# ------------------------------------------------------------------------------- 8. L9 firewall hardening
THRESHOLDS_FILE = f"{THRESHOLDS}.py"


@pytest.mark.parametrize("name", ALL_FILES)
def test_no_dynamic_loading_or_dynamic_code(name):
    """The thresholds module's own definition strings name the banned machinery; its code is still checked."""
    assert dynamic_problems(tree_of(name), strings=name != THRESHOLDS_FILE) == [], name


@pytest.mark.parametrize("name", ALL_FILES)
def test_no_evaluator_only_names(name):
    assert evaluator_name_problems(tree_of(name), strings=name != THRESHOLDS_FILE) == [], name


@pytest.mark.parametrize("name", ALL_FILES)
def test_lint_readers_are_named_only_by_the_thresholds_module(name):
    assert lint_reader_problems(Path(name).stem, tree_of(name)) == [], name


@pytest.mark.parametrize("name", ALL_FILES)
def test_function_level_v2_imports_are_the_exact_geometry_statement(name):
    assert lazy_import_problems(Path(name).stem, tree_of(name)) == [], name


@pytest.mark.parametrize("name", ALL_FILES)
def test_m0_lint_firewall_rule_is_clean(name):
    assert th.firewall_violations(tree_of(name), Path(name).stem) == [], name


def run_check(check, stem, tree):
    return {"forbidden": lambda: forbidden_import_problems(tree), "dynamic": lambda: dynamic_problems(tree),
            "evaluator": lambda: evaluator_problems(tree), "calls": lambda: call_problems(stem, tree),
            "readers": lambda: lint_reader_problems(stem, tree), "lazy": lambda: lazy_import_problems(stem, tree),
            "graph": lambda: graph_problems(stem, tree)}[check]()


# The Wave 0 gate's probes, verbatim (impl/v2/wave0/gate/probes/firewall_probe.py). Before L9 the rows marked W1
# passed both this file's checks and M0's forbidden_import rule. Each row: (source, check here, M0 lint rule).
GATE_PROBES = {
    "plain_import_truth": ("import gpt_field.agent_speedup_v2_truth as t\n", "forbidden", "forbidden_import"),
    "from_pkg_contacts": ("from gpt_field import agent_speedup_v2_contacts\n", "forbidden", "forbidden_import"),
    "relative_tasks": ("from . import agent_speedup_v2_tasks\n", "forbidden", "forbidden_import"),
    "importlib_literal": ("import importlib\nT = importlib.import_module('gpt_field.agent_speedup_v2_truth')\n",
                          "dynamic", "dynamic_import"),                                                    # W1
    "importlib_in_func": ("import importlib\ndef f():\n    return importlib.import_module('.agent_speedup_v2_sim', "
                          "'gpt_field')\n", "dynamic", "dynamic_import"),                                  # W1
    "dunder_import": ("def f():\n    return __import__('gpt_field.agent_speedup_v2_truth')\n", "dynamic",
                      "dynamic_import"),                                                                   # W1
    "spec_from_file_oracle": ("import importlib.util\ndef f():\n    return importlib.util.spec_from_file_location("
                              "'o', 'scripts/agent_speedup_v2_oracle.py')\n", "dynamic", "dynamic_import"),  # W1
    "sys_modules": ("import sys\ndef f():\n    return sys.modules['gpt_field.agent_speedup_v2_truth']\n", "dynamic",
                    "dynamic_import"),                                                                     # W1
    "getattr_truth": ("def f(sim):\n    return getattr(sim, 'truth')()\n", "evaluator", "evaluator_name"),  # W1
    "bddl_open": ("def f(root):\n    return open(root + '/libero/bddl_files/x.bddl').read()\n", "evaluator",
                  "bddl_access"),
    "libero_path_join": ("import os\ndef f(root):\n    return open(os.path.join(root, 'libero', 'libero', "
                         "'bddl_files'))\n", "evaluator", "bddl_access"),
    "truth_attr_noncall": ("def f(sim):\n    g = sim.truth\n    return g()\n", "evaluator", "evaluator_name"),  # W1
    "oracle_param": ("def f(answers):\n    return answers.answer('plan', {})\n", "calls", "scripted_answer_call"),
    "v1_runtime": ("def f():\n    import gpt_field.agent_speedup_v1 as v\n    return v\n", "forbidden",
                   "forbidden_import"),
}
# More evasions of the same rules (dynamic loading and code, evaluator names, lint readers, lazy-import form).
PLANTED_L9 = [
    ("from importlib import import_module\n", "dynamic", "dynamic_import"),
    ("import runpy\nX = runpy.run_path('scripts/evaluate_agent_speedup_v2.py')\n", "dynamic", "dynamic_import"),
    ("import pkgutil\n", "dynamic", "dynamic_import"),
    ("import builtins\n", "dynamic", "dynamic_import"),
    ("def f():\n    return __builtins__['eval']\n", "dynamic", "dynamic_import"),
    ("def f(g):\n    return g.__globals__['__builtins__']\n", "dynamic", "dynamic_import"),
    ("def f():\n    return globals()['__builtins__']\n", "dynamic", "dynamic_import"),
    ("from sys import modules\n", "dynamic", "dynamic_import"),
    ("import sys\ndef f():\n    return getattr(sys, 'modules')\n", "dynamic", "dynamic_import"),
    ("import sys as s\ndef f():\n    return vars(s)\n", "dynamic", "dynamic_import"),
    ("import sys\nsys.path.insert(0, 'scripts')\n", "dynamic", "dynamic_import"),
    ("import os\ndef f():\n    return os.sys.modules\n", "dynamic", "dynamic_import"),
    ("def f(s):\n    exec(s)\n", "dynamic", "dynamic_code"),
    ("def f(s):\n    return eval(s)\n", "dynamic", "dynamic_code"),
    ("def f(s):\n    return compile(s, 'f', 'exec')\n", "dynamic", "dynamic_code"),
    ("RUN = eval\n", "dynamic", "dynamic_code"),
    ("def f(process):\n    return hasattr(process, 'oracle')\n", "evaluator", "evaluator_name"),
    ("def f(sim):\n    return vars(sim)['settle_truth']\n", "evaluator", "evaluator_name"),
    ("import operator\nREAD = operator.attrgetter('goal_satisfied')\n", "evaluator", "evaluator_name"),
    ("def f(world):\n    return world.contact_log\n", "evaluator", "evaluator_name"),
    ("def f(world):\n    return world.body_pose('x')\n", "evaluator", "evaluator_name"),
    ("def f(sim):\n    return sim.geometry()\n", "evaluator", "evaluator_name"),
    ("def f(oracle):\n    return oracle\n", "evaluator", "evaluator_name"),
    ("class OracleAnswersLike:\n    pass\n", "evaluator", "evaluator_name"),
    ("from . import agent_speedup_v2_thresholds as th\ndef f(root):\n    return th.lint_vocabulary(root)\n",
     "readers", "lint_reader"),
    ("from . import agent_speedup_v2_thresholds as th\nREAD = th.bddl_vocabulary\n", "readers", "lint_reader"),
    ("from .agent_speedup_v2_thresholds import bddl_vocabulary\n", "readers", "lint_reader"),
    ("from . import agent_speedup_v2_thresholds as th\ndef f():\n    return getattr(th, 'lint_module')\n",
     "readers", "lint_reader"),
    ("from . import agent_speedup_v2_thresholds as th\ndef f(word):\n    return word in th.PART_NOUNS\n",
     "readers", "lint_reader"),
    ("def f():\n    from .agent_speedup_v2_geometry import structure_axis\n    return structure_axis\n", "lazy",
     "lazy_import"),
    ("def f():\n    from . import agent_speedup_v2_geometry\n    return agent_speedup_v2_geometry\n", "lazy",
     "lazy_import"),
    ("def f():\n    from . import agent_speedup_v2_geometry as g\n    return g\n", "lazy", "lazy_import"),
    ("def f():\n    from gpt_field import agent_speedup_v2_geometry as geometry\n    return geometry\n", "lazy",
     "lazy_import"),
    ("def f():\n    import gpt_field.agent_speedup_v2_geometry as geometry\n    return geometry\n", "lazy",
     "lazy_import"),
    ("def f():\n    from . import agent_speedup_v2_geometry as geometry, agent_speedup_v2_types\n", "lazy",
     "lazy_import"),
    ("def f():\n    from .agent_speedup_v2_types import Seed\n    return Seed\n", "lazy", "lazy_import"),
    ("class C:\n    def f(self):\n        from .agent_speedup_v2_geometry import CameraModel\n", "lazy",
     "lazy_import"),
]
# Patterns that stay legal: (module stem, source).
ALLOWED_L9 = [
    (SCENE, "import re\nPATTERN = re.compile('x')\n"),
    (SCENE, "import ast\ndef f(s):\n    return ast.literal_eval(s)\n"),
    (TYPES, "def f(o):\n    for m in ('to_dict', 'summary'):\n        call = getattr(o, m, None)\n"
            "    object.__setattr__(o, 'value', call)\n    return o.__dict__\n"),
    (SCENE, "import sys\nEPS = sys.float_info.epsilon\n"),
    (SCENE, "def f():\n    from . import agent_speedup_v2_geometry as geometry\n    return geometry.structure_axis\n"),
    (OBSERVERS, "class W:\n    def f(self):\n        from . import agent_speedup_v2_geometry as geometry\n"
                "        return geometry.hand_motion_mask\n"),
    (RELATIONS, "from . import agent_speedup_v2_geometry as geometry\n"
                "from .agent_speedup_v2_geometry import CameraModel\n"),
    (ROBOT, "def f():\n    from .agent_speedup_v2_thresholds import CONSTANT_LABELS\n    return CONSTANT_LABELS\n"),
    (POLICY, "from . import agent_speedup_v2_thresholds as th\ndef f(s, ctx):\n"
             "    return th.tokens(s), th.constants_sha256(), th.Threshold, ctx.answer\n"),
    (CLIENT, "class ScriptedV2Transport:\n    def __init__(self, answers):\n        self.answers = answers\n\n"
             "    def send(self, kind, meta):\n        return self.answers.answer(kind, meta)\n"),
    (SCENE, "'''Never reads evaluator truth, the ORACLE or a BDDL name; never runs text as code.'''\n"),
    (PLAN, "SOURCES = ('plan_cell', 'fine', 'coarse')\n"),
]


@pytest.mark.parametrize("name", sorted(GATE_PROBES))
def test_gate_probes_are_caught_by_the_firewall_and_by_the_lint(name):
    source, check, rule = GATE_PROBES[name]
    tree = ast.parse(source)
    assert run_check(check, SCENE, tree), (name, check)
    assert firewall_problems(SCENE, tree), name
    assert rule in lint_firewall_rules(tree), (name, rule, lint_firewall_rules(tree))


@pytest.mark.parametrize("source, check, rule", PLANTED_L9)
def test_l9_planted_evasions_are_caught(source, check, rule):
    tree = ast.parse(source)
    assert run_check(check, SCENE, tree), (source, check)
    assert rule in lint_firewall_rules(tree), (source, rule, lint_firewall_rules(tree))


@pytest.mark.parametrize("stem, source", ALLOWED_L9)
def test_l9_allowed_patterns_stay_legal(stem, source):
    tree = ast.parse(source)
    assert firewall_problems(stem, tree) == [], source
    assert th.firewall_violations(tree, stem) == [], source


def test_scripted_answers_call_only_in_the_client():
    tree = ast.parse("def f(answers):\n    return answers.answer('plan', {})\n")
    assert call_problems(PRIMITIVES, tree) and call_problems(POLICY, tree) and call_problems(CLIENT, tree) == []
    assert "scripted_answer_call" in lint_firewall_rules(tree, POLICY)
    assert "scripted_answer_call" not in lint_firewall_rules(tree, CLIENT)


def test_lint_reader_rule_exempts_only_the_thresholds_module():
    tree = ast.parse("def f(root):\n    return bddl_vocabulary(root), lint_vocabulary(root)\n")
    assert lint_reader_problems(THRESHOLDS, tree) == [] and lint_reader_problems(TYPES, tree)
    assert "lint_reader" not in lint_firewall_rules(tree, THRESHOLDS)
    assert "lint_reader" in lint_firewall_rules(tree, ROBOT) and "lint_reader" in lint_firewall_rules(tree, CLIENT)


def test_lint_and_firewall_lists_agree():
    """The two layers name the same machinery, so neither drifts silently."""
    assert th.DYNAMIC_IMPORT_MODULES == DYNAMIC_MODULES and th.DYNAMIC_IMPORT_NAMES == DYNAMIC_NAMES
    assert th.DYNAMIC_CODE_BUILTINS == CODE_BUILTINS and th.SYS_IMPORT_ATTRIBUTES == SYS_LOADER_ATTRIBUTES
    assert th.EVALUATOR_ONLY_TOKENS == EVALUATOR_TOKENS and th.EVALUATOR_ONLY_CALLS == SIM_EVALUATOR_CALLS
    assert all(th.evaluator_only_name(name) for name in EVALUATOR_NAMES)
    assert set(th.EVALUATOR_ONLY_NAMES) <= EVALUATOR_NAMES
    assert th.SCRIPTED_ANSWER_METHODS == SCRIPTED_ANSWER_CALLS and set(th.BDDL_PATH_MARKERS) == set(BDDL_STRINGS)
    assert th.LINT_READERS == LINT_READERS and th.LAZY_IMPORT_STATEMENT == LAZY_GEOMETRY
    assert {"truth", "oracle", "answer_key", "goal_satisfied"} <= EVALUATOR_NAMES                  # L9's examples


def evaluator_accessors(path, classes):
    """Public method names of ``classes`` in an evaluator-side source file (AST; never imported)."""
    tree = ast.parse(path.read_text())
    return {item.name for node in tree.body if isinstance(node, ast.ClassDef) and node.name in classes
            for item in node.body if isinstance(item, ast.FunctionDef) and not item.name.startswith("_")}


def test_simulator_evaluator_accessors_are_all_covered():
    """Every evaluator-only method of the v2 simulator (worker and process client) is caught as a call, as an
    attribute read (except ``geometry``, the lazy module alias) and as a getattr string."""
    names = evaluator_accessors(PACKAGE / "agent_speedup_v2_sim.py", {"V2LiberoWorker", "V2SimProcess"})
    accessors = names - {"start", "close", "reset", "step"}
    assert {"truth", "settle_truth", "answer_key", "geometry", "oracle"} <= accessors, sorted(names)
    for name in sorted(accessors):
        call = ast.parse(f"def f(sim):\n    return sim.{name}()\n")
        assert evaluator_problems(call) and "evaluator_name" in lint_firewall_rules(call), name
        if name not in SIM_EVALUATOR_CALLS:
            dynamic = ast.parse(f"def f(sim):\n    return getattr(sim, {name!r})\n")
            assert evaluator_problems(dynamic) and "evaluator_name" in lint_firewall_rules(dynamic), name


def test_oracle_stub_and_fake_oracle_are_unnameable():
    oracle_path = ROOT / "scripts" / "agent_speedup_v2_oracle.py"
    classes = {node.name for node in ast.parse(oracle_path.read_text()).body if isinstance(node, ast.ClassDef)}
    fakes = {node.name for node in ast.parse((ROOT / "tests" / "agent_speedup_v2_fakes.py").read_text()).body
             if isinstance(node, ast.ClassDef) and "Oracle" in node.name}
    for name in sorted(classes | fakes | {"oracle_enabled", "require_oracle_sim", "truth_points", "surface_truth"}):
        tree = ast.parse(f"def f(x):\n    return x.{name}\n")
        assert evaluator_name_problems(tree) and "evaluator_name" in lint_firewall_rules(tree), name


RUNTIME_PROBE = ("import importlib, json, sys\nimportlib.import_module('gpt_field.' + sys.argv[1])\n"
                 "print(json.dumps(sorted(sys.modules)))\n")


def test_importing_a_cm_or_mim_loads_no_evaluator_module():
    """The gate's transitive probe as a test: a fresh interpreter that imports one CM / MIM has loaded no truth,
    simulator, LIBERO / robosuite / mujoco / BDDL, ORACLE, v1 agent, script or test module."""
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    running = {name: subprocess.Popen([sys.executable, "-c", RUNTIME_PROBE, Path(name).stem], cwd=ROOT, env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for name in ALL_FILES}
    for name, process in running.items():
        out, err = process.communicate(timeout=300)
        assert process.returncode == 0, (name, err[-2000:])
        loaded = json.loads(out.strip().splitlines()[-1])
        assert f"gpt_field.{Path(name).stem}" in loaded, name
        assert [m for m in loaded if forbidden(m)] == [], name


# ------------------------------------------------------------------------------------- ORACLE: never in live mode
def test_oracle_unavailable_in_live_mode():
    from gpt_field import agent_speedup_v2_sim as sim
    from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config
    assert "oracle" not in sim.DEFAULT_MONITOR_OPTIONS
    assert not sim.validate_monitor_options(None).get("oracle", False)
    with pytest.raises(ValueError):
        sim.validate_monitor_options({"oracle": 1})                     # never truthy by accident
    config, monitor = build_v2_sim_config("libero_10_t2")
    live_options = sim.validate_monitor_options(monitor)
    assert not live_options.get("oracle", False)                        # the registry's live options
    process = sim.V2SimProcess(config, monitor_options=monitor)         # constructed, never started
    if hasattr(process, "oracle"):
        with pytest.raises(PermissionError):
            process.oracle({"kind": "joints", "names": []})
    oracle_path = ROOT / "scripts" / "agent_speedup_v2_oracle.py"
    if oracle_path.is_file():
        spec = importlib.util.spec_from_file_location("agent_speedup_v2_oracle_firewall", oracle_path)
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        with pytest.raises(PermissionError):
            oracle.OracleAnswers(process, "libero_10_t2")              # refused without the oracle option


def test_runner_does_not_load_the_oracle_and_refuses_live():
    runner_path = ROOT / "scripts" / "evaluate_agent_speedup_v2.py"
    tree = ast.parse(runner_path.read_text())
    module_level = [t for t, _, lazy, _ in imports(tree) if not lazy]
    assert not any("oracle" in t.lower() for t in module_level)
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('runner_firewall', {str(runner_path)!r})\n"
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n"
            "loaded = sorted(m for m in sys.modules if 'oracle' in m.lower() or 'smoke_scenarios' in m)\n"
            "print(loaded)\n"
            "try:\n    module._default_client_factory(False)\nexcept ValueError:\n    print('refused')\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": "/tmp"})
    assert result.returncode == 0, result.stderr[-2000:]
    lines = result.stdout.strip().splitlines()
    assert lines[-2] == "[]" and lines[-1] == "refused", result.stdout


# ------------------------------------------------------------------------------------------- 2. AC-1(a)
def load_runner():
    spec = importlib.util.spec_from_file_location("evaluate_agent_speedup_v2_firewall",
                                                  ROOT / "scripts/evaluate_agent_speedup_v2.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def smoke_manifest(runner, root, key):
    """The run manifest of a one-step zero-call plumbing smoke for ``key`` against M0's fake simulator."""
    from agent_speedup_v2_fakes import FakeV2Sim
    from gpt_field.agent_speedup_v2_tasks import resolve_v2_task
    task = resolve_v2_task(key)
    sim_config = runner.build_sim_config(key)[0]
    fake = FakeV2Sim(prompt=task.prompt, task_id=task.task_id, chained=sim_config.collision_check_mode != "none")
    output = root / key
    runner.execute(output, config=runner.configuration(), sim_config=sim_config, task_key=key, episode_index=0,
                   sim_factory=lambda cfg: fake, video=False, smoke=True, smoke_steps=1)
    return json.loads((output / "manifest.json").read_text())["configuration"]


def test_one_constants_register_in_every_key_manifest(tmp_path):
    """AC-1(a) as the runner records it: the manifest of every registry key (built with a fake simulator) carries
    the same constants register and hash, the same instruction identity and the same lint identity, and its
    configured task is that key's official sentence."""
    from gpt_field.agent_speedup_v2_tasks import list_v2_tasks
    runner = load_runner()
    identities = {}
    for spec in list_v2_tasks():
        config = smoke_manifest(runner, tmp_path, spec.key)
        assert config["task"] == spec.prompt and config["task_key"] == spec.key, spec.key
        meta = config["agent_speedup_v2"]
        identities[spec.key] = (meta["constants_sha256"], meta["instructions_sha256"],
                                json.dumps(meta["constant_labels"], sort_keys=True),
                                json.dumps(meta["lint"], sort_keys=True), meta["thresholds_version"])
    assert len(identities) == len(list_v2_tasks()) >= 2
    assert len(set(identities.values())) == 1, sorted(identities)
    constants_sha, instructions_sha, labels, _, version = next(iter(identities.values()))
    assert constants_sha == th.constants_sha256() and json.loads(labels) == th.constants_register()
    assert version == th.THRESHOLDS_VERSION
    if module_exists(CLIENT) and instructions_sha is not None:
        assert instructions_sha == importlib.import_module(f"gpt_field.{CLIENT}").instructions_sha256()


def test_one_constants_register_for_every_key():
    runner = load_runner()
    first = runner.agent_metadata(runner.DEFAULT_LIBERO_ROOT)
    assert first["constants_sha256"] == th.constants_sha256()
    register = th.constants_register()
    for name in ALL_FILES:                                              # the hash covers the merged LABELS
        for key in th.module_labels(PACKAGE / name):
            assert key in register, key
    if module_exists(CLIENT):
        client = importlib.import_module(f"gpt_field.{CLIENT}")
        digest = client.instructions_sha256()
        assert isinstance(digest, str) and len(digest) == 64
        assert not inspect.signature(client.instructions_sha256).parameters     # no task input
        if first["instructions_sha256"] is not None:
            assert first["instructions_sha256"] == digest


# ------------------------------------------------------------------------------------------- 3. AC-1(b)
TASKISH = set(th.TASK_TEXT_IDENTIFIERS) | {"task", "task_name", "suite"}


def test_client_constructor_requires_the_official_sentence():
    if not module_exists(CLIENT):
        pytest.skip("the v2 client arrives in Wave 1 (BC)")
    client = importlib.import_module(f"gpt_field.{CLIENT}")
    parameters = inspect.signature(client.AgentSpeedUpV2Client.__init__).parameters
    task = parameters.get("task_prompt")
    assert task is not None and task.kind is inspect.Parameter.KEYWORD_ONLY and task.default is inspect.Parameter.empty
    assert not {"task_key", "task_id", "task_spec"} & set(parameters)
    with pytest.raises(TypeError):
        client.AgentSpeedUpV2Client()
    from gpt_field.agent_speedup_v2_tasks import list_v2_tasks
    prompts = {spec.key: spec.prompt for spec in list_v2_tasks()}
    if hasattr(client, "preflight_payloads"):
        for key, prompt in prompts.items():
            text = json.dumps(client.preflight_payloads(prompt))
            assert prompt in text, key
            others = [p for p in prompts.values() if p != prompt and p not in prompt]
            assert not [p for p in others if p in text], key


def test_policy_constructor_takes_no_task_input():
    if not module_exists(POLICY):
        pytest.skip("the v2 policy arrives in Wave 2 (BX)")
    policy = importlib.import_module(f"gpt_field.{POLICY}")
    parameters = list(inspect.signature(policy.AgentSpeedUpV2Policy.__init__).parameters)
    assert parameters[:2] == ["self", "client"]
    assert not set(parameters) & TASKISH, parameters


# ------------------------------------------------------------------------------------------- 4. AC-1(c)
def test_t0_lint_and_literal_audit_in_every_wave():
    report = th.lint_control_modules(LIBERO)
    assert report["passed"], {k: v for k, v in {**report["modules"], **report["interface_modules"]}.items() if v}
    for name in th.CONTROL_MODULES + th.PRESENT_INTERFACE_MODULES:
        assert th.literal_audit(PACKAGE / name) == [], name
    assert th.LABEL_MERGE_ERRORS == {}


# ------------------------------------------------------------------------------------------- 5. AC-5
BANNED_WORDS = {"left", "right", "up", "down", "forward", "backward", "north", "south", "east", "west", "clockwise",
                "counterclockwise", "cw", "ccw", "deg", "degree", "angle", "rad", "radian", "yaw", "pitch", "roll",
                "offset", "distance", "height", "depth", "cm", "mm", "meter", "xyz", "dx", "dy", "dz"}
COMPLETION_WORDS = {"done", "complete", "completed", "success", "finished"}
NUMERIC_FIELDS = {"w", "x", "point_id", "cell_id"}


def scan_schema(schema, problems, path="$"):
    if isinstance(schema, dict):
        if schema.get("type") in ("integer", "number"):
            field = path.rsplit(".", 1)[-1]
            if field not in NUMERIC_FIELDS or schema.get("type") != "integer" or not (
                    schema.get("minimum") == 0 and schema.get("maximum") == 64):
                problems.append(("numeric field", path))
        for key, value in schema.items():
            if key == "properties":
                for name, sub in value.items():
                    words = set(th.tokens(name))
                    if words & (BANNED_WORDS | COMPLETION_WORDS):
                        problems.append(("property name", f"{path}.{name}"))
                    scan_schema(sub, problems, f"{path}.{name}")
            elif key == "enum":
                for item in value:
                    if isinstance(item, str) and set(th.tokens(item)) & (BANNED_WORDS | COMPLETION_WORDS):
                        problems.append(("enum value", f"{path}={item}"))
            elif key == "description" and isinstance(value, str):
                if set(th.tokens(value)) & COMPLETION_WORDS:
                    problems.append(("description", path))
            else:
                scan_schema(value, problems, path)
    elif isinstance(schema, list):
        for item in schema:
            scan_schema(item, problems, path)


def test_ac5_schema_scan():
    if not module_exists(PLAN):
        pytest.skip("the plan schemas arrive in Wave 1 (BC)")
    plan = importlib.import_module(f"gpt_field.{PLAN}")
    problems = []
    for kind, schema in plan.schemas().items():
        scan_schema(schema, problems, kind)
    assert problems == []


def test_ac5_instruction_texts_carry_no_vocabulary():
    if not module_exists(CLIENT):
        pytest.skip("the v2 client arrives in Wave 1 (BC)")
    client = importlib.import_module(f"gpt_field.{CLIENT}")
    vocab, known = set(th.lint_vocabulary(LIBERO)["vocabulary"]), set(th.LINT_ALLOWLIST)
    texts = [client.instructions(kind) for kind in client.REQUEST_KINDS] + [json.dumps(client.WORKED_EXAMPLE)]
    for text in texts:
        hits = set().union(*(th.token_hits(t, vocab, known) for t in th.tokens(text)))
        assert not hits, hits


# Payload values the model itself supplied or that are the official sentence (contract §C12 test 2): the "task"
# value, the point request's "part" text, the status block's "scene" and its [pid, text] part pairs.
ECHOED_KEYS = {"task", "part", "scene"}
LAST_OUTCOME_KEYS = {"reason", "evidence_text", "thresholds"}


def executor_strings(value, prompt, found, key=None):
    """Every executor-written string of a payload: dict keys and string values, recursing into JSON-encoded text
    (the request text is ``ordered_dumps`` of a dict); skips image data URLs, the echoed values (ECHOED_KEYS, the
    texts of status-block ``parts`` pairs) and every occurrence of the official sentence; collects every
    ``last_outcome`` value into ``found["last_outcome"]``."""
    if isinstance(value, dict):
        for name, item in value.items():
            found["strings"].append(str(name))
            if name in ECHOED_KEYS:
                continue
            if name == "last_outcome":
                found["last_outcome"].append(item)
            if name == "parts" and isinstance(item, list):
                for pair in item:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        executor_strings(pair[0], prompt, found)               # the pid; the text is echoed
                    else:
                        executor_strings(pair, prompt, found)
                continue
            executor_strings(item, prompt, found, name)
    elif isinstance(value, (list, tuple)):
        for item in value:
            executor_strings(item, prompt, found, key)
    elif isinstance(value, str):
        if value.startswith("data:"):
            return
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                decoded = json.loads(stripped)
            except ValueError:
                decoded = None
            if isinstance(decoded, (dict, list)):
                executor_strings(decoded, prompt, found, key)
                return
        found["strings"].append(value.replace(prompt, " "))


def test_ac5_client_payload_scan():
    """AC-5 / contract §C12 test 2 at the cross-module level: for every registry key, every executor-written string
    of every preflight payload is free of lint-vocabulary tokens (this catches inherited legacy instruction texts),
    and a repair payload's last_outcome carries exactly reason, evidence_text and thresholds [C-45]."""
    if not module_exists(CLIENT):
        pytest.skip("the v2 client arrives in Wave 1 (BC)")
    client = importlib.import_module(f"gpt_field.{CLIENT}")
    if not hasattr(client, "preflight_payloads"):
        pytest.fail("contract §C12: the client module provides preflight_payloads(task_prompt)")
    from gpt_field.agent_speedup_v2_tasks import list_v2_tasks
    vocab, known = set(th.lint_vocabulary(LIBERO)["vocabulary"]), set(th.LINT_ALLOWLIST)
    for spec in list_v2_tasks():
        payloads = client.preflight_payloads(spec.prompt)
        assert isinstance(payloads, dict) and {"plan", "point_fine", "point_coarse", "repair"} <= set(payloads), spec.key
        found = {"strings": [], "last_outcome": []}
        executor_strings(payloads, spec.prompt, found)
        hits = {}
        for text in found["strings"]:
            words = set().union(*(th.token_hits(t, vocab, known) for t in th.tokens(text)))
            if words:
                hits[text[:80]] = sorted(words)
        assert not hits, (spec.key, hits)
        repair = {"strings": [], "last_outcome": []}
        executor_strings(payloads["repair"], spec.prompt, repair)
        assert repair["last_outcome"], (spec.key, "the repair payload carries a last_outcome field")
        for outcome in found["last_outcome"]:
            assert outcome is None or (isinstance(outcome, dict) and set(outcome) == LAST_OUTCOME_KEYS), (spec.key,
                                                                                                      outcome)


def test_payload_scan_helper_on_a_planted_payload():
    prompt = "put the alpha on the beta"
    text = json.dumps({"request": "repair", "task": prompt, "plan": {"parts": [["A", "the wooden bowl"]],
                       "scene": "a moka pot", "steps": []}, "last_outcome": {"reason": "hold_lost",
                       "evidence_text": "part A", "thresholds": [["G6_hold_floor_mm", 0.5]]}, "holding": None})
    payload = {"instructions": "Answer with JSON.", "input": [{"role": "user", "content": [
        {"type": "input_text", "text": text}, {"type": "input_image", "image_url": "data:image/png;base64,bW9rYQ=="}]}]}
    found = {"strings": [], "last_outcome": []}
    executor_strings(payload, prompt, found)
    joined = " ".join(found["strings"])
    assert "wooden" not in joined and "moka" not in joined and prompt not in joined and "base64" not in joined
    assert "hold_lost" in joined and "G6_hold_floor_mm" in joined and "Answer with JSON." in joined
    assert found["last_outcome"] == [{"reason": "hold_lost", "evidence_text": "part A",
                                      "thresholds": [["G6_hold_floor_mm", 0.5]]}]
    found = {"strings": [], "last_outcome": []}
    executor_strings({"instructions": "keep the bowl-on-plate rule"}, prompt, found)   # a legacy text is caught
    vocab, known = set(th.lint_vocabulary(LIBERO)["vocabulary"]), set(th.LINT_ALLOWLIST)
    assert set().union(*(th.token_hits(t, vocab, known) for s in found["strings"] for t in th.tokens(s)))


# ------------------------------------------------------------------------------------------- 6. AC-6
def check_outcome_record(record):
    assert record["reason"] in kernel.REASONS and record["status"] in kernel.OUTCOME_STATUSES, record
    assert isinstance(record["evidence"], dict)
    assert isinstance(record["thresholds"], list)
    for item in record["thresholds"]:
        assert item["label"] in th.LABEL_CLASSES and item["derivation"].strip() and item["name"], item
    json.dumps(record, allow_nan=False)


def test_outcome_records_carry_reason_evidence_and_labelled_thresholds():
    outcome = kernel.RelationOutcome("held", "succeeded", "hold_pass", {"obj": "A"}, {"gap_mm": 25.4},
                                     [kernel.th_max("G5_object_present_gap_mm", kernel.Term("DD", 0.5, "G5_FLOOR_MM"),
                                                    unit="mm")], "part A held", 0, 10)
    check_outcome_record(outcome.record(1, 10, 0))


def test_ac6_fake_end_to_end_outcomes():
    needed = (POLICY, RELATIONS, CLIENT, PLAN, GEOMETRY, SCENE, PLANNER, PRIMITIVES, OBSERVERS)
    if not all(module_exists(stem) for stem in needed):
        pytest.skip("the fake end-to-end run needs every v2 module (Wave 2)")
    fakes = importlib.import_module("agent_speedup_v2_fakes")
    if not all(hasattr(fakes, name) for name in ("FakeTwoViewSim", "task2_like_scene", "FakeOracleAnswers")):
        pytest.skip("FakeTwoViewSim / FakeOracleAnswers not in the fakes yet")
    from gpt_field.agent_speedup_v2_tasks import resolve_v2_task
    client_module = importlib.import_module(f"gpt_field.{CLIENT}")
    policy_module = importlib.import_module(f"gpt_field.{POLICY}")
    prompt = resolve_v2_task("libero_10_t2").prompt
    world = fakes.FakeTwoViewSim(fakes.task2_like_scene(), prompt=prompt)
    client = client_module.AgentSpeedUpV2Client(
        task_prompt=prompt, transport=client_module.ScriptedV2Transport(fakes.FakeOracleAnswers(world)))
    policy = policy_module.AgentSpeedUpV2Policy(client)
    policy.reset()
    observation = world.reset()
    for _ in range(1300):
        decision = policy.act(observation)
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
        if world.goal_satisfied():
            break
    assert policy.outcomes
    for record in policy.outcomes:
        check_outcome_record(record)


# ------------------------------------------------------------------------------------------- 7. calls and Align
@pytest.mark.parametrize("name", ALL_FILES)
def test_align_never_points_only_the_policy_calls_the_client_and_no_pointing_request(name):
    assert call_problems(Path(name).stem, tree_of(name)) == [], name
