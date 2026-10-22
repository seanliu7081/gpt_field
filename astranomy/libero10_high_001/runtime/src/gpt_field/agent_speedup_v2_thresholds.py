"""agent_SpeedUp_v2 threshold policy: the Threshold record, the one constants register and the T0 lint (spec §5.1).

Rule: no number in a v2 control module may be specific to a task. Every number the executor compares against is
a ``Threshold`` with a label and a derivation, logged in every outcome record that used it:

- ROBOT  a robot asset fact or a ruling fact;
- ONLINE measured in this episode by the robot's own motion or sensing;
- DD     a labelled design default: k x an ONLINE or ROBOT quantity, or a floor justified from robot-only traces;
- RULING set by a user ruling.

There is no evaluator-tuned class (R-3 stays refused).

``CONSTANT_LABELS`` is the single constants register: every numeric module-level constant of every v2 control
module (``CONTROL_MODULES``) has an entry ``"<module>.<NAME>": {"label", "derivation"}``. It is hashed into the
run manifest (``constants_sha256``) and must be identical for every registry key (AC-1).

T0 lint (``lint_module``):
1. every assignment at module level or in a class body whose value contains a numeric literal anywhere
   (dicts, calls with keyword arguments, conditional expressions, ...) has a ``CONSTANT_LABELS`` entry
   ("<module>.<NAME>" or "<module>.<Class>.<NAME>"), and so has every numeric default of a function argument
   ("<module>.<function>.<arg>", rule ``numeric_default``);
2. no token of the lint vocabulary appears in an identifier, a string, an imported module path or an imported
   name of a control module; a token also matches through its plural (-s / -es) and through a two-word
   compound written without a separator, with one half in the vocabulary and the other a vocabulary or
   allowlisted word (an all-caps or run-together pair of task nouns);
   attributes are skipped only for a fixed set of third-party library modules (``np.x``, ``math.x``);
3. a control module never reads the task sentence (no prompt / task identifiers), and never imports the
   evaluator modules (truth, sim, contacts, tasks), ``libero`` or any v1 module (rule ``forbidden_import``);
4. TH-EXT-6 (lead decision L9): no CM / MIM loads code by name or runs text as code (importlib and its relatives,
   ``__import__``, ``sys.modules``, exec, eval, compile), names an evaluator-only accessor (truth, ORACLE,
   answer_key, goal_satisfied, ...; a call, an attribute read or a getattr string), calls the ScriptedAnswers
   protocol outside the MIM, names a BDDL path or this module's BDDL / lint readers, or imports a v2 module inside
   a function other than as ``from . import agent_speedup_v2_geometry as geometry`` (``firewall_violations``).

Lint vocabulary: the underscore- and digit-split tokens of every object category, fixture category and region
name in all LIBERO BDDL files (all suites), plus the words of every official task sentence, plus the checked-in
``PART_NOUNS`` (object-part nouns that BDDL names do not contain; a controller named after a part is
task-shaped), minus the checked-in ``LINT_ALLOWLIST`` of generic words the executor legitimately uses. The
vocabulary, the allowlist and the part nouns and their sha256 go into the manifest (D2, decided 2026-10-18, lead
default; the user may override; spec §5.1 item 4). The allowlist choices are documented next to
``LINT_ALLOWLIST`` (comments: this docstring is itself linted).

The lint does not scan the strings of the vocabulary definitions themselves (``LINT_ALLOWLIST``, ``PART_NOUNS``)
in this module, and only in this module: the same assignment in any other control module is linted.

TH-EXT (contract §C4.9, kernel builder): ``CONTROL_MODULES`` = M0's two modules plus every planned v2 control
module whose file exists (computed at import); the model-interface module (the client) is linted with task-text
identifiers allowed; each v2 module declares its own labels in one pure-literal ``LABELS`` dict, merged into
``CONSTANT_LABELS`` by AST (never by importing the module); a module whose ``LABELS`` cannot be merged makes
``constants_register`` / ``constants_sha256`` raise, so no run manifest carries an incomplete register;
``lint_module`` also flags orphan labels; ``literal_audit`` lists float literals in function bodies whose magnitude
is not structural, numeric literals in defaults and class bodies, and numeric literals in module-level statements
other than plain name assignments (``if`` / ``try`` / ``for`` / ``with`` blocks, augmented or subscript
assignments), which the label rule cannot see; ``THRESHOLD_IDS`` is an informational transcription of the contract's
threshold-id list (owner, runtime names, label classes, rule): it holds no value, no control code reads it, and the
owner modules' labelled constants stay the only authority.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

# Version history: v3 M0; v4 ratified at W4 (Appendix L 12); v5 TH-EXT-6 firewall rules (Appendix L 9); v6 design
# review 1 decision §3: the new DD constants of geometry, scene, planner, primitives and relations (V6_CONSTANTS,
# labelled in their owner modules) and the amended D4, D7, D8, D10, D15, V1, V5, G12, G13, L1, M1, O3, P4, P5, P7,
# P8, P11 and P12 rules in THRESHOLD_IDS; v7 design review 2 decision §3 (K-6): the D10-OE orientation and collapse
# constants of geometry and scene (V7_CONSTANTS), the retired D10 deciders (V7_RETIRED_DECIDERS, still logged) and
# the amended D8, D10, L1 and P2 rules in THRESHOLD_IDS.
THRESHOLDS_VERSION = "agent-speedup-v2-thresholds-v7"
LABEL_CLASSES = ("ROBOT", "ONLINE", "DD", "RULING")
# Control modules of v2 (paths relative to src/gpt_field). TH-EXT-1: M0's two modules plus every planned control
# module whose file exists, computed at import, so a module is linted (labels, defaults, class bodies, forbidden
# imports, vocabulary, task text) from the moment its file exists. The client is the one model-interface module:
# linted the same way except that task-text identifiers are allowed (it must take the official sentence). The
# optional video module is a runner-side tool and is in neither list (contract §C1.4).
M0_CONTROL_MODULES = ("agent_speedup_v2_robot.py", "agent_speedup_v2_thresholds.py")
PLANNED_CONTROL_MODULES = ("agent_speedup_v2_types.py", "agent_speedup_v2_geometry.py", "agent_speedup_v2_scene.py",
                           "agent_speedup_v2_planner.py", "agent_speedup_v2_primitives.py",
                           "agent_speedup_v2_observers.py", "agent_speedup_v2_relations.py", "agent_speedup_v2_plan.py",
                           "agent_speedup_v2.py")
MODEL_INTERFACE_MODULES = ("agent_speedup_v2_client.py",)
PACKAGE_DIR = Path(__file__).resolve().parent
CONTROL_MODULES = M0_CONTROL_MODULES + tuple(name for name in PLANNED_CONTROL_MODULES
                                             if (PACKAGE_DIR / name).is_file())
PRESENT_INTERFACE_MODULES = tuple(name for name in MODEL_INTERFACE_MODULES if (PACKAGE_DIR / name).is_file())
# Name of the per-module label register (contract §C1.5 item 3): {"NAME" | "Class.NAME" | "function.arg":
# (label, derivation)}, a pure literal.
LOCAL_LABELS_NAME = "LABELS"
# Generic words the executor legitimately uses although they occur in BDDL names or task sentences. D2 (decided
# 2026-10-18, lead default; the user may override):
# - 'rim' stays allowlisted. It enters the raw vocabulary only through the category glazed_rim_porcelain_ramekin,
#   and it names the measured top edge of ANY cavity: a geometric quantity that the generic place-in and support
#   code compares against (spec §4.8 inside, M9 "bottom <= rim - M9", M10, SupportClassify perched_on_rim, the
#   rim_pinch contact primitive), not a part of a particular object.
# - 'box', 'bin', 'flat' and 'layer' are generic programming / geometry words (a bounding box, a histogram bin, a
#   flat plane fill, a HeightField layer). They enter the raw vocabulary through region and category names
#   (box_region, "cream cheese box", bin, flat_stove, top_layer); the object nouns around them stay banned.
LINT_ALLOWLIST = (
    "and", "at", "back", "behind", "between", "bin", "both", "bottom", "box", "center", "close", "contain",
    "flat", "floor", "from", "front", "in", "init", "inside", "it", "layer", "left", "middle", "new", "next",
    "object", "of", "off", "on", "open", "other", "pick", "place", "push", "put", "region", "right", "rim", "side",
    "stack", "target", "the", "them", "to", "top", "turn", "two", "under", "up",
)
# Object-part nouns added to the vocabulary (D2): not in any BDDL name, but a control module that names them is
# task-shaped. Parts are addressed by the plan's part ids and their measured shape, never by these words.
# 'cap' is deliberately NOT listed: it is the spec's and the robot facts' generic word for a command limit (R10
# "rotation cap 0.15 rad/step" = ROTATION_CAP_RAD_PER_STEP, K2 "cap 0.1"), and no BDDL name has a cap part. Adding
# it would require renaming that ROBOT fact.
PART_NOUNS = ("button", "burner", "door", "handle", "knob", "lever", "lid", "spout")
# Module-level assignments that define the vocabulary; their strings are not scanned in THIS module only.
VOCABULARY_DEFINITIONS = ("LINT_ALLOWLIST", "PART_NOUNS")
TASK_TEXT_IDENTIFIERS = ("prompt", "task_prompt", "task_key", "task_id", "task_spec", "language", "sentence")
MIN_TOKEN_CHARS = 2
COMPOUND_MIN_CHARS = 3
# Third-party / standard-library modules whose attributes (np.array, math.pi) are not lint tokens.
THIRD_PARTY_MODULES = frozenset({"ast", "cv2", "collections", "dataclasses", "hashlib", "json", "math", "mujoco",
                                 "np", "numpy", "pathlib", "re", "typing"})
# Modules a control module must never import: evaluator truth and infrastructure, LIBERO, v1.
FORBIDDEN_IMPORT_COMPONENTS = frozenset({"agent_speedup_v2_truth", "agent_speedup_v2_sim", "agent_speedup_v2_contacts",
                                         "agent_speedup_v2_tasks", "libero"})
FORBIDDEN_IMPORT_PREFIXES = ("agent_speedup_v1",)
# TH-EXT-6 (lead decision L9, Wave 0 gate W1): firewall hardening, every CM and MIM (``firewall_violations``).
# - rule dynamic_import: the import machinery that loads a module by name, the builtins table, and the loader state
#   of ``sys`` (any use of ``sys`` other than an attribute outside SYS_IMPORT_ATTRIBUTES);
DYNAMIC_IMPORT_MODULES = frozenset({"builtins", "imp", "importlib", "pkgutil", "runpy", "zipimport"})
DYNAMIC_IMPORT_NAMES = frozenset({"__builtins__", "__globals__", "__import__", "__loader__", "__spec__",
                                  "__subclasses__", "exec_module", "find_module", "find_spec", "import_module",
                                  "importlib", "load_module", "module_from_spec", "run_module", "run_path",
                                  "spec_from_file_location", "spec_from_loader"})
SYS_IMPORT_ATTRIBUTES = frozenset({"meta_path", "modules", "path", "path_hooks", "path_importer_cache"})
# - rule dynamic_code: the builtins that run text as code, as plain names (re.compile and ast.literal_eval stay legal);
DYNAMIC_CODE_BUILTINS = frozenset({"compile", "eval", "exec"})
# - rule evaluator_name: an identifier, attribute, imported path component or (dotted) identifier-shaped string
#   constant that names an evaluator-only accessor (simulator / fake / truth module / ORACLE): one of
#   EVALUATOR_ONLY_NAMES, or any name with an EVALUATOR_ONLY_TOKENS token (truth, settle_truth, OracleAnswers, ...),
#   so getattr(sim, "truth") and a bare attribute read are caught as well as a call;
EVALUATOR_ONLY_NAMES = frozenset({"answer_key", "body_pose", "colour_region_changed", "contact_log", "evaluate_goal",
                                  "failing_checks", "goal_satisfied", "parse_check", "physical_summary",
                                  "release_and_settle"})
EVALUATOR_ONLY_TOKENS = frozenset({"oracle", "truth"})
#   plus, as a method call only, the simulator's evaluator geometry accessor (``sim.geometry()``; the name itself is
#   the lazily imported module's alias);
EVALUATOR_ONLY_CALLS = frozenset({"geometry"})
# - rule scripted_answer_call: ``x.answer(...)`` is the ScriptedAnswers (ORACLE) protocol; only the MIM's scripted
#   transport calls it;
SCRIPTED_ANSWER_METHODS = frozenset({"answer"})
# - rule bddl_access: BDDL file paths in a string constant (only this module reads BDDL, for the vocabulary);
BDDL_PATH_MARKERS = (".bddl", "bddl_files", "get_libero_path", "libero/libero")
# - rule lint_reader: the readers of BDDL, of module sources and of the lint lists are this module's own; no other
#   CM / MIM names them (a call, a reference, an import or a getattr string);
LINT_READERS = frozenset({"LINT_ALLOWLIST", "PART_NOUNS", "bddl_vocabulary", "firewall_violations",
                          "lint_control_modules", "lint_module", "lint_vocabulary", "literal_audit", "local_labels",
                          "merged_labels", "module_labels", "v6_constant_status", "v7_constant_status"})
# - rule lazy_import: a function-level import of a v2 module is written exactly as this statement (contract §C2.2);
#   M0's own modules keep their function-level imports.
LAZY_IMPORT_STATEMENT = "from . import agent_speedup_v2_geometry as geometry"
# Module-level assignments whose strings name the banned things; not scanned by firewall_violations in THIS module.
FIREWALL_DEFINITIONS = ("FORBIDDEN_IMPORT_COMPONENTS", "FORBIDDEN_IMPORT_PREFIXES", "DYNAMIC_IMPORT_MODULES",
                        "DYNAMIC_IMPORT_NAMES", "SYS_IMPORT_ATTRIBUTES", "DYNAMIC_CODE_BUILTINS", "EVALUATOR_ONLY_NAMES",
                        "EVALUATOR_ONLY_TOKENS", "EVALUATOR_ONLY_CALLS", "SCRIPTED_ANSWER_METHODS", "BDDL_PATH_MARKERS", "LINT_READERS",
                        "LAZY_IMPORT_STATEMENT", "FIREWALL_DEFINITIONS")


@dataclass(frozen=True)
class Threshold:
    """Every number the executor compares against; logged in evidence with its label and derivation."""
    name: str
    value: float
    label: Literal["ROBOT", "ONLINE", "DD", "RULING"]
    derivation: str

    def __post_init__(self):
        if self.label not in LABEL_CLASSES:
            raise ValueError(f"threshold label must be one of {LABEL_CLASSES}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("threshold name must be a nonempty string")
        if not isinstance(self.derivation, str) or not self.derivation.strip():
            raise ValueError("threshold derivation must be a nonempty string")
        value = float(self.value)
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("threshold value must be finite")
        object.__setattr__(self, "value", value)

    def to_dict(self):
        return asdict(self)


def _entry(label, derivation):
    if label not in LABEL_CLASSES:
        raise ValueError(label)
    return {"label": label, "derivation": derivation}


R = "agent_speedup_v2_robot."
T = "agent_speedup_v2_thresholds."
CONSTANT_LABELS = {
    # ROBOT facts R1-R14 (spec §4.4)
    R + "GRIP_SITE_ALONG_R2_M": _entry("ROBOT", "R1: grip site = hand + 0.097 R2 (robot_tool_geometry.py:49; "
                                                "robot_facts_v2.json asset)"),
    R + "WRIST_CAMERA_ALONG_R0_M": _entry("ROBOT", "R2 (ruling R-2): wrist camera = hand + 0.05 R0, optical axis "
                                                   "R2; image right = R1, up = R0 (wrist_servo.py:29, :56-66)"),
    R + "FOCAL_PX_AT_512": _entry("ROBOT", "R3 (ruling R-2f granted): f = 256 / tan(fovy/2), fovy 75 deg -> "
                                           "333.6 px at 512 px; cross-checked by the ONLINE f-hat (D3) in M1"),
    R + "FOCAL_REFERENCE_IMAGE_PX": _entry("ROBOT", "R3: image width at which the focal length is stated"),
    R + "PAD_SPAN_ALONG_APPROACH_MM": _entry("ROBOT", "R4: pads span -11.6 .. +4.4 mm along the approach about the "
                                                      "grip (finger body 0.0524 + pad centre 0.041 +- 0.008)"),
    R + "PAD_LENGTH_R0_MM": _entry("ROBOT", "R4: pad length along R0 (2 x 0.008 m half-size)"),
    R + "PAD_THICKNESS_R1_MM": _entry("ROBOT", "R4: pad thickness along the jaw (2 x 0.004 m half-size)"),
    R + "FINGERTIP_REACH_MM": _entry("ROBOT", "R5: fingertip reach beyond the grip, pinned 9.3 mm (finger hull "
                                              "9.25 mm in robot_facts_v2.json; a full-opening descent stalled "
                                              "9.4 mm above its support, spec CP)"),
    R + "PALM_UNDERSIDE_MM": _entry("ROBOT", "R6: palm underside behind the grip, pinned 31.0 mm (convex hull of "
                                             "the hand collision mesh: 31.04 mm)"),
    R + "PAD_SEPARATION_OFFSET_MM": _entry("ROBOT", "R9: pad separation = finger gap - 1.0 mm (asset: "
                                                    "robot_facts_v2.json pad_separation_offset_mm, gap 2q minus "
                                                    "twice the pad inner face; contact check: a 16.3 mm bar reads "
                                                    "17.45-17.48 mm, a 24.3 mm ridge reads 25.39 mm; spec R9)"),
    R + "TRANSLATION_PER_UNIT_M": _entry("ROBOT", "R10: OSC_POSE 0.05 m per unit translation (action.py:16)"),
    R + "ROTATION_PER_UNIT_RAD": _entry("ROBOT", "R10: OSC_POSE 0.5 rad per unit rotation (action.py:17)"),
    R + "ROTATION_CAP_RAD_PER_STEP": _entry("ROBOT", "R10: rotation cap 0.15 rad per step (action.py:48)"),
    R + "CONTROL_HZ": _entry("ROBOT", "R10: 20 Hz control"),
    R + "GRIPPER_RAMP_PER_SUBSTEP": _entry("ROBOT", "R11: gripper current_action += 0.01 sign(a) per substep; "
                                                    "action 0 holds (robosuite panda_gripper format_action)"),
    R + "JOINT7_LIMIT_RAD": _entry("ROBOT", "R12: joint 7 range +-2.8973 (panda robot.xml:225; robot_facts_v2.json)"),
    R + "NOMINAL_START_Q7_RAD": _entry("ROBOT", "R12: nominal init q7 = pi/4 (mounted_panda.py:35-38)"),
    R + "START_Q7_PERTURBATION_RAD": _entry("ROBOT", "R12: init perturbation allowance 0.15 rad (measured start "
                                                     "q7 0.768-0.818)"),
    R + "VERIFIED_MOUNT_BASE_XYZ": _entry("ROBOT", "R13 (ruling R-5): MountedPanda base (-0.66, 0, 0.912) "
                                                   "(agent_speedup_v1.py:92-96)"),
    R + "SHOULDER_ABOVE_BASE_M": _entry("ROBOT", "R13: shoulder (link2 anchor) 0.333 m above the base (shoulder z "
                                                 "1.245; robot_facts_v2.json link2)"),
    R + "WRIST_FROM_GRIP_M": _entry("ROBOT", "R13: wrist point = grip - 0.204 R2 (agent_speedup_v1.py:94)"),
    R + "FULL_STRETCH_M": _entry("ROBOT", "R13: full arm stretch ~0.804 m (spec v1 live run 001)"),
    R + "START_OFFSET_FROM_BASE_M": _entry("ROBOT", "R14 (ruling R-5b): robot start pose relative to its base, "
                                                    "(0.4537, 0.0002, 0.2655) m (identical init_qpos)"),
    R + "START_OFFSET_SPREAD_M": _entry("ROBOT", "R14: start-pose spread bound (1.1, 1.9, 1.8) cm: the measured "
                                                 "maxima (1.08, 1.85, 1.72) cm of |start grip - base - offset| "
                                                 "over the 10 libero_10 resets, rounded up"),
    # RULING (spec §5.3)
    R + "VERIFIED_MOUNT_GATE_M": _entry("RULING", "R-5 gate: half of the 9.0 cm distance from the verified mount "
                                                  "to the nearest other LIBERO mount (-0.75, 0, 0.912); 1.6x the "
                                                  "2.8 cm start-pose spread bound"),
    R + "FRONTAL_STRETCH_LIMIT_M": _entry("DD", "v1 frontal guard: refuse frontal hovers beyond 0.77 m "
                                                "(agent_speedup_v1.py:95; v1 live run 001 trapped from 0.788)"),
    R + "FRONTAL_STRETCH_STOP_M": _entry("DD", "v1 frontal guard: stop frontal moves at 0.785 m "
                                               "(agent_speedup_v1.py:96)"),
    R + "UNVERIFIED_GUARD_TIGHTENING_M": _entry("RULING", "R-5b: frontal guards tightened by the 2.8 cm start-pose "
                                                          "spread bound in unverified arenas (0.742 / 0.757 m)"),
    R + "FRONTAL_MAX_VERTICAL": _entry("DD", "v1: |approach . z| below 0.5 is a frontal hand "
                                             "(agent_speedup_v1.py:97)"),
    # lint scaffolding
    T + "MIN_TOKEN_CHARS": _entry("DD", "lint tokenizer: tokens shorter than 2 characters are ignored"),
    T + "COMPOUND_MIN_CHARS": _entry("DD", "lint compounds: the vocabulary half of a two-word compound has >= 3 "
                                           "characters"),
    T + "STRUCTURAL_FLOATS": _entry("DD", "contract [C-16]: the float literals allowed inside function bodies "
                                          "without a label (identities, halves and doubles, mm per m, numeric "
                                          "epsilons)"),
}
# M0's central entries before the per-module LABELS are merged (TH-EXT-2).
CENTRAL_LABELS = {name: dict(entry) for name, entry in CONSTANT_LABELS.items()}
# Function-body float literals that need no label (TH-EXT-5, [C-16]). The audit compares the magnitude: a negated
# structural value (-0.5, -2.0, -1e-9) is structural too, because -(0.5 * x) and (-0.5) * x cannot be told apart.
STRUCTURAL_FLOATS = (0.0, 1.0, -1.0, 0.5, 2.0, 1000.0, 1e-6, 1e-9, 1e-12)


def constants_register():
    """The one constants register (M0 entries plus every merged module LABELS). ValueError while a listed module's
    LABELS could not be merged (``LABEL_MERGE_ERRORS``): an incomplete register must never reach a run manifest."""
    if LABEL_MERGE_ERRORS:
        raise ValueError("constants register incomplete; LABELS not merged for: "
                         + "; ".join(f"{name}: {error}" for name, error in sorted(LABEL_MERGE_ERRORS.items())))
    return {name: dict(entry) for name, entry in sorted(CONSTANT_LABELS.items())}


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def constants_sha256():
    return _sha(json.dumps(constants_register(), sort_keys=True, separators=(",", ":")))


# ---------------- lint vocabulary ----------------
def tokens(text):
    """Underscore-, digit-, punctuation- and camelCase-split lowercase tokens of at least MIN_TOKEN_CHARS."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(text))
    return [t for t in re.split(r"[^a-z]+", spaced.lower()) if len(t) >= MIN_TOKEN_CHARS]


def bddl_vocabulary(libero_root):
    """Object/fixture categories, region names and task-sentence words of every BDDL file (all suites)."""
    root = Path(libero_root) / "libero" / "libero" / "bddl_files"
    files = sorted(root.glob("*/*.bddl"))
    if not files:
        raise FileNotFoundError(f"no BDDL files under {root}")
    categories, regions, words = set(), set(), set()
    for path in files:
        text = path.read_text()
        spoken = re.search(r"\(:language([^)]*)\)", text)
        if spoken:
            words.update(tokens(spoken.group(1)))
        for block in ("objects", "fixtures"):
            match = re.search(r"\(:" + block + r"(.*?)\)", text, re.S)
            if match:
                for _name, category in re.findall(r"(\w+)\s*-\s*(\w+)", match.group(1)):
                    categories.update(tokens(category))
        match = re.search(r"\(:regions(.*?)\n\s*\(:fixtures", text, re.S)
        if match:
            for name in re.findall(r"\(\s*(\w+)\s*\n\s*\(:target", match.group(1)):
                regions.update(tokens(name))
    return {"files": len(files), "categories": sorted(categories), "regions": sorted(regions),
            "sentence_words": sorted(words)}


def lint_vocabulary(libero_root, extra_sentences=()):
    source = bddl_vocabulary(libero_root)
    raw = set(source["categories"]) | set(source["regions"]) | set(source["sentence_words"]) | set(PART_NOUNS)
    for line in extra_sentences:
        raw.update(tokens(line))
    vocabulary = sorted(raw - set(LINT_ALLOWLIST))
    return {"vocabulary": vocabulary, "allowlist": sorted(LINT_ALLOWLIST), "part_nouns": sorted(PART_NOUNS),
            "bddl_files": source["files"], "vocabulary_sha256": _sha("\n".join(vocabulary)),
            "allowlist_sha256": _sha("\n".join(sorted(LINT_ALLOWLIST))),
            "part_nouns_sha256": _sha("\n".join(sorted(PART_NOUNS)))}


# ---------------- lint ----------------
def _numeric(node):
    """True when a numeric literal (not a bool) occurs anywhere in the expression."""
    return any(isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)
               for n in ast.walk(node))


def _assigned_numeric_names(body):
    names = []
    for node in body:
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None or not _numeric(value):
            continue
        for target in targets:
            for leaf in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]):
                if isinstance(leaf, ast.Name):
                    names.append(leaf.id)
    return names


def module_numeric_constants(tree):
    """Numeric constants at module level ("NAME") and in class bodies, nested classes included ("Class.NAME")."""
    names = list(_assigned_numeric_names(tree.body))

    def visit(body, prefix):
        for node in body:
            if isinstance(node, ast.ClassDef):
                qualified = f"{prefix}{node.name}"
                names.extend(f"{qualified}.{name}" for name in _assigned_numeric_names(node.body))
                visit(node.body, qualified + ".")
    visit(tree.body, "")
    return names


def numeric_defaults(tree):
    """'function.arg' for every function argument whose default contains a numeric literal."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        args = node.args
        positional = [*args.posonlyargs, *args.args]
        pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
        pairs += [(arg, default) for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is not None]
        name = getattr(node, "name", "<lambda>")
        found.extend(f"{name}.{arg.arg}" for arg, default in pairs if _numeric(default))
    return found


def _imported_modules(tree):
    """Local names bound to third-party library modules (their attributes are not lint tokens)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in THIRD_PARTY_MODULES:
                    names.add((alias.asname or alias.name).split(".")[0])
    return names


def _import_paths(tree):
    """(line, dotted path) of every imported module and every name imported from a module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.lineno, node.module
            for alias in node.names:
                yield node.lineno, (f"{node.module}.{alias.name}" if node.module else alias.name)


def forbidden_imports(tree):
    """One violation per import statement that reaches a forbidden module (or a name inside one)."""
    out, lines = [], set()
    for line, path in _import_paths(tree):
        parts = path.split(".")
        if line not in lines and any(part in FORBIDDEN_IMPORT_COMPONENTS or part.startswith(FORBIDDEN_IMPORT_PREFIXES)
                                     for part in parts):
            lines.add(line)
            out.append({"rule": "forbidden_import", "line": line, "text": path})
    return out


# ------------------------------------------------------------------------- TH-EXT-6: firewall hardening (L9)
_DOTTED_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
M0_STEMS = tuple(Path(name).stem for name in M0_CONTROL_MODULES)
INTERFACE_STEMS = tuple(Path(name).stem for name in MODEL_INTERFACE_MODULES)


def _definition_strings(tree, names):
    """ids of the string constants inside the module-level assignments named in ``names``."""
    exempt = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                exempt |= {id(n) for n in ast.walk(node.value) if isinstance(n, ast.Constant)}
    return exempt


def _firewall_names(tree, exempt_strings):
    """(kind, name, line) for every identifier ("name": names, definitions, arguments, keywords, import aliases),
    attribute, imported path component ("import") and component of a dotted identifier-shaped string constant
    ("string"; the constants in ``exempt_strings`` are skipped)."""
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name):
            yield "name", node.id, line
        elif isinstance(node, ast.Attribute):
            yield "attribute", node.attr, line
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield "name", node.name, line
        elif isinstance(node, ast.arg):
            yield "name", node.arg, line
        elif isinstance(node, ast.keyword) and node.arg:
            yield "name", node.arg, line
        elif isinstance(node, ast.ImportFrom) and node.module:
            for part in node.module.split("."):
                yield "import", part, line
        elif isinstance(node, ast.alias):
            for part in node.name.split("."):
                yield "import", part, line
            if node.asname:
                yield "name", node.asname, line
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in exempt_strings
              and _DOTTED_IDENTIFIER.fullmatch(node.value)):
            for part in node.value.split("."):
                yield "string", part, line


def evaluator_only_name(name):
    """True when ``name`` names an evaluator-only accessor (rule evaluator_name)."""
    return name in EVALUATOR_ONLY_NAMES or bool(set(tokens(name)) & EVALUATOR_ONLY_TOKENS)


def _function_level_imports(tree):
    """Import statements nested in a function body (class bodies inside functions included)."""
    found = []

    def visit(node, nested):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and nested:
            found.append(node)
        nested = nested or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for child in ast.iter_child_nodes(node):
            visit(child, nested)
    visit(tree, False)
    return found


def _imports_v2_module(node):
    names = [alias.name for alias in node.names]
    paths = names if isinstance(node, ast.Import) else [node.module or ""] + names
    return any(part.startswith("agent_speedup_v2") for path in paths for part in path.split("."))


def firewall_violations(tree, module_name):
    """TH-EXT-6 (L9): the firewall rules every CM and MIM obeys (one violation per rule, line and name):

    - ``dynamic_import``: an import of DYNAMIC_IMPORT_MODULES (importlib and its relatives, builtins), a name in
      DYNAMIC_IMPORT_NAMES (``__import__``, ``import_module``, ``spec_from_file_location``, ``__builtins__``, ...)
      as an identifier, attribute or identifier-shaped string, any read of SYS_IMPORT_ATTRIBUTES of ``sys``
      (``sys.modules``, ``sys.path``; ``from sys import modules``), a bare ``sys`` reference (``getattr(sys, ...)``,
      ``vars(sys)``) and any ``.sys`` attribute (``os.sys``);
    - ``dynamic_code``: ``exec``, ``eval`` or ``compile`` as a name (called or not);
    - ``evaluator_name``: see EVALUATOR_ONLY_NAMES / EVALUATOR_ONLY_TOKENS (calls, attribute reads, getattr strings)
      and the method calls in EVALUATOR_ONLY_CALLS;
    - ``scripted_answer_call``: ``x.answer(...)`` outside the MIM;
    - ``bddl_access``: a BDDL path marker in a string constant, outside this module;
    - ``lint_reader``: a LINT_READERS name outside this module;
    - ``lazy_import``: a function-level import of a v2 module that is not LAZY_IMPORT_STATEMENT, outside M0's modules.
    Module-level geometry imports are the firewall test's business (§C2.2 graph)."""
    own = module_name == Path(__file__).stem
    exempt = _definition_strings(tree, FIREWALL_DEFINITIONS) if own else frozenset()
    out, seen = [], set()

    def add(rule, line, text):
        key = (rule, line, str(text))
        if key not in seen:
            seen.add(key)
            out.append({"rule": rule, "line": line, "text": str(text)[:80]})
    sys_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DYNAMIC_IMPORT_MODULES:
                    add("dynamic_import", node.lineno, alias.name)
                if alias.name == "sys":
                    sys_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if node.module.split(".")[0] in DYNAMIC_IMPORT_MODULES:
                add("dynamic_import", node.lineno, node.module)
            if node.module == "sys":
                for alias in node.names:
                    if alias.name in SYS_IMPORT_ATTRIBUTES or alias.name == "*":
                        add("dynamic_import", node.lineno, f"sys.{alias.name}")
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in sys_names:
            parent = parents.get(id(node))
            if not (isinstance(parent, ast.Attribute) and parent.value is node
                    and parent.attr not in SYS_IMPORT_ATTRIBUTES):
                add("dynamic_import", node.lineno, f"sys.{parent.attr}" if isinstance(parent, ast.Attribute)
                    else "sys")
        elif isinstance(node, ast.Attribute) and node.attr == "sys":
            add("dynamic_import", node.lineno, "sys")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in SCRIPTED_ANSWER_METHODS and module_name not in INTERFACE_STEMS:
                add("scripted_answer_call", node.lineno, node.func.attr)
            if node.func.attr in EVALUATOR_ONLY_CALLS:
                add("evaluator_name", node.lineno, node.func.attr)
        elif (not own and isinstance(node, ast.Constant) and isinstance(node.value, str)
              and any(marker in node.value for marker in BDDL_PATH_MARKERS)):
            add("bddl_access", node.lineno, node.value)
    for kind, name, line in _firewall_names(tree, exempt):
        if name in DYNAMIC_IMPORT_NAMES:
            add("dynamic_import", line, name)
        if kind == "name" and name in DYNAMIC_CODE_BUILTINS:
            add("dynamic_code", line, name)
        if evaluator_only_name(name):
            add("evaluator_name", line, name)
        if not own and name in LINT_READERS:
            add("lint_reader", line, name)
    if module_name not in M0_STEMS:
        for node in _function_level_imports(tree):
            if _imports_v2_module(node) and ast.unparse(node) != LAZY_IMPORT_STATEMENT:
                add("lazy_import", node.lineno, ast.unparse(node))
    return out


def token_hits(token, vocabulary, known=()):
    """Vocabulary words a token stands for: itself, its -s / -es singular, or one half of a two-word compound
    whose other half is a vocabulary or known (allowlisted) word."""
    variants = {token}
    if token.endswith("s") and len(token) > MIN_TOKEN_CHARS + 1:
        variants.add(token[:-1])
    if token.endswith("es") and len(token) > MIN_TOKEN_CHARS + 2:
        variants.add(token[:-2])
    hits = {v for v in variants if v in vocabulary}
    words = set(vocabulary) | set(known)
    for variant in variants:
        for cut in range(MIN_TOKEN_CHARS, len(variant) - MIN_TOKEN_CHARS + 1):
            head, tail = variant[:cut], variant[cut:]
            for word, other in ((head, tail), (tail, head)):
                if len(word) >= COMPOUND_MIN_CHARS and word in vocabulary and other in words:
                    hits.add(word)
    return hits


def _vocabulary_definition_strings(tree):
    """ids of the string constants inside this module's own vocabulary definitions (module-level assignments
    named in VOCABULARY_DEFINITIONS)."""
    exempt = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in VOCABULARY_DEFINITIONS for t in targets):
                exempt |= {id(n) for n in ast.walk(node.value) if isinstance(n, ast.Constant)}
    return exempt


def _identifiers_and_strings(tree, exempt_strings=frozenset()):
    """Identifiers, string constants, imported names and module paths; attributes of third-party library modules
    (ast.X, np.X) and the string constants in ``exempt_strings`` (node ids) are skipped."""
    modules = _imported_modules(tree)
    for line, path in _import_paths(tree):
        yield "import", path, line
    for node in ast.walk(tree):
        if id(node) in exempt_strings:
            continue
        if isinstance(node, ast.Name):
            yield "identifier", node.id, node.lineno
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in modules:
                continue
            yield "identifier", node.attr, node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield "identifier", node.name, node.lineno
        elif isinstance(node, ast.arg):
            yield "identifier", node.arg, node.lineno
        elif isinstance(node, ast.keyword) and node.arg:
            yield "identifier", node.arg, getattr(node, "lineno", 0)
        elif isinstance(node, ast.alias):
            yield "identifier", node.name, getattr(node, "lineno", 0)
            if node.asname:
                yield "identifier", node.asname, getattr(node, "lineno", 0)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield "string", node.value, node.lineno


# ---------------------------------------------------------------- TH-EXT-2: per-module LABELS registers
LABEL_MERGE_ERRORS = {}
# "NAME", "Class.NAME", "function.arg"; a lambda default is keyed "<lambda>.arg", exactly as numeric_defaults
# reports it (the last component is always an identifier).
_LABEL_KEY = re.compile(r"(?:(?:[A-Za-z_]\w*|<lambda>)\.)*[A-Za-z_]\w*")


def _labels_assignment(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == LOCAL_LABELS_NAME for t in targets):
            return node.value
    return None


def local_labels(tree):
    """The module's top-level ``LABELS`` literal as {key: (label, derivation)}, keys in the three M0 forms without
    the module prefix ("NAME", "Class.NAME", "function.arg"); {} when absent. ValueError when it is not a pure dict
    literal (no names, f-strings, operators or unpacking), repeats a key, or holds an entry that is not a
    (label in LABEL_CLASSES, nonempty derivation) pair."""
    value = _labels_assignment(tree)
    if value is None:
        return {}
    if not isinstance(value, ast.Dict):
        raise ValueError(f"{LOCAL_LABELS_NAME} must be a dict literal")
    seen = set()
    for key in value.keys:
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            raise ValueError(f"{LOCAL_LABELS_NAME} keys must be string literals (no unpacking)")
        if key.value in seen:
            raise ValueError(f"{LOCAL_LABELS_NAME} key defined twice: {key.value}")
        seen.add(key.value)
    try:
        literal = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise ValueError(f"{LOCAL_LABELS_NAME} must be a pure literal: {exc}") from exc
    out = {}
    for key, entry in literal.items():
        if not _LABEL_KEY.fullmatch(key):
            raise ValueError(f"{LOCAL_LABELS_NAME} key {key!r} is not NAME, Class.NAME, function.arg or "
                             "<lambda>.arg")
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"{LOCAL_LABELS_NAME}[{key!r}] must be a (label, derivation) pair")
        label, derivation = entry
        if label not in LABEL_CLASSES:
            raise ValueError(f"{LOCAL_LABELS_NAME}[{key!r}] label must be one of {LABEL_CLASSES}")
        if not isinstance(derivation, str) or not derivation.strip():
            raise ValueError(f"{LOCAL_LABELS_NAME}[{key!r}] needs a nonempty derivation")
        out[key] = (label, derivation)
    return out


def module_labels(path, *, module_name=None):
    """TH-EXT-2: a module's LABELS register as CONSTANT_LABELS entries ("<stem>.<key>" -> {"label",
    "derivation"}), read from the AST; the module is never imported."""
    path = Path(path)
    stem = module_name or path.stem
    tree = ast.parse(path.read_text(), filename=str(path))
    return {f"{stem}.{key}": _entry(label, derivation) for key, (label, derivation) in local_labels(tree).items()}


def _collect_labels(package_dir):
    """(CENTRAL_LABELS + every existing listed module's LABELS, {module file: error})."""
    labels = {name: dict(entry) for name, entry in CENTRAL_LABELS.items()}
    errors = {}
    for name in PLANNED_CONTROL_MODULES + MODEL_INTERFACE_MODULES:
        path = Path(package_dir) / name
        if not path.is_file():
            continue
        try:
            entries = module_labels(path)
        except (ValueError, SyntaxError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        twice = sorted(set(entries) & set(labels))
        if twice:
            errors[name] = "constant label defined twice: " + ", ".join(twice)
            continue
        labels.update(entries)
    return labels, errors


def merged_labels(package_dir=None):
    """CENTRAL_LABELS merged with the LABELS of every existing listed module (CM or MIM) of ``package_dir``;
    ValueError on an invalid LABELS literal or a key defined twice."""
    labels, errors = _collect_labels(Path(package_dir) if package_dir is not None else PACKAGE_DIR)
    if errors:
        raise ValueError("; ".join(f"{name}: {error}" for name, error in sorted(errors.items())))
    return labels


def _merge_local_labels():
    """Import-time merge into CONSTANT_LABELS. A module whose LABELS is invalid (or repeats a key) contributes no
    labels and is recorded in LABEL_MERGE_ERRORS, which fails lint_control_modules; the import itself never fails,
    so one half-written module cannot break every other module's import during a parallel build."""
    labels, errors = _collect_labels(PACKAGE_DIR)
    LABEL_MERGE_ERRORS.clear()
    LABEL_MERGE_ERRORS.update(errors)
    CONSTANT_LABELS.update({name: entry for name, entry in labels.items() if name not in CONSTANT_LABELS})


# ------------------------------------------------------------------------------- TH-EXT-3 / 4: the lint
def lint_module(path, vocabulary, *, labels=None, module_name=None, allow_task_text=False):
    """T0 lint of one control module; returns a list of violations (empty = pass).

    ``labels=None`` uses CONSTANT_LABELS plus the module's own LABELS literal; an explicit ``labels`` is used as
    given. ``allow_task_text=True`` (the model-interface module) skips only the task-text identifier rule. Every
    LABELS key that names no numeric module constant, class attribute or function default is an orphan
    (TH-EXT-4). The firewall rules of ``firewall_violations`` (TH-EXT-6) apply to every module."""
    path = Path(path)
    module_name = module_name or path.stem
    tree = ast.parse(path.read_text(), filename=str(path))
    vocab = set(vocabulary)
    known = set(LINT_ALLOWLIST)
    violations = []
    try:
        own = local_labels(tree)
    except ValueError as exc:
        own = {}
        violations.append({"rule": "invalid_labels", "name": f"{module_name}.{LOCAL_LABELS_NAME}",
                           "text": str(exc)[:160]})
    if labels is None:
        labels = {**CONSTANT_LABELS, **{f"{module_name}.{key}": _entry(*entry) for key, entry in own.items()}}

    def labelled(key):
        entry = labels.get(key)
        return isinstance(entry, dict) and entry.get("label") in LABEL_CLASSES and bool(entry.get("derivation"))
    constants, defaults = module_numeric_constants(tree), numeric_defaults(tree)
    for name in constants:
        key = f"{module_name}.{name}"
        if not labelled(key):
            violations.append({"rule": "unlabelled_constant", "name": key})
    for name in defaults:
        key = f"{module_name}.{name}"
        if not labelled(key):
            violations.append({"rule": "numeric_default", "name": key})
    numeric_keys = set(constants) | set(defaults)
    for key in own:
        if key not in numeric_keys:
            violations.append({"rule": "orphan_label", "name": f"{module_name}.{key}"})
    violations.extend(forbidden_imports(tree))
    violations.extend(firewall_violations(tree, module_name))
    # Only the lint's own module may define the vocabulary lists without being flagged by them.
    exempt = _vocabulary_definition_strings(tree) if module_name == Path(__file__).stem else frozenset()
    for kind, text, line in _identifiers_and_strings(tree, exempt):
        hits = sorted(set().union(*(token_hits(t, vocab, known) for t in tokens(text))))
        if hits:
            violations.append({"rule": "lint_vocabulary", "kind": kind, "line": line, "tokens": hits,
                               "text": text[:80]})
        if not allow_task_text and kind == "identifier" and text.lower() in TASK_TEXT_IDENTIFIERS:
            violations.append({"rule": "task_text_identifier", "line": line, "text": text})
    return violations


def lint_control_modules(libero_root, package_dir=None, extra_sentences=(), *, require_all=False):
    """T0 lint of every existing control module (``report["modules"]``, the CONTROL_MODULES of the package) and of
    the existing model-interface modules (``report["interface_modules"]``, task text allowed). ``passed`` needs
    both clean and no label-merge error; with ``require_all`` every planned module and the client must also exist
    (``report["missing"]``). The default package uses the import-time module lists and CONSTANT_LABELS; another
    ``package_dir`` is scanned and merged afresh."""
    default = package_dir is None or Path(package_dir).resolve() == PACKAGE_DIR
    package_dir = PACKAGE_DIR if default else Path(package_dir).resolve()
    vocab = lint_vocabulary(libero_root, extra_sentences)
    if default:
        modules, interface = CONTROL_MODULES, PRESENT_INTERFACE_MODULES
        labels, errors = CONSTANT_LABELS, dict(LABEL_MERGE_ERRORS)
    else:
        modules = tuple(name for name in M0_CONTROL_MODULES + PLANNED_CONTROL_MODULES
                        if (package_dir / name).is_file())
        interface = tuple(name for name in MODEL_INTERFACE_MODULES if (package_dir / name).is_file())
        labels, errors = _collect_labels(package_dir)

    def lint(name, allow_task_text):
        found = lint_module(package_dir / name, vocab["vocabulary"], labels=labels, module_name=Path(name).stem,
                            allow_task_text=allow_task_text)
        if name in errors:
            found.append({"rule": "label_merge", "name": name, "text": errors[name][:160]})
        return found
    report = {name: lint(name, False) for name in modules}
    interface_report = {name: lint(name, True) for name in interface}
    linted = set(modules) | set(interface)
    missing = [name for name in M0_CONTROL_MODULES + PLANNED_CONTROL_MODULES + MODEL_INTERFACE_MODULES
               if name not in linted]
    passed = (all(not v for v in report.values()) and all(not v for v in interface_report.values())
              and not (require_all and missing))
    return {"modules": report, "interface_modules": interface_report, "missing": missing,
            "require_all": bool(require_all), "passed": passed,
            "vocabulary_sha256": vocab["vocabulary_sha256"], "allowlist_sha256": vocab["allowlist_sha256"],
            "part_nouns_sha256": vocab["part_nouns_sha256"], "vocabulary_size": len(vocab["vocabulary"])}


# ------------------------------------------------------------------------------ TH-EXT-5: literal audit
def _numeric_literal(node):
    """Value of a numeric literal node with any unary sign folded in; None for anything else."""
    sign = 1
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        sign = -sign if isinstance(node.op, ast.USub) else sign
        node = node.operand
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return sign * node.value
    return None


def _plain_module_assignment(node):
    """A top-level Assign / AnnAssign whose targets are all plain names (tuple / list unpacking of names included):
    exactly the statements M0's ``unlabelled_constant`` rule sees."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return False
    leaves = [leaf for target in targets
              for leaf in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target])]
    return all(isinstance(leaf, ast.Name) for leaf in leaves)


def literal_audit(path):
    """TH-EXT-5 [C-16]: every float literal in a function body whose magnitude is not in STRUCTURAL_FLOATS (rule
    ``float_literal``), every numeric literal in a function default or a class body (rule ``use_named_constant``:
    use a labelled module constant, None, or a required field), and every numeric literal in a module-level
    statement that is not a plain name assignment (rule ``module_compound_literal``: an ``if`` / ``try`` / ``for`` /
    ``while`` / ``with`` block, an augmented, subscript or attribute assignment, an expression), which the label
    rule cannot see. Plain module-level assignments are the lint's business (``unlabelled_constant``); definitions
    nested in a compound statement are audited by the same function / class rules."""
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []

    def visit(node, context, owner):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            name = getattr(node, "name", "<lambda>")
            args = node.args
            positional = [*args.posonlyargs, *args.args]
            pairs = list(zip(positional[len(positional) - len(args.defaults):], args.defaults))
            pairs += [(arg, default) for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is not None]
            for arg, default in pairs:
                for sub in ast.walk(default):
                    value = _numeric_literal(sub)
                    if value is not None and isinstance(sub, ast.Constant):
                        found.append({"rule": "use_named_constant", "where": "default", "name": f"{name}.{arg.arg}",
                                      "line": sub.lineno, "value": value})
            for decorator in getattr(node, "decorator_list", ()):
                visit(decorator, context, owner)
            for child in (node.body if isinstance(node.body, list) else [node.body]):
                visit(child, "function", name)
            return
        if isinstance(node, ast.ClassDef):
            for child in [*node.bases, *(k.value for k in node.keywords), *node.decorator_list]:
                visit(child, context, owner)
            for child in node.body:
                visit(child, "class", node.name)
            return
        if context != "module":
            value = _numeric_literal(node)
            if value is not None:
                if context == "class":
                    found.append({"rule": "use_named_constant", "where": "class_body", "name": owner,
                                  "line": node.lineno, "value": value})
                elif context == "compound":
                    found.append({"rule": "module_compound_literal", "where": "module_statement", "name": owner,
                                  "line": node.lineno, "value": value})
                elif isinstance(value, float) and abs(value) not in STRUCTURAL_FLOATS:
                    found.append({"rule": "float_literal", "where": "function_body", "name": owner,
                                  "line": node.lineno, "value": value})
                return
        for child in ast.iter_child_nodes(node):
            visit(child, context, owner)
    for node in tree.body:
        plain = _plain_module_assignment(node) or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                                     ast.ClassDef))
        visit(node, "module" if plain else "compound", None if plain else type(node).__name__)
    return found


# ------------------------------------------------------------------- spec §5.2 threshold ids (contract §C16)
SLICE_STATUSES = ("yes", "record", "reserved", "test")
_P = "agent_speedup_v2_"


def _tid(owners, names, labels, status, rule):
    if status not in SLICE_STATUSES or any(label not in LABEL_CLASSES for label in labels):
        raise ValueError(f"bad threshold id entry: {rule}")
    return {"owners": tuple(owners), "names": tuple(names), "labels": tuple(labels), "slice": status, "rule": rule}


_ROBOT, _PRIM, _GEO, _SCENE = (_P + "robot",), (_P + "primitives",), (_P + "geometry",), (_P + "scene",)
_PLAN, _OBS, _REL, _POLICY = (_P + "planner",), (_P + "observers",), (_P + "relations",), ("agent_speedup_v2",)
_RO, _ON, _DD, _RU = "ROBOT", "ONLINE", "DD", "RULING"
# INFORMATIONAL: a transcription of contract §C16 (every spec §5.2 / §4.4 threshold id with its owner module(s), the
# runtime Threshold names the owner logs, the label classes a winning term may carry, the slice status and the rule,
# as amended by the contract and its Appendix L lead decisions), for diagnosis reports and reviews. It is not an
# authority: no control code reads it, it is not part of constants_sha256, and the rule texts are summaries. The
# values and their labels live only in the owner modules' labelled constants (LABELS); where this list and a module
# section of the contract disagree, the module section and the owner's LABELS win.
THRESHOLD_IDS = {
    # ROBOT facts (spec §4.4; M0 robot module, wrapped by the kernel RobotModel)
    "R1": _tid(_ROBOT, ("GRIP_SITE_ALONG_R2_M",), (_RO,), "yes", "grip site = hand + 0.097 R2"),
    "R2": _tid(_ROBOT, ("WRIST_CAMERA_ALONG_R0_M",), (_RO,), "yes",
               "wrist camera = hand + 0.05 R0, optical axis R2; image right = R1, up = R0; centred principal point"),
    "R3": _tid(_ROBOT, ("FOCAL_PX_AT_512",), (_RO,), "yes", "f = 333.6 px at 512 px (ruling R-2f)"),
    "R4": _tid(_ROBOT, ("PAD_LENGTH_R0_MM", "PAD_THICKNESS_R1_MM"), (_RO,), "yes",
               "pads span -11.6 .. +4.4 mm along the approach about the grip; 8 mm thick, 16 mm long along R0"),
    "R5": _tid(_ROBOT, ("FINGERTIP_REACH_MM",), (_RO,), "yes", "fingertip reach 9.3 mm beyond the grip"),
    "R6": _tid(_ROBOT, ("PALM_UNDERSIDE_MM",), (_RO,), "yes", "palm underside 31.0 mm behind the grip (profile p(xi))"),
    "R7": _tid(_ROBOT, (), (_RO,), "yes", "hand footprint R1 -104.0 .. +100.4 mm, R0 +-31.6 mm"),
    "R8": _tid(_ROBOT, (), (_RO,), "yes", "pad faces and finger joints 0-40 mm each (finger landing footprints)"),
    "R9": _tid(_ROBOT, ("PAD_SEPARATION_OFFSET_MM",), (_RO,), "yes", "pad separation = finger gap - 1.0 mm"),
    "R10": _tid(_ROBOT, ("ROTATION_CAP_RAD_PER_STEP",), (_RO,), "yes",
                "0.05 m per unit translation, rotation cap 0.15 rad per step, 20 Hz"),
    "R11": _tid(_ROBOT, (), (_RO,), "yes", "gripper ramp 0.01 sign(a) per substep; action 0 holds"),
    "R12": _tid(_ROBOT, ("JOINT7_LIMIT_RAD", "YAW_BUDGET_RAD"), (_RO,), "yes",
                "net hand-yaw budget about the approach 2.8973 - pi/4 - 0.15 = 1.96 rad from the start"),
    "R13": _tid(_ROBOT, ("WRIST_FROM_GRIP_M", "FULL_STRETCH_M", "VERIFIED_MOUNT_GATE_M", "FRONTAL_STRETCH_LIMIT_M",
                         "FRONTAL_STRETCH_STOP_M"), (_RO, _RU), "yes",
                "verified mount base, shoulder 0.333 m above it, wrist = grip - 0.204 R2, stretch 0.804 m (R-5)"),
    "R14": _tid(_ROBOT, (), (_RO, _RU), "yes", "start pose relative to the base and its spread bound (R-5b)"),
    # gripper and contact (G)
    "G1": _tid(_PRIM, ("G1_open_gap_mm",), (_ON,), "yes", "open gap g_open from the Calibrate+ open hold"),
    "G2": _tid(_PRIM, ("G2_empty_k15_mm",), (_ON,), "yes", "empty-close curve gap(k) from Calibrate+"),
    "G3": _tid(_PRIM, ("G3_touch_threshold_mm",), (_DD, _ON), "yes", "finger touch max(0.10 mm, 3 sigma_in)"),
    "G4": _tid(_PRIM, ("G4_sigma_close_mm",), (_ON, _DD), "yes",
               "sigma_close = |delta gap(15)| / sqrt 2 over two empty closes (a third when they disagree beyond the "
               "G5 floor); numerical floor [C-8]"),
    "G5": _tid(_PRIM, ("G5_object_present_gap_mm",), (_ON, _DD), "yes",
               "object present: gap(15) > empty(15) + max(0.5 mm, 4 sigma_close)"),
    "G6": _tid(_PRIM, ("G6_hold_floor_mm",), (_ON, _DD), "yes",
               "hold floor = empty floor + max(0.5 mm, 6 sigma_close); co-motion O3 required as well"),
    "G7": _tid(_PRIM, ("G7_carry_total_rise_mm", "G7_carry_step_rise_mm"), (_ON, _DD), "yes",
               "held contact while carrying: rise >= max(1.0 mm, 6 sigma_carry) in total or >= max(0.4 mm, "
               "4 sigma_carry) per step; provisional floors until re-derived from robot-only traces"),
    "G8": _tid(_PRIM, ("G8_lower_step_rise_mm",), (_ON, _DD), "yes",
               "support cue while lowering: per-step rise >= max(0.3 mm, 4 sigma_lower), or a stall, or a touch "
               "(provisional floor)"),
    "G9": _tid(_PRIM, ("G9_open_finger_contact_mm",), (_ON, _DD), "record",
               "open-finger contact during reaches |delta gap| >= max(1.0 mm, 10 sigma_in) (v1 WaypointReach)"),
    "G10": _tid((), (), (), "reserved", "slip gap jump (slides only) max(0.2 mm, 5 sigma_carry)"),
    "G12": _tid(_PRIM, ("G12_stall_ratio",), (_ON, _DD), "record",
                "stall: 2-step realized/commanded < 0.6 x the free ratio, clipped 0.18-0.40 (v1 AnchoredMove); v6 "
                "RC-8: hover probe legs decide from step 5 (PROBE_STALL_MIN_STEPS) with the own free ratio clipped "
                "0.08-0.40 (PROBE_FREE_RATIO_FLOOR); Calibrate+ legs, GuardedLower and Touchdown unchanged"),
    "G13": _tid(_PLAN, ("G13_width_band_mm",), (_RO, _ON, _DD), "yes",
                "width consistency: gap - 1.0 mm within [w_min - max(3 sigma_w, 3 mm), w_max + max(3 sigma_w, 3 mm)]; "
                "v6 G13+ (free parts, right after the close and before the load test): misaligned when (gap - 1.0) "
                "- w_min > max(3 mm, 3 sigma_w) + 16 mm sin(YAW_GRID_RAD / 2)"),
    # depth and geometry (D)
    "D1": _tid(_GEO, ("D1_min_shift_px", "D1_max_residual_px"), (_DD,), "yes",
               "kappa probe validity: shift >= 6 px, residual < 0.1 shift (floor 0.6 px), leg agreement 0.35"),
    "D2": _tid(_PRIM, (), (_DD,), "yes", "hover probe amplitude +-2 cm out and back along R1, then R0"),
    "D3": _tid(_GEO, ("D3_f_agreement",), (_ON, _DD), "yes",
               "f-hat by looming (vertical +-4 cm); agreement with R3 within max(5 %, 3 sigma)"),
    "D4": _tid(_GEO, (), (_ON,), "yes", "sigma of a kappa height from fit residuals and the 1.5 px pixel noise; v6 "
                                         "(ruling 4b): D4_MODEL_SIGMA_M is diagnostic only, the model term is "
                                         "applied once per part (TOP_MODEL_M 3 mm, TOP_MODEL_THIN_M 5 mm when the "
                                         "top band has fewer than TOP_BAND_SUPPORT_N members)"),
    "D5": _tid(_SCENE, ("D5_fingertip_reach_m",), (_RO, _ON), "yes",
               "touch height: surface = grip z - t_f; sigma_touch = one stall window of realized motion"),
    "D6": _tid((), (), (), "reserved", "dense flow validity"),
    "D7": _tid(_SCENE, ("D7_colour_band", "D7_fill_height_m"), (_DD, _ON), "yes",
               "unknown cells: plane fill at plane z + 3 sigma_plane within 2 cm of the plane cells, no D9 edge and "
               "|delta Lab| <= 3 sigma_plane; otherwise an obstacle at max(known neighbours within 2 cm) + 3 sigma; v6: "
               "the obstacle sigma is min(sigma_parent, HF_CELL_M) (RC-9a, D7_PARENT_SIGMA_CAP_M); D7b fills are "
               "unverified for contacts (RC-1); DSW (S-5) withdraws dense-lattice hover samples above support + "
               "D8 floor outside the region dilated by one cell and refills; part-band fills (S-6) connected to the "
               "region within PB_REACH_M of the record point are the part's halo, excluded in P7 only"),
    "D8": _tid(_SCENE, ("D8_same_plane_m",), (_DD, _ON), "yes", "same plane |delta h| <= max(3 mm, 3 sigma_h); "
                                                                "v6: the part-level model term never widens it; v7 "
                                                                "S-14: a whole registration whose top reads below "
                                                                "an existing lower-bound top of the part by more "
                                                                "than max(D8_FLOOR_M, 3 sigma_h) keeps the lower "
                                                                "bound and top_upper_z (top_inconsistent)"),
    "D9": _tid(_SCENE, ("D9_gradient_ratio", "D9_depth_step_m"), (_DD, _ON), "yes",
               "region edge: |grad I| > 4 x the local median, or a depth step > 3 sigma_h"),
    "D10": _tid(_SCENE, ("D10_coherence", "D10_aspect"), (_DD,), "yes",
                "elongated: structure-tensor coherence >= 0.5 and aspect >= 1.5; v6 S-4': accepted only with >= "
                "D10_MIN_REGION_CELLS region cells and the image or mask tensor axis within YAW_GRID_RAD / 2 of the "
                "region-cell PCA axis, otherwise refused with axis_check source axis_inconsistent; v7 S-11 (D10-OE, "
                "axis_check method outline_edges) replaces S-4' and the coherence, aspect and cell gates: the "
                "outline axis (share >= D10_OUTLINE_SHARE_MIN, second peak <= D10_OUTLINE_SECOND_MAX, two-sided) "
                "is accepted within AXIS_AGREE_MAX_RAD of a gated edge axis (share >= D10_EDGE_SHARE_MIN) in the "
                "region dilated D10_EDGE_REGION_DILATE_PX or the top-band hull dilated D10_EDGE_HULL_DILATE_PX; a "
                "failed outline gate is not_elongated (reason outline_gate), no agreeing edge axis_inconsistent "
                "(no_edge_agreement); S-12: an accepted axis that disagrees with an axis remembered for the part "
                "since part_moved is refused (views_disagree) and the memory conflicted, a refusal under an "
                "unconflicted memory carries the remembered axis (kept_earlier); S-13: a top-band hull above "
                "REGION_COLLAPSE_RATIO x the region refuses the axis and its width (region_collapsed)"),
    "D11": _tid(_SCENE, ("D11_rms_px",), (_DD,), "yes", "circle fit accepted when rms <= max(1.5 px, 3 % r)"),
    "D12": _tid((), (), (), "reserved", "cavity"),
    "D13": _tid((), (), (), "reserved", "mount side"),
    "D14": _tid(_GEO, ("D14_residual_deg",), (_DD,), "yes", "external Jacobian residual 30 deg (v1)"),
    "D15": _tid(_GEO, ("D15_min_std_grey",), (_DD,), "yes", "feature texture test: patch std >= 4 grey (21 px); "
                                                             "v6 probe-tracker mode: band-passed patch std >= 1.0 "
                                                             "grey (D15_PROBE_MIN_STD, 11 px templates)"),
    # contact planning (P)
    "P1": _tid(_PLAN, ("P1_e_lat_mm",), (_ON, _RO), "yes", "lateral error budget 2 sigma_align + sigma_w + 1.0 mm"),
    "P2": _tid(_PLAN, ("P2_width_feasible_mm", "P2_width_infeasible_mm"), (_RO, _ON), "yes",
               "width three-way: feasible w + 2 e_lat <= sep; infeasible w - 2 sigma_w >= sep - 2 mm; else marginal; "
               "v7 P-I: a record whose width_by_yaw is () (S-13 collapsed region: width unmeasured, K-5) answers "
               "geometry_unmeasured"),
    "P3": _tid(_PLAN, ("P3_pad_overlap_mm",), (_RO,), "yes", "pad overlap >= 8 mm (half the pad)"),
    "P4": _tid(_PLAN, ("P4_known_fraction",), (_DD,), "yes",
               "landing: each finger column has >= 70 % known cells (fills count); its highest surface is the landing; "
               "v6 RC-1a: the landing is the verified maximum (measured cells, D7a fills, keep-outs) or the support, "
               "and D7b fills widen the expected band below the P3 cap (unverified_grip_z)"),
    "P5": _tid(_PLAN, ("P5_sweep_margin_m",), (_RO, _ON), "yes",
               "closing sweep: no height above landing + 3 sigma_h between the landing and the part edge; v6 RC-1b: "
               "D7b fills and measured cells within |along jaw| <= w/2 + e_lat + cell/2 are close_verified, and a "
               "candidate admitted only through unverified or close-verified cells is marginal"),
    "P6": _tid(_PLAN, ("P6_target_grip_z_m", "P6_palm_margin_m", "P6_landing_margin_m"), (_RO, _ON), "yes",
               "height stop g* = top - 11.6 mm - max(2 mm, 1.5 sigma_h); palm g* + p_min >= top + m_palm with "
               "m_palm = max(5 mm, 2 sigma_h) for free objects and 0 for fixtures; both column landings <= g* - "
               "4.4 mm - max(3 mm, 3 sigma_h); for a free object feasible only while sigma_h <= 5.54 mm, refine the "
               "height first otherwise (Appendix L 3)"),
    "P7": _tid(_PLAN, ("P7_footprint_margin_m",), (_RO, _ON), "yes",
               "hand-footprint clearance excluding the part region and the finger slot <= grip z + p_min - "
               "max(5 mm, 3 sigma_h); v6 RC-1c/d: no sigma counted twice at a D7b argmax, and the scene's "
               "part_fill_cells are excluded"),
    "P8": _tid(_PLAN, ("P8_band_m",), (_ON, _DD), "yes", "contact classification band expected +- max(5 mm, "
                                                          "2 sigma_h); v6 P-B: on a fixture-prior part a "
                                                          "touch_stop contact below every band and above the P9 "
                                                          "bound is expected with landing_measured"),
    "P9": _tid(_PLAN, ("P9_bound_grip_z_m",), (_ON, _DD), "yes", "approach bound expected contact + max(3 cm, "
                                                                  "3 sigma_h)"),
    "P10": _tid(_PLAN, ("P10_retreat_m",), (_RO,), "yes",
                "clearance retreat centres the pad band on [contact surface, part top], clipped 0-8 mm"),
    "P11": _tid(_PLAN, ("P11_hover_clearance_m",), (_DD, _ON), "yes",
                "hover: fingertips >= max(5 cm, min(3 sigma_h + 2 cm, 10 cm)) above the part top (v6 P-E: the "
                "P11_CAP_M cap; a binding cap flags first_stage; a registered record's own top and region are the "
                "reference)"),
    "P12": _tid(_PLAN, ("P12_turn_clearance_m",), (_RO, _ON), "yes",
                "turned-hand clearance: the swept hand clears the height field by max(1 cm, 3 sigma_h); v6 P-F: a "
                "lower-bound part's cells are raised to its top_upper_z"),
    "P13": _tid(_PLAN, (), (), "yes", "rule: never a palm stop on a free (or unknown) object [C-47]"),
    "P14": _tid(_REL, ("P14_segment_m",), (_DD,), "yes", "ladder 20 mm segments, 3 settle steps (v1)"),
    "P15": _tid(_PRIM, ("P15_command_m", "P15_lateral_cap_m"), (_DD,), "yes",
                "guarded command 10 mm per step, lateral cap 3 mm per step (v1 AnchoredMove)"),
    "P16": _tid(_PLAN, ("P16_column_inflate_m",), (_RO, _ON), "yes",
                "finger-descent column: each finger's lowest footprint at the measured opening, inflated by e_lat"),
    "P17": _tid(_PLAN, (), (_RO,), "yes",
                "held-object visibility: prefer the 180 deg twin with R0 . (held centre - grip) >= 0; score +0.3"),
    # motion and placement (M)
    "M1": _tid(_PLAN, ("M1_transit_clearance_m",), (_DD, _ON), "yes",
               "transit: lowest held point >= swept max height + max(2 cm, 3 sigma_h); v6 P-F: a lower-bound "
               "part's cells are raised to its top_upper_z"),
    "M2": _tid(_PLAN, ("M2_segment_m",), (_DD,), "yes", "held segment length <= 7.5 cm"),
    "M3": _tid(_PRIM, ("M3_reorient_rad",), (_DD,), "yes", "held reorientation <= 15 deg (v1)"),
    "M4": _tid(_PLAN, ("M4_aim_tol_m",), (_ON, _DD), "yes", "aim tolerance max(5 mm, 0.5 m_free)"),
    "M5": _tid(_PLAN + _PRIM, ("M5_drift_abort_m", "M5_saturated_steps"), (_ON, _DD), "yes",
               "lowering drift abort at max(5 mm, 0.5 m_free); <= 2 re-aims"),
    "M6": _tid(_PLAN, ("M6_fast_until_grip_z_m",), (_ON,), "yes",
               "fast-to-slow switch 3 cm + 3 sigma_h above the expected support"),
    "M7": _tid(_PLAN, ("M7_bound_grip_z_m",), (_ON, _DD), "yes",
               "lowering bound S_dest + b_low - max(1 cm, 2 tol_h)"),
    "M8": _tid(_PLAN, ("M8_tol_h_m",), (_ON, _DD), "yes",
               "support height tol_h = max(1 cm, 3 sqrt(sigma_dest^2 + sigma_b^2 + sigma_touch^2))"),
    "M9": _tid((), (), (), "reserved", "inside: bottom <= rim - max(1 cm, 2 sigma_rim)"),
    "M10": _tid((), (), (), "reserved", "ambiguous cavity: rim - floor < 2 tol_h -> floor touch probe first"),
    "M11": _tid(_OBS, ("M11_rest_diff_grey",), (_DD, _ON), "yes",
                "at rest: frame difference over the object region <= max(3 grey, 3 sigma_static) for 10 frames"),
    "M12": _tid(_PRIM, ("M12_released_gap_mm",), (_DD,), "yes", "released: gap >= g_open - 2 mm (v1)"),
    "M13": _tid(_PRIM, ("M13_retreat_m",), (_DD,), "yes", "retreat 3 cm along -R2 (v1)"),
    "M14": _tid(_SCENE, ("M14_gap_change_mm", "M14_iou", "M14_shift_px"), (_ON, _DD), "yes",
                "touchdown trigger: lift gap change > max(1 mm, 6 sigma_close), or IoU < 0.8, or template shift > "
                "max(4 px, 3 sigma_track), or not wrist_visible"),
    "M15": _tid((), (), (), "reserved", "fits(p) travel stop"),
    "M16": _tid(_PLAN, ("M16_free_radius_m",), (_ON, _DD), "yes",
                "free target spot: centre of the largest free disk of radius r_obj + max(5 mm, sigma_offset)"),
    "M17": _tid(_SCENE, ("M17_sigma_offset_m",), (_DD,), "yes",
                "offset sigma fallback: IoU < 0.8 or not wrist_visible -> max(measured, 0.5 r_obj)"),
    "M18": _tid((), (), (), "reserved", "axis-specific fit"),
    # mechanisms (K)
    "K1w": _tid(_OBS, ("K1w_free_rate_mm", "K1w_blocked_rate_mm", "K1w_unwedge_steps"), (_RO, _ON, _DD), "yes",
                "twist sense: 12-step segment (extend to 16); early = mean gap rate over rates 1-4 counting from 0 "
                "(the first step's rate skipped, Appendix L 6), late = mean of the last 4; free if late <= "
                "max(0.1 l w, 4 sigma_g, 0.25 early); blocked if late >= max(0.5 l w, 0.75 early) in the windows "
                "ending at steps 11 and 12; unwedge within ceil(1.5 x the reversed segment) steps [C-21]"),
    "K2": _tid(_PRIM, ("K2_twist_rad",), (_RO, _DD), "yes", "twist increment <= 0.1 rad per step command"),
    "K3": _tid(_PRIM, ("K3_orbit_offset_m",), (_DD, _ON), "yes",
               "orbit follow when the fitted hinge offset > max(3 mm, the Align tolerance); refit gated [C-49]"),
    "K4w": _tid(_OBS, ("K4w_stop_rate_mm",), (_RO, _ON, _DD), "yes",
                "end stop after a free classification: two 4-step windows ending at consecutive steps with mean gap "
                "rate >= max(0.5 l w, 5 x the median rate since the free classification), or a K7 part stop [C-30]"),
    "K5": _tid(_PRIM, ("K5_margin_rad",), (_DD,), "yes", "turn margin after the effect onset + max(0.1 rad, "
                                                           "0.25 theta_e)"),
    "K6": _tid(_REL, ("K6_yaw_budget_rad",), (_RO,), "yes",
               "turn cap = the R12 yaw budget left about the approach (kernel RobotModel.yaw_budget_left); the value "
               "is the robot's (robot_thresholds() logs the full budget as YAW_BUDGET_RAD), the turn controller that "
               "compares against it logs K6_yaw_budget_rad (contract §C16 names the robot as the owner of the value)"),
    "K7": _tid(_OBS, ("K7_rotation_deg", "K7_slip_rate_mm"), (_ON, _DD), "yes",
               "part rotation relative to the hand >= max(3 deg, 3 sigma_theta), or a gap fall <= -0.5 l w for >= 2 "
               "steps outside an expected unwedge, or the gap reaching G6"),
    "K8": _tid((), (), (), "reserved", "slide end stop"),
    "K9": _tid(_OBS, ("K9_change",), (_DD, _ON), "yes",
               "spring-back: part orientation or position change <= max(3 deg, 3 mm, 3 sigma) over 10 frames"),
    "K10": _tid((), (), (), "reserved", "DOF probe"),
    "K11": _tid(_PRIM, ("K11_tug_m",), (_DD, _ON), "yes",
                "axis tug: lateral deviation > max(3 mm, 2 sin(theta/2) sigma_c + 3 sigma_pos) for 4 steps"),
    # observers (O)
    "O1": _tid(_OBS, ("O1_pixel_diff", "O1_fraction"), (_ON, _DD), "yes",
               "effect (Appendix L 8): share of the pointed region's pixels whose chroma |delta(a*, b*)| > max(12, "
               "4 sigma_chroma) against the pre-action reference; effect when the share >= max(floor, k x the "
               "false-change rate measured while the robot moves near the region) for 3 frames; theta_e at the "
               "first frame of the run [C-28]"),
    "O2": _tid(_OBS, ("O2_iou",), (_DD, _ON), "yes", "rigid hold: silhouette IoU >= 0.8; centroid shift <= "
                                                       "max(4 px, 3 sigma_track)"),
    "O3": _tid(_OBS, ("O3_wrist_px", "O3_external_px"), (_DD, _ON), "yes",
               "co-motion: part template within 10 px in the wrist view and the external template moving by "
               "J_ext dp +- 3 sigma; v6 O-1: the carry load test also detects rotation of the carried part's "
               "rigid patch, and a tilt during the VerifyHold lift is grip_unstable (open, re-grasp)"),
    "O4": _tid(_OBS, ("O4_tilt_deg",), (_DD, _ON), "yes",
               "upright: hand orientation within M3 of the grasp and an in-hand tilt change <= 20 deg"),
    "O5": _tid(_PLAN + _OBS, ("O5_inflate_m", "O5_top_m"), (_DD, _ON), "yes",
               "keep-out: the part region inflated by max(2 cm, 3 sigma_loc) [C-23], from its support to top + "
               "max(2 cm, 3 sigma_h)"),
    "O6": _tid((), (), (), "reserved", "push arrival"),
    "O7": _tid(_OBS, ("O7_false_rate", "O7_remaining_fraction", "O7_remaining_px"), (_DD, _ON), "yes",
               "effect region validity [C-26], Appendix L 2: the sweep runs in the known sense only; >= 50 % of the "
               "region and >= 400 px remain outside the swept hand and the obj / grip regions, false-change rate "
               "(motion rate) below 1 %; DD, no robot-only data yet, re-derive at V5"),
    "O8": _tid(_OBS, ("O8_min_rotation_rad",), (_DD,), "yes", "minimum part rotation before an onset counts 5 deg"),
    # pointing and tracking (V)
    "V1": _tid(_GEO, ("V1_min_peak",), (_DD,), "yes", "NCC lost when peak < 0.6 for 2 frames; re-template "
                                                       "0.70 <= peak < 0.85 (v1); v6 probe-tracker mode: min "
                                                       "peak 0.9 (V1_PROBE_MIN_PEAK)"),
    "V2": _tid(_REL, ("V2_align_tol_px",), (_DD,), "record", "Align converged <= min(6 px, 3 mm kappa), >= 2 px, "
                                                              "3 frames (v1)"),
    "V3": _tid(_GEO, ("V3_reacquire_px",), (_DD,), "yes", "re-acquisition +-48 px, peak >= 0.6, within 3 sigma"),
    "V4": _tid(_REL, ("V4_repoints",), (_DD,), "yes", "2 re-points per part per view"),
    "V5": _tid(_PLAN, ("V5_grasp_margin_m", "V5_axis_agree"), (_DD, _ON), "yes",
               "answer checks: grasp point >= 3 sigma_h above both finger-landing columns; surface planar within "
               "D8; elongated agree 0.8, coherence 0.5, rms 2; v6 S-10: the elongated check reads the scene's D10 "
               "and axis_check result"),
    "V6": _tid((_P + "client",), ("V6_crop_px",), (_DD,), "yes", "fine crop 128 px x4, 8x8 points (v1)"),
    "V7": _tid(("test",), (), (), "test", "FeatureTracker equivalence |delta uv| <= 0.1 px against NCCTrack"),
    # LookAt (L)
    "L1": _tid(_GEO, ("L1_locator_px",), (_ON, _DD), "yes",
               "hand locator consistency: displacement residual against J_ext dp <= max(3 px, 3 sigma); v6 G-5: "
               "with n <= 3 locatable moves the absolute bound sqrt 2 x L1_SIGMA_K x L1_FLOOR_PX = 12.7 px; with "
               "2 agreeing locations fit and flag calibration_degraded; v7 G-8 (HandLocator V2v): the across "
               "coordinate is the mean of the two finger-band pieces (each >= HL_MIN_BLOB_PX), accepted when the "
               "measured separation is within PROJECTOR_AFFINE_FRACTION x the predicted one + L1_FLOOR_PX, else "
               "refused (finger_pair_too_few, finger_pair_one_piece, finger_pair_inconsistent)"),
    "L2": _tid((), (), (), "reserved", "RayServo converged"),
    "L3": _tid((), (), (), "reserved", "first surface on the ray"),
    "L4": _tid((), (), (), "reserved", "ray-depth consistency"),
    "L5": _tid((), (), (), "reserved", "answer ray band"),
    # budgets (B)
    "B1": _tid(_POLICY, ("B1_calls_per_act",), (_DD,), "yes", "<= 2 model calls per act, 1 during a rejection "
                                                              "storm (v1)"),
    "B2": _tid(_REL, ("B2_local_retries",), (_DD,), "yes", "2 local retries per relation (v1)"),
    "B3": _tid(_POLICY, ("B3_reviews",), (_DD,), "yes", "<= 2 reviews per episode"),
    "B4": _tid(_PRIM, ("B4_quiet",), (_DD,), "yes",
               "settle quietness <= 0.03 deg per step and <= 0.3 mm per step for 2 steps, <= 14 steps (v1)"),
    "B5": _tid(("evaluate_agent_speedup_v2",), (), (_DD,), "yes",
               "stuck alert only: >= 10 of the last 12 failed / rejected; a met relation failing to re-establish "
               "more than 5 times; invalid_plan at once"),
}
_THRESHOLD_ID = re.compile(r"([RGDPMKOVLB])(\d+)(w?)(?=_|$)", re.IGNORECASE)


def threshold_id(name):
    """The spec §5.2 id a constant or runtime threshold name starts with ("K1W_DECIDE_STEP" -> "K1w",
    "G5_object_present_gap_mm" -> "G5"), or None."""
    match = _THRESHOLD_ID.match(str(name))
    if not match:
        return None
    letter, number, suffix = match.groups()
    key = f"{letter.upper()}{int(number)}{'w' if suffix else ''}"
    return key if key in THRESHOLD_IDS else None


def threshold_ids_sha256():
    return _sha(json.dumps(THRESHOLD_IDS, sort_keys=True, separators=(",", ":")))


# ------------------------------------------------------- THRESHOLDS_VERSION v6 (design review 1 decision §3, K-2)
V6_STATES = ("module_absent", "pending", "alias", "labelled", "mislabelled", "unlabelled")


def _version_entry(version, owners, ids, labels, decision, derivation):
    if (len(owners) != 1 or not labels or any(label not in LABEL_CLASSES for label in labels)
            or any(i not in THRESHOLD_IDS for i in ids) or not decision.strip() or not derivation.strip()):
        raise ValueError(f"bad {version} constant entry: {decision}")
    return {"owner": owners[0], "ids": tuple(ids), "labels": tuple(labels), "decision": decision,
            "derivation": derivation}


def _v6(owners, ids, labels, decision, derivation):
    return _version_entry("v6", owners, ids, labels, decision, derivation)


# INFORMATIONAL, like THRESHOLD_IDS: the named constants that design review 1 (decision §3) adds or re-states for
# THRESHOLDS_VERSION v6, each with its owner module, the spec ids whose rule it amends, the label classes the decision
# gives it and the decided value as text. It holds no number and is not part of constants_sha256: the value and its
# label live only in the owner's LABELS (the T0 lint and literal_audit apply there as to every constant).
# ``v6_constant_status`` reports, from the owner's AST, whether each name is labelled with one of its classes. The
# decision's other §3 rows add no name: DSW (D8_FLOOR_M and one cell), the D10 / RC-7 agreement (the kernel's
# AXIS_AGREE_MAX_RAD = YAW_GRID_RAD / 2) and the G13+ excess (G13 floor and k, R4 pad length, the yaw grid).
V6_CONSTANTS = {
    "FT_PROBE_RADIUS_PX": _v6(_GEO, (), (_DD,), "5 px (11 px templates)",
                              "7-11 px templates attach within 0.2 mm on the thin grip part (task 2 C); 21 px "
                              "templates lag about 3 px on its tube sides"),
    "FT_DENSE_STEP_PX": _v6(_GEO, (), (_DD,), "4 px",
                            "a 6 px lattice breaks the thin grip part at yaw -90 deg (p90 23.8 mm)"),
    "FT_DENSE_WINDOW_PX": _v6(_GEO, (), (_DD,), "40 px (= TOP_WINDOW_PX)",
                              "about 1.2-2 cm at hover depth: the band rule window"),
    "TOP_WINDOW_PX": _v6(_SCENE, (), (_DD,), "40 px (= FT_DENSE_WINDOW_PX)",
                         "the band rule window, equal to the dense lattice window"),
    "FT_BANDPASS_SIGMAS": _v6(_GEO, ("D15",), (_DD,), "(0.8, 3.0) px, gain 4",
                              "the probe tracker's band-pass prefilter comparison"),
    "D15_PROBE_MIN_STD": _v6(_GEO, ("D15",), (_DD,), "1.0 grey on the band-passed patch",
                             "band-pass prefilter comparison; re-derive ONLINE (k x band-passed sigma_static) at V5"),
    "V1_PROBE_MIN_PEAK": _v6(_GEO, ("V1",), (_DD,), "0.9",
                             "dense seeds: thin grip part p90 12.4 -> 3.2 mm"),
    "FT_TRACKED_SEARCH_PX": _v6(_GEO, (), (_DD,), "12 px",
                                "remedy 1 E1: the same accuracy over 100 hovers at half the time"),
    "TOP_GATE_M": _v6(_SCENE, (), (_DD,), "2 mm", "band-top ablation: the sample sigma_z gate"),
    "TOP_BAND_K": _v6(_SCENE, (), (_DD,), "3 (= BAND_MIN_SAMPLES)", "the existing band member minimum"),
    "TOP_BAND_SIGMA_K": _v6(_SCENE, (), (_DD,), "2.0", "band-top ablation (3 mm floor, k 2)"),
    "TOP_BAND_SUPPORT_N": _v6(_SCENE, (), (_DD,), "10", "band-top ablation: n < 10 -> p90 4.5 mm"),
    "TOP_MODEL_M": _v6(_SCENE, ("D4",), (_DD,), "3 mm (top band n >= TOP_BAND_SUPPORT_N)",
                       "remedy 1 E2: thin grip part 2 sigma coverage 0.979, 3 sigma 1.000, P6 pass 1.00, other parts "
                       "1.000; judge live sigma_h 3.0 mm"),
    "TOP_MODEL_THIN_M": _v6(_SCENE, ("D4",), (_DD,), "5 mm (top band n < TOP_BAND_SUPPORT_N)",
                            "remedy 1 E2: 3 sigma coverage 1.000; judge live sigma_h 5.0-5.1 mm, below the P6 limit "
                            "5.54 mm"),
    "D10_MIN_REGION_CELLS": _v6(_SCENE, ("D10",), (_DD,), "4",
                                "the smallest region whose second moment is defined with 2 x 2 cells; S-4' refused "
                                "5/5 wrong axes and kept 14/14 good ones"),
    "PB_REACH_M": _v6(_SCENE, ("D7",), (_RO, _DD), "0.05 m",
                      "R7 hand half-footprint along R0 (31.6 mm) + D7_OBSTACLE_RADIUS_M (20 mm)"),
    "D7_PARENT_SIGMA_CAP_M": _v6(_SCENE, ("D7",), (_DD,), "HF_CELL_M (5 mm)", "the field's resolution (RC-9a)"),
    "P11_CAP_M": _v6(_PLAN, ("P11",), (_DD,), "0.10 m",
                     "v1 hovers of 5-10 cm worked; judge: band support at the chain hovers 3/3/7 -> 6/27/16"),
    "PROBE_STALL_MIN_STEPS": _v6(_PRIM, ("G12",), (_DD, _ON), "5 steps",
                                 "v1 measures a move's own free ratio at step 5 (DD -> ONLINE)"),
    "PROBE_FREE_RATIO_FLOOR": _v6(_PRIM, ("G12",), (_DD,), "0.08",
                                  "robot-only: above 0.067 (a leg starting in contact reads <= 0.04 and must stall at "
                                  "step 5) and below 0.083 (the lowest own free ratio of 48 free legs)"),
    "HOVER_LOCATE_M": _v6(_REL, (), (_DD,), "0.03 m",
                          "rulings item 4: the validated 3 cm settled vertical move (median 4.8 px)"),
    "SIGMA_VC_M": _v6(_PLAN, ("P8", "P9"), (_DD,), "0.004 m (unchanged)", "ruling 4a: 4 mm at part tops"),
    "D4_MODEL_SIGMA_M": _v6(_GEO, ("D4",), (_DD,), "0.009 m (unchanged, diagnostic only)",
                            "ruling 4b: never applied per sample; the part-level term is TOP_MODEL_M"),
}


def _constants_view(path):
    """(numeric module-level names, all plainly assigned module-level names, the LABELS register) of a module, read
    from its AST; an invalid LABELS reads as empty (the lint reports it)."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    assigned = set()
    for node in tree.body:
        if _plain_module_assignment(node):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned |= {leaf.id for target in targets
                         for leaf in (target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target])}
    try:
        labels = local_labels(tree)
    except ValueError:
        labels = {}
    return set(_assigned_numeric_names(tree.body)), assigned, labels


def v6_constant_status(package_dir=None):
    """K-2 report for every V6_CONSTANTS name, from the owner module's AST (never imported): {"owner", "state",
    "label", "allowed"}. States: module_absent (no owner file yet), pending (not defined yet), alias (assigned at
    module level from another named constant, no literal of its own), labelled (a numeric constant labelled with one
    of the decision's classes), mislabelled (labelled with another class), unlabelled (numeric without a label)."""
    return _constant_status(V6_CONSTANTS, package_dir)


def _constant_status(register, package_dir=None):
    """The owner-AST report of ``v6_constant_status`` / ``v7_constant_status`` for one versioned register."""
    package_dir = PACKAGE_DIR if package_dir is None else Path(package_dir)
    views, out = {}, {}
    for name, entry in sorted(register.items()):
        owner = entry["owner"]
        if owner not in views:
            path = package_dir / f"{owner}.py"
            views[owner] = _constants_view(path) if path.is_file() else None
        view, label = views[owner], None
        if view is None:
            state = "module_absent"
        elif name in view[0]:
            label = view[2].get(name, (None, None))[0]
            state = "unlabelled" if label is None else ("labelled" if label in entry["labels"] else "mislabelled")
        elif name in view[1]:
            state = "alias"
        else:
            state = "pending"
        out[name] = {"owner": owner, "state": state, "label": label, "allowed": list(entry["labels"])}
    return out


# ------------------------------------------------------- THRESHOLDS_VERSION v7 (design review 2 decision §3, K-6)
V7_STATES = V6_STATES


def _v7(owners, ids, labels, decision, derivation):
    return _version_entry("v7", owners, ids, labels, decision, derivation)


# INFORMATIONAL, like V6_CONSTANTS: the named constants that design review 2 (decision §3) adds or re-states for
# THRESHOLDS_VERSION v7, with the owner module, the spec ids whose rule they amend, the decision's label classes and
# the decided value as text (no number; not part of constants_sha256; the value and its label live only in the
# owner's LABELS). ``v7_constant_status`` reports them from the owner's AST. The decision's other §3 rows add no
# name: the orientation peak window (YAW_GRID_RAD / 2), the D10-OE and S-12 view agreement (the kernel's
# AXIS_AGREE_MAX_RAD), the S-15 seed distance (q_now + TRACK_SIGMA_PX), the S-16 hand mask margin max(4 px,
# ceil(2 sigma_proj)) with PROJECTOR_AFFINE_FRACTION, and V2v (HL_MIN_BLOB_PX, PROJECTOR_AFFINE_FRACTION,
# L1_FLOOR_PX, L1_TIP_BAND_PX and the robot footprint R5, R6, R7). Every DD value was fitted on one task's two grip
# parts: re-derive on the cavity-edge and sliding-fixture regressions before M7 (decision §3).
V7_CONSTANTS = {
    "ORIENT_SMOOTH_DEG": _v7(_GEO, ("D10",), (_DD,), "3.0 deg circular smoothing of the 1 deg doubled-angle "
                                                     "orientation histogram",
                             "axis lens (review 2) orientation_peak: 1 deg bins, peak refined within YAW_GRID_RAD / 2"),
    "OUTLINE_TANGENT_PTS": _v7(_GEO, ("D10",), (_DD,), "4 outline points each side (about +-4 px tangents)",
                               "axis lens outline_orientation: length-weighted tangents of the largest outline"),
    "D10_OUTLINE_SHARE_MIN": _v7(_SCENE, ("D10",), (_DD,), "0.20",
                                 "axis lens sensitivity: 0 wrong axes at >= 0.20, one 16.4 deg wrong at 0.15"),
    "D10_OUTLINE_SECOND_MAX": _v7(_SCENE, ("D10",), (_DD,), "0.8 (second peak / main peak)",
                                  "axis lens outline gates R0 / R1; judge live2: the 8.3 deg wrong axis had 0.83"),
    "D10_EDGE_SHARE_MIN": _v7(_SCENE, ("D10",), (_DD,), "0.20", "axis lens: the edge histogram share gate"),
    "D10_EDGE_TOP_FRACTION": _v7(_GEO, ("D10",), (_DD,), "0.25 (the strongest quarter of the ROI gradients)",
                                 "axis lens edge_hist"),
    "D10_EDGE_BLUR_PX": _v7(_GEO, ("D10",), (_DD,), "1.0 px luma blur sigma before the Sobel gradient",
                            "axis lens edge_hist"),
    "D10_EDGE_REGION_DILATE_PX": _v7(_SCENE, ("D10",), (_DD,), "6 px", "axis lens ROI form region"),
    "D10_EDGE_HULL_DILATE_PX": _v7(_SCENE, ("D10",), (_DD,), "8 px", "axis lens ROI form hull (top-band members)"),
    "REGION_COLLAPSE_RATIO": _v7(_SCENE, ("D10", "P2"), (_DD,), "2.0 (band hull pixels / region pixels)",
                                 "axis lens collapse guard: the collapses of grip part A were bands of 75-94 "
                                 "members on 0-1 cells; judge live: 0 collapsed widths delivered"),
    "COLLAPSE_HULL_DILATE_PX": _v7(_SCENE, ("D10", "P2"), (_DD,), "2 px", "axis lens collapse guard band hull"),
    "FT_TRACKED_SEARCH_PX": _v7(_GEO, (), (_DD,), "12 px, opt-in and not adopted (decision §5 sign-off)",
                                "adaptive search changed tracks on 40/40 hovers and 2428 D1 decisions while the exact "
                                "path ran 131/131 stages within 40 ms; turning it on needs an RS-8 with off-part "
                                "parity"),
}
# INFORMATIONAL: decision §3 "retired as deciders". These scene constants no longer decide D10 (S-11 replaces S-4'
# and its cell PCA); they stay logged in axis_check as diagnosis, and the planner's P-C region axis still reads the
# aspect and cell minimum through its own aliases.
V7_RETIRED_DECIDERS = {
    "D10_MIN_COHERENCE": {"owner": _P + "scene", "ids": ("D10",), "ruling": "structure-tensor coherence gate retired "
                                                                         "(logged)"},
    "D10_MIN_ASPECT": {"owner": _P + "scene", "ids": ("D10",), "ruling": "region aspect gate retired as a D10 decider "
                                                                      "(logged; the P-C region axis keeps it)"},
    "D10_MIN_REGION_CELLS": {"owner": _P + "scene", "ids": ("D10",), "ruling": "S-4' cell minimum retired (logged; the "
                                                                            "P-C region axis keeps it)"},
}


def v7_constant_status(package_dir=None):
    """K-6 report for every V7_CONSTANTS name, from the owner module's AST (never imported), in the states of
    ``v6_constant_status`` (V7_STATES)."""
    return _constant_status(V7_CONSTANTS, package_dir)


_merge_local_labels()
