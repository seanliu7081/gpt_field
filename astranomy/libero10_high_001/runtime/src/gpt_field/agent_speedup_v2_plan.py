"""agent_SpeedUp_v2 relation plan as data (contract §C11; spec §3.3, §3.4, §3.7, §3.9).

Contents: the exact wire schemas (plan and replan §3.3, repair and review §3.7, pointing as in v1: one bounded
integer id), a strict conformance check of model JSON against those schemas, the structural parse into the
kernel's plan records, the executor-side plan validator (§3.4), part roles, the BDDL-like rendering for logs and
video (§3.9, part labels by the slug rule [C-44]), the compact plan status block sent with repair, replan and
review requests, and the repair answer's parse and validation.

The module holds no task text and no example plans (those live only in tests). Every issue text is
executor-written, at most ISSUE_TEXT_MAX_CHARS long, and names parts by id only: it never repeats a part text.
Schemas are built inline (no $ref, $defs or pattern), every object is closed and every key is required; property
order is the wire order (``scene`` first in a plan, ``why`` first in a repair answer), so the model writes its
description before its choices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re

from .agent_speedup_v2_types import (DO_KINDS, HOLD_TRUE_RELS, KEEP_KINDS, MOVE_PRIORS, NEED_KINDS, PIDS,
                                     PLAN_SOURCES, POINT_STAGES, REASONS, REPAIR_ACTIONS, SENSES, UNTIL_KINDS,
                                     HoldTrue, Keep, Plan, PlanPart, PlanStep, RepairAnswer, StepState, Until)

PLAN_SCHEMA_VERSION = "agent-speedup-v2-plan-schema-1"

LABELS = {
    "PLAN_MIN_PARTS": ("DD", "spec §3.3 schema bound: a plan names at least one part"),
    "PLAN_MAX_PARTS": ("DD", "spec §3.3 schema bound: at most 8 parts, one per id A-H"),
    "PLAN_MAX_STEPS": ("DD", "spec §3.3 schema bound: at most 6 relation steps"),
    "PLAN_MAX_GRIP": ("DD", "spec §3.3 schema bound: up to 2 grip parts in preference order"),
    "PLAN_MAX_KEEP": ("DD", "spec §3.3 schema bound: up to 3 keep constraints per step"),
    "PLAN_MAX_UNTIL": ("DD", "spec §3.3 schema bound: up to 3 completion criteria per step"),
    "PLAN_MAX_HOLD_TRUE": ("DD", "spec §3.3 schema bound: up to 3 already-true relations"),
    "SCENE_MIN_CHARS": ("DD", "spec §3.3 schema bound: scene is a nonempty string"),
    "SCENE_MAX_CHARS": ("DD", "spec §3.3 rev 2 schema bound: scene capped at 80 characters (120 before) to save about "
                              "12 visible tokens per plan (spec §3.8)"),
    "TEXT_MIN_CHARS": ("DD", "spec §3.3 schema bound: a part text has at least 3 characters"),
    "TEXT_MAX_CHARS": ("DD", "spec §3.3 schema bound: a part text has at most 60 characters"),
    "TXT_MAX_CHARS": ("DD", "spec §3.3 schema bound: an until txt has at most 40 characters"),
    "CELL_NOT_VISIBLE": ("DD", "spec §3.3: cell id 0 means not visible in that image"),
    "CELL_MAX": ("DD", "spec §3.3: 8x8 grid cells and crop points numbered 1-64 row-major from the top left"),
    "WHY_MIN_CHARS": ("DD", "spec §3.7 schema bound: why is a nonempty string"),
    "WHY_MAX_CHARS": ("DD", "spec §3.7 schema bound: why has at most 100 characters"),
    "ISSUE_TEXT_MAX_CHARS": ("DD", "contract §C11: one executor-written issue text is at most 120 characters"),
    "ISSUES_TEXT_MAX_CHARS": ("DD", "contract §C11 issues_text: the error text of the one re-ask (spec §3.4) is at "
                                    "most 300 characters"),
    "BDDL_LABEL_MAX_CHARS": ("DD", "[C-44] BDDL-like part labels (logs and video only) are cut to 32 characters at "
                                   "a hyphen boundary"),
}

PLAN_MIN_PARTS = 1
PLAN_MAX_PARTS = 8
PLAN_MAX_STEPS = 6
PLAN_MAX_GRIP = 2
PLAN_MAX_KEEP = 3
PLAN_MAX_UNTIL = 3
PLAN_MAX_HOLD_TRUE = 3
SCENE_MIN_CHARS = 1
SCENE_MAX_CHARS = 80
TEXT_MIN_CHARS = 3
TEXT_MAX_CHARS = 60
TXT_MAX_CHARS = 40
CELL_NOT_VISIBLE = 0
CELL_MAX = 64
WHY_MIN_CHARS = 1
WHY_MAX_CHARS = 100
ISSUE_TEXT_MAX_CHARS = 120
ISSUES_TEXT_MAX_CHARS = 300
BDDL_LABEL_MAX_CHARS = 32

PLAN_ISSUE_CODES = ("duplicate_pid", "undefined_pid", "obj_equals_ref", "turn_needs_criterion", "turn_part_not_rotary",
                    "slide_needs_sense", "slide_part_not_sliding", "needs_ref", "obj_not_free", "fits_part_not_free",
                    "upright_part_not_free", "clear_of_is_obj", "rotation_criterion_not_turn", "until_not_allowed",
                    "effect_part_is_operand", "criterion_needs_part")
# Repair answers (validate_repair) add these codes; an inserted step is checked with the plan codes above.
REPAIR_ISSUE_CODES = ("part_required", "step_required", "step_limit", "no_next_step")
ISSUE_CODES = PLAN_ISSUE_CODES + REPAIR_ISSUE_CODES

# §3.3 / §3.4 rules as data
ALLOWED_UNTIL = {"turn": ("effect", "stop", "quarter_turn", "half_turn"), "slide": ("stop", "fits", "effect"),
                 "place_on": ("effect",), "place_in": ("effect",), "hold": (), "push_to": (), "release": ()}
ROTATION_CRITERIA = ("quarter_turn", "half_turn")
CRITERIA_WITH_PART = ("effect", "fits")
FREE_MOVES = ("free", "unknown")            # [C-22]: a free object is free or unknown, verified later by co-motion
ROTARY_MOVES = ("hinged", "unknown")
SLIDING_MOVES = ("sliding", "unknown")
SLIDE_SENSES = ("open", "close")
REF_KINDS = ("place_on", "place_in", "push_to")
CARRIED_KINDS = ("place_on", "place_in", "push_to", "hold")
# The grasping relations: with grip == [] the hand grips obj itself (kernel PlanStep.grip_parts, StepContext.from_step,
# contract §C3.6 "grip == [] means grip = (obj,)"), so obj gets the grip role. push_to (contact beside the object)
# and release grip nothing.
DEFAULT_GRIP_KINDS = ("hold", "turn", "slide", "place_on", "place_in")
POINTING_KEYS = {"fine": "point_id", "coarse": "cell_id"}
STATUS_BLOCK_KEYS = ("scene", "parts", "steps", "hold_true", "holding")

# BDDL-like rendering: the atom each relation step establishes (logs only)
GOAL_ATOMS = {"place_on": "on", "place_in": "in", "push_to": "at", "hold": "held", "release": "released"}
SLIDE_GOAL_ATOMS = {"open": "open", "close": "closed", "any": "slid"}
TURN_GOAL_ATOM = "turned"


class PlanParseError(ValueError):
    """A structural defect of model JSON against its schema (plan, replan, repair or review)."""


# ------------------------------------------------------------------------------------------------- schemas
def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def _enum(values) -> dict:
    return {"type": "string", "enum": list(values)}


def _object(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def _array(items: dict, *, max_items: int, min_items: int | None = None) -> dict:
    out = {"type": "array"}
    if min_items is not None:
        out["minItems"] = min_items
    out["maxItems"] = max_items
    out["items"] = items
    return out


def _string(*, max_chars: int, min_chars: int | None = None) -> dict:
    out = {"type": "string"}
    if min_chars is not None:
        out["minLength"] = min_chars
    out["maxLength"] = max_chars
    return out


def _cell() -> dict:
    return {"type": "integer", "minimum": CELL_NOT_VISIBLE, "maximum": CELL_MAX}


def pid_schema() -> dict:
    """{"type": "string", "enum": ["A".."H"]}, written inline at every use (no $ref / $defs / pattern)."""
    return _enum(PIDS)


def part_item_schema() -> dict:
    return _object({"id": pid_schema(), "text": _string(min_chars=TEXT_MIN_CHARS, max_chars=TEXT_MAX_CHARS),
                    "move": _enum(MOVE_PRIORS), "w": _cell(), "x": _cell()})


def hold_true_item_schema() -> dict:
    return _object({"rel": _enum(HOLD_TRUE_RELS), "a": pid_schema(), "b": _nullable(pid_schema())})


def step_item_schema() -> dict:
    """§3.3 step item; keys in the order do, obj, grip, ref, sense, keep, until."""
    keep = _object({"c": _enum(KEEP_KINDS), "p": pid_schema()})
    until = _object({"o": _enum(UNTIL_KINDS), "p": _nullable(pid_schema()),
                     "txt": _nullable(_string(max_chars=TXT_MAX_CHARS))})
    return _object({"do": _enum(DO_KINDS), "obj": pid_schema(),
                    "grip": _array(pid_schema(), min_items=0, max_items=PLAN_MAX_GRIP),
                    "ref": _nullable(pid_schema()), "sense": _enum(SENSES),
                    "keep": _array(keep, max_items=PLAN_MAX_KEEP), "until": _array(until, max_items=PLAN_MAX_UNTIL)})


def plan_schema() -> dict:
    """§3.3 exact; property order scene, parts, hold_true, steps."""
    return _object({"scene": _string(min_chars=SCENE_MIN_CHARS, max_chars=SCENE_MAX_CHARS),
                    "parts": _array(part_item_schema(), min_items=PLAN_MIN_PARTS, max_items=PLAN_MAX_PARTS),
                    "hold_true": _array(hold_true_item_schema(), max_items=PLAN_MAX_HOLD_TRUE),
                    "steps": _array(step_item_schema(), min_items=0, max_items=PLAN_MAX_STEPS)})


def repair_schema() -> dict:
    """§3.7 exact; property order why, action, part, step (step = the §3.3 step item)."""
    return _object({"why": _string(min_chars=WHY_MIN_CHARS, max_chars=WHY_MAX_CHARS), "action": _enum(REPAIR_ACTIONS),
                    "part": _nullable(pid_schema()), "step": _nullable(step_item_schema())})


def pointing_stage(stage) -> str:
    """'fine' / 'coarse' for a v1 stage name or a request kind ('point_fine' / 'point_coarse')."""
    if isinstance(stage, str) and stage.startswith("point_"):
        stage = stage[len("point_"):]
    if not isinstance(stage, str) or stage not in POINT_STAGES:
        raise ValueError("pointing stage must be fine or coarse (or point_fine / point_coarse)")
    return stage


def pointing_schema(stage: str) -> dict:
    """{"point_id"} (fine) / {"cell_id"} (coarse): one integer 0..64, 0 = not visible (v1 pattern)."""
    return _object({POINTING_KEYS[pointing_stage(stage)]: _cell()})


def schemas() -> dict[str, dict]:
    """The output schema of every client request kind (plan, replan, point_fine, point_coarse, repair, review)."""
    by_kind = {"plan": plan_schema, "replan": plan_schema, "point_fine": lambda: pointing_schema("fine"),
               "point_coarse": lambda: pointing_schema("coarse"), "repair": repair_schema, "review": repair_schema}
    return {kind: by_kind[kind]() for kind in NEED_KINDS}


def schemas_sha256() -> str:
    """sha256 of every schema in wire (insertion) order, with the schema version; the manifest's plan schema id."""
    material = json.dumps({"version": PLAN_SCHEMA_VERSION, "schemas": schemas()}, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------- schema conformance
def conform(data, schema: Mapping, path: str = "$") -> None:
    """Strict check of JSON data against a schema of this module's subset (closed objects, every key required,
    anyOf with null, string enums and lengths, bounded integers, bounded arrays). PlanParseError names the first
    defect by its JSON path; the message never repeats the offending value."""
    if "anyOf" in schema:
        failures = []
        for option in schema["anyOf"]:
            try:
                conform(data, option, path)
                return
            except PlanParseError as exc:
                if option.get("type") != "null":
                    failures.append(str(exc))
        raise PlanParseError(failures[0] if len(failures) == 1 else f"{path}: matches none of the allowed forms")
    kind = schema.get("type")
    if kind == "null":
        if data is not None:
            raise PlanParseError(f"{path}: must be null")
    elif kind == "string":
        if not isinstance(data, str):
            raise PlanParseError(f"{path}: must be a string")
        if "enum" in schema and data not in schema["enum"]:
            raise PlanParseError(f"{path}: not one of the allowed values")
        if len(data) < schema.get("minLength", 0) or len(data) > schema.get("maxLength", len(data)):
            raise PlanParseError(f"{path}: length outside [{schema.get('minLength', 0)}, "
                                 f"{schema.get('maxLength', 'any')}]")
    elif kind == "integer":
        if type(data) is not int:
            raise PlanParseError(f"{path}: must be an integer")
        if data < schema.get("minimum", data) or data > schema.get("maximum", data):
            raise PlanParseError(f"{path}: outside [{schema.get('minimum')}, {schema.get('maximum')}]")
    elif kind == "array":
        if not isinstance(data, (list, tuple)):
            raise PlanParseError(f"{path}: must be an array")
        if len(data) < schema.get("minItems", 0) or len(data) > schema.get("maxItems", len(data)):
            raise PlanParseError(f"{path}: item count outside [{schema.get('minItems', 0)}, "
                                 f"{schema.get('maxItems', 'any')}]")
        for index, item in enumerate(data):
            conform(item, schema["items"], f"{path}[{index}]")
    elif kind == "object":
        if not isinstance(data, Mapping):
            raise PlanParseError(f"{path}: must be an object")
        keys = set(data)
        expected = set(schema["required"])
        if keys != expected:
            missing = [k for k in schema["required"] if k not in keys]
            extra = sorted(str(k) for k in keys - expected)
            raise PlanParseError(f"{path}: " + (f"missing {', '.join(missing)}" if missing else "")
                                 + ("; " if missing and extra else "")
                                 + (f"{len(extra)} unexpected key(s)" if extra else ""))
        for name, sub in schema["properties"].items():
            conform(data[name], sub, f"{path}.{name}")
    else:
        raise ValueError(f"unsupported schema construct at {path}")


# -------------------------------------------------------------------------------------------------- parsing
def _parse_step(data: Mapping) -> PlanStep:
    return PlanStep(data["do"], data["obj"], tuple(data["grip"]), data["ref"], data["sense"],
                    tuple(Keep(k["c"], k["p"]) for k in data["keep"]),
                    tuple(Until(u["o"], u["p"], u["txt"]) for u in data["until"]))


def parse_plan(data: Mapping, *, source: str = "plan") -> Plan:
    """Structural parse of a plan answer into kernel records; PlanParseError on any schema defect. Semantic
    validation (§3.4) is ``validate_plan``'s, run by the StepRunner."""
    if source not in PLAN_SOURCES:
        raise ValueError(f"source must be one of {PLAN_SOURCES}")
    conform(data, plan_schema())
    try:
        parts = tuple(PlanPart(p["id"], p["text"], p["move"], p["w"], p["x"]) for p in data["parts"])
        hold_true = tuple(HoldTrue(h["rel"], h["a"], h["b"]) for h in data["hold_true"])
        steps = tuple(_parse_step(s) for s in data["steps"])
        return Plan(data["scene"], parts, hold_true, steps, source)
    except ValueError as exc:                 # unreachable after conform; kept as a structural error
        raise PlanParseError(str(exc)[:ISSUE_TEXT_MAX_CHARS]) from None


def parse_repair(data: Mapping) -> RepairAnswer:
    """Structural parse of a repair or review answer; PlanParseError on any schema defect."""
    conform(data, repair_schema())
    try:
        step = None if data["step"] is None else _parse_step(data["step"])
        return RepairAnswer(data["why"], data["action"], data["part"], step)
    except ValueError as exc:
        raise PlanParseError(str(exc)[:ISSUE_TEXT_MAX_CHARS]) from None


# ------------------------------------------------------------------------------------------------ validation
def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - len("...")] + "..."


@dataclass(frozen=True)
class PlanIssue:
    """One validation finding; ``text`` is executor-written (<= ISSUE_TEXT_MAX_CHARS) and names parts by id."""
    code: str
    step: int | None
    pid: str | None
    text: str

    def __post_init__(self):
        if self.code not in ISSUE_CODES:
            raise ValueError(f"unknown plan issue code {self.code!r}")
        if self.step is not None and (isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0):
            raise ValueError("PlanIssue.step must be a nonnegative integer or None")
        if self.pid is not None and self.pid not in PIDS:
            raise ValueError("PlanIssue.pid must be a part id or None")
        if not isinstance(self.text, str) or not self.text or len(self.text) > ISSUE_TEXT_MAX_CHARS:
            raise ValueError(f"PlanIssue.text must be a nonempty string of <= {ISSUE_TEXT_MAX_CHARS} characters")

    def to_dict(self) -> dict:
        return {"code": self.code, "step": self.step, "pid": self.pid, "text": self.text}


def _issue(code, step, pid, text) -> PlanIssue:
    return PlanIssue(code, step, pid, _clip(text, ISSUE_TEXT_MAX_CHARS))


def _step_issues(i: int, step: PlanStep, moves: Mapping[str, str]) -> list[PlanIssue]:
    out = []

    def defined(pid, field):
        if pid in moves:
            return True
        out.append(_issue("undefined_pid", i, pid, f"step {i} {field} names part {pid}, which is not defined"))
        return False

    obj_ok = defined(step.obj, "obj")
    for pid in step.grip:
        defined(pid, "grip")
    if step.ref is not None:
        defined(step.ref, "ref")
        if step.ref == step.obj:
            out.append(_issue("obj_equals_ref", i, step.ref, f"step {i}: ref is the same part as obj ({step.obj})"))
    move = moves.get(step.obj)
    if step.do == "turn":
        if not step.until:
            out.append(_issue("turn_needs_criterion", i, step.obj, f"step {i}: turn needs at least one until criterion"))
        if obj_ok and move not in ROTARY_MOVES:
            out.append(_issue("turn_part_not_rotary", i, step.obj,
                              f"step {i}: turn obj {step.obj} has move {move}; turn needs hinged or unknown"))
    if step.do == "slide":
        if step.sense not in SLIDE_SENSES:
            out.append(_issue("slide_needs_sense", i, step.obj, f"step {i}: slide needs sense open or close"))
        if obj_ok and move not in SLIDING_MOVES:
            out.append(_issue("slide_part_not_sliding", i, step.obj,
                              f"step {i}: slide obj {step.obj} has move {move}; slide needs sliding or unknown"))
    if step.do in REF_KINDS:
        if step.ref is None:
            out.append(_issue("needs_ref", i, step.obj, f"step {i}: {step.do} needs a ref part"))
        if obj_ok and move not in FREE_MOVES:
            out.append(_issue("obj_not_free", i, step.obj,
                              f"step {i}: {step.do} obj {step.obj} has move {move}; it needs free or unknown"))
    for keep in step.keep:
        if not defined(keep.p, keep.c):
            continue
        if keep.c == "upright" and moves[keep.p] not in FREE_MOVES:
            out.append(_issue("upright_part_not_free", i, keep.p,
                              f"step {i}: upright part {keep.p} has move {moves[keep.p]}; it needs free or unknown"))
        if keep.c == "clear_of" and keep.p == step.obj:
            out.append(_issue("clear_of_is_obj", i, keep.p, f"step {i}: clear_of names the step's own obj {keep.p}"))
    operands = {step.obj, *step.grip_parts}
    allowed = ALLOWED_UNTIL[step.do]
    for until in step.until:
        if until.o in ROTATION_CRITERIA and step.do != "turn":
            out.append(_issue("rotation_criterion_not_turn", i, until.p,
                              f"step {i}: {until.o} is allowed only on turn, not on {step.do}"))
        elif until.o not in allowed:
            listing = ", ".join(allowed) if allowed else "none"
            out.append(_issue("until_not_allowed", i, until.p,
                              f"step {i}: until {until.o} is not allowed on {step.do} (allowed: {listing})"))
        if until.p is not None and not defined(until.p, f"until {until.o}"):
            continue
        if until.o in CRITERIA_WITH_PART and until.p is None:
            out.append(_issue("criterion_needs_part", i, None, f"step {i}: until {until.o} needs a part p"))
            continue
        if until.o == "effect" and until.p in operands:
            out.append(_issue("effect_part_is_operand", i, until.p,
                              f"step {i}: effect part {until.p} is the step's obj or a grip part"))
        if until.o == "fits" and until.p is not None and moves[until.p] not in FREE_MOVES:
            out.append(_issue("fits_part_not_free", i, until.p,
                              f"step {i}: fits part {until.p} has move {moves[until.p]}; it needs free or unknown"))
    return out


def validate_plan(plan: Plan) -> list[PlanIssue]:
    """The §3.4 rules (contract §C11): unique ids and defined references, obj != ref, turn / slide / place / push
    preconditions on the move priors, keep and until rules per do kind. [] = valid."""
    if not isinstance(plan, Plan):
        raise TypeError("validate_plan takes a Plan")
    issues = []
    counts: dict[str, int] = {}
    for part in plan.parts:
        counts[part.pid] = counts.get(part.pid, 0) + 1
    for pid, n in counts.items():
        if n > 1:
            issues.append(_issue("duplicate_pid", None, pid, f"part id {pid} is defined {n} times"))
    moves = {}
    for part in plan.parts:
        moves.setdefault(part.pid, part.move)
    for j, claim in enumerate(plan.hold_true):
        for field, pid in (("a", claim.a), ("b", claim.b)):
            if pid is not None and pid not in moves:
                issues.append(_issue("undefined_pid", None, pid,
                                     f"hold_true {j} {field} names part {pid}, which is not defined"))
    for i, step in enumerate(plan.steps):
        issues.extend(_step_issues(i, step, moves))
    return issues


def issues_text(issues: Sequence[PlanIssue]) -> str:
    """The re-ask error text (<= ISSUES_TEXT_MAX_CHARS): "code: text" items joined by "; ", cut at an item."""
    items = [f"{issue.code}: {issue.text}" for issue in issues]
    text = ""
    for item in items:
        candidate = item if not text else f"{text}; {item}"
        if len(candidate) > ISSUES_TEXT_MAX_CHARS:
            return text or _clip(item, ISSUES_TEXT_MAX_CHARS)
        text = candidate
    return text


def plan_validation_event(issues: Sequence[PlanIssue], *, reask: bool, final: bool = False) -> dict:
    """``plan_validation`` trace payload {ok, issues, reask, final} (§C18.1)."""
    return {"ok": not issues, "issues": [issue.to_dict() for issue in issues], "reask": bool(reask),
            "final": bool(final)}


# ----------------------------------------------------------------------------------------------- part roles
def part_roles(plan: Plan) -> dict[str, frozenset[str]]:
    """ROLES per defined part: grip (a step's grip list, or obj of a grasping step - hold, turn, slide, place_on,
    place_in - with an empty grip list, as ``PlanStep.grip_parts`` resolves it), carried (obj of place_on / place_in
    / push_to / hold), ref, effect (an until.effect part), keepout (a keep.clear_of part), hold_true (named by an
    already-true claim). Parts with no role map to an empty set."""
    roles: dict[str, set[str]] = {part.pid: set() for part in plan.parts}

    def add(pid, role):
        if pid in roles:
            roles[pid].add(role)

    for step in plan.steps:
        grip = step.grip or ((step.obj,) if step.do in DEFAULT_GRIP_KINDS else ())
        for pid in grip:
            add(pid, "grip")
        if step.do in CARRIED_KINDS:
            add(step.obj, "carried")
        if step.ref is not None:
            add(step.ref, "ref")
        for until in step.until:
            if until.o == "effect" and until.p is not None:
                add(until.p, "effect")
        for keep in step.keep:
            if keep.c == "clear_of":
                add(keep.p, "keepout")
    for claim in plan.hold_true:
        add(claim.a, "hold_true")
        if claim.b is not None:
            add(claim.b, "hold_true")
    return {pid: frozenset(pid_roles) for pid, pid_roles in roles.items()}


# ------------------------------------------------------------------------------------------- BDDL-like text
def part_label(text: str, pid: str) -> str:
    """[C-44] slug rule: lowercase, every run of non-letters -> '-', strip leading / trailing '-', cut to
    BDDL_LABEL_MAX_CHARS at a '-' boundary (a hard cut when the first word is longer); the pid when empty."""
    slug = re.sub(r"[^a-z]+", "-", str(text).lower()).strip("-")
    if len(slug) > BDDL_LABEL_MAX_CHARS:
        head = slug[:BDDL_LABEL_MAX_CHARS]
        if slug[BDDL_LABEL_MAX_CHARS] != "-" and "-" in head:
            head = head[:head.rindex("-")]
        slug = head.strip("-")
    return slug or pid


def _atom(*items) -> str:
    return "(" + " ".join(items) + ")"


def _joined(op: str, atoms: Sequence[str]) -> str:
    return atoms[0] if len(atoms) == 1 else _atom(op, *atoms)


def _until_atom(until: Until) -> str:
    items = [until.o]
    if until.p is not None:
        items.append(until.p)
    if until.txt is not None:
        items.append(json.dumps(until.txt))
    return _atom(*items)


def _step_text(step: PlanStep) -> str:
    items = [step.do, step.obj]
    if step.grip:
        items += [":grip", *step.grip]
    if step.ref is not None:
        items += [":ref", step.ref]
    if step.sense != "any":
        items += [":sense", step.sense]
    if step.keep:
        items += [":keep", _joined("and", [_atom(k.c, k.p) for k in step.keep])]
    if step.until:
        items += [":until", _joined("or", [_until_atom(u) for u in step.until])]
    return _atom(*items)


def _goal_atoms(plan: Plan) -> list[str]:
    atoms = []
    for step in plan.steps:
        if step.do == "turn":
            effects = [_atom("effect", u.p, step.obj) for u in step.until if u.o == "effect" and u.p is not None]
            atoms.extend(effects or [_atom(TURN_GOAL_ATOM, step.obj)])
        elif step.do == "slide":
            atoms.append(_atom(SLIDE_GOAL_ATOMS[step.sense], step.obj))
        elif step.ref is not None and step.do in REF_KINDS:
            atoms.append(_atom(GOAL_ATOMS[step.do], step.obj, step.ref))
        else:
            atoms.append(_atom(GOAL_ATOMS[step.do], step.obj))
    for claim in plan.hold_true:
        atoms.append(_atom(claim.rel, claim.a, *(() if claim.b is None else (claim.b,))))
    return atoms


def render_bddl(plan: Plan) -> str:
    """§3.9 BDDL-like rendering (logs and video only): (:parts ...), (:goal ...), (:steps ...)."""
    parts = " ".join(_atom(p.pid, part_label(p.text, p.pid), p.move) for p in plan.parts)
    goal = "(:goal  " + _atom("and", *_goal_atoms(plan)) + ")"
    steps = [_step_text(step) for step in plan.steps]
    lines = [f"(:parts {parts})", goal]
    if steps:
        lines.append("(:steps " + "\n        ".join(steps) + ")")
    else:
        lines.append("(:steps)")
    return "\n".join(lines)


def compact_chars(plan_json: Mapping) -> int:
    """Compact-JSON length, exactly ``json.dumps(separators=(",", ":"))`` (default ensure_ascii, so a non-ASCII
    character counts as its escape) as in the spec §3.8 budgets; allow_nan=False only refuses non-JSON numbers."""
    return len(json.dumps(plan_json, separators=(",", ":"), allow_nan=False))


def relation_plan_event(plan: Plan) -> dict:
    """``relation_plan`` trace payload {plan, bddl, compact_chars, source} (§C18.1)."""
    wire = plan.to_json()
    return {"plan": wire, "bddl": render_bddl(plan), "compact_chars": compact_chars(wire), "source": plan.source}


# -------------------------------------------------------------------------------------------- status block
def _holding(holding):
    if holding is not None and holding not in PIDS:
        raise ValueError("holding must be a part id or None")
    return holding


def plan_status_block(plan: Plan, states: Sequence[StepState], *, holding: str | None) -> dict:
    """Compact plan with step statuses for repair, replan and review requests:
    {"scene", "parts": [[pid, text]], "steps": [{"i", "do", "obj", "grip", "ref", "status", "reason"}],
     "hold_true": [{"rel", "a", "b"}], "holding"}. ``states`` empty -> the plan's steps, all pending. Reasons are
    REASONS codes (never free text), so the block carries no executor text beyond codes and ids."""
    if not isinstance(plan, Plan):
        raise TypeError("plan_status_block takes a Plan")
    rows = []
    if states:
        for state in states:
            if not isinstance(state, StepState):
                raise TypeError("states must be StepState records")
            if state.reason is not None and state.reason not in REASONS:
                raise ValueError(f"step {state.i} reason {state.reason!r} is not a REASONS code")
            rows.append({"i": state.i, "do": state.step.do, "obj": state.step.obj, "grip": list(state.step.grip),
                         "ref": state.step.ref, "status": state.status, "reason": state.reason})
    else:
        rows = [{"i": i, "do": s.do, "obj": s.obj, "grip": list(s.grip), "ref": s.ref, "status": "pending",
                 "reason": None} for i, s in enumerate(plan.steps)]
    return {"scene": plan.scene, "parts": [[p.pid, p.text] for p in plan.parts], "steps": rows,
            "hold_true": [{"rel": h.rel, "a": h.a, "b": h.b} for h in plan.hold_true],
            "holding": _holding(holding)}


# ------------------------------------------------------------------------------------------------- repairs
def with_inserted(plan: Plan, i: int, step: PlanStep) -> Plan:
    """The plan with ``step`` inserted before step i (i == len(steps) appends); the source is kept."""
    if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i <= len(plan.steps):
        raise ValueError(f"insertion index must be in 0..{len(plan.steps)}")
    if not isinstance(step, PlanStep):
        raise TypeError("with_inserted takes a PlanStep")
    return Plan(plan.scene, plan.parts, plan.hold_true, plan.steps[:i] + (step,) + plan.steps[i:], plan.source)


def validate_repair(answer: RepairAnswer, plan: Plan, *, failed_i: int | None, review: bool) -> list[PlanIssue]:
    """use_part / repoint need an existing part; insert_step needs a step that validates in place (before the failed
    step, or appended when there is none) within PLAN_MAX_STEPS; swap_next needs a step after the failed one.
    retry, skip_step, replan and hold_still need nothing. Unused part / step fields are ignored, not rejected.
    [] = valid.

    No failed step. A review (``review=True``) is sent only when every step is done (spec §3.7), so it has no
    failed step and ``failed_i`` must be None (ValueError otherwise); a repair raised by a lost hold_true claim
    (§C3.7) has none either. Then: insert_step appends its step; swap_next is rejected (no_next_step: there is
    nothing to exchange); retry, skip_step, use_part and repoint validate (contract §C11: a review allows every
    action) but name no step, so the StepRunner (§C3.6 step 7) decides what they act on; replan and hold_still need
    nothing, and a review's hold_still is logged by the StepRunner as an unverified claim."""
    if not isinstance(answer, RepairAnswer) or not isinstance(plan, Plan):
        raise TypeError("validate_repair takes a RepairAnswer and a Plan")
    if failed_i is not None and (isinstance(failed_i, bool) or not isinstance(failed_i, int)
                                 or not 0 <= failed_i < len(plan.steps)):
        raise ValueError("failed_i must index a plan step or be None")
    if review and failed_i is not None:
        raise ValueError("a review has no failed step: pass failed_i=None")
    defined = {part.pid for part in plan.parts}
    action = answer.action
    if action in ("use_part", "repoint"):
        if answer.part is None:
            return [_issue("part_required", failed_i, None, f"{action} needs a part id")]
        if answer.part not in defined:
            return [_issue("undefined_pid", failed_i, answer.part,
                           f"{action} names part {answer.part}, which is not defined")]
        return []
    if action == "insert_step":
        if answer.step is None:
            return [_issue("step_required", failed_i, None, "insert_step needs a step")]
        if len(plan.steps) >= PLAN_MAX_STEPS:
            return [_issue("step_limit", failed_i, None, f"the plan already has {len(plan.steps)} steps, the most "
                                                         "allowed")]
        at = failed_i if failed_i is not None else len(plan.steps)
        return validate_plan(with_inserted(plan, at, answer.step))
    if action == "swap_next":
        if failed_i is None or failed_i + 1 >= len(plan.steps):
            text = ("swap_next needs a failed step to exchange; a review has none" if review
                    else "swap_next needs a step after the failed step")
            return [_issue("no_next_step", failed_i, None, text)]
        return []
    if action not in REPAIR_ACTIONS:                   # unreachable: RepairAnswer validates the action
        raise ValueError(f"unknown repair action {action!r}")
    return []
