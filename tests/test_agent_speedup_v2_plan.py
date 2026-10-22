"""T1 tests of the relation plan module (contract §C11; spec §3.3, §3.4, §3.7, §3.9). No network, no simulator.

Fixtures (test-only; the module itself holds no task text): the spec §3.9 task-2 and drawer example plans, with
their scenes re-cut to the 80-character cap.
"""

import ast
import copy
import json
from pathlib import Path
import re

import pytest

from gpt_field import agent_speedup_v2_plan as P
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field import agent_speedup_v2_types as kernel

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
PLAN_PATH = th.PACKAGE_DIR / "agent_speedup_v2_plan.py"
PID = {"type": "string", "enum": ["A", "B", "C", "D", "E", "F", "G", "H"]}

# spec §3.3 / §3.7, transcribed verbatim (PID inline; STEP = the §3.3 step item)
SPEC_STEP = {
    "type": "object", "additionalProperties": False,
    "required": ["do", "obj", "grip", "ref", "sense", "keep", "until"],
    "properties": {
        "do": {"type": "string", "enum": ["hold", "turn", "slide", "place_on", "place_in", "push_to", "release"]},
        "obj": PID,
        "grip": {"type": "array", "minItems": 0, "maxItems": 2, "items": PID},
        "ref": {"anyOf": [PID, {"type": "null"}]},
        "sense": {"type": "string", "enum": ["any", "open", "close"]},
        "keep": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False, "required": ["c", "p"],
            "properties": {"c": {"type": "string", "enum": ["upright", "clear_of"]}, "p": PID}}},
        "until": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False, "required": ["o", "p", "txt"],
            "properties": {
                "o": {"type": "string", "enum": ["effect", "stop", "fits", "quarter_turn", "half_turn"]},
                "p": {"anyOf": [PID, {"type": "null"}]},
                "txt": {"anyOf": [{"type": "string", "maxLength": 40}, {"type": "null"}]}}}}}}
SPEC_PLAN = {
    "type": "object", "additionalProperties": False,
    "required": ["scene", "parts", "hold_true", "steps"],
    "properties": {
        "scene": {"type": "string", "minLength": 1, "maxLength": 80},
        "parts": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "text", "move", "w", "x"],
            "properties": {
                "id": PID,
                "text": {"type": "string", "minLength": 3, "maxLength": 60},
                "move": {"type": "string", "enum": ["free", "fixed", "hinged", "sliding", "unknown"]},
                "w": {"type": "integer", "minimum": 0, "maximum": 64},
                "x": {"type": "integer", "minimum": 0, "maximum": 64}}}},
        "hold_true": {"type": "array", "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False, "required": ["rel", "a", "b"],
            "properties": {
                "rel": {"type": "string", "enum": ["on", "in", "open", "closed", "effect"]},
                "a": PID,
                "b": {"anyOf": [PID, {"type": "null"}]}}}},
        "steps": {"type": "array", "minItems": 0, "maxItems": 6, "items": SPEC_STEP}}}
SPEC_REPAIR = {
    "type": "object", "additionalProperties": False,
    "required": ["why", "action", "part", "step"],
    "properties": {
        "why": {"type": "string", "minLength": 1, "maxLength": 100},
        "action": {"type": "string",
                   "enum": ["retry", "repoint", "use_part", "insert_step", "skip_step", "swap_next", "replan",
                            "hold_still"]},
        "part": {"anyOf": [PID, {"type": "null"}]},
        "step": {"anyOf": [SPEC_STEP, {"type": "null"}]}}}

# spec §3.9 examples (scenes re-cut to the 80-character cap, §C11 test 3)
TASK2_EXAMPLE = {
    "scene": "stove off; black knob with lever below the stove; moka pot at top edge, handle loop toward the frypan "
             "handle"[:80],
    "parts": [
        {"id": "A", "text": "stove knob: lever on top of the knob", "move": "hinged", "w": 31, "x": 30},
        {"id": "B", "text": "stove: ring burner plate", "move": "fixed", "w": 47, "x": 39},
        {"id": "C", "text": "moka pot: black handle loop", "move": "free", "w": 59, "x": 36},
        {"id": "D", "text": "moka pot: whole pot", "move": "free", "w": 60, "x": 44},
        {"id": "E", "text": "frying pan: long handle", "move": "free", "w": 43, "x": 35}],
    "hold_true": [],
    "steps": [
        {"do": "turn", "obj": "A", "grip": ["A"], "ref": None, "sense": "any", "keep": [],
         "until": [{"o": "effect", "p": "B", "txt": "burner rings turn red"}, {"o": "stop", "p": None, "txt": None}]},
        {"do": "place_on", "obj": "D", "grip": ["C"], "ref": "B", "sense": "any",
         "keep": [{"c": "upright", "p": "D"}, {"c": "clear_of", "p": "E"}], "until": []}]}
DRAWER_EXAMPLE = {
    "scene": "cabinet with three closed drawers; white bowl with dark pattern in front of it beside a plate"[:80],
    "parts": [
        {"id": "A", "text": "cabinet: handle bar of the TOP drawer", "move": "sliding", "w": 0, "x": 21},
        {"id": "B", "text": "white bowl with dark pattern: whole bowl", "move": "free", "w": 28, "x": 45},
        {"id": "C", "text": "white bowl with dark pattern: rim", "move": "free", "w": 28, "x": 45},
        {"id": "D", "text": "cabinet: inside of the top drawer", "move": "sliding", "w": 0, "x": 21}],
    "hold_true": [],
    "steps": [
        {"do": "slide", "obj": "A", "grip": ["A"], "ref": None, "sense": "open", "keep": [],
         "until": [{"o": "fits", "p": "B", "txt": None}, {"o": "stop", "p": None, "txt": None}]},
        {"do": "place_in", "obj": "B", "grip": ["C"], "ref": "D", "sense": "any", "keep": [{"c": "upright", "p": "B"}],
         "until": []}]}
SPEC_GOAL_LINE = "(:goal  (and (effect B A) (on D B)))"
SPEC_STEPS_LINES = ('(:steps (turn A :grip A :until (or (effect B "burner rings turn red") (stop)))\n'
                    "        (place_on D :grip C :ref B :keep (and (upright D) (clear_of E))))")


def task2():
    return copy.deepcopy(TASK2_EXAMPLE)


def walk(schema, path="$"):
    yield path, schema
    if isinstance(schema, dict):
        for key, value in schema.items():
            yield from walk(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for index, value in enumerate(schema):
            yield from walk(value, f"{path}[{index}]")


def codes(data):
    return [issue.code for issue in P.validate_plan(P.parse_plan(data))]


# ---------------------------------------------------------------------------------------- 1. schema strictness
def test_schemas_equal_the_spec_text_exactly():
    assert P.plan_schema() == SPEC_PLAN
    assert P.repair_schema() == SPEC_REPAIR
    assert P.step_item_schema() == SPEC_STEP
    assert P.pid_schema() == PID
    assert P.pointing_schema("fine") == {"type": "object", "additionalProperties": False, "required": ["point_id"],
                                         "properties": {"point_id": {"type": "integer", "minimum": 0, "maximum": 64}}}
    assert P.pointing_schema("point_coarse")["required"] == ["cell_id"]
    with pytest.raises(ValueError):
        P.pointing_schema("medium")


def test_every_object_closed_every_key_required_no_refs_and_wire_order():
    for kind, schema in P.schemas().items():
        for path, node in walk(schema):
            if isinstance(node, dict):
                assert not {"$ref", "$defs", "pattern", "definitions"} & set(node), (kind, path)
                if node.get("type") == "object":
                    assert node["additionalProperties"] is False, (kind, path)
                    assert node["required"] == list(node["properties"]), (kind, path)
    assert list(P.plan_schema()["properties"]) == ["scene", "parts", "hold_true", "steps"]
    assert list(P.repair_schema()["properties"]) == ["why", "action", "part", "step"]
    assert list(P.step_item_schema()["properties"]) == ["do", "obj", "grip", "ref", "sense", "keep", "until"]
    assert list(P.schemas()) == ["plan", "replan", "point_fine", "point_coarse", "repair", "review"]
    assert P.schemas()["replan"] == P.schemas()["plan"] and P.schemas()["review"] == P.schemas()["repair"]
    # schema builders return fresh objects (callers may mutate their copy)
    first = P.plan_schema()
    first["properties"]["scene"]["maxLength"] = 1
    assert P.plan_schema()["properties"]["scene"]["maxLength"] == 80


def test_schemas_sha256_is_stable_and_order_sensitive():
    digest = P.schemas_sha256()
    assert len(digest) == 64 and digest == P.schemas_sha256()
    assert P.PLAN_SCHEMA_VERSION == "agent-speedup-v2-plan-schema-1"


# ----------------------------------------------------------------------------------------------- 2. AC-5 scan
BANNED = {"left", "right", "up", "down", "forward", "backward", "north", "south", "east", "west", "clockwise",
          "counterclockwise", "cw", "ccw", "deg", "degree", "angle", "rad", "radian", "yaw", "pitch", "roll", "offset",
          "distance", "height", "depth", "cm", "mm", "meter", "xyz", "dx", "dy", "dz"}
COMPLETION = {"done", "complete", "completed", "success", "finished"}
NUMERIC_FIELDS = {"w", "x", "point_id", "cell_id"}
WHITELIST = {"any", "open", "close", "quarter_turn", "half_turn", "stop", "fits", "effect", "upright", "clear_of",
             "on", "in", "closed"}


def ac5_problems(schema, path="$", problems=None):
    problems = [] if problems is None else problems
    if isinstance(schema, dict):
        if schema.get("type") in ("integer", "number"):
            field = path.rsplit(".", 1)[-1]
            if not (field in NUMERIC_FIELDS and schema["type"] == "integer" and schema.get("minimum") == 0
                    and schema.get("maximum") == 64):
                problems.append(("numeric field", path))
        for key, value in schema.items():
            if key == "properties":
                for name, sub in value.items():
                    if set(th.tokens(name)) & (BANNED | COMPLETION) or name in COMPLETION:
                        problems.append(("property", f"{path}.{name}"))
                    ac5_problems(sub, f"{path}.{name}", problems)
            elif key == "enum":
                for item in value:
                    words = set(th.tokens(item))
                    if item not in WHITELIST and words & (BANNED | COMPLETION):
                        problems.append(("enum", f"{path}={item}"))
                    if item in COMPLETION:
                        problems.append(("completion enum", f"{path}={item}"))
            elif key == "description":
                if set(th.tokens(value)) & COMPLETION:
                    problems.append(("description", path))
            else:
                ac5_problems(value, path, problems)
    elif isinstance(schema, list):
        for item in schema:
            ac5_problems(item, path, problems)
    return problems


def test_ac5_scan_over_all_schemas():
    for kind, schema in P.schemas().items():
        assert ac5_problems(schema) == [], kind
    numeric = [path for kind, schema in P.schemas().items() for path, node in walk(schema)
               if isinstance(node, dict) and node.get("type") in ("integer", "number")]
    assert numeric and all(path.endswith((".w", ".x", ".point_id", ".cell_id")) for path in numeric)


def test_ac5_scan_catches_planted_defects():
    planted = {"type": "object", "properties": {
        "dx": {"type": "number"}, "done": {"type": "boolean"},
        "side": {"type": "string", "enum": ["left", "right"]},
        "w": {"type": "integer", "minimum": 0, "maximum": 99}}}
    found = {kind for kind, _ in ac5_problems(planted)}
    assert {"numeric field", "property", "enum"} <= found


# --------------------------------------------------------------------------------------------- 3. example plans
def test_spec_examples_parse_validate_and_fit_the_compact_budgets():
    for example, budget in ((TASK2_EXAMPLE, 834), (DRAWER_EXAMPLE, 741)):
        assert len(example["scene"]) <= 80
        plan = P.parse_plan(example)
        assert P.validate_plan(plan) == []
        assert plan.to_json() == example                    # lossless round trip in wire order
        assert P.compact_chars(example) == len(json.dumps(example, separators=(",", ":")))
        assert P.compact_chars(example) <= budget
    accented = dict(TASK2_EXAMPLE, scene="stove off; moka pot with a café label")    # the contract's formula,
    assert P.compact_chars(accented) == len(json.dumps(accented, separators=(",", ":")))  # default ensure_ascii
    assert P.compact_chars(accented) == len(json.dumps(accented, ensure_ascii=False, separators=(",", ":"))) + 5


def test_render_bddl_goal_and_steps_lines_equal_the_spec():
    text = P.render_bddl(P.parse_plan(TASK2_EXAMPLE))
    lines = text.split("\n")
    assert lines[1] == SPEC_GOAL_LINE
    assert "\n".join(lines[2:]) == SPEC_STEPS_LINES
    expected_parts = " ".join(f"({p['id']} {P.part_label(p['text'], p['id'])} {p['move']})"
                              for p in TASK2_EXAMPLE["parts"])
    assert lines[0] == f"(:parts {expected_parts})"


def test_slug_rule():
    assert P.part_label("stove knob: lever on top of the knob", "A") == "stove-knob-lever-on-top-of-the"
    assert P.part_label("moka pot: black handle loop", "C") == "moka-pot-black-handle-loop"
    assert P.part_label("cabinet: handle bar of the TOP drawer", "A") == "cabinet-handle-bar-of-the-top"
    assert P.part_label("  ++ 12 ", "E") == "E"                              # no letters -> the pid
    assert P.part_label("x" * 40, "B") == "x" * 32                          # one long word: hard cut
    exact = "abcd-" * 6 + "ef"                                               # 32 characters
    assert P.part_label(exact, "A") == exact
    assert P.part_label(exact + "-tail", "A") == exact                       # a cut at the boundary keeps the word
    for text in ("a" * 31 + " bcd", "one two three four five six seven", "x"):
        label = P.part_label(text, "A")
        assert len(label) <= P.BDDL_LABEL_MAX_CHARS and not label.startswith("-") and not label.endswith("-")
        assert set(label) <= set("abcdefghijklmnopqrstuvwxyz-")


def test_render_other_relations_and_empty_steps():
    plan = P.parse_plan(DRAWER_EXAMPLE)
    text = P.render_bddl(plan)
    assert "(:goal  (and (open A) (in B D)))" in text
    assert "(slide A :grip A :sense open :until (or (fits B) (stop)))" in text
    assert "(place_in B :grip C :ref D :keep (upright B))" in text
    empty = dict(task2(), steps=[], hold_true=[{"rel": "effect", "a": "B", "b": "A"}])
    rendered = P.render_bddl(P.parse_plan(empty))
    assert "(:goal  (and (effect B A)))" in rendered and rendered.endswith("(:steps)")
    event = P.relation_plan_event(plan)
    assert set(event) == {"plan", "bddl", "compact_chars", "source"} and event["source"] == "plan"
    json.dumps(event, allow_nan=False)


# ------------------------------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("scene"),
    lambda d: d.update(extra=1),
    lambda d: d.update(scene=""),
    lambda d: d.update(scene="s" * 81),
    lambda d: d["parts"][0].update(move="rotating"),
    lambda d: d["parts"][0].update(w=65),
    lambda d: d["parts"][0].update(w=-1),
    lambda d: d["parts"][0].update(w=True),
    lambda d: d["parts"][0].update(x=3.0),
    lambda d: d["parts"][0].update(text="ab"),
    lambda d: d["parts"][0].update(text="t" * 61),
    lambda d: d["parts"][0].update(id="Z"),
    lambda d: d["parts"].clear(),
    lambda d: d["parts"].extend([copy.deepcopy(d["parts"][0])] * 4),
    lambda d: d["steps"][0].update(grip=["A", "B", "C"]),
    lambda d: d["steps"][0].update(ref="Q"),
    lambda d: d["steps"][0].update(sense="clockwise"),
    lambda d: d["steps"][0]["until"][0].update(txt="t" * 41),
    lambda d: d["steps"][0]["until"][0].pop("txt"),
    lambda d: d["steps"][0].update(keep=[{"c": "upright", "p": "D"}] * 4),
    lambda d: d["steps"].extend([copy.deepcopy(d["steps"][0])] * 5),
    lambda d: d["hold_true"].append({"rel": "under", "a": "A", "b": None}),
    lambda d: d.update(steps="turn A"),
])
def test_parse_plan_rejects_structural_defects(mutate):
    data = task2()
    mutate(data)
    with pytest.raises(P.PlanParseError) as raised:
        P.parse_plan(data)
    assert len(str(raised.value)) <= P.ISSUE_TEXT_MAX_CHARS
    assert "moka" not in str(raised.value) and "stove" not in str(raised.value)   # values are never echoed


def test_parse_plan_records_and_sources():
    plan = P.parse_plan(TASK2_EXAMPLE, source="replan")
    assert isinstance(plan, kernel.Plan) and plan.source == "replan"
    assert plan.part("A") == kernel.PlanPart("A", "stove knob: lever on top of the knob", "hinged", 31, 30)
    turn = plan.steps[0]
    assert turn.grip == ("A",) and turn.until[0] == kernel.Until("effect", "B", "burner rings turn red")
    assert plan.steps[1].keep == (kernel.Keep("upright", "D"), kernel.Keep("clear_of", "E"))
    with pytest.raises(ValueError):
        P.parse_plan(TASK2_EXAMPLE, source="other")
    empty_grip = task2()
    empty_grip["steps"][0]["grip"] = []
    assert P.parse_plan(empty_grip).steps[0].grip_parts == ("A",)


def test_conform_messages_name_the_path():
    data = task2()
    data["steps"][1]["ref"] = "Q"
    with pytest.raises(P.PlanParseError, match=r"\$\.steps\[1\]\.ref"):
        P.parse_plan(data)
    data = task2()
    data["parts"][2]["surprise"] = "moka"
    with pytest.raises(P.PlanParseError, match="unexpected"):
        P.parse_plan(data)


# --------------------------------------------------------------------------------------------- 4. validator
def _mut(fn):
    data = task2()
    fn(data)
    return data


INVALID = {
    "duplicate_pid": lambda d: d["parts"].append(dict(d["parts"][4])),
    "undefined_pid": lambda d: d["steps"][1].update(grip=["F"]),
    "obj_equals_ref": lambda d: d["steps"][1].update(ref="D"),
    "turn_needs_criterion": lambda d: d["steps"][0].update(until=[]),
    "turn_part_not_rotary": lambda d: d["parts"][0].update(move="fixed"),
    "slide_needs_sense": lambda d: (d["parts"][0].update(move="sliding"),
                                    d["steps"][0].update(do="slide", until=[{"o": "stop", "p": None, "txt": None}])),
    "slide_part_not_sliding": lambda d: d["steps"][0].update(do="slide", sense="open",
                                                             until=[{"o": "stop", "p": None, "txt": None}]),
    "needs_ref": lambda d: d["steps"][1].update(ref=None),
    "obj_not_free": lambda d: d["parts"][3].update(move="fixed") or d["steps"][1].update(
        keep=[{"c": "clear_of", "p": "E"}]),
    "fits_part_not_free": lambda d: (d["parts"][0].update(move="sliding"),
                                     d["steps"][0].update(do="slide", sense="open",
                                                          until=[{"o": "fits", "p": "B", "txt": None}])),
    "upright_part_not_free": lambda d: d["steps"][1].update(keep=[{"c": "upright", "p": "B"},
                                                                  {"c": "clear_of", "p": "E"}]),
    "clear_of_is_obj": lambda d: d["steps"][1].update(keep=[{"c": "clear_of", "p": "D"},
                                                            {"c": "clear_of", "p": "E"}]),
    "rotation_criterion_not_turn": lambda d: d["steps"][1].update(until=[{"o": "quarter_turn", "p": None,
                                                                          "txt": None}]),
    "until_not_allowed": lambda d: d["steps"][1].update(until=[{"o": "stop", "p": None, "txt": None}]),
    "effect_part_is_operand": lambda d: d["steps"][0]["until"][0].update(p="A"),
    "criterion_needs_part": lambda d: d["steps"][0]["until"][0].update(p=None),
    "unused_object": lambda d: d["steps"][1].update(keep=[{"c": "upright", "p": "D"}]),          # E in no step
    "grip_object_mismatch": lambda d: d["steps"][1].update(grip=["E"]),                         # another object
    "grip_part_not_free": lambda d: d["parts"][2].update(move="fixed"),             # the pot's loop is not a fixture
    "duplicate_until": lambda d: d["steps"][0]["until"].append({"o": "effect", "p": "B", "txt": "glows"}),
}


def test_one_failing_fixture_per_issue_code():
    assert set(INVALID) == set(P.PLAN_ISSUE_CODES) and len(P.PLAN_ISSUE_CODES) == len(set(P.PLAN_ISSUE_CODES))
    vocab = set(th.lint_vocabulary(LIBERO)["vocabulary"])
    for code, fn in INVALID.items():
        issues = P.validate_plan(P.parse_plan(_mut(fn)))
        assert {issue.code for issue in issues} == {code}, (code, [i.to_dict() for i in issues])
        for issue in issues:
            assert 0 < len(issue.text) <= P.ISSUE_TEXT_MAX_CHARS
            hits = set().union(*(th.token_hits(t, vocab, th.LINT_ALLOWLIST) for t in th.tokens(issue.text)))
            assert not hits, (code, issue.text)                     # executor text: ids and codes only
            json.dumps(issue.to_dict(), allow_nan=False)


def test_validator_details():
    issues = P.validate_plan(P.parse_plan(_mut(INVALID["undefined_pid"])))
    assert issues[0].step == 1 and issues[0].pid == "F"
    hold = task2()
    hold["hold_true"] = [{"rel": "on", "a": "D", "b": "G"}]
    issues = P.validate_plan(P.parse_plan(hold))
    assert [(i.code, i.step, i.pid) for i in issues] == [("undefined_pid", None, "G")]
    unknown = task2()                                                # unknown counts as rotary, free and sliding
    for part in unknown["parts"]:
        part["move"] = "unknown"
    assert P.validate_plan(P.parse_plan(unknown)) == []
    assert P.validate_plan(P.parse_plan(DRAWER_EXAMPLE)) == []
    # every do kind with an empty until list is allowed except turn
    for do in ("hold", "release", "push_to"):
        data = task2()
        data["parts"] = [data["parts"][1], data["parts"][3]] if do == "push_to" else [data["parts"][3]]
        data["steps"] = [{"do": do, "obj": "D", "grip": [], "ref": "B" if do == "push_to" else None, "sense": "any",
                          "keep": [], "until": []}]
        assert P.validate_plan(P.parse_plan(data)) == [], do
    effect_on_place = task2()
    effect_on_place["steps"][1]["until"] = [{"o": "effect", "p": "B", "txt": "glows"}]
    assert P.validate_plan(P.parse_plan(effect_on_place)) == []
    with pytest.raises(TypeError):
        P.validate_plan(TASK2_EXAMPLE)


def neutral(parts, steps, hold_true=()):
    """A plan of (id, text, move) parts; cells are irrelevant to the validator."""
    return {"scene": "test scene", "parts": [{"id": i, "text": t, "move": m, "w": 1, "x": 1} for i, t, m in parts],
            "hold_true": list(hold_true), "steps": list(steps)}


def step(do, obj, *, grip=(), ref=None, sense="any", keep=(), until=()):
    return {"do": do, "obj": obj, "grip": list(grip), "ref": ref, "sense": sense, "keep": list(keep),
            "until": list(until)}


def until(o, p=None, txt=None):
    return {"o": o, "p": p, "txt": txt}


def issue_rows(data):
    return [(i.code, i.step, i.pid) for i in P.validate_plan(P.parse_plan(data))]


def test_unused_object_flags_a_listed_object_that_no_step_uses():
    # libero_10_t0 live plan 001 attempt 0 (planck_001): the tomato sauce named, never planned
    t0 = neutral([("A", "alphabet soup: whole blue-labeled can", "free"), ("B", "alphabet soup: top rim", "free"),
                  ("C", "tomato sauce: whole can", "free"), ("D", "tomato sauce: top rim", "free"),
                  ("E", "basket: open cavity", "free")],
                 [step("place_in", "A", grip=["B"], ref="E")], hold_true=[{"rel": "open", "a": "E", "b": None}])
    assert issue_rows(t0) == [("unused_object", None, "C")]                  # one issue per object, its first part
    used = copy.deepcopy(t0)
    used["steps"].append(step("place_in", "C", ref="E"))
    assert issue_rows(used) == []                                            # D: an unused part of a used object
    # a use is any step role (obj, grip, ref, keep, until) or an already-true claim
    parts = [("A", "music box: crank", "hinged"), ("B", "lamp: shade", "fixed"), ("C", "toy car: whole car", "free"),
             ("D", "rug: star patch", "fixed")]
    base = [step("turn", "A", keep=[{"c": "clear_of", "p": "C"}], until=[until("effect", "B", "lights up")])]
    assert issue_rows(neutral(parts, base, [{"rel": "on", "a": "C", "b": "D"}])) == []
    assert issue_rows(neutral(parts, base)) == [("unused_object", None, "D")]
    # a qualified head of a used object is that object; a text without an object head is never flagged
    qualified = neutral([("A", "stove knob: lever", "hinged"), ("B", "stove: plate", "fixed"),
                         ("C", "a loose sponge", "free")], [step("turn", "A", until=[until("stop")])])
    assert issue_rows(qualified) == []
    empty = neutral(parts, [], [{"rel": "on", "a": "C", "b": "D"}])          # no steps: claims are the only uses
    assert issue_rows(empty) == [("unused_object", None, "A"), ("unused_object", None, "B")]


def test_grip_object_mismatch_and_object_heads():
    parts = [("A", "microwave: open door", "hinged"), ("B", "microwave door: outer frame edge", "hinged"),
             ("C", "the white mug: whole mug", "free"), ("D", "white mug: loop", "free"),
             ("E", "gray mug: loop", "free"), ("F", "a plain sponge", "free"), ("G", "microwave: cavity", "fixed")]
    keep = [{"c": "clear_of", "p": "E"}]
    ok = neutral(parts, [step("turn", "A", grip=["B"], until=[until("stop")], keep=keep),
                         step("place_in", "C", grip=["D", "C"], ref="G", keep=[{"c": "clear_of", "p": "F"}])])
    assert issue_rows(ok) == []                                  # qualified heads and articles: the same object
    wrong = copy.deepcopy(ok)
    wrong["steps"][1]["grip"] = ["E", "F"]                       # another mug; F has no object head (unknown)
    wrong["steps"][1]["keep"] = [{"c": "clear_of", "p": "D"}]
    assert issue_rows(wrong) == [("grip_object_mismatch", 1, "E")]
    assert P.object_words("The Gray Mug: loop") == ("gray", "mug") and P.object_words("no head") is None
    assert P.object_words(": loop") is None and P.same_object(None, ("mug",))
    assert P.same_object(("microwave",), ("microwave", "door")) and not P.same_object(("white", "mug"),
                                                                                        ("gray", "mug"))


def test_grip_part_not_free_only_for_a_carried_free_obj():
    # planck_002 (prompt 2 before the move-prior clarification): "moka pot: black loop handle" came back fixed
    fixed_loop = _mut(INVALID["grip_part_not_free"])
    assert issue_rows(fixed_loop) == [("grip_part_not_free", 1, "C")]
    unknown_loop = task2()
    unknown_loop["parts"][2]["move"] = "unknown"
    assert issue_rows(unknown_loop) == []                                     # unknown counts as free [C-22]
    unknown_obj = _mut(INVALID["grip_part_not_free"])
    unknown_obj["parts"][3]["move"] = "unknown"                              # obj not known to be free: no claim
    assert issue_rows(unknown_obj) == []
    drawer = copy.deepcopy(DRAWER_EXAMPLE)                                    # a slide grips a sliding part: fine
    assert issue_rows(drawer) == []


def test_duplicate_until_once_per_repeated_criterion():
    parts = [("A", "music box: crank", "hinged"), ("B", "music box: dancer", "hinged"), ("C", "music box: dome", "fixed")]
    repeated = neutral(parts, [step("turn", "A", until=[until("effect", "B", "spins"), until("effect", "B", "turns"),
                                                        until("effect", "B", "moves")])])
    assert issue_rows(repeated) == [("duplicate_until", 0, "B")]
    stops = neutral(parts, [step("turn", "A", until=[until("stop"), until("stop"), until("effect", "C", "glows")])])
    assert issue_rows(stops) == [("duplicate_until", 0, None)]
    distinct = neutral(parts, [step("turn", "A", until=[until("effect", "B", "spins"), until("effect", "C", "glows"),
                                                        until("stop")])])
    assert issue_rows(distinct) == []


def test_new_issue_codes_on_the_live_plan_check_answers():
    """planck_001 shapes: the repeated effect criterion (libero_10_t1 attempt 1) and the grip of a qualified head
    (libero_10_t9: obj 'microwave: door', grip 'microwave door: frame edge', the same object)."""
    t9 = neutral([("A", "yellow-and-white mug: whole mug", "free"), ("B", "yellow-and-white mug: loop", "free"),
                  ("C", "microwave: interior cavity", "fixed"), ("D", "microwave: open hinged door", "hinged"),
                  ("E", "microwave door: thin outer frame edge", "hinged"), ("F", "gray mug: whole mug", "free")],
                 [step("place_in", "A", grip=["B"], ref="C", keep=[{"c": "clear_of", "p": "F"}]),
                  step("turn", "D", grip=["E"], sense="close", until=[until("stop")])])
    assert issue_rows(t9) == []
    t1 = copy.deepcopy(t9)
    t1["steps"][0]["until"] = [until("effect", "C", "holds the mug")] * 3
    assert issue_rows(t1) == [("duplicate_until", 0, "C")]


# ----------------------------------------------------------------------------------------------- sanitizing
@pytest.mark.parametrize("raw, clean", [
    ("moka pot: whole pot", "moka pot: whole pot"),                       # printable ASCII: unchanged
    ("rests on  ", "rests on  "),
    ("black rotary\u200e\u200d", "black rotary"),                         # direction / zero-width marks at the cut
    ("rests inside the\u7bee\u7bee", "rests inside the"),                 # another script at the cut
    ("bowl between the plate and the\u94f6\u8272", "bowl between the plate and the"),
    ("separate gray mug on\u2202", "separate gray mug on"),
    ("released mug at\u2032", "released mug at"),
    ("pot\u2019s \u201cloop\u201d \u2013 left", "pot's \"loop\" - left"),
    ("caf\u00e9 label\u2026", "cafe label..."),
    ("cafe\u0301 label", "cafe label"),                                    # a decomposed accent keeps the word whole
    ("a\ud83d tail", "a tail"),                                         # a lone surrogate half (partial code point)
    ("tab\there\nnext", "tab here next"),
    ("\u200b\u200b", ""),
    ("\ue000x\U000e0001y", "xy"),                                         # private use and tag characters
])
def test_sanitize_text(raw, clean):
    assert P.sanitize_text(raw) == clean
    assert all(" " <= ch <= "~" for ch in P.sanitize_text(raw))


def test_sanitize_text_cuts_a_lengthened_text_and_reports_unsafe_characters():
    assert P.sanitize_text("ab\u2026", 4) == "ab.." and P.sanitize_text("abc", 2) == "abc"   # ASCII is never cut
    for code in list(range(0, 0x3000, 7)) + [0xd800, 0xdfff, 0xfeff, 0x200b, 0x202e, 0x2066, 0x1f600]:
        text = "x" + chr(code) + "y"
        out = P.sanitize_text(text, 10)
        assert all(" " <= ch <= "~" for ch in out) and len(out) <= 10, hex(code)
    assert P.unsafe_characters("a\u200bb\x07c\ud800\u00e9") == ["U+200B", "U+0007", "U+D800"]
    assert P.unsafe_characters("plain text, caf\u00e9") == []
    with pytest.raises(TypeError):
        P.sanitize_text(None)


def test_parse_plan_sanitizes_model_texts_and_refuses_unreadable_ones():
    data = task2()
    data["scene"] = "moka pot beside a stove with a black rotary\u200e\u200d"
    data["parts"][2]["text"] = "moka pot: black handle loop and the\u94f6\u8272"
    data["steps"][0]["until"][0]["txt"] = "burner rings turn red\u7bee"
    plan = P.parse_plan(data)
    assert plan.scene == "moka pot beside a stove with a black rotary"
    assert plan.part("C").text == "moka pot: black handle loop and the"
    assert plan.steps[0].until[0].txt == "burner rings turn red"
    wire = plan.to_json()
    assert json.dumps(wire, ensure_ascii=False).isascii() and P.validate_plan(plan) == []
    blank = task2()
    blank["steps"][0]["until"][0]["txt"] = "\u200b\u200d"                  # a description of nothing: no description
    assert P.parse_plan(blank).steps[0].until[0].txt is None
    kept_empty = task2()
    kept_empty["steps"][0]["until"][0]["txt"] = ""
    assert P.parse_plan(kept_empty).steps[0].until[0].txt == ""          # the model's own empty string stays
    for path, mutate in (("$.parts[1].text", lambda d: d["parts"][1].update(text="ab\u7bee\u7bee")),
                         ("$.scene", lambda d: d.update(scene="\u200b"))):
        bad = task2()
        mutate(bad)
        with pytest.raises(P.PlanParseError, match=re.escape(path)) as raised:
            P.parse_plan(bad)
        assert len(str(raised.value)) <= P.ISSUE_TEXT_MAX_CHARS
    answer = P.parse_repair(repair("retry", why="the grip slipped\u200f"))
    assert answer.why == "the grip slipped"
    with pytest.raises(P.PlanParseError, match=r"\$\.why"):
        P.parse_repair(repair("retry", why="\u200f"))
    inserted = P.parse_repair(repair("insert_step", step=dict(INSERT_BAD, until=[until("stop", txt="end\u7bee")])))
    assert inserted.step.until[0].txt == "end"


# ----------------------------------------------------------------------------------------- re-ask retention
def retention(previous, corrected, issues=None):
    before, after = P.parse_plan(previous), P.parse_plan(corrected)
    found = P.validate_plan(before) if issues is None else issues
    return [(i.code, i.step, i.pid) for i in P.reask_retention_issues(before, after, found)]


def test_reask_retention_keeps_unnamed_steps():
    previous = _mut(INVALID["effect_part_is_operand"])                   # step 0 named
    assert retention(previous, task2()) == []
    assert retention(previous, dict(task2(), steps=task2()["steps"][1:])) == []      # the named step may go
    dropped = dict(task2(), steps=task2()["steps"][:1])
    assert retention(previous, dropped) == [("step_dropped", 1, "D")]
    (issue,) = P.reask_retention_issues(P.parse_plan(previous), P.parse_plan(dropped),
                                        P.validate_plan(P.parse_plan(previous)))
    assert issue.text.startswith("step 1 (place_on D)") and len(issue.text) <= P.ISSUE_TEXT_MAX_CHARS
    assert P.issues_text([issue]).startswith("step_dropped: step 1")
    # ids may be renamed when texts stay, texts reworded when ids stay; another object under the same id is a drop
    renamed = task2()
    renamed["parts"][2]["id"], renamed["parts"][3]["id"] = "D", "C"
    renamed["steps"][1].update(obj="C", grip=["D"])
    renamed["steps"][1]["keep"][0]["p"] = "C"
    assert retention(previous, renamed) == []
    reworded = task2()
    reworded["parts"][3]["text"] = "moka pot: silver body"
    assert retention(previous, reworded) == []
    other = task2()
    other["parts"][3]["text"] = "teapot: whole teapot"
    assert retention(previous, other) == [("step_dropped", 1, "D")]
    resensed = copy.deepcopy(DRAWER_EXAMPLE)
    resensed["steps"][0]["sense"] = "close"
    named_place = [P.PlanIssue("needs_ref", 1, "B", "step 1: place_in needs a ref part")]
    assert retention(DRAWER_EXAMPLE, resensed, named_place) == [("step_dropped", 0, "A")]


def test_reask_retention_exemptions():
    issues = [P.PlanIssue("effect_part_is_operand", 2, "D", "step 2: effect part D is the step's obj")]
    with_hold = dict(task2(), steps=[task2()["steps"][0], HOLD, task2()["steps"][1],
                                     step("release", "D")])
    plain = task2()
    assert retention(with_hold, plain, issues) == []          # the unnecessary hold and the release may go
    lone_hold = dict(task2(), steps=[task2()["steps"][0], HOLD])
    assert retention(lone_hold, task2(), []) == [("step_dropped", 1, "D")]    # a hold that is the goal stays
    plan_level = [P.PlanIssue("unused_object", None, "E", "part E: its object is in no step")]
    assert retention(task2(), dict(task2(), steps=task2()["steps"][:1]), plan_level) == []   # step 1 keeps E
    twice = dict(task2(), steps=[task2()["steps"][1], task2()["steps"][1]])
    assert retention(twice, dict(task2(), steps=[task2()["steps"][1]]), []) == [("step_dropped", 1, "D")]
    with pytest.raises(TypeError):
        P.reask_retention_issues(task2(), P.parse_plan(task2()), [])


def test_reask_retention_on_the_live_t3_reask():
    """libero_10_t3 (planck_001): the issue named only step 2, and the regenerated answer dropped the slide close
    (and the slide open); both are step_dropped, the release is not."""
    parts = [("A", "black bowl: whole bowl", "free"), ("B", "black bowl: thin rim", "free"),
             ("C", "cabinet: open bottom drawer cavity", "sliding"), ("D", "cabinet: bottom drawer handle", "sliding"),
             ("E", "bottle: dark bottle beside the drawer", "free")]
    keep = [{"c": "upright", "p": "A"}, {"c": "clear_of", "p": "E"}]
    first = neutral(parts, [
        step("slide", "C", grip=["D"], sense="open", keep=keep[1:], until=[until("fits", "A"), until("stop")]),
        step("hold", "A", grip=["B"], keep=keep),
        step("place_in", "A", grip=["B"], ref="C", keep=keep, until=[until("effect", "A", "rests inside")]),
        step("slide", "C", grip=["D"], sense="close", keep=keep[1:], until=[until("stop")]),
        step("release", "C")])
    assert [i[0] for i in issue_rows(first)] == ["effect_part_is_operand"]
    answer = neutral(parts, [step("hold", "A", grip=["B"], keep=keep),
                             step("place_in", "A", grip=["B"], ref="C", keep=keep,
                                  until=[until("effect", "C", "holds the bowl")])])
    assert retention(first, answer) == [("step_dropped", 0, "C"), ("step_dropped", 3, "C")]
    targeted = copy.deepcopy(first)
    targeted["steps"][2]["until"] = [until("effect", "C", "holds the bowl")]
    assert retention(first, targeted) == [] and issue_rows(targeted) == []


def test_validate_repair_judges_an_insertion_by_what_it_adds():
    mismatched = task2()
    mismatched["steps"][1]["grip"] = ["C", "E"]                            # an adopted plan already off (use_part)
    plan = P.parse_plan(mismatched)
    assert [i.code for i in P.validate_plan(plan)] == ["grip_object_mismatch"]
    ok = P.parse_repair(repair("insert_step", step=INSERT_OK))
    assert P.validate_repair(ok, plan, failed_i=1, review=False) == []
    assert P.validate_repair(ok, plan, failed_i=0, review=False) == []     # the old finding moves to step 2
    bad = P.parse_repair(repair("insert_step", step=dict(INSERT_OK, do="hold", grip=["E"])))
    assert [(i.code, i.step) for i in P.validate_repair(bad, plan, failed_i=0, review=False)] == [
        ("grip_object_mismatch", 0)]


def test_plan_issue_record_and_issues_text():
    with pytest.raises(ValueError):
        P.PlanIssue("unknown_code", 0, "A", "x")
    with pytest.raises(ValueError):
        P.PlanIssue("needs_ref", 0, "A", "x" * 121)
    with pytest.raises(ValueError):
        P.PlanIssue("needs_ref", -1, "A", "x")
    many = [P.PlanIssue("needs_ref", i, "A", f"step {i}: place_on needs a ref part " + "z" * 60) for i in range(6)]
    text = P.issues_text(many)
    assert 0 < len(text) <= P.ISSUES_TEXT_MAX_CHARS and text.startswith("needs_ref: step 0")
    assert P.issues_text([]) == ""
    assert P.issues_text(many[:1]) == f"needs_ref: {many[0].text}"
    event = P.plan_validation_event(many[:2], reask=True)
    assert event["ok"] is False and event["reask"] is True and len(event["issues"]) == 2
    assert P.plan_validation_event([], reask=False)["ok"] is True


# --------------------------------------------------------------------------------------------- 5. part roles
def test_part_roles_on_the_task2_example():
    roles = P.part_roles(P.parse_plan(TASK2_EXAMPLE))
    assert roles == {"A": frozenset({"grip"}), "B": frozenset({"effect", "ref"}), "C": frozenset({"grip"}),
                     "D": frozenset({"carried"}), "E": frozenset({"keepout"})}
    assert all(role in kernel.ROLES for pid_roles in roles.values() for role in pid_roles)


def test_part_roles_defaults_and_hold_true():
    data = task2()
    data["steps"][0]["grip"] = []                       # default grip of a turn = obj
    data["steps"][1]["grip"] = []                       # place_on with an empty grip list grips obj itself
    data["hold_true"] = [{"rel": "on", "a": "E", "b": "B"}]
    plan = P.parse_plan(data)
    roles = P.part_roles(plan)
    assert roles["A"] == {"grip"} and roles["D"] == {"grip", "carried"} and roles["C"] == frozenset()
    assert roles["E"] == {"keepout", "hold_true"} and roles["B"] == {"effect", "ref", "hold_true"}
    # the grip role agrees with the kernel's grip resolution (PlanStep.grip_parts / StepContext.from_step)
    for step in plan.steps:
        assert all("grip" in roles[pid] for pid in step.grip_parts)


@pytest.mark.parametrize("do, ref, obj_roles", [
    ("hold", None, {"grip", "carried"}), ("place_in", "B", {"grip", "carried"}), ("slide", None, {"grip"}),
    ("push_to", "B", {"carried"}), ("release", None, set()),
])
def test_part_roles_empty_grip_list_per_relation(do, ref, obj_roles):
    data = task2()
    move = {"slide": "sliding"}.get(do, "free")
    data["parts"][3]["move"] = move
    sense = "open" if do == "slide" else "any"
    data["steps"] = [{"do": do, "obj": "D", "grip": [], "ref": ref, "sense": sense, "keep": [], "until": []}]
    roles = P.part_roles(P.parse_plan(data))
    assert roles["D"] == obj_roles, do


# ------------------------------------------------------------------------------------------ status block
def test_plan_status_block_shape_and_states():
    plan = P.parse_plan(TASK2_EXAMPLE)
    block = P.plan_status_block(plan, [], holding=None)
    assert list(block) == list(P.STATUS_BLOCK_KEYS)
    assert block["parts"][0] == ["A", "stove knob: lever on top of the knob"]
    assert [row["status"] for row in block["steps"]] == ["pending", "pending"]
    states = [kernel.StepState(0, plan.steps[0], "done", "turned_by_effect", {}, {}, 120, 1, [3]),
              kernel.StepState(1, plan.steps[1], "failed", "hold_lost", {}, {}, 80, 2, [5])]
    block = P.plan_status_block(plan, states, holding="C")
    assert block["steps"][1] == {"i": 1, "do": "place_on", "obj": "D", "grip": ["C"], "ref": "B", "status": "failed",
                                 "reason": "hold_lost"}
    assert block["holding"] == "C" and block["hold_true"] == []
    json.dumps(block, allow_nan=False)
    with pytest.raises(ValueError):
        P.plan_status_block(plan, [kernel.StepState(0, plan.steps[0], "failed", "the moka pot fell", {}, {}, 1, 1)],
                            holding=None)
    with pytest.raises(ValueError):
        P.plan_status_block(plan, [], holding="Z")


# -------------------------------------------------------------------------------------------- 6. repairs
def repair(action, part=None, step=None, why="the grip slipped"):
    return {"why": why, "action": action, "part": part, "step": step}


INSERT_OK = {"do": "release", "obj": "D", "grip": [], "ref": None, "sense": "any", "keep": [], "until": []}
HOLD = {"do": "hold", "obj": "D", "grip": ["C"], "ref": None, "sense": "any", "keep": [], "until": []}
INSERT_BAD = {"do": "turn", "obj": "B", "grip": [], "ref": None, "sense": "any", "keep": [], "until": []}


@pytest.mark.parametrize("data, failed_i, review, expected", [
    (repair("retry"), 1, False, set()),
    (repair("skip_step"), 1, False, set()),
    (repair("replan"), 0, False, set()),
    (repair("hold_still"), None, True, set()),
    (repair("hold_still"), 1, False, set()),
    (repair("repoint", part="C"), 1, False, set()),
    (repair("repoint"), 1, False, {"part_required"}),
    (repair("use_part", part="D"), 1, False, set()),
    (repair("use_part", part="H"), 1, False, {"undefined_pid"}),
    (repair("use_part"), 1, False, {"part_required"}),
    (repair("insert_step", step=INSERT_OK), 1, False, set()),
    (repair("insert_step", step=INSERT_OK), None, True, set()),
    (repair("insert_step"), 1, False, {"step_required"}),
    (repair("insert_step", step=INSERT_BAD), 1, False, {"turn_needs_criterion", "turn_part_not_rotary"}),
    (repair("swap_next"), 0, False, set()),
    (repair("swap_next"), 1, False, {"no_next_step"}),
    (repair("swap_next"), None, True, {"no_next_step"}),
    (repair("retry", part="A", step=INSERT_OK), 1, False, set()),          # unused fields are ignored
])
def test_validate_repair_each_action(data, failed_i, review, expected):
    plan = P.parse_plan(TASK2_EXAMPLE)
    answer = P.parse_repair(data)
    issues = P.validate_repair(answer, plan, failed_i=failed_i, review=review)
    assert {issue.code for issue in issues} == expected, [i.to_dict() for i in issues]


def test_insert_step_positions_and_limit():
    plan = P.parse_plan(TASK2_EXAMPLE)
    step = P.parse_repair(repair("insert_step", step=INSERT_OK)).step
    assert P.with_inserted(plan, 1, step).steps == (plan.steps[0], step, plan.steps[1])
    assert P.with_inserted(plan, 2, step).steps[-1] == step and P.with_inserted(plan, 0, step).steps[0] == step
    with pytest.raises(ValueError):
        P.with_inserted(plan, 3, step)
    full = task2()
    full["steps"] = full["steps"] * 3
    issues = P.validate_repair(P.parse_repair(repair("insert_step", step=INSERT_OK)), P.parse_plan(full),
                               failed_i=0, review=False)
    assert [i.code for i in issues] == ["step_limit"]
    with pytest.raises(ValueError):
        P.validate_repair(P.parse_repair(repair("retry")), plan, failed_i=2, review=False)


def test_validate_repair_review_has_no_failed_step():
    plan = P.parse_plan(TASK2_EXAMPLE)
    with pytest.raises(ValueError):                          # a review is sent only when every step is done
        P.validate_repair(P.parse_repair(repair("retry")), plan, failed_i=1, review=True)
    for action in kernel.REPAIR_ACTIONS:                    # every action stays allowed except swap_next
        data = repair(action, part="C" if action in ("use_part", "repoint") else None,
                      step=INSERT_OK if action == "insert_step" else None)
        codes = [i.code for i in P.validate_repair(P.parse_repair(data), plan, failed_i=None, review=True)]
        assert codes == (["no_next_step"] if action == "swap_next" else []), action
    (issue,) = P.validate_repair(P.parse_repair(repair("swap_next")), plan, failed_i=None, review=True)
    assert "review" in issue.text and issue.step is None
    (issue,) = P.validate_repair(P.parse_repair(repair("swap_next")), plan, failed_i=None, review=False)
    assert "review" not in issue.text                       # a repair raised by a lost hold_true claim


@pytest.mark.parametrize("bad", [
    repair("teleport"), repair("retry", why=""), repair("retry", why="w" * 101), repair("use_part", part="Z"),
    {"why": "x", "action": "retry", "part": None}, dict(repair("retry"), extra=1),
    repair("insert_step", step={"do": "turn"}), repair("retry", why=7),
])
def test_parse_repair_rejects_structural_defects(bad):
    with pytest.raises(P.PlanParseError):
        P.parse_repair(bad)


def test_parse_repair_records():
    answer = P.parse_repair(repair("insert_step", step=INSERT_OK))
    assert isinstance(answer, kernel.RepairAnswer) and answer.step.do == "release"
    assert answer.to_json() == repair("insert_step", step=INSERT_OK)


# --------------------------------------------------------------------------------------------------- T0 lint
def test_plan_module_passes_the_t0_lint_firewall_and_literal_audit():
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    assert th.lint_module(PLAN_PATH, vocab) == []
    assert th.literal_audit(PLAN_PATH) == []
    if hasattr(th, "firewall_violations"):
        assert th.firewall_violations(ast.parse(PLAN_PATH.read_text()), "agent_speedup_v2_plan") == []
    assert "agent_speedup_v2_plan.py" in th.CONTROL_MODULES
    labels = th.module_labels(PLAN_PATH)
    assert labels and all(key in th.constants_register() for key in labels)


def test_plan_module_imports_only_the_kernel_and_holds_no_task_text():
    tree = ast.parse(PLAN_PATH.read_text())
    internal = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            internal.add(node.module)
    assert internal == {"agent_speedup_v2_types"}
    from gpt_field.agent_speedup_v2_tasks import list_v2_tasks
    source = PLAN_PATH.read_text()
    for spec in list_v2_tasks():
        assert spec.prompt not in source
    assert "start_poses" not in source and "measured_tasks" not in source


def test_oracle_and_fake_plan_templates_validate():
    import agent_speedup_v2_fakes as fakes
    plan = P.parse_plan(fakes.TASK2_ORACLE_PLAN)
    assert P.validate_plan(plan) == []
    path = Path(__file__).resolve().parents[1] / "scripts" / "agent_speedup_v2_oracle.py"
    assert path.is_file()


def test_runner_repair_issues_repoint_limit_and_retry_unchanged():
    """Handle lens BX-1 and BRR-2: the StepRunner refuses a repoint of a part that already failed REPOINT_LIMIT
    repoints within one step (repoint_limit) and a retry of a refusal that repeated without motion from an unchanged
    hand state (retry_unchanged). Both are repair issue codes with executor-written texts naming parts by id; the
    repoint text is the lens's measured wording."""
    assert {"repoint_limit", "retry_unchanged"} <= set(P.REPAIR_ISSUE_CODES)
    assert not {"repoint_limit", "retry_unchanged"} & set(P.PLAN_ISSUE_CODES)
    assert len(P.ISSUE_CODES) == len(set(P.ISSUE_CODES))
    issue = P.repoint_limit_issue("D", 2, 1)
    assert (issue.code, issue.step, issue.pid) == ("repoint_limit", 1, "D")
    assert issue.text == "repoint of part D failed 2 times at this object; choose use_part, replan or skip_step"
    unchanged = P.retry_unchanged_issue(3, 0)
    assert (unchanged.code, unchanged.step, unchanged.pid) == ("retry_unchanged", 0, None)
    assert "3" in unchanged.text and "retry" in unchanged.text and "without motion" in unchanged.text
    assert P.issues_text([issue]) == "repoint_limit: " + issue.text
    for item in (issue, unchanged, P.retry_unchanged_issue(10 ** 6, None), P.repoint_limit_issue("H", 10 ** 6, None)):
        assert len(item.text) <= P.ISSUE_TEXT_MAX_CHARS and item.text.isascii()
    with pytest.raises(ValueError):
        P.repoint_limit_issue("Q", 2, 0)
    with pytest.raises(ValueError):
        P.retry_unchanged_issue(0, 0)                     # a retry refusal needs a repeated refusal
