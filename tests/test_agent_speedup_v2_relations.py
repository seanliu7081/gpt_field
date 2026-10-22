"""T1 tests of the agent_SpeedUp_v2 relation controllers (contract §C10.7; decisions 1 and 2 §2.6; lead ruling 3).

No simulator, no network, no model. Three layers:
1. module hygiene (T0 lint, labels, restated v1 constants, evidence text, registries);
2. controller DECISION LOGIC on a scripted driver: the controllers' sub-steps (hover, plan, Align, descent, close,
   load test, twist segments, lowering, ...) are replaced by scripted stand-ins, so each branch of §C10.2-§C10.4
   (never close above the band, never open when perched, the reversal, the budget pre-check, the grip fallback,
   the re-point after seed_outside_region, ...) is exercised exactly;
3. physics on FakeTwoViewSim through the real policy loop (§C3.4) with the real client over ScriptedV2Transport
   (FakeOracleAnswers): Calibrate+ on the fake, and the task-2-like chain with a truth-augmented survey recorder
   injected through ``RelationContext.survey_recorder`` (the fake's rendered textures are too weak for the real
   start survey to register the destination and the keep-out part; see ``TruthAugmentedRecorder``).
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import agent_speedup_v2_fakes as fakes
from gpt_field import agent_speedup_v2_geometry as geometry
from gpt_field import agent_speedup_v2_planner as planner
from gpt_field import agent_speedup_v2_primitives as primitives
from gpt_field import agent_speedup_v2_robot as robot_facts
from gpt_field import agent_speedup_v2_relations as R
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field import agent_speedup_v2_types as kernel
from gpt_field.agent_speedup_v2_observers import InvariantMonitor, StaysChecker
from gpt_field.agent_speedup_v2_types import (DepthSample, HandPose, Keep, Meas, ObsReport, PartRecord, PointAnswer,
                                              PointNeed, PointReply, RelationOutcome, RobotModel, StepContext, Until,
                                              WristCamera, CallRecord)
from gpt_field.contact_skills import GripperCalibration, StepView, matrix_to_quaternion_xyzw
from gpt_field.observation import PolicyObservation

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
RELATIONS_PATH = Path(th.PACKAGE_DIR) / "agent_speedup_v2_relations.py"
ROBOT = RobotModel()


# ================================================================================================ 1. hygiene
def test_relations_module_passes_the_t0_lint_audit_and_firewall():
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    assert th.lint_module(RELATIONS_PATH, vocab) == []
    assert th.literal_audit(RELATIONS_PATH) == []
    tree = ast.parse(RELATIONS_PATH.read_text())
    assert th.firewall_violations(tree, "agent_speedup_v2_relations") == []
    assert "agent_speedup_v2_relations.py" in th.CONTROL_MODULES and not th.LABEL_MERGE_ERRORS
    assert "agent_speedup_v2_relations.HOVER_LOCATE_M" in th.CONSTANT_LABELS
    status = th.v6_constant_status()["HOVER_LOCATE_M"]
    assert status["state"] == "labelled" and status["label"] == "DD"
    # every Align is built with allow_pointing=False; nothing here calls the client or asks for pointing
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "Align":
            flags = [kw for kw in node.keywords if kw.arg == "allow_pointing"]
            assert len(flags) == 1 and flags[0].value.value is False
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("request_point", "request_plan", "request_repair")


def test_restated_v1_constants_equal_their_sources():
    from gpt_field import agent_speedup_v1 as v1
    from gpt_field import agent_speedup_v2_client as client
    assert R.LIFT_TEST_M == v1.LIFT_TEST_M
    assert R.GRASP_SETTLE_MAX_POSE_CORRECTION_M == v1.GRASP_SETTLE.max_pose_correction_m
    assert R.GRASP_SETTLE == v1.GRASP_SETTLE and R.OPEN_SETTLE == v1.OPEN_SETTLE
    assert R.OPEN_SETTLE_MIN_ACTIONS == v1.OPEN_SETTLE.min_open_actions
    assert (R.P14_LADDER_SEGMENT_M, R.P14_SETTLE_STEPS, R.P14_FIRST_SETTLE_MAX_STEPS) == (
        v1.LADDER_SEGMENT_M, v1.LADDER_SETTLE_STEPS, v1.FIRST_SETTLE_MAX_STEPS)
    assert R.POST_ORIENT_HOLD_STEPS == v1.POST_ORIENT_HOLD_STEPS
    assert R.HELD_CONTACT_BACKOFF_M == v1.HELD_CONTACT_BACKOFF_M
    assert R.MIN_MOVE_OFFSET_M == v1.MIN_MOVE_OFFSET_M
    assert (R.REACH_LIMIT_MEMORY, R.REACH_LIMIT_RADIUS_M, R.REACH_LIMIT_PUSH_M, R.REACH_LIMIT_MAX_LIFT_M) == (
        v1.REACH_LIMIT_MEMORY, v1.REACH_LIMIT_RADIUS_M, v1.REACH_LIMIT_PUSH_M, v1.REACH_LIMIT_MAX_LIFT_M)
    assert R.OPEN_TOLERANCE_MM == v1.OPEN_TOLERANCE_MM
    assert R.ORIENT_TOLERANCE_RAD == pytest.approx(math.radians(3.0))
    # rulings item 9 / lead ruling 3: coarse first above 64 / sqrt(2 ln 2) px (about 54 px), half the V6 crop
    assert R.FINE_CROP_HALF_PX == client.FINE_CROP_PX / 2.0
    assert R.COARSE_FIRST_SIGMA_PX == pytest.approx(54.35, abs=0.01)


def test_controllers_registry_and_reserved_relations():
    assert set(R.CONTROLLERS) == set(kernel.DO_KINDS)
    ctx = make_ctx()
    for do in ("slide", "place_in", "push_to"):
        step = StepContext(0, do, "A", ("A",), "B" if do != "slide" else None, "any", (), (), 1)
        outcome = drive(R.CONTROLLERS[do](ctx, step), ctx)
        assert (outcome.relation, outcome.status, outcome.reason) == (do, "failed", "relation_not_implemented")
    assert [o.reason for o in ctx.pending_outcomes] == ["relation_not_implemented"] * 3


def test_relation_context_records_once_and_counts_rejections():
    ctx = make_ctx()
    out = R._outcome(ctx, "held", "failed", "empty_close", pids=R._pids("D", ["C"]), evidence={"x": 1})
    assert ctx.pending_outcomes == [out]
    R._outcome(ctx, "held", "failed", "empty_close", pids=R._pids("D", ["C"]), evidence={}, record=False)
    assert len(ctx.pending_outcomes) == 1
    for _ in range(3):
        ctx.note_rejection("point_check")
    assert ctx.consecutive_rejections == 3
    ctx.note_accepted("point_fine")
    assert ctx.consecutive_rejections == 0
    with pytest.raises(TypeError):
        ctx.record({"relation": "held"})


def test_evidence_text_names_parts_by_pid_in_cm_and_degrees_without_vocabulary():
    vocab = set(th.lint_vocabulary(LIBERO)["vocabulary"])
    allow = set(th.LINT_ALLOWLIST)
    ths = [kernel.th_value("M8_tol_h_m", 0.01, "DD", "M8"), kernel.th_value("K5_margin_rad", 0.1, "DD", "K5")]
    cases = [("supported_on", "failed", "support_above_destination", {"obj": "D", "grip": ["C"], "ref": "B"},
              {"text_items": [["bottom above surface", 0.056, "m"]], "plan_text": "moka pot handle on the burner"}),
             ("turned", "failed", "effect_not_observed", {"obj": "A", "grip": ["A"], "ref": None},
              {"text_items": [["part angle", 1.57, "rad"]]}),
             ("held", "succeeded", "hold_pass", {"obj": "D", "grip": ["C"], "ref": None},
              {"text_items": [["finger gap", 14.1, "mm"]]})]
    for relation, status, reason, pids, evidence in cases:
        outcome = RelationOutcome(relation, status, reason, pids, evidence, ths, "", 0, 0)
        text = R.evidence_text(outcome)
        assert len(text) <= kernel.EVIDENCE_TEXT_MAX_CHARS and f"part {pids['obj']}" in text
        assert "moka" not in text and "burner" not in text and "handle" not in text
        hits = set().union(*(th.token_hits(t, vocab, allow) for t in th.tokens(text)))
        assert not hits, (text, hits)
        assert "[" not in text                                     # no signed vectors
    assert "5.6 cm" in R.evidence_text(RelationOutcome(*cases[0][:4], cases[0][4], ths, "", 0, 0))
    assert "90 deg" in R.evidence_text(RelationOutcome(*cases[1][:4], cases[1][4], ths, "", 0, 0))
    assert "1.4 cm" in R.evidence_text(RelationOutcome(*cases[2][:4], cases[2][4], ths, "", 0, 0))
    long = RelationOutcome("held", "failed", "no_feasible_candidate", {"obj": "D", "grip": ["C"], "ref": "B"},
                           {"text_items": [["x" * 90, 1.0, "m"]] * 5}, ths * 3, "", 0, 0)
    assert len(R.evidence_text(long)) == kernel.EVIDENCE_TEXT_MAX_CHARS


# ======================================================================================= 2. scripted driver
class ListSink:
    def __init__(self):
        self.rows = []

    def emit(self, event, /, **payload):
        assert event in kernel.EVENTS, event
        json.dumps(kernel.jsonable(payload), allow_nan=False)
        self.rows.append((event, payload))

    def names(self):
        return [name for name, _ in self.rows]


class StubScene:
    """The SceneModel surface the controllers touch when their sub-steps are scripted."""

    def __init__(self, parts=()):
        self.parts = {p.pid: p for p in parts}
        self.keepouts = {}
        self.moved = []
        self.held = None
        self.hf = None
        self.projector = None
        self.planes = []
        self.baselines = {}
        self.last_registration = {}
        self.added_keepouts = []

    def part_moved(self, pid):
        self.moved.append(pid)

    def registration_thresholds(self, pid):
        return []

    def needs_survey_hover(self, pid):
        return False

    def release_held(self, view=None):
        self.held = None
        return {}

    def add_keepout(self, keepout):
        self.added_keepouts.append(keepout)
        self.keepouts[f"{keepout.kind}:{keepout.pid}"] = keepout

    def hand_swept_mask(self, camera, view, yaw_range_rad, *, dilate_px=None):
        return np.zeros((512, 512), dtype=bool)

    def drop(self, pid):
        self.dropped = getattr(self, "dropped", []) + [pid]
        part = self.parts.get(pid)
        if part is not None:
            self.parts[pid] = PartRecord(part.pid, part.text, part.move_prior, part.roles, status="dropped")
        self.moved.append(pid)

    def predicted_uv(self, pid, view, camera="wrist"):
        return None

    def region_mask(self, pid, camera, view):
        return None

    def part_fill_cells(self, pid):
        return getattr(self, "fills", {}).get(pid)

    def part_ring_cells(self, pid):
        return getattr(self, "rings", {}).get(pid)


def gripper_ths():
    cal = GripperCalibration.default()
    return {"g3": kernel.th_value("G3_touch_threshold_mm", 0.1, "DD", "G3"),
            "g5": kernel.th_value("G5_object_present_gap_mm", 0.5, "DD", "G5"),
            "g6": kernel.th_value("G6_hold_floor_mm", cal.empty_floor_mm + 0.5, "DD", "G6"), "quantities": {}}


def blank_view():
    world = fakes.FakeTwoViewSim(fakes.task2_like_scene(), prompt=fakes.TASK2_PROMPT, images=False)
    return StepView(world.observation())


_VIEW = None


def make_ctx(parts=(), **overrides):
    global _VIEW
    if _VIEW is None:
        _VIEW = blank_view()
    ctx = R.RelationContext(robot=ROBOT, scene=StubScene(parts), monitor=InvariantMonitor(), stays=StaysChecker(),
                            events=ListSink(), overrides=dict(overrides))
    ctx.view = _VIEW
    ctx.cal = SimpleNamespace(gripper=GripperCalibration.default(), sigma_close_mm=0.05, sigma_g_mm_per_step=0.002,
                              free_yaw_rate_rad_per_step=0.0229, camera=SimpleNamespace(f_used=333.6),
                              external=None, locator=None, projector=None, fingertip_band=None,
                              sigma_px={"wrist": 0.4, "external": 0.4}, sigma_static_grey=0.4,
                              start_quaternion=tuple(_VIEW.quaternion_xyzw), mount={})
    ctx.memory["gripper"] = gripper_ths()
    return ctx


def scripted(value=None):
    """A sub-step stand-in: a generator that yields nothing and returns ``value`` (or value(**kwargs))."""
    def step(*args, **kwargs):
        return value(*args, **kwargs) if callable(value) else copy.deepcopy(value)
        yield
    return step


def drive(gen, ctx, *, points=None, factory_reply=None, limit=500, primitive_for=None):
    """A copy of the §C3.4 loop without a simulator: a factory gets a scripted (outcome, None) reply (the factory
    is NOT constructed, so no primitive runs; ``primitive_for(factory)`` may supply the reply's primitive stand-in),
    a PointNeed gets ``points(need)``. Returns the generator's value; the yielded items are kept in
    ``ctx.memory['yielded']``."""
    yielded = ctx.memory.setdefault("yielded", [])
    try:
        item = next(gen)
        for _ in range(limit):
            yielded.append(item)
            if isinstance(item, PointNeed):
                reply = points(item)
            elif callable(item):
                primitive = primitive_for(item) if callable(primitive_for) else None
                if callable(factory_reply):
                    reply = (factory_reply(item), primitive)
                else:
                    reply = (factory_reply or {"status": "succeeded", "reason": "scripted", "evidence": {},
                                               "actions": 0}, primitive)
            else:
                raise AssertionError(f"unexpected yield {item!r}")
            item = gen.send(reply)
        raise AssertionError("the controller did not finish")
    except StopIteration as stop:
        return stop.value


def answer(need, uv, quantization=8.0):
    stage = "fine" if need.kind == "point_fine" else "coarse"
    reply = PointReply(stage, need.camera, None if uv is None else 1, uv, quantization, None, None, None,
                       need.quarter_turns)
    return PointAnswer(need, reply, False, CallRecord(need.kind, 0.0, None, None, True, None))


def part(pid="A", *, move="hinged", roles=("grip",), registered=True, elongated=True, band_n=40, axis=None,
         collapsed=False, top=0.955):
    return PartRecord(pid, f"part {pid}", move, frozenset(roles), xyz=np.array([-0.2, 0.2, top]),
                      cov=np.eye(3) * 1e-6, top_z=Meas(top, 0.003, "kappa"), support_z=Meas(0.918, 0.002, "kappa"),
                      region_xy=np.array([[-0.24, 0.19], [-0.16, 0.19], [-0.16, 0.21], [-0.24, 0.21]]),
                      width_by_yaw=((0.0, Meas(0.0243, 0.0015, "kappa")),),
                      elongation=(np.array([1.0, 0.0]), 0.6) if elongated else None,
                      registered_empty_hand=True, registration_step=1,
                      status="registered" if registered else "seeded", sigma_h_m=0.0031,
                      top_band={"n": band_n, "stat_m": 0.0005, "model_m": 0.003},
                      axis_check=axis if axis is not None else {
                          "image_axis": [1.0, 0.0], "region_axis": [1.0, 0.0], "agree_rad": 0.0, "accepted": True,
                          "source": "d10", "region_collapsed": collapsed})


def contact_plan(pid="A", *, yaw=-math.pi / 2, rule="height_stop", marginal=False, move="hinged"):
    jaw = (math.cos(yaw), math.sin(yaw), 0.0)
    expected = (planner.ExpectedContact("none_before_target", 0.944, 0.014, None, close_limit_grip_z=0.956),)
    return planner.ContactPlan(
        pid=pid, family="top_down", depth_rule=rule, purpose="turn", jaw_world=jaw, approach_world=(0.0, 0.0, -1.0),
        quaternion_xyzw=tuple(float(v) for v in ROBOT.quaternion_for(np.array(jaw), np.array([0.0, 0.0, -1.0]))),
        yaw_turn_rad=0.0, grip_point_xyz=(-0.2, 0.2, 0.955), hover_grip_z=1.02, target_grip_z=0.944,
        bound_grip_z=0.914, expected=expected, clearance_retreat_m=0.0, columns=(),
        width={"w_mm": 24.3, "sigma_w_mm": 1.5, "sep_mm": 77.5, "e_lat_mm": 8.5,
               "class": "marginal" if marginal else "feasible"}, checks={}, align_mode="across_bar", score=1.0,
        thresholds=(kernel.th_value("P6_target_grip_z_m", 0.944, "ONLINE", "P6"),), move_prior=move)


def infeasible(pid, reason="column_blocked", refine_axis=False):
    return planner.Infeasible(pid, reason, (), (kernel.th_value("P3_pad_overlap_mm", 8.0, "ROBOT", "P3"),),
                              refine_axis=refine_axis)


class GraspScript:
    """Scripted grasp sub-steps; records what ran."""

    def __init__(self, monkeypatch, *, plans, classes, closes=(), loads=(), hover_reason=None, record=None):
        self.log = []
        self.plans, self.classes = list(plans), list(classes)
        self.closes, self.loads = list(closes), list(loads)
        self.excludes, self.keepouts, self.close_kwargs = [], [], []
        rec = record or part()

        def grip_hover(ctx, pid, **kw):
            self.log.append(("hover", pid))
            return {"record": None if hover_reason else rec, "reason": hover_reason, "calls": 1}

        def plan_(ctx, record, *, purpose, obj, exclude=(), sigma_align_m=None, extra_keepouts=()):
            self.excludes.append(list(exclude))
            self.keepouts.append(list(extra_keepouts))
            self.log.append(("plan", sigma_align_m is not None))
            return self.plans.pop(0) if self.plans else infeasible(record.pid, "no_feasible_candidate")

        def classify(plan, z, event):
            kind = self.classes.pop(0)
            return {"class": kind, "matched": 0 if kind == "expected" else None, "event": event, "thresholds": [],
                    "contact_grip_z": z, "landing_measured": False, "surface_z": None}

        def close(ctx, plan, record, **kw):
            self.log.append(("close", plan.yaw_rad))
            self.close_kwargs.append({k: list(v) if isinstance(v, list) else v for k, v in kw.items()})
            spec = self.closes.pop(0) if self.closes else (True, False)
            if isinstance(spec, dict):                                   # a scripted PC-1 / GC-1 abort
                return {"present": False, "gap_mm": float(ctx.view.gap_mm), "close": {"reason": spec["aborted"]},
                        "width": None, "drag_m": 0.0, "plan": plan, **copy.deepcopy(spec)}
            present, misaligned = spec
            return {"present": present, "gap_mm": 25.4 if present else 1.0, "close": {"object_present": present},
                    "width": {"consistent": True, "misaligned": misaligned, "thresholds": []}, "drag_m": 0.001}

        def load(ctx, *, grip, obj, attempt):
            self.log.append(("load", grip))
            spec = self.loads.pop(0) if self.loads else {"ok": True, "reason": "hold_pass"}
            return {"held": None, "evidence": {}, **spec}

        def logged(name, value=None):
            def step(*args, **kwargs):
                self.log.append((name,))
                return copy.deepcopy(value)
                yield
            return step

        monkeypatch.setattr(R, "_grip_hover", scripted(grip_hover))
        monkeypatch.setattr(R, "_plan", plan_)
        monkeypatch.setattr(R.planner, "classify_contact", classify)
        monkeypatch.setattr(R, "_ensure_open", scripted(None))
        monkeypatch.setattr(R, "_orient", logged("orient", {"ok": True, "skipped": True}))
        monkeypatch.setattr(R, "_align", logged("align", {"ok": True, "reason": None, "sigma_align_m": None,
                                                         "track": None, "jacobian": None, "offset": [0.0, -0.05]}))
        monkeypatch.setattr(R, "_descend", logged("descend", {"event": "target_reached", "contact_grip_z": 0.944}))
        monkeypatch.setattr(R, "close_and_classify", scripted(close))
        monkeypatch.setattr(R, "_load_test", scripted(load))
        monkeypatch.setattr(R, "_return_to_hover", logged("return_to_hover", {"reason": "travel_reached"}))
        monkeypatch.setattr(R, "_release_grip", logged("release", {"reason": "released"}))
        monkeypatch.setattr(R, "_lift", logged("lift", {"reason": "travel_reached"}))
        monkeypatch.setattr(R, "_support_pick", lambda ctx, obj, grip: Meas(0.9, 0.001, "plane"))
        original_put_back = R._put_back

        def put_back(ctx, lift_start_z):
            self.log.append(("put_back",))
            return (yield from original_put_back(ctx, lift_start_z))
        monkeypatch.setattr(R, "_put_back", put_back)

    def names(self):
        return [row[0] for row in self.log]


def test_grasp_never_closes_after_blocked_above_part_and_takes_the_next_candidate(monkeypatch):
    first, second = contact_plan(yaw=-math.pi / 2), contact_plan(yaw=-math.pi / 2 + kernel.YAW_GRID_RAD)
    script = GraspScript(monkeypatch, plans=[first, second], classes=["blocked_above_part", "expected"])
    ctx = make_ctx()
    outcome = drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.relation, outcome.status, outcome.reason) == ("held", "succeeded", "held_for_turn")
    names = script.names()
    assert names.count("close") == 1 and names.index("return_to_hover") < names.index("close")
    assert script.excludes[1] == [first.candidate_key()]           # the refused candidate is excluded
    assert outcome.local_retries == 1                              # F6-A (ruling L9): a descent touch is a retry
    assert ctx.monitor.keys("held") == ["hold_floor"]               # turn: G6 on the bus, no co-motion
    assert ctx.pending_outcomes == [outcome] and ctx.holding is None
    assert "contact_plan" in ctx.events.names() and "contact" in ctx.events.names()


def test_grasp_empty_close_retries_toward_the_centre_line_then_fails(monkeypatch):
    plan = contact_plan()
    script = GraspScript(monkeypatch, plans=[plan], classes=["expected"] * 4,
                         closes=[(False, False)] * (R.B2_LOCAL_RETRIES + 1))
    ctx = make_ctx()
    outcome = drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "empty_close")
    assert script.names().count("close") == R.B2_LOCAL_RETRIES + 1
    assert outcome.local_retries == R.B2_LOCAL_RETRIES + 1
    shifted = {tuple(a["plan"]["grip_point_xyz"]) for a in outcome.evidence["attempts"]}
    assert len(shifted) > 1                                         # the grip point moved along the jaw


def test_grasp_g13_misaligned_excludes_the_candidate_and_its_twin(monkeypatch):
    first, second = contact_plan(move="free"), contact_plan(yaw=0.3, move="free")
    script = GraspScript(monkeypatch, plans=[first, second], classes=["expected", "expected"],
                         closes=[(True, True), (True, False)])
    ctx = make_ctx()
    outcome = drive(R.grasp(ctx, "C", obj_pid="D", purpose="carry"), ctx)
    assert outcome.status == "succeeded" and outcome.reason == "hold_pass"
    excluded = script.excludes[1]
    assert first.candidate_key() in excluded and R._twin(first) in excluded
    assert ctx.monitor.keys("held")                                  # carry bus registered


def test_grasp_infeasible_and_unreachable_parts_send_the_caller_to_the_next_grip_part(monkeypatch):
    GraspScript(monkeypatch, plans=[infeasible("C", "column_blocked")], classes=[])
    ctx = make_ctx()
    step = StepContext(1, "place_on", "D", ("C", "D"), "B", "any", (), (), 1)
    calls = []
    original = R.grasp

    def counting(ctx, grip_pid, **kw):
        calls.append(grip_pid)
        return (yield from original(ctx, grip_pid, **kw))
    monkeypatch.setattr(R, "grasp", counting)
    outcome = drive(R._grasp_with_fallback(ctx, step, purpose="carry"), ctx)
    assert calls == ["C", "D"] and outcome.evidence["next_grip"] is True
    assert [o.reason for o in ctx.pending_outcomes] == ["column_blocked", "no_feasible_candidate"]


def test_grasp_grip_not_on_obj_releases_repoints_once_then_moves_on(monkeypatch):
    plan = contact_plan(move="free")
    GraspScript(monkeypatch, plans=[plan, plan, plan], classes=["expected"] * 3,
                loads=[{"ok": False, "reason": "grip_not_on_obj"}] * 2)
    rounds = []
    monkeypatch.setattr(R, "grasp_point_round", scripted(lambda ctx, pid, **kw: (
        rounds.append(kw.get("previous")), {"record": part("C", move="free"), "reason": None})[1]))
    ctx = make_ctx()
    outcome = drive(R.grasp(ctx, "C", obj_pid="D", purpose="carry"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "grip_not_on_obj") and outcome.evidence["next_grip"]
    assert len(rounds) == 1 and rounds[0]                            # ONE local re-point with the measured reason


def test_grasp_precondition_unmet_when_holding_another_part():
    ctx = make_ctx()
    ctx.holding = SimpleNamespace(pid="D", grip_pid="C")
    outcome = drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "precondition_unmet")
    assert ctx.memory["yielded"] == []


# ----------------------------------------------------------------------------------------------- pointing
def test_point_part_fine_first_then_coarse_after_a_zero_answer():
    ctx = make_ctx()
    replies = iter([None, (300.0, 200.0), (310.0, 204.0)])
    asked = []

    def points(need):
        asked.append((need.kind, need.center_uv))
        return answer(need, next(replies), 8.0 if need.kind == "point_fine" else 32.0)
    out = drive(R.point_part(ctx, "A", camera="wrist", purpose="grasp_point", center_uv=(250.0, 250.0),
                             sigma_uv_px=10.0, previous_rejected=None), ctx, points=points)
    assert [k for k, _ in asked] == ["point_fine", "point_coarse", "point_fine"]
    assert asked[2][1] == (300.0, 200.0)                             # the second fine crop is centred on the cell
    assert out["seed"].source == "fine" and out["seed"].uv == (310.0, 204.0) and out["calls"] == 3


def test_point_part_coarse_first_when_the_prediction_sigma_exceeds_the_crop_reach():
    ctx = make_ctx()
    asked = []

    def points(need):
        asked.append(need.kind)
        return answer(need, (200.0, 200.0))
    drive(R.point_part(ctx, "C", camera="wrist", purpose="grasp_point", center_uv=(250.0, 250.0),
                       sigma_uv_px=R.COARSE_FIRST_SIGMA_PX + 1.0, previous_rejected=None), ctx, points=points)
    assert asked[0] == "point_coarse"
    asked.clear()
    drive(R.point_part(ctx, "C", camera="wrist", purpose="grasp_point", center_uv=(250.0, 250.0),
                       sigma_uv_px=R.COARSE_FIRST_SIGMA_PX - 1.0, previous_rejected=None), ctx, points=points)
    assert asked[0] == "point_fine"


def test_point_part_zero_zero_is_part_not_found():
    ctx = make_ctx()
    out = drive(R.point_part(ctx, "A", camera="wrist", purpose="grasp_point", center_uv=(250.0, 250.0),
                             sigma_uv_px=5.0, previous_rejected=None), ctx, points=lambda need: answer(need, None))
    assert out["seed"] is None and out["reason"] == "part_not_found" and out["calls"] == 2


def test_grasp_point_round_checks_after_the_probe_and_reasks_once_with_the_measured_reason(monkeypatch):
    ctx = make_ctx([part("A", registered=False)])
    order = []
    monkeypatch.setattr(R, "_probe_and_register", scripted(lambda ctx, pid, seed, label: (
        order.append(("probe", label)), {"record": part("A"), "failure": None, "label": label, "probe": {}})[1]))
    monkeypatch.setattr(R, "_pointed_visibility", scripted(lambda ctx, pid, seed, **kw: (seed, None)))
    checks = iter([{"ok": False, "reason": "pointed_not_grasp_point", "values": {"above_m": [0.004, 0.03],
                                                                                "margin_m": 0.009},
                    "thresholds": []}, {"ok": True, "reason": None, "values": {}, "thresholds": []}])
    monkeypatch.setattr(R.planner, "check_point_answer", lambda *a, **k: (order.append(("check",)), next(checks))[1])
    monkeypatch.setattr(R, "_inputs", lambda ctx, **kw: None)
    ctx.scene.predicted_uv = lambda pid, view, camera="wrist": (np.array([250.0, 250.0]), 5.0)
    needs = []

    def points(need):
        needs.append(need)
        return answer(need, (252.0, 249.0))
    out = drive(R.grasp_point_round(ctx, "A", purpose="turn"), ctx, points=points)
    assert out["record"] is not None and out["reason"] is None
    assert order == [("probe", "hover_round_0"), ("check",), ("probe", "hover_round_1"), ("check",)]
    assert needs[0].previous_rejected is None and "0.4 cm" in needs[1].previous_rejected
    assert len(needs[1].previous_rejected) <= kernel.PREVIOUS_REJECTED_MAX_CHARS
    assert ctx.consecutive_rejections == 0                           # the accepted answer reset the storm count


def test_seed_outside_region_gets_one_repoint_at_the_hover(monkeypatch):
    ctx = make_ctx([part("C", move="free", registered=False)])
    failures = iter([{"reason": "part_not_found", "why": R.SEED_REFUSAL}, {"reason": "part_not_found",
                                                                          "why": R.SEED_REFUSAL}])
    monkeypatch.setattr(R, "_probe_and_register", scripted(lambda ctx, pid, seed, label: {
        "record": None, "failure": next(failures), "label": label, "probe": {}}))
    monkeypatch.setattr(R, "_pointed_visibility", scripted(lambda ctx, pid, seed, **kw: (seed, None)))
    ctx.scene.predicted_uv = lambda pid, view, camera="wrist": (np.array([250.0, 250.0]), 80.0)
    needs = []
    out = drive(R.grasp_point_round(ctx, "C", purpose="carry"), ctx,
                points=lambda need: (needs.append(need), answer(need, (240.0, 240.0)))[1])
    assert out["record"] is None and out["reason"] == "part_not_found"
    # sigma 80 px > COARSE_FIRST_SIGMA_PX: coarse then fine, twice (the first pointing and the ONE re-point)
    assert [n.kind for n in needs] == ["point_coarse", "point_fine"] * 2
    assert "outside" in needs[2].previous_rejected


def probe_outcome(reason):
    leg = {"stage": "r0_back", "reason": "contact_stall" if reason == "probe_contact" else "travel_reached"}
    return {"status": "succeeded" if reason == "probe_complete" else "failed", "reason": reason,
            "evidence": {"legs": [leg], "net_displacement_m": [0.0, 0.0, 0.01]}, "actions": 86}


@pytest.mark.parametrize("reasons, attempts, final", [
    (["probe_contact", "probe_complete"], 2, "probe_complete"),
    (["probe_contact", "probe_contact"], 2, "probe_contact"),
    (["probe_complete"], 1, "probe_complete"),
])
def test_a_probe_that_ends_early_is_repeated_once_from_its_start_pose(reasons, attempts, final):
    ctx = make_ctx()
    replies = iter(probe_outcome(r) for r in reasons)
    result, _, summaries = drive(R._run_probe(ctx, lambda v: None), ctx, factory_reply=lambda item: next(replies))
    assert result["reason"] == final and len(summaries) == attempts
    assert summaries[0]["legs"][0]["stage"] == "r0_back"
    if attempts > 1:
        assert summaries[0]["return"] == "goal_reached"   # the stub view never moved: no return reach needed


FRONTAL_EXTRA_M = 0.30          # a frontal wrist sits 0.204 m behind the grip: move the hover out beyond any reach


def stretch_hover(x, z):
    return {"hover_grip_xyz": [x, -0.0723, z], "clearance_m": 0.1, "first_stage": True,
            "first_stage_reasons": ["p11_cap", "lower_bound"], "reason": None}


@pytest.mark.parametrize("x, z, expected", [
    (0.0117, 1.2823, "kept"),        # the survey hover that probed cleanly (wrist-shoulder 0.73 m)
    (0.0297, 1.3647, "lowered"),     # near the first-stage hover whose probe stalled at 0.786 m
    (0.12, 1.3338, "refused"),       # no height keeps the wrist within the limit
])
def test_the_stretch_guard_lowers_or_refuses_hovers_beyond_the_mount_limit(x, z, expected):
    ctx = make_ctx()
    ctx.cal.mount = {"shoulder_xyz": (-0.66, 0.0, 1.245), "frontal_stretch_limit_m": 0.77}
    record = part("C", move="free", registered=False, top=1.0)
    out = R._stretch_cap(ctx, record, stretch_hover(x, z))
    info = out["stretch"]
    if expected == "kept":
        assert out["hover_grip_xyz"][2] == z and info["lowered_m"] == 0.0
    elif expected == "lowered":
        new_z = out["hover_grip_xyz"][2]
        assert info["floor_z"] <= new_z < z and info["lowered_m"] > 0.01
        approach = np.asarray(ctx.view.R)[:, 2]
        for probe_x in (x, x + primitives.D2_PROBE_M):
            grip = np.array([probe_x, -0.0723, new_z])
            assert robot_facts.wrist_to_shoulder_m(grip, approach, (-0.66, 0.0, 1.245)) <= 0.77 + 1e-6
        assert robot_facts.wrist_to_shoulder_m(grip - [primitives.D2_PROBE_M, 0, -R.HOVER_LOCATE_M], approach,
                                               (-0.66, 0.0, 1.245)) <= 0.77 + 1e-6
        assert out["first_stage"]
    else:
        # BRR-11 kept as a disclosed deviation (top-down hands too; a lowering-only variant never reached its hover on
        # the real sim): refused, and a frontal hand as well
        assert out["hover_grip_xyz"] is None and out["reason"] == "reach_limit" and not info["frontal"]
        frontal_ctx = make_ctx()
        frontal_ctx.cal.mount = dict(ctx.cal.mount)
        # a frontal hand (approach R2 = +x, horizontal) far out: refused as before
        frontal_ctx.view = SimpleNamespace(R=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]))
        refused = R._stretch_cap(frontal_ctx, record, stretch_hover(x + FRONTAL_EXTRA_M, z))
        assert refused["stretch"]["frontal"] and refused["hover_grip_xyz"] is None
        assert refused["reason"] == "reach_limit"
    # no mount: unchanged
    ctx.cal.mount = {}
    assert R._stretch_cap(ctx, record, stretch_hover(x, z))["hover_grip_xyz"][2] == z


def test_the_grasp_point_check_uses_the_planned_jaw_of_a_part_without_a_long_axis(monkeypatch):
    ctx = make_ctx()
    seen = {}

    def fake_plan(ctx, record, *, purpose, obj, exclude=(), sigma_align_m=None):
        seen["args"] = (record.pid, purpose, obj)
        return contact_plan(record.pid, yaw=0.3)
    monkeypatch.setattr(R, "_plan", fake_plan)
    yaw, source = R._check_jaw(ctx, part("C", move="free", elongated=False), kind="grasp_point", purpose="carry",
                               obj="D")
    assert source == "planned" and abs(yaw - 0.3) < 1e-9 and seen["args"] == ("C", "carry", "D")
    # an elongated record keeps the planner's own rule (across its long axis); other kinds are not planned
    assert R._check_jaw(ctx, part("C"), kind="grasp_point", purpose="carry", obj="D") == (None, None)
    assert R._check_jaw(ctx, part("C", elongated=False), kind="surface_point", purpose="carry", obj="D") == (None,
                                                                                                           None)
    monkeypatch.setattr(R, "_plan", lambda ctx, record, **kw: infeasible(record.pid))
    assert R._check_jaw(ctx, part("C", elongated=False), kind="grasp_point", purpose="carry", obj="D") == (
        None, "column_blocked")


# --------------------------------------------------------------------------------------------- BR-1 matrix
COLLAPSED = {"image_axis": [1.0, 0.0], "region_axis": None, "agree_rad": None, "accepted": False,
             "source": kernel.NOT_ELONGATED, "region_collapsed": True}

@pytest.mark.parametrize("record, grip, expected", [
    (part(band_n=5), True, ["band_support"]),
    (part(band_n=5), False, ["band_support"]),
    (part(axis={"image_axis": None, "region_axis": None, "agree_rad": 1.4, "accepted": False,
                "source": kernel.AXIS_INCONSISTENT}, elongated=False), True, ["axis_inconsistent"]),
    (part(axis={"image_axis": None, "region_axis": None, "agree_rad": 1.4, "accepted": False,
                "source": kernel.AXIS_INCONSISTENT}, elongated=False), False, []),
    (part(axis={"image_axis": None, "region_axis": None, "agree_rad": 1.4, "accepted": False,
                "source": kernel.NOT_ELONGATED}, elongated=False), True, []),
    (part(axis=COLLAPSED, elongated=False), True, ["region_collapsed"]),
    (part(axis=COLLAPSED, elongated=False), False, []),
    (part(), True, []),
])
def test_br1_trigger_matrix(record, grip, expected):
    assert R._br1_triggers(make_ctx(), record, record.pid, grip=grip) == expected


def test_br1_refine_axis_trigger_comes_from_the_planner():
    ctx = make_ctx()
    assert R._br1_triggers(ctx, part(), "A", grip=True, plan_result=infeasible("A", "geometry_unmeasured",
                                                                               refine_axis=True)) == ["refine_axis"]


# ===================================================================================================== turn
class TurnScript:
    def __init__(self, monkeypatch, *, segments, hinges, grasp_status=("succeeded", "held_for_turn"), post=None,
                 effect_region=None):
        self.log = []
        self.segments, self.hinges = list(segments), list(hinges)

        def grasp_fallback(ctx, step, *, purpose):
            self.log.append(("grasp",))
            return R._outcome(ctx, "held", grasp_status[0], grasp_status[1], pids=R._pids(step.obj, list(step.grip)),
                              evidence={"next_grip": False}, start=ctx.step)

        def twist(ctx, *, sense, part, effect, exclude, after_reversal=False, blocked_steps=None, label):
            reason = self.segments.pop(0)
            self.log.append(("segment", sense, after_reversal, blocked_steps, reason))
            return {"result": {"reason": reason, "status": "succeeded" if reason == "free_sense" else "failed",
                               "evidence": {}, "actions": 12}, "reason": reason, "sense": sense, "validity": None,
                    "steps": 12, "segment": label}

        def hinge(ctx, *, sense, part, effect, theta_target, theta_start):
            reason, theta = self.hinges.pop(0)
            self.log.append(("hinge", sense, theta_start, reason))
            return ({"reason": reason, "status": "succeeded", "evidence": {"theta_part_rad": theta}, "actions": 20},
                    None)

        def logged(name, value=None):
            def step(*args, **kwargs):
                self.log.append((name,))
                return copy.deepcopy(value)
                yield
            return step
        monkeypatch.setattr(R, "_grasp_with_fallback", scripted(grasp_fallback))
        monkeypatch.setattr(R, "grasp", scripted(lambda ctx, grip, **kw: grasp_fallback(ctx, SimpleNamespace(
            obj=kw["obj_pid"], grip=[grip]), purpose="turn")))
        monkeypatch.setattr(R, "_twist_segment", scripted(twist))
        monkeypatch.setattr(R, "_hinge", scripted(hinge))
        monkeypatch.setattr(R, "_turn_release", logged("release", {"reason": "released", "evidence": {}}))
        monkeypatch.setattr(R, "_turn_post_check", logged("post_check", post or {"spring_back": "met",
                                                                                 "persists": "met"}))
        monkeypatch.setattr(R, "_turn_stays", lambda ctx, step, **kw: self.log.append(("stays", kw["effect"] is not None)))
        monkeypatch.setattr(R, "_articulated_keepout", lambda ctx, pid, **kw: self.log.append(("keepout", pid)))
        monkeypatch.setattr(R, "_external_region", lambda ctx, pid: effect_region if pid == "B" else None)
        monkeypatch.setattr(R, "_sense_validity", lambda ctx, effect, **kw: {"valid": True, "remaining_fraction": 0.99})

    def names(self):
        return [row[0] for row in self.log]


def turn_step(*until):
    return StepContext(0, "turn", "A", ("A",), None, "any", (), tuple(until), 1)


def effect_mask():
    mask = np.zeros((512, 512), dtype=bool)
    mask[300:340, 360:420] = True
    return mask


def test_turn_half_turn_fails_yaw_budget_before_any_motion():
    ctx = make_ctx()
    outcome = drive(R.turn(ctx, turn_step(Until("half_turn", None, None))), ctx)
    assert (outcome.relation, outcome.status, outcome.reason) == ("turned", "failed", "yaw_budget")
    assert ctx.memory["yielded"] == [] and outcome.thresholds[0].name == "K6_yaw_budget_rad"


def test_turn_default_sense_blocked_then_reversed_free_then_the_effect_margin(monkeypatch):
    script = TurnScript(monkeypatch, segments=["blocked_sense", "free_sense"],
                        hinges=[("effect_margin_reached", 0.38)], effect_region=effect_mask())
    ctx = make_ctx()
    outcome = drive(R.turn(ctx, turn_step(Until("effect", "B", None), Until("stop", None, None))), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_by_effect")
    segments = [row for row in script.log if row[0] == "segment"]
    assert segments[0][1:] == (1, False, None, "blocked_sense")       # +1 about the approach first [C-25]
    assert segments[1][1:] == (-1, True, 12, "free_sense")            # reversal with the blocked steps [C-21]
    assert ("hinge", -1, 0.0, "effect_margin_reached") in script.log
    assert script.names()[-4:] == ["release", "post_check", "stays", "keepout"]
    assert ("stays", True) in script.log and outcome.evidence["free_sense"] == -1
    assert [o.relation for o in ctx.pending_outcomes] == ["held", "turned"]


def test_turn_first_sense_override_is_free_at_once(monkeypatch):
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[("end_stop", 1.2)])
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("stop", None, None))), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_to_stop")
    assert [row[1] for row in script.log if row[0] == "segment"] == [-1]


def test_turn_both_senses_blocked_is_not_rotatable_and_releases(monkeypatch):
    script = TurnScript(monkeypatch, segments=["blocked_sense", "blocked_sense"], hinges=[])
    ctx = make_ctx()
    outcome = drive(R.turn(ctx, turn_step(Until("stop", None, None))), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "not_rotatable")
    assert "release" in script.names() and "hinge" not in script.names()


UNCHANGED_AFTER_RELEASE = {"spring_back": "met", "persists": "violated"}


def test_turn_yaw_budget_with_an_effect_criterion_is_effect_not_observed(monkeypatch):
    """The yaw budget ends the twist without an onset and the post-release confirmation finds the region unchanged
    (blocker-1 spec BR-2): effect_not_observed, flag effect_region_unchanged."""
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[("yaw_budget", 1.7)],
                        effect_region=effect_mask(), post=UNCHANGED_AFTER_RELEASE)
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("effect", "B", None))), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "effect_not_observed")
    assert outcome.evidence["theta_part_rad"] == pytest.approx(1.7) and "deg" in outcome.evidence_text
    assert outcome.evidence["effect_region_unchanged"] is True and "effect region unchanged" in outcome.evidence_text
    assert script.names()[-3:] == ["release", "post_check", "keepout"] and ("stays", True) not in script.log


def test_turn_stop_without_a_matching_criterion_is_not_claimed(monkeypatch):
    TurnScript(monkeypatch, segments=["free_sense"], hinges=[("end_stop", 1.0)], effect_region=effect_mask(),
               post=UNCHANGED_AFTER_RELEASE)
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("effect", "B", None))), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "effect_not_observed")
    assert outcome.evidence["stop"] == "end_stop"


def test_turn_slip_regrasps_and_continues_from_the_measured_angle(monkeypatch):
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[("turn_slip", 0.4), ("quarter_turn_reached",
                                                                                           1.6)])
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("quarter_turn", None, None))), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_quarter") and outcome.local_retries == 1
    hinges = [row for row in script.log if row[0] == "hinge"]
    assert hinges[1][2] == pytest.approx(0.4)                         # continued from the measured angle
    assert script.names().count("grasp") == 2


def test_turn_axis_tug_recentres_once_then_fails(monkeypatch):
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[("axis_tug", 0.3), ("axis_tug", 0.5)])
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("stop", None, None))), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "axis_tug")
    assert script.names().count("grasp") == 2


def test_turn_spring_back_after_release_fails_the_turn(monkeypatch):
    TurnScript(monkeypatch, segments=["free_sense"], hinges=[("end_stop", 1.0)], post={"spring_back": "violated"})
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("stop", None, None))), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "spring_back")


def test_a_turn_whose_grasp_fails_reports_the_grasp_cause_not_its_own_budget(monkeypatch):
    """Integration (live smokes 001/002): a turn that fails in its grasp carried only its K6 budget pre-check, so the
    model read 'column_blocked; K6_yaw_budget_rad 113 deg' as a motion-limit failure. The turn now carries the
    grasp's thresholds and text flags, plus the flag 'grasp failed'."""
    TurnScript(monkeypatch, segments=[], hinges=[])
    p5 = kernel.th_value("P5_sweep_margin_m", 0.009, "ONLINE", "P5_SIGMA_K*sigma_h")

    def failed_grasp(ctx, step, *, purpose):
        return R._outcome(ctx, "held", "failed", "column_blocked", pids=R._pids(step.obj, list(step.grip)),
                          evidence={"next_grip": False, "text_flags": ["aligned candidate refused column_blocked"]},
                          thresholds=[p5], start=ctx.step)
    monkeypatch.setattr(R, "_grasp_with_fallback", scripted(failed_grasp))
    ctx = make_ctx()
    outcome = drive(R.turn(ctx, turn_step(Until("effect", "B", None))), ctx)
    assert (outcome.relation, outcome.status, outcome.reason) == ("turned", "failed", "column_blocked")
    assert [t.name for t in outcome.thresholds] == ["P5_sweep_margin_m"]
    assert "grasp failed" in outcome.evidence_text and "K6" not in outcome.evidence_text
    assert "aligned candidate refused column blocked" in outcome.evidence_text
    assert outcome.evidence["failed_in"] == "grasp"


# test_grasp_names_the_refusal_after_align_and_puts_the_decisive_check_first: replaced by the CI-2 refusal test
# (test_agent_speedup_v2_cv5.py::test_ci2_a_refused_revalidation_names_its_own_reason_and_returns_to_the_grasp_hover)

def test_an_aborted_ladder_descent_reports_its_stop_height_so_a_height_stop_never_closes_above_the_band():
    """BI review F1: a ladder segment that fails (an AnchoredMove 'budget' while blocked on top) returned contact z
    None, which classify_contact reads as 'reached g*' for a height_stop plan (class 'expected': close). The abort
    now reports the hand's stop height; above the band that is 'stopped_early' (no close)."""
    ctx = make_ctx()
    plan = contact_plan(marginal=True)
    assert float(ctx.view.position[2]) > plan.expected[0].upper_z        # the hand stops above the band
    replies = iter([{"status": "succeeded", "reason": "quiet", "evidence": {}, "actions": 3},
                    {"status": "failed", "reason": "budget", "evidence": {}, "actions": 40}])
    result = drive(R.ladder_descent(ctx, {"track": None, "jacobian": None, "offset": None}, plan,
                                    quaternion=plan.quaternion_xyzw), ctx, factory_reply=lambda item: next(replies))
    assert result["event"] == "budget" and result["contact_grip_z"] == pytest.approx(float(ctx.view.position[2]))
    klass = planner.classify_contact(plan, result["contact_grip_z"], result["event"])
    assert klass["class"] == "stopped_early"
    assert planner.classify_contact(plan, None, "budget")["class"] == "expected"     # what the None used to give


def test_an_infeasible_grasp_names_the_failing_check_before_the_width_checks(monkeypatch):
    """Integration (live smoke 002): 'held failed for part A: column_blocked; P1_e_lat_mm 9.16, P2_width..' named the
    width checks the candidate passed; the failing rule's own threshold (P5 for height_stop:P5) now comes first."""
    p1 = kernel.th_value("P1_e_lat_mm", 9.1, "ONLINE", "P1")
    p2 = kernel.th_value("P2_width_feasible_mm", 59.2, "ONLINE", "P2")
    p5 = kernel.th_value("P5_sweep_margin_m", 0.009, "ONLINE", "P5_SIGMA_K*sigma_h")
    blocked = planner.CandidateRejection("top_down", -math.pi / 2, "height_stop:P5", "column_blocked", {},
                                         (p1, p2, p5))
    GraspScript(monkeypatch, plans=[planner.Infeasible("A", "column_blocked", (blocked,), (p1, p2, p5))], classes=[])
    ctx = make_ctx()
    outcome = drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "column_blocked")
    assert [t.name for t in outcome.thresholds][:3] == ["P5_sweep_margin_m", "P1_e_lat_mm", "P2_width_feasible_mm"]
    assert "P5_sweep_margin_m" in outcome.evidence_text


# ================================================================================================= place_on
def held_frame():
    return kernel.HeldFrame(pid="D", grip_pid="C", grasp_pose=tuple(_VIEW.state if _VIEW else [0.0] * 8),
                            offset_h=np.array([0.08, 0.0, 0.0]), vacated_cells=None, sigma_offset_m=0.003,
                            b_low=Meas(0.12, 0.002, "kappa"), r_obj_m=0.033, extent_axes=None, tilt_deg=None,
                            wrist_visible=True, silhouette=None, ext_template=None, gap_ref_mm=14.0,
                            sigma_carry_mm=None, support_z_pick=Meas(0.9, 0.001, "plane"), g_close=1.02,
                            touchdown=None)


class PlaceScript:
    def __init__(self, monkeypatch, *, supports, lowers=None):
        self.log = []
        self.supports = list(supports)
        self.lowers = list(lowers or [])

        def logged(name, value=None):
            def step(*args, **kwargs):
                self.log.append((name,))
                return copy.deepcopy(value)
                yield
            return step

        def lower(ctx, lowering, aim):
            self.log.append(("lower",))
            return self.lowers.pop(0) if self.lowers else {"reason": "support_event", "event": "contact_stall",
                                                           "g_event": 1.051, "lateral_error_m": 0.002}

        def classify(lowering, g_event, held, lateral, *, event):
            return {"class": self.supports.pop(0), "S_dest": 0.9305, "bottom_z": 0.95, "dz_m": 0.02,
                    "thresholds": []}
        monkeypatch.setattr(R, "held_move", logged("carry", {"ok": True, "reason": "goal_reached"}))
        monkeypatch.setattr(R, "_aim_check", logged("aimcheck", {"external": "agrees", "wrist":
                                                                 "aimcheck_unavailable"}))
        monkeypatch.setattr(R, "_lower", scripted(lower))
        monkeypatch.setattr(R, "_lift", logged("lift", {"reason": "travel_reached"}))
        monkeypatch.setattr(R, "_release_grip", logged("release", {"reason": "released"}))
        monkeypatch.setattr(R, "_rest_check", logged("rest", {"verdict": "met"}))
        monkeypatch.setattr(R, "_external_point", lambda ctx, held: None)
        monkeypatch.setattr(R, "_external_region", lambda ctx, pid: effect_mask())
        monkeypatch.setattr(R.planner, "aim_target", lambda *a, **k: {
            "ok": True, "target_obj_xy": [-0.046, 0.206], "target_grip_xy": [-0.046, 0.126], "tol_m": (0.005, 0.005),
            "m_free_m": 0.05, "thresholds": []})
        monkeypatch.setattr(R.planner, "transit_path", lambda *a, **k: {
            "ok": True, "waypoints": [[0.0, 0.0, 1.1], [-0.046, 0.126, 1.12]], "carry_z": 1.12, "thresholds": []})
        monkeypatch.setattr(R.planner, "lowering_plan", lambda *a, **k: {
            "S_dest": 0.9305, "fast_until_grip_z": 1.08, "bound_grip_z": 1.03, "drift_abort_m": 0.01,
            "tol_h_m": 0.01, "thresholds": []})
        monkeypatch.setattr(R.planner, "support_classify", classify)

    def names(self):
        return [row[0] for row in self.log]


def destination():
    return PartRecord("B", "part B", "fixed", frozenset({"ref"}), xyz=np.array([-0.046, 0.206, 0.9305]),
                      cov=np.eye(3) * 1e-6, top_z=Meas(0.9305, 0.001, "kappa"), support_z=Meas(0.9, 0.001, "plane"),
                      region_xy=np.array([[-0.14, 0.11], [0.05, 0.11], [0.05, 0.30], [-0.14, 0.30]]),
                      registered_empty_hand=True, registration_step=1, status="registered", sigma_h_m=0.001)


def place_step():
    return StepContext(1, "place_on", "D", ("C",), "B", "any", (Keep("upright", "D"),), (), 1)


def holding_ctx():
    ctx = make_ctx([destination(), part("C", move="free"), part("D", move="free", roles=("carried",))])
    ctx.holding = held_frame()
    return ctx


def test_place_on_never_opens_when_perched_and_fails_after_b2(monkeypatch):
    script = PlaceScript(monkeypatch, supports=["perched"] * (R.B2_LOCAL_RETRIES + 1))
    ctx = holding_ctx()
    ctx.held_estimator = SimpleNamespace(relocalize=lambda held, view: held)
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert (outcome.relation, outcome.status, outcome.reason) == ("supported_on", "failed",
                                                                 "support_above_destination")
    assert "release" not in script.names()
    assert script.names().count("lift") == R.B2_LOCAL_RETRIES and outcome.local_retries == R.B2_LOCAL_RETRIES
    assert "bottom above surface" in outcome.evidence_text


def test_place_on_on_surface_releases_rests_and_registers_its_stays(monkeypatch):
    script = PlaceScript(monkeypatch, supports=["on_surface"])
    ctx = holding_ctx()
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "placed")
    assert script.names() == ["carry", "aimcheck", "lower", "release", "rest"]
    assert ctx.stays.keys() == ["stays:supported_on:D:B"] and ctx.stays.entry("stays:supported_on:D:B")["step_i"] == 1
    assert "support" in ctx.events.names() and "transit" in ctx.events.names()


def test_place_on_drift_lifts_and_reaims_within_b2(monkeypatch):
    drift = {"reason": "lateral_drift", "event": None, "g_event": None, "lateral_error_m": 0.02}
    script = PlaceScript(monkeypatch, supports=["on_surface"], lowers=[drift])
    ctx = holding_ctx()
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert outcome.status == "succeeded" and outcome.local_retries == 1
    assert script.names()[:5] == ["carry", "aimcheck", "lower", "lift", "carry"]


def test_place_on_no_support_lifts_to_clearance_and_repairs(monkeypatch):
    script = PlaceScript(monkeypatch, supports=["no_support_within_bound"])
    ctx = holding_ctx()
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "no_support_within_bound")
    assert "release" not in script.names() and script.names()[-1] == "lift"


def test_place_on_while_holding_needs_a_destination_registered_with_an_empty_hand(monkeypatch):
    PlaceScript(monkeypatch, supports=[])
    ctx = holding_ctx()
    ctx.scene.parts["B"] = dataclasses.replace(destination(), registered_empty_hand=False)
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "destination_not_registered")
    assert ctx.memory["yielded"] == []


def test_destination_precondition_falls_back_on_the_base_plane():
    ctx = make_ctx()
    ctx.scene.planes = [{"z": Meas(0.9, 0.0005, "plane")}]
    record = dataclasses.replace(destination(), support_z=None)
    assert planner.destination_precondition(record)["ok"] is False
    out = R._destination_check(ctx, record)
    assert out["ok"] is True and out["support_source"] == "base_plane"


# =================================================================================================== release
def test_release_with_an_empty_open_hand_records_released_without_moving():
    ctx = make_ctx()
    outcome = drive(R.release(ctx, None), ctx)
    assert (outcome.relation, outcome.status, outcome.reason) == ("released", "succeeded", "released")
    assert ctx.memory["yielded"] == [] and ctx.pending_outcomes == [outcome]


def test_release_clears_the_held_scope_before_opening(monkeypatch):
    ctx = make_ctx()
    ctx.holding = held_frame()
    order = []
    ctx.monitor.add("hold_floor", SimpleNamespace(start=lambda v: None, update=lambda *a, **k: None), view=ctx.view,
                    scope="held")

    def fake_run(factory):
        order.append(("held_keys", list(ctx.monitor.keys("held"))))
        return {"reason": "released", "status": "succeeded", "evidence": {}}, None
        yield
    monkeypatch.setattr(R, "_run", fake_run)

    def put_down(ctx_, held):
        order.append(("put_down", list(ctx_.monitor.keys("held"))))
        return {"touched": True, "reason": "touch", "support_z": 0.9}
        yield
    monkeypatch.setattr(R, "_put_down", put_down)
    outcome = drive(R.release(ctx, None), ctx)
    assert order == [("put_down", ["hold_floor"]), ("held_keys", [])]
    assert outcome.reason == "released" and ctx.holding is None and outcome.evidence["put_down"]["touched"]


def test_release_of_a_carried_object_never_opens_without_a_support_cue(monkeypatch):
    """F4 (cross-module review): the implicit release of a carried object (the BX F7 rerun after
    no_support_within_bound, the hand at clearance height) sets it down on its pick support first; without a support
    cue the hand stays closed and the release fails no_support_within_bound."""
    ctx = make_ctx()
    ctx.holding = held_frame()
    opened = []

    def fake_release_grip(ctx_, **kwargs):
        opened.append(True)
        return {"reason": "released"}
        yield
    monkeypatch.setattr(R, "_release_grip", fake_release_grip)

    def put_down(ctx_, held):
        return {"touched": False, "reason": "bound_reached", "support_z": 0.9}
        yield
    monkeypatch.setattr(R, "_put_down", put_down)
    outcome = drive(R.release(ctx, None), ctx)
    assert (outcome.relation, outcome.status, outcome.reason) == ("released", "failed", "no_support_within_bound")
    assert opened == [] and ctx.holding is not None and outcome.evidence["release_without_support"] is True


# ================================================================================ 3. physics on the fakes
class TruthAugmentedRecorder:
    """Test-side ``ctx.survey_recorder``: the real geometry.SurveyRecorder plus DepthSamples of the construction
    view read from the fake's depth map (grid every TRUTH_GRID_PX on textured, depth-step-free patches; relative
    noise TRUTH_NOISE). The fake's rendered textures are too weak for the real start survey to register the
    destination and the keep-out part (planes[0] becomes the destination plane), which is a perception fidelity
    limit of the fake, not of the controllers under test."""

    TRUTH_GRID_PX = 24
    HALF_PX = 6
    MIN_STD = 2.0
    NEAR_M = 0.16
    TRUTH_NOISE = 0.004
    world = None

    def __init__(self, view, robot, **kwargs):
        self._real = geometry.SurveyRecorder(view, robot, **kwargs)
        self._extra = self.truth_samples(view, int(kwargs.get("start_step") or 0))

    def truth_samples(self, view, step):
        rng = np.random.default_rng(3)
        gray = np.asarray(view.wrist_gray_upright, float)
        depth = self.world.depth_map("wrist")
        pose = HandPose.from_view(view)
        cam = WristCamera(self.world.f_wrist)
        out, r = [], self.HALF_PX
        h, w = gray.shape
        for v in range(self.TRUTH_GRID_PX // 2, h - r, self.TRUTH_GRID_PX):
            for u in range(self.TRUTH_GRID_PX // 2, w - r, self.TRUTH_GRID_PX):
                zpatch = depth[v - r:v + r + 1, u - r:u + r + 1]
                z = float(depth[v, u])
                if not np.isfinite(zpatch).all() or z < self.NEAR_M or (zpatch.max() - zpatch.min()) / z > 0.01:
                    continue
                if gray[v - r:v + r + 1, u - r:u + r + 1].std() < self.MIN_STD:
                    continue
                zn = z * (1.0 + self.TRUTH_NOISE * rng.standard_normal())
                xyz = cam.backproject(pose, (float(u), float(v)), zn)
                out.append(DepthSample(tuple(float(x) for x in xyz), self.TRUTH_NOISE * z,
                                       self.TRUTH_NOISE * z / 2.0, "kappa", (float(u), float(v)), 90000 + len(out),
                                       step, self.world.f_wrist / zn))
        return out

    def on_transition(self, stage, before, after):
        self._real.on_transition(stage, before, after)

    def result(self, camera):
        result = self._real.result(camera)
        if camera is None:
            return result
        return dataclasses.replace(result, samples=list(result.samples) + list(self._extra))

    def __getattr__(self, name):
        return getattr(self._real, name)


def fake_policy(world, plan, *, setup=None, episode=None):
    """The real policy, client (ScriptedV2Transport over FakeOracleAnswers) and relation layer on ``world``; the
    episode factory injects the truth-augmented recorder."""
    from gpt_field.agent_speedup_v2 import AgentSpeedUpV2Policy, StepRunner
    from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, ScriptedV2Transport
    recorder = type("Recorder", (TruthAugmentedRecorder,), {"world": world})
    client = AgentSpeedUpV2Client(task_prompt=fakes.TASK2_PROMPT,
                                  transport=ScriptedV2Transport(fakes.FakeOracleAnswers(world, plan=plan)))

    def factory(ctx):
        ctx.survey_recorder = recorder
        return episode(ctx) if episode is not None else StepRunner(ctx).run()
    return AgentSpeedUpV2Policy(client, episode_factory=factory), client


def run_world(policy, world, steps, *, stop=None):
    observation = world.observation()
    for k in range(1, steps + 1):
        decision = policy.act(observation)
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
        if stop is not None and stop(policy, world):
            return k
    return steps


def test_calibrate_plus_on_the_fake_sets_the_calibration_and_records_calibrated():
    world = fakes.FakeTwoViewSim(fakes.task2_like_scene(), prompt=fakes.TASK2_PROMPT)

    def episode(ctx):
        cal = yield from R.calibrate_plus(ctx)
        assert ctx.cal is cal
        while True:
            yield (lambda v: R.HoldPrimitive(v, steps=5, gripper="open", name="hold"))
    policy, client = fake_policy(world, None, episode=episode)
    run_world(policy, world, 400, stop=lambda p, w: p.ctx.cal is not None)
    ctx = policy.ctx
    assert ctx.cal is not None and ctx.cal.external is not None and ctx.cal.projector is not None
    assert abs(ctx.cal.camera.f_used - world.f_wrist) <= 0.01 * world.f_wrist
    assert ctx.cal.quarter_turns == 2 and np.asarray(ctx.cal.fingertip_band).any()
    assert [r["relation"] for r in policy.outcomes] == ["calibrated"]
    assert policy.outcomes[0]["status"] == "succeeded" and policy.outcomes[0]["thresholds"]
    assert ctx.scene.hf is not None and {"g3", "g5", "g6"} <= set(ctx.memory["gripper"])
    assert list(client.transport.kinds) == []                        # Calibrate+ makes no call


def chain_plan():
    """The §3.9 task-2 plan without the keep-out part E (its fake start region is the whole support, see
    TruthAugmentedRecorder) and with a quarter-turn criterion besides the stop (the fake destination's tiny
    start region leaves the effect watch invalid under O7)."""
    plan = copy.deepcopy(fakes.TASK2_ORACLE_PLAN)
    plan["parts"] = [p for p in plan["parts"] if p["id"] != "E"]
    plan["steps"][0]["until"] = [{"o": "quarter_turn", "p": None, "txt": None}, {"o": "stop", "p": None, "txt": None}]
    plan["steps"][1]["keep"] = [k for k in plan["steps"][1]["keep"] if k.get("p") != "E"]
    return plan


def test_fake_task2_chain_turns_the_rotary_part_and_places_the_object():
    world = fakes.FakeTwoViewSim(fakes.task2_like_scene(), prompt=fakes.TASK2_PROMPT)
    policy, client = fake_policy(world, chain_plan())
    steps = run_world(policy, world, 1500, stop=lambda p, w: w.goal_satisfied())
    relations = [(r["relation"], r["status"], r["reason"]) for r in policy.outcomes]
    succeeded = [name for name, status, _ in relations if status == "succeeded"]
    assert succeeded[0] == "calibrated"
    assert ("turned", "succeeded", "turned_quarter") in relations, relations
    assert world.joint("knob") >= 0.5
    for record in policy.outcomes:                                   # AC-6 on every recorded outcome
        assert record["reason"] in kernel.REASONS and record["thresholds"] is not None
        assert len(record["evidence_text"]) <= kernel.EVIDENCE_TEXT_MAX_CHARS
        json.dumps(record, allow_nan=False)
    if not world.goal_satisfied():
        placed = [r for r in policy.outcomes if r["relation"] == "supported_on"]
        # every place attempt must be a refusal before any carry: the destination survey (geometry_unmeasured) or,
        # since C-7' (gap round SAL-R2) passes the fake's flush destination on its lateral cue, the grasp (no attempt)
        assert placed and all(r["status"] == "failed" and not (r["evidence"] or {}).get("attempts")
                              and (r["reason"] == "geometry_unmeasured"
                                   or (r["evidence"] or {}).get("failed_in") == "grasp")
                              for r in placed), (steps, relations, list(client.transport.kinds))
        pytest.xfail("fake fidelity: the fake's low-texture support leaves the destination flush with its support "
                     "(step ~0), so [C-7] refused it before the carry; C-7' passes it laterally and the grasp of "
                     "the thin part re-registered after the turn is refused before any contact (reach_limit at "
                     "the seed-lift hover); the real-sim place_t2 smoke passes the precondition")


# ============================================================================ review fixes (BRR, 2026-10-19)
def effect_turn_step(**kw):
    return dataclasses.replace(turn_step(Until("effect", "B", None), Until("stop", None, None)), **kw)


def test_turn_failure_after_a_twist_carries_resume_and_marks_the_turned_part(monkeypatch):
    """BRR-1 / BRR-9: a turn that twisted (free sense -1 measured) and failed effect_not_observed returns ``resume``
    (free sense, part angle) and keeps it per part; the articulated keep-out is added at the angle turned and the
    turned part's stale record is dropped, its hinge-centre hover point kept as the seed lift."""
    script = TurnScript(monkeypatch, segments=["blocked_sense", "free_sense"], hinges=[("yaw_budget", 1.6)],
                        effect_region=effect_mask(), post=UNCHANGED_AFTER_RELEASE)
    ctx = make_ctx([part("A")])
    outcome = drive(R.turn(ctx, effect_turn_step()), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "effect_not_observed")
    assert outcome.resume["free_sense"] == -1 and outcome.resume["theta_part_rad"] == pytest.approx(1.6)
    assert ctx.memory["turned"]["A"]["free_sense"] == -1
    assert ("keepout", "A") in script.log and ctx.scene.dropped == ["A"]
    assert ctx.scene.parts["A"].status == "dropped" and "A" in ctx.memory["seed_lift"]
    assert outcome.evidence["turned_marks"]["dropped"] == ["A"]
    json.dumps(kernel.jsonable(outcome.resume), allow_nan=False)


def test_turn_resume_with_the_effect_already_held_succeeds_without_any_motion(monkeypatch):
    """BRR-1: a resumed turn whose effect already holds against the start baseline (stays_check_for) does not twist
    the part back: succeeded turned_by_effect, flag effect_held_before_twist, no grasp, no primitive at all, the
    stays entry registered."""
    script = TurnScript(monkeypatch, segments=[], hinges=[], effect_region=effect_mask())
    met = ObsReport("region_stays", "met", 1.4, None, {"fraction": 0.3}, (), 1)
    monkeypatch.setattr(R, "stays_check_for", lambda ctx, step: ("stays:turned:A", lambda view: met))
    ctx = make_ctx([part("A")])
    ctx.memory["turned"] = {"A": {"pid": "A", "free_sense": -1, "theta_part_rad": 1.6, "theta_net_rad": 1.93}}
    outcome = drive(R.turn(ctx, effect_turn_step()), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_by_effect")
    assert outcome.evidence["effect_held_before_twist"] and "effect held before twist" in outcome.evidence_text
    assert ctx.memory["yielded"] == [] and script.log == []
    assert ctx.stays.keys() == ["stays:turned:A"] and ctx.stays.entry("stays:turned:A")["step_i"] == 0


def test_turn_resume_keeps_the_free_sense_and_never_reverses(monkeypatch):
    """BRR-1: a repair retry (StepContext.resume) twists in the KEPT free sense (-1) although the default first
    sense is +1, continues the part angle (HingeTwist theta_start), and blocked in the kept sense is the stop of
    that sense: turned_to_stop with a stop criterion, effect_not_observed without; never a reversal."""
    resume = {"pid": "A", "free_sense": -1, "theta_part_rad": 1.6, "theta_net_rad": 1.9}
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[("end_stop", 1.95)], effect_region=effect_mask())
    ctx = make_ctx([part("A")])
    outcome = drive(R.turn(ctx, effect_turn_step(resume=resume)), ctx)
    assert [row[1] for row in script.log if row[0] == "segment"] == [-1]
    assert [row for row in script.log if row[0] == "hinge"][0][1:3] == (-1, pytest.approx(1.6))
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_to_stop") and outcome.resume is None

    blocked = TurnScript(monkeypatch, segments=["blocked_sense"], hinges=[], effect_region=effect_mask())
    ctx = make_ctx([part("A")])
    outcome = drive(R.turn(ctx, effect_turn_step(resume=resume)), ctx)
    segments = [row for row in blocked.log if row[0] == "segment"]
    assert segments == [("segment", -1, False, None, "blocked_sense")]            # no reversal toward +1
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_to_stop")
    assert "kept_sense_blocked" in outcome.evidence["flags"] and "stop in kept sense" in outcome.evidence_text
    assert "release" in blocked.names() and ("stays", False) in blocked.log

    effect_only = TurnScript(monkeypatch, segments=["blocked_sense"], hinges=[], effect_region=effect_mask())
    ctx = make_ctx([part("A")])
    step = dataclasses.replace(turn_step(Until("effect", "B", None)), resume=resume)
    outcome = drive(R.turn(ctx, step), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "effect_not_observed")
    assert [row[1] for row in effect_only.log if row[0] == "segment"] == [-1]
    assert outcome.resume == resume                                # nothing moved: the earlier record holds


def test_turn_non_slip_preemptions_release_and_fail_with_their_own_reason(monkeypatch):
    """BRR-6 (turn): a keep-out or upright preemption during a twist is not a slip: release in place, failed with
    the invariant's reason, no re-grasp."""
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[("preempted:keepout_contact", 0.5)])
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("stop", None, None))), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "keepout_contact")
    assert script.names().count("grasp") == 1 and "release" in script.names()
    script = TurnScript(monkeypatch, segments=["preempted:object_rotated_in_hand"], hinges=[])
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("stop", None, None))), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "object_rotated_in_hand")
    assert script.names().count("grasp") == 1


def test_a_repeated_motionless_refusal_holds_before_it_returns():
    """BRR-2: a failure without any motion is flagged motionless; the same refusal again from an unchanged hand
    state holds MOTIONLESS_HOLD_STEPS (doubled per repeat) before it returns, with the repeat in the evidence
    text; a changed hand state starts the count again."""
    ctx = make_ctx()
    ctx.holding = SimpleNamespace(pid="D", grip_pid="C")
    step = StepContext(0, "hold", "A", ("A",), None, "any", (), (), 1)
    first = drive(R.CONTROLLERS["hold"](ctx, step), ctx)
    assert first.reason == "precondition_unmet" and first.evidence["motionless"]
    assert "motionless_repeat" not in first.evidence and ctx.memory["yielded"] == []
    holds = []
    for repeat in (1, 2):
        ctx.memory["yielded"] = []
        again = drive(R.CONTROLLERS["hold"](ctx, step), ctx)
        assert again.evidence["motionless_repeat"] == repeat and "same refusal again, repeats" in again.evidence_text
        (factory,) = ctx.memory["yielded"]
        hold = factory(ctx.view)
        assert isinstance(hold, R.HoldPrimitive) and hold.name == "motionless_repeat_hold"
        holds.append(again.evidence["motionless_hold_steps"])
        assert "MOTIONLESS_hold_steps" in [t.name for t in again.thresholds]
    assert holds == [R.MOTIONLESS_HOLD_STEPS, 2 * R.MOTIONLESS_HOLD_STEPS]
    moved = np.asarray(ctx.view.position, dtype=float) + [0.02, 0.0, 0.0]
    ctx.view = SimpleNamespace(position=moved, quaternion_xyzw=ctx.view.quaternion_xyzw)
    ctx.memory["yielded"] = []
    fresh = drive(R.CONTROLLERS["hold"](ctx, step), ctx)
    assert fresh.evidence["motionless"] and "motionless_repeat" not in fresh.evidence
    assert ctx.memory["yielded"] == []


def test_a_failed_load_test_lowers_the_object_back_before_opening(monkeypatch):
    """BRR-3: after a failed load test (grip_unstable after the 2.5 cm lift) the object is lowered back onto its
    pick support (a slow Touchdown over the realized lift) BEFORE the release, and the parts are marked moved before
    the next candidate is planned."""
    plan = contact_plan(move="free")
    ctx = make_ctx([part("C", move="free"), part("D", move="free", roles=("carried",))])
    lift_start = float(ctx.view.position[2]) - R.LIFT_TEST_M
    script = GraspScript(monkeypatch, plans=[plan, contact_plan(yaw=0.4, move="free")], classes=["expected"] * 2,
                         loads=[{"ok": False, "reason": "grip_unstable", "lift_start_z": lift_start}])
    outcome = drive(R.grasp(ctx, "C", obj_pid="D", purpose="carry"), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "hold_pass") and outcome.local_retries == 1
    names = script.names()
    i = names.index("put_back")
    assert names[i - 1] == "load" and names[i + 1] == "release" and names[i + 2] == "return_to_hover"
    assert {"C", "D"} <= set(ctx.scene.moved)
    (factory,) = [item for item in ctx.memory["yielded"] if callable(item)]
    touchdown = factory(ctx.view)
    assert isinstance(touchdown, primitives.Touchdown) and touchdown.name == "put_back"
    assert touchdown.travel_m == pytest.approx(R.LIFT_TEST_M + primitives.TOUCHDOWN_EXTRA_M)
    assert touchdown.hold_floor_mm == pytest.approx(ctx.cal.gripper.empty_floor_mm)
    put = outcome.evidence["attempts"][0]["put_back"]
    assert put["lowered"] and put["lift_m"] == pytest.approx(R.LIFT_TEST_M)
    # nothing to lower when the hand never rose (a refusal before the lift)
    ctx2 = make_ctx()
    assert drive(R._put_back(ctx2, float(ctx2.view.position[2])), ctx2)["lowered"] is False
    assert ctx2.memory["yielded"] == []


def test_the_pick_support_is_measured_or_the_carry_is_refused_before_any_contact(monkeypatch):
    """BRR-10: the pick support is the object's (else the grip part's) measured support, else the scene's base
    plane when it lies below the object's top and the hand; nothing else. Without one, grasp(carry) refuses
    geometry_unmeasured at the hover: no plan, no contact, no close."""
    bare = [dataclasses.replace(part("C", move="free"), support_z=None),
            dataclasses.replace(part("D", move="free", roles=("carried",)), support_z=None)]
    ctx = make_ctx([part("C", move="free"), bare[1]])
    assert R._support_pick(ctx, "D", "C").value == pytest.approx(0.918)       # the grip part's own support
    ctx = make_ctx(bare)
    assert R._support_pick(ctx, "D", "C") is None                               # no planes
    ctx.scene.planes = [{"z": Meas(0.9, 0.0005, "plane")}]
    assert R._support_pick(ctx, "D", "C").value == pytest.approx(0.9)
    ctx.scene.planes = [{"z": Meas(0.96, 0.0005, "plane")}]                     # above the object's top
    assert R._support_pick(ctx, "D", "C") is None
    ctx.scene.planes = [Meas(0.9, 0.0005, "plane")]                             # not a plane record
    assert R._support_pick(ctx, "D", "C") is None
    script = GraspScript(monkeypatch, plans=[contact_plan(move="free")], classes=["expected"])
    monkeypatch.setattr(R, "_support_pick", lambda ctx, obj, grip: None)
    ctx = make_ctx(bare)
    outcome = drive(R.grasp(ctx, "C", obj_pid="D", purpose="carry"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "geometry_unmeasured")
    assert outcome.evidence["support_unmeasured"] and script.names() == ["hover"]
    assert "support unmeasured" in outcome.evidence_text


def test_an_undecided_comotion_or_an_unmeasured_rotation_is_flagged_not_hidden():
    """BRR-8: the load test passes on VerifyHold but a co-motion verdict that stayed pending / missing and an O-1
    rotation verdict switched off (no wrist patch) are flagged in the result."""
    ctx = make_ctx([part("C", move="free"), part("D", move="free", roles=("carried",))])
    held = held_frame()
    ctx.held_estimator = SimpleNamespace(prior_at_close=lambda *a, **k: held, relocalize=lambda h, v: h,
                                         lift_recorder=lambda h, v: None,
                                         after_lift=lambda h, rec, **k: (h, {"touchdown": {"needed": False}}))
    ctx.scene.vacate_held = lambda h, step: 0
    out = drive(R._load_test(ctx, grip="C", obj="D", attempt=1), ctx,
                factory_reply={"status": "succeeded", "reason": "hold_pass", "evidence": {"min_gap_mm": 13.9},
                               "actions": 6})
    assert out["ok"] and out["reason"] == "hold_pass"
    assert {"comotion_unverified", "rotation_unmeasured"} <= set(out["flags"])
    assert out["evidence"]["comotion_decided"] is False


def test_the_grasp_outcome_text_names_an_unverified_hold(monkeypatch):
    GraspScript(monkeypatch, plans=[contact_plan(move="free")], classes=["expected"],
                loads=[{"ok": True, "reason": "hold_pass", "flags": ["comotion_unverified"]}])
    ctx = make_ctx()
    outcome = drive(R.grasp(ctx, "C", obj_pid="D", purpose="carry"), ctx)
    assert outcome.reason == "hold_pass" and "comotion unverified" in outcome.evidence_text
    assert outcome.evidence["comotion_unverified"] is True


def test_the_last_ladder_refresh_must_be_within_v2_before_the_close(monkeypatch):
    """BRR-16 / decision 1 §2.6: the last ladder refresh that servoed must report its residual within V2; a
    setpoint_cutoff refresh does not count; outside V2 -> no close, back to the hover, align_no_convergence."""
    rows = [{"reason": "aligned", "residual_px": 1.0, "tolerance_px": 2.0},
            {"reason": "align_no_convergence", "residual_px": 4.5, "tolerance_px": 2.0},
            {"reason": "setpoint_cutoff", "residual_px": 9.0, "tolerance_px": 2.0}]
    last = R._last_servo_refresh(rows)
    assert last["reason"] == "align_no_convergence" and last["within_v2"] is False
    assert R._last_servo_refresh(rows[:1])["within_v2"] is True and R._last_servo_refresh(rows[2:]) is None
    script = GraspScript(monkeypatch, plans=[contact_plan()], classes=["expected"])
    monkeypatch.setattr(R, "_descend", scripted({"event": "bound_reached", "contact_grip_z": 0.944,
                                                 "last_refresh": dict(last)}))
    ctx = make_ctx()
    outcome = drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "align_no_convergence")
    assert "close" not in script.names() and script.names()[-1] == "return_to_hover"
    assert "px" in outcome.evidence_text


def test_place_on_while_holding_refuses_a_failed_destination_precondition_before_any_carry(monkeypatch):
    """BRR-5: holding, with the destination registered but its [C-7] precondition failing (sigma_top too large for
    its 3 cm step): failed geometry_unmeasured before any carry, lowering or release (the survey needs an empty
    hand)."""
    script = PlaceScript(monkeypatch, supports=["on_surface"])
    ctx = holding_ctx()
    ctx.scene.parts["B"] = dataclasses.replace(destination(), top_z=Meas(0.9305, 0.02, "kappa"), sigma_h_m=0.02)
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "geometry_unmeasured")
    assert script.names() == [] and ctx.memory["yielded"] == []
    assert "empty hand" in outcome.evidence_text


def test_a_confirmed_aimcheck_correction_is_kept_by_the_lowering_and_the_held_frame(monkeypatch):
    """BRR-4: AimCheck's measured correction d moves the lowering's grip target by d (GuardedLower would otherwise
    pull the hand back) and the HeldFrame offset by R^T(-d) (a re-aim keeps it); the scene's hand-attached copies
    move with it."""
    PlaceScript(monkeypatch, supports=["on_surface"])
    d = [0.004, -0.003]
    monkeypatch.setattr(R, "_aim_check", scripted({"external": "corrected", "correction_xy": d, "move_ok": True}))
    seen = []

    def lower(ctx, lowering, aim):
        seen.append(list(aim["target_grip_xy"]))
        return {"reason": "support_event", "event": "contact_stall", "g_event": 1.051, "lateral_error_m": 0.001}
    monkeypatch.setattr(R, "_lower", scripted(lower))
    ctx = holding_ctx()
    shifted = []
    ctx.held_estimator = SimpleNamespace(_shift_attached=lambda held, pose, drag: shifted.append(np.asarray(drag)))
    before = np.asarray(ctx.holding.offset_h, dtype=float).copy()
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert outcome.status == "succeeded"
    assert seen == [pytest.approx([-0.046 + d[0], 0.126 + d[1]])]
    R_ = HandPose.from_view(ctx.view).R
    assert np.allclose(ctx.holding.offset_h, before + R_.T @ np.r_[-np.asarray(d), 0.0])
    assert len(shifted) == 1 and np.allclose(shifted[0], np.r_[-np.asarray(d), 0.0])
    assert outcome.evidence["attempts"][0]["aim_corrected_grip_xy"] == pytest.approx(seen[0])


def upright_ctx(tilt):
    """Holding D via C with a REAL UprightObserver on the held bus whose in-hand tilt is ``tilt["deg"]``."""
    ctx = holding_ctx()
    ctx.grasp_quaternion = tuple(float(v) for v in ctx.view.quaternion_xyzw)
    from gpt_field.agent_speedup_v2_observers import UprightObserver
    observer = UprightObserver(robot=ROBOT, grasp_quaternion=ctx.grasp_quaternion, held=ctx.holding,
                               tilt_fn=lambda view: tilt["deg"])
    ctx.monitor.add("upright:D", observer, view=ctx.view, preempt=True, scope="held")
    return ctx, observer


def carry_sequence(monkeypatch, results):
    calls = []

    def carry(ctx, waypoints, *, kind="carry"):
        calls.append(kind)
        return copy.deepcopy(results.pop(0)) if results else {"ok": True, "reason": "goal_reached"}
        yield
    monkeypatch.setattr(R, "held_move", carry)
    return calls


ROTATED = {"ok": False, "reason": "object_rotated_in_hand", "preemption": {"invariant": "upright:D"}}


def test_place_on_restores_the_grasp_orientation_after_an_upright_violation(monkeypatch):
    """BRR-6 / contract §C10.2 step 11: an upright violation during the carry (the hand turned, the in-hand tilt
    within O4): Orient back to the grasp orientation, the SAME observer re-judges (met) and returns to the bus
    unrestarted, the carry is retried and the object placed."""
    PlaceScript(monkeypatch, supports=["on_surface"])
    calls = carry_sequence(monkeypatch, [dict(ROTATED)])
    tilt = {"deg": 4.0}
    ctx, observer = upright_ctx(tilt)
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "placed") and outcome.local_retries == 1
    assert calls == ["carry", "carry"]
    entry = ctx.monitor.observer("upright:D")
    assert isinstance(entry, R._Resumed) and entry.inner is observer and observer.tilt0 == 4.0
    orient = [item(ctx.view) for item in ctx.memory["yielded"] if callable(item)]
    assert orient and isinstance(orient[0], R.Orient) and orient[0].name == "orient_upright_restore"
    restore = outcome.evidence["attempts"][0]["recovery"][0]["restore"]
    assert restore["ok"] and restore["recheck"]["verdict"] == "met"


def test_place_on_puts_a_tilted_object_down_and_regrasps_once_then_fails(monkeypatch):
    """BRR-6: an in-hand tilt beyond O4 that the orientation restore does not clear: put the object down (a slow
    Touchdown onto the surface below it), release, re-grasp ONCE, retry; the same violation again ->
    object_rotated_in_hand, the object still held (never opened in the air)."""
    script = PlaceScript(monkeypatch, supports=["on_surface"])
    calls = carry_sequence(monkeypatch, [dict(ROTATED), dict(ROTATED)])
    tilt = {"deg": 0.0}
    ctx, observer = upright_ctx(tilt)
    tilt["deg"] = 25.0                                    # beyond O4_TILT_MAX_DEG (20 deg) relative to the start
    regrasps = []

    def regrasp(ctx, step, *, purpose):
        regrasps.append(purpose)
        ctx.holding = held_frame()
        return R._outcome(ctx, "held", "succeeded", "hold_pass", pids=R._pids("D", ["C"]), evidence={},
                          start=ctx.step)
        yield
    monkeypatch.setattr(R, "_grasp_with_fallback", regrasp)
    touched = SimpleNamespace(touch_grip_z=1.02)                     # the put-down met the support (a support cue)
    outcome = drive(R.place_on(ctx, place_step()), ctx, primitive_for=lambda item: touched)
    assert (outcome.status, outcome.reason) == ("failed", "object_rotated_in_hand")
    assert regrasps == ["carry"] and calls == ["carry", "carry"]
    assert script.names().count("release") == 1                    # the put-down release only; never a placement
    built = [item(ctx.view) for item in ctx.memory["yielded"] if callable(item)]
    downs = [p for p in built if isinstance(p, primitives.Touchdown)]
    assert len(downs) == 1 and downs[0].name == "put_down"
    # the bound reaches the lowest support it can meet (the pick support, 0.90 m): object lowest point - 0.90 m
    lowest = HandPose.from_view(ctx.view).grip[2] - held_frame().b_low.value
    assert downs[0].travel_m == pytest.approx(lowest - 0.9 + primitives.TOUCHDOWN_EXTRA_M)
    record = outcome.evidence["attempts"][0]["recovery"][0]
    assert record["restore"]["recheck"]["cause"] == "in_hand_tilt" and record["regrasp"]["status"] == "succeeded"
    assert record["put_down"]["touched"] is True and not getattr(ctx.scene, "dropped", [])


def test_a_put_down_that_meets_no_support_never_opens(monkeypatch):
    """BRR-6 safety ([F] real task 2: a put-down that ended at its bound without a support cue opened the hand and
    the pot fell over): without a support cue the hand stays closed, the upright invariant goes back on the bus
    unrestarted, and the step fails object_rotated_in_hand while still holding."""
    script = PlaceScript(monkeypatch, supports=["on_surface"])
    carry_sequence(monkeypatch, [dict(ROTATED)])
    tilt = {"deg": 0.0}
    ctx, observer = upright_ctx(tilt)
    tilt["deg"] = 25.0
    outcome = drive(R.place_on(ctx, place_step()), ctx, primitive_for=lambda item: SimpleNamespace(touch_grip_z=None))
    assert (outcome.status, outcome.reason) == ("failed", "object_rotated_in_hand")
    assert "release" not in script.names() and ctx.holding is not None
    entry = ctx.monitor.observer("upright:D")
    assert isinstance(entry, R._Resumed) and entry.inner is observer
    assert outcome.evidence["attempts"][0]["recovery"][0]["put_down"]["touched"] is False


def test_held_move_backs_off_after_a_keepout_preemption():
    """BRR-6: a keep-out preemption (O5) during a held move backs off HELD_CONTACT_BACKOFF_M (straight up after a
    horizontal move) and fails keepout_contact, with the invariant named."""
    ctx = holding_ctx()
    ctx.grasp_quaternion = tuple(float(v) for v in ctx.view.quaternion_xyzw)
    replies = iter([{"status": "failed", "reason": "preempted:keepout_contact", "actions": 3,
                     "evidence": {"invariant": "keepout", "report": {"evidence": {}}}},
                    {"status": "succeeded", "reason": "travel_reached", "evidence": {}, "actions": 4}])
    goal = np.asarray(ctx.view.position, dtype=float) + [0.05, 0.0, 0.0]
    out = drive(R.held_move(ctx, [goal]), ctx, factory_reply=lambda item: next(replies))
    assert out["reason"] == "keepout_contact" and out["backoff"] == "travel_reached"
    assert out["preemption"]["invariant"] == "keepout"
    backoff = ctx.memory["yielded"][1](ctx.view)
    assert np.allclose(backoff.axis, [0.0, 0.0, 1.0]) and backoff.travel_m == pytest.approx(R.HELD_CONTACT_BACKOFF_M)


def test_sense_validity_thresholds_are_records_in_events_and_evidence():
    """BRR-15: the O7 validity carries threshold dicts (JSON records), not Threshold reprs."""
    from gpt_field.agent_speedup_v2_observers import EffectWatch
    ctx = make_ctx()
    effect = EffectWatch(region=effect_mask(), false_rate=0.0, sigma_px=0.4)
    validity = R._sense_validity(ctx, effect, sense=-1, steps=10, exclude=[], phase="sense")
    assert validity["valid"] and all(isinstance(t, dict) and "name" in t for t in validity["thresholds"])
    assert "Threshold(" not in json.dumps(kernel.jsonable(validity))


def test_the_support_ambiguous_class_is_flagged_in_the_evidence_text(monkeypatch):
    """BRR-16: the ambiguous support class (no kernel code of its own) fails lateral_drift with the flag
    support_ambiguous in its evidence and evidence text; never opened."""
    script = PlaceScript(monkeypatch, supports=["ambiguous"])
    ctx = holding_ctx()
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "lateral_drift")
    assert outcome.evidence["support_ambiguous"] and "support ambiguous" in outcome.evidence_text
    assert "release" not in script.names()


CARRY_PLAN = {"scene": "pot with a handle loop beside a plate region",
              "parts": [{"id": "B", "text": "plate region", "move": "fixed", "w": 0, "x": 0},
                        {"id": "C", "text": "pot handle loop", "move": "free", "w": 0, "x": 0},
                        {"id": "D", "text": "whole pot", "move": "free", "w": 0, "x": 0}],
              "hold_true": [],
              "steps": [{"do": "place_on", "obj": "D", "grip": ["C"], "ref": "B", "sense": "any",
                         "keep": [{"c": "upright", "p": "D"}], "until": []}]}


def fake_body_start_region(patch):
    """Gap round MC-R3 PO1s on the fake (fake fidelity, not a rule change): the fake's start-view registration of the
    66 mm moka body under-reads it (region 38 x 22 mm, ext_start 19.7 mm vs 45.8 mm on the real t2 record), so the true
    handle point (79.5 mm from the body centre) lies 1.2 mm beyond ext_start + PB_REACH_M; on the recorded real runs the
    smallest PO1s margin is 37 mm. Only the ``far_from_object`` refusal is lifted; PO3 / PO5 / PO6 stay on."""
    from gpt_field import agent_speedup_v2_scene as scene_module
    original = scene_module.SceneModel._part_object_check

    def check(self, *args, **kwargs):
        refusal, thresholds = original(self, *args, **kwargs)
        if refusal is not None and refusal.get("why") == "far_from_object":
            return None, thresholds
        return refusal, thresholds
    patch.setattr(scene_module.SceneModel, "_part_object_check", check)


def test_fake_carry_chain_runs_the_real_carry_path_unpatched():
    """BRR-7: the carry half of a task-2-like chain on FakeTwoViewSim, nothing patched (the truth-augmented survey
    recorder only): calibrate -> plan -> register -> place_on D via the thin part C onto B: the real grasp with the
    load test (VerifyHold, co-motion, after_lift, Touchdown, vacate_held), the held-scope bus, the held transit,
    AimCheck, GuardedLower fast and slow, SupportClassify on_surface, release, rest and the stays entry. Also the
    regression of the co-motion re-anchor (the pick-time external template is lost mid-carry while held).
    The blocker-2 framing of the grip part's pointing hover (BP-1 / BR-2) is switched off here: on the fake it moves
    the thin part's measurement view and its grasp then needs retries (a single high-sigma cell in the P7 footprint,
    an Align template loss), which ``test_fake_carry_chain_with_the_pointing_frame_completes`` covers; this test keeps
    the first-attempt carry path it guards."""
    world = fakes.FakeTwoViewSim(fakes.free_object_with_thin_part_scene(), prompt=fakes.TASK2_PROMPT)
    policy, client = fake_policy(world, CARRY_PLAN)
    patch = pytest.MonkeyPatch()
    patch.setattr(R, "_frames_grip_part", lambda pid, obj: False)
    fake_body_start_region(patch)
    try:
        run_world(policy, world, 800, stop=lambda p, w: p.ctx.stays.keys() == ["stays:supported_on:D:B"])
    finally:
        patch.undo()
    relations = [(r["relation"], r["status"], r["reason"]) for r in policy.outcomes]
    assert ("held", "succeeded", "hold_pass") in relations, relations
    assert relations[-1] == ("supported_on", "succeeded", "placed"), relations
    assert world.goal_satisfied() and world.held() is None
    assert list(client.transport.kinds).count("repair") == 0
    held = [r for r in policy.outcomes if r["relation"] == "held"][-1]
    load = held["evidence"]["attempts"][-1]["load"]
    assert load["verify_hold"]["reason"] == "hold_pass" and load["after_lift"] is not None
    bus = held["evidence"]["bus"]
    assert bus[0] == "hold_floor" and any(k.startswith("comotion:") for k in bus)
    placed = policy.outcomes[-1]
    attempt = placed["evidence"]["attempts"][-1]
    assert attempt["carry"]["ok"] and attempt["aimcheck"]["external"] in ("agrees", "transient", "corrected",
                                                                          "aimcheck_unavailable")
    assert [p["phase"] for p in attempt["lower"]["phases"]] == ["fast", "slow"]
    assert attempt["support"]["class"] == "on_surface" and placed["evidence"]["release"]["reason"] == "released"
    assert placed["evidence"]["rest"]["verdict"] != "violated"
    for record in policy.outcomes:
        assert record["reason"] in kernel.REASONS and len(record["evidence_text"]) <= kernel.EVIDENCE_TEXT_MAX_CHARS
        json.dumps(record, allow_nan=False)


# ---------------------------------------------------------------------- relation wiring on the fake (BRR-13)
def drive_world(gen, ctx, world, *, limit=2000):
    """The §C3.4 loop on a FakeTwoViewSim without the policy: each yielded factory is constructed on the current
    view and run to done (one world step per action; the invariant bus observed after every transition, as the
    policy does). A PointNeed fails the test (these scenarios make no model call)."""
    ctx.view = StepView(world.observation())
    steps = 0
    item = next(gen)
    while True:
        assert callable(item) and not isinstance(item, PointNeed), item
        primitive = item(ctx.view)
        while not primitive.done:
            action = primitive.next_action(ctx.view)
            after = StepView(world.step(action))
            feedback = primitive.observe_transition(ctx.view, action, after)
            ctx.monitor.observe(ctx.view, action, after, {"primitive_feedback": feedback, "primitive": primitive.name})
            ctx.view = after
            steps += 1
            ctx.step = steps
            assert steps <= limit, "the controller did not finish"
        try:
            item = gen.send((primitive.outcome(), primitive))
        except StopIteration as stop:
            return stop.value


def round_part_ctx(**scene_kw):
    """The round (no wedge) rotary fake, clamped by the scripted truth grasp, and a registered record of the part
    whose rigid patch lies on the disk texture (the observer-test patch that PartRotationObserver decides on)."""
    scene = fakes.rotary_part_scene(lever=False, **scene_kw)
    world = fakes.FakeTwoViewSim(scene, eef=scene.info["start_eef"])
    fakes.scripted_grasp_rotary(world)
    view = StepView(world.observation())
    cx, cy = fakes.TASK2_KNOB_CENTRE_XY
    record = PartRecord("A", "part A", "hinged", frozenset({"grip"}), xyz=np.array([cx, cy, 0.955]),
                        cov=np.eye(3) * 1e-6, top_z=Meas(0.955, 0.003, "kappa"), support_z=Meas(0.918, 0.002, "kappa"),
                        registered_empty_hand=True, registration_step=0, status="registered", sigma_h_m=0.003,
                        rigid_patch={"uv": (180.0, 340.0), "pose": tuple(float(v) for v in view.state)})
    ctx = make_ctx([record])
    ctx.view = view
    ctx.cal.start_quaternion = tuple(float(v) for v in view.quaternion_xyzw)
    return world, ctx


def test_turn_on_the_round_part_fake_decides_by_part_rotation_and_stops_without_a_wedge(monkeypatch):
    """BRR-13 / §C10.7 item 4 through the relation layer (real TwistSegment, HingeTwist, ReleaseVerify, the real
    WedgeObserver and PartRotationObserver as BR builds them; only the grasp is the scripted truth clamp). Default
    first sense +1: blocked on the round part, decided by part rotation (the gap reads free: no wedge); on this fake
    the pads stay slipped, so the reversal is blocked too (the part truly did not turn) -> not_rotatable. First
    sense -1: free; HingeTwist runs into the part's limit and ends end_stop with part_stop_without_wedge, decided by
    part rotation -> turned_to_stop, the keep-out added and the stale record dropped."""
    def clamped(ctx, step, *, purpose):
        return R._outcome(ctx, "held", "succeeded", "held_for_turn", pids=R._pids(step.obj, list(step.grip)),
                          evidence={"scripted_truth_clamp": True}, start=ctx.step)
        yield
    monkeypatch.setattr(R, "_grasp_with_fallback", clamped)
    world, ctx = round_part_ctx(q_hi=0.35)
    outcome = drive_world(R.turn(ctx, turn_step(Until("stop", None, None))), ctx, world)
    first, reversal = outcome.evidence["segments"][:2]
    assert (first["sense"], first["reason"], first["decided_by"]) == (1, "blocked_sense", "part_rotation")
    assert "blocked_by_part_rotation" in first["flags"] and reversal["sense"] == -1
    assert (outcome.status, outcome.reason) == ("failed", "not_rotatable") and world.joint("knob") < 0.1

    world, ctx = round_part_ctx(q_hi=0.35)
    ctx.overrides = {"first_sense": -1}
    outcome = drive_world(R.turn(ctx, turn_step(Until("stop", None, None))), ctx, world)
    first = outcome.evidence["segments"][0]
    assert (first["sense"], first["reason"]) == (-1, "free_sense")
    hinge = outcome.evidence["hinge"][-1]
    assert (hinge["reason"], hinge["decided_by"]) == ("end_stop", "part_rotation")
    assert "part_stop_without_wedge" in (hinge["flags"] or [])
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_to_stop"), outcome.evidence_text
    assert world.joint("knob") >= 0.3 and world.held() is None
    assert ctx.scene.added_keepouts and ctx.scene.added_keepouts[-1].kind == "articulated"
    assert ctx.scene.dropped == ["A"] and ctx.memory["turned"]["A"]["free_sense"] == -1


def o7_views(region, changed):
    """Upright external frames: a textured base, then frames whose chroma changed on ``changed`` only."""
    rng = np.random.default_rng(0)
    base = np.repeat(rng.integers(60, 120, region.shape + (1,)).astype(np.uint8), 3, axis=2)
    on = base.astype(int).copy()
    on[changed] += np.array([40, -20, -20])
    on = np.clip(on, 0, 255).astype(np.uint8)
    from gpt_field.observation import PolicyObservation
    wrist = th_png(np.zeros((8, 8, 3), np.uint8))
    state = tuple(float(v) for v in fakes.TASK2_START_STATE8)
    return [StepView(PolicyObservation(th_png(frame), wrist, state, fakes.TASK2_PROMPT)) for frame in [base] + [on] * 5]


def th_png(rgb):
    import base64
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[:, ::-1], dtype=np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.mark.parametrize("overlap, valid", [(0.30, True), (0.60, False)])
def test_o7_validity_through_the_relation_layer(overlap, valid):
    """BRR-13 / §C10.7 item 5 with the real EffectWatch: the hand-swept mask covers 30 % (valid: O1 then runs on the
    remainder only and onsets on a change confined to it) or 60 % of the effect region (effect_region_invalid: the
    effect is not watched in that phase)."""
    from gpt_field.agent_speedup_v2_observers import EffectWatch
    region = effect_mask()
    rows = np.flatnonzero(region.any(axis=1))
    cut = rows[0] + int(round(overlap * len(rows)))
    swept = np.zeros_like(region)
    swept[rows[0]:cut] = region[rows[0]:cut]
    ctx = make_ctx()
    ctx.scene.hand_swept_mask = lambda camera, view, yaw_range_rad, dilate_px=None: swept
    effect = EffectWatch(region=region, false_rate=0.0, sigma_px=0.4)
    validity = R._sense_validity(ctx, effect, sense=-1, steps=40, exclude=[], phase="free")
    assert validity["valid"] is valid
    assert validity["remaining_fraction"] == pytest.approx(1.0 - overlap, abs=0.02)
    if not valid:
        assert validity["reason"] == "effect_region_invalid"
        return
    views = o7_views(region, region & ~swept)                  # the change lies on the remainder only
    effect.start(views[0])
    R._sense_validity(ctx, effect, sense=-1, steps=40, exclude=[], phase="free")
    reports = [effect.update(views[k - 1], None, views[k], {"theta_part_rad": 1.0}) for k in range(1, len(views))]
    assert reports[-1].verdict == "met" and reports[-1].evidence["phase"] == "free"


def test_an_invalid_effect_region_drops_the_effect_but_the_stop_still_ends_the_turn(monkeypatch):
    """§C10.7 item 5 (second half) with the real _sense_validity: 60 % overlap -> effect_region_invalid, HingeTwist
    gets no effect watch, the stop criterion ends the turn: turned_to_stop."""
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[], effect_region=effect_mask())
    monkeypatch.setattr(R, "_sense_validity", REAL_SENSE_VALIDITY)
    region = effect_mask()
    rows = np.flatnonzero(region.any(axis=1))
    swept = np.zeros_like(region)
    swept[rows[0]:rows[0] + int(0.6 * len(rows))] = True
    effects = []

    def hinge(ctx, *, sense, part, effect, theta_target, theta_start):
        effects.append(effect)
        return ({"reason": "end_stop", "status": "succeeded", "evidence": {"theta_part_rad": 1.2}, "actions": 20}, None)
        yield
    monkeypatch.setattr(R, "_hinge", hinge)
    ctx = make_ctx(first_sense=-1)
    ctx.scene.hand_swept_mask = lambda camera, view, yaw_range_rad, dilate_px=None: swept
    outcome = drive(R.turn(ctx, effect_turn_step()), ctx)
    assert outcome.evidence["effect_region_invalid"] and effects == [None]
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_to_stop")
    assert ("stays", False) in script.log


REAL_SENSE_VALIDITY = R._sense_validity


class Violated:
    """A bus observer stand-in that is violated with ``reason`` at every update."""
    name = "stand_in"

    def __init__(self, reason):
        self.reason = reason

    def start(self, view):
        return None

    def update(self, before, action, after, context=None):
        return ObsReport(self.name, "violated", None, self.reason, {}, (), 1)


def test_the_put_down_returns_to_the_pick_support_and_refuses_a_keepout_touch(monkeypatch):
    """BRR-6 safety ([F] real task 2: a tilt found 2 cm into the carry was set down where it was, on the neighbouring
    keep-out): the put-down first carries the object back over its pick grip xy (the carry's transit rules), then
    the Touchdown toward the lowest reachable support; a keep-out report queued at the touch is no free support
    (touched False: the caller keeps holding)."""
    ctx = holding_ctx()
    state = list(ctx.view.state)
    state[0] += 0.05                                                  # picked 5 cm away along x
    held = dataclasses.replace(held_frame(), grasp_pose=tuple(state))
    paths, moves = [], []

    def transit(held_, start, target_xy, *args, **kwargs):
        paths.append([float(v) for v in target_xy])
        return {"ok": True, "waypoints": [list(start), [float(target_xy[0]), float(target_xy[1]), float(start[2])]]}
    monkeypatch.setattr(R.planner, "transit_path", transit)

    def move(ctx, waypoints, *, kind="carry"):
        moves.append(kind)
        return {"ok": True, "reason": "goal_reached"}
        yield
    monkeypatch.setattr(R, "held_move", move)
    out = drive(R._put_down(ctx, held), ctx, primitive_for=lambda item: SimpleNamespace(touch_grip_z=0.95))
    assert paths == [pytest.approx(state[:2])] and moves == ["put_down"]
    assert out["returned"]["ok"] and out["touched"] is True and out["support_z"] == pytest.approx(0.9)
    (factory,) = [item for item in ctx.memory["yielded"] if callable(item)]
    assert isinstance(factory(ctx.view), primitives.Touchdown)
    # a keep-out contact queued at the touch: not a free support
    ctx2 = holding_ctx()
    ctx2.monitor.add("keepout", Violated("keepout_contact"), view=ctx2.view, preempt=True, scope="held")
    ctx2.monitor.observe(ctx2.view, None, ctx2.view, {})
    out = drive(R._put_down(ctx2, held_frame()), ctx2, primitive_for=lambda item: SimpleNamespace(touch_grip_z=0.95))
    assert out["touched"] is False and out["keepout_contact"] is True


def test_a_lost_hold_opens_the_empty_hand_and_drops_the_stale_records(monkeypatch):
    """BRR-2 class: lost_in_transport with the finger gap below G6 (nothing in the hand): open, ctx.holding None, the
    parts' view memory reset (records kept: the release writes them back), so a retry grasps again instead of
    carrying an empty hand."""
    script = PlaceScript(monkeypatch, supports=[])
    carry_sequence(monkeypatch, [{"ok": False, "reason": "lost_in_transport", "primitive_reason": "hold_lost"}])
    ctx = holding_ctx()
    floor = float(R._g6(ctx).value)
    ctx.view = SimpleNamespace(**{k: getattr(ctx.view, k) for k in ("position", "quaternion_xyzw", "R", "state")},
                               gap_mm=floor - 1.0)
    monkeypatch.setattr(R, "_release_grip", scripted(lambda ctx: (setattr(ctx, "holding", None),
                                                                   {"reason": "released"})[1]))
    outcome = drive(R.place_on(ctx, place_step()), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "lost_in_transport")
    assert ctx.holding is None and outcome.evidence["hold_lost_released"] == "released"
    assert not getattr(ctx.scene, "dropped", []) and {"C", "D"} <= set(ctx.scene.moved)   # records kept, memory reset


def test_a_failure_without_net_motion_counts_as_motionless():
    """BRR-2: a failure whose primitives left the hand where it started (a still-violated invariant preempting the
    first action of every retry) is motionless too: the repeat holds."""
    ctx = make_ctx()

    def one_step_refusal(ctx, step):
        start = int(ctx.step)
        yield (lambda v: R.HoldPrimitive(v, steps=1, gripper="open", name="one_step"))
        ctx.step = int(ctx.step) + 1
        return R._outcome(ctx, "supported_on", "failed", "lost_in_transport", pids=R._pids("D", ["C"], "B"),
                          evidence={}, start=start)
    guarded = R._guarded(one_step_refusal)
    step = place_step()
    first = drive(guarded(ctx, step), ctx)
    assert first.steps == 1 and first.evidence["motionless"] and "motionless_repeat" not in first.evidence
    second = drive(guarded(ctx, step), ctx)
    assert second.evidence["motionless_repeat"] == 1 and second.evidence["motionless_hold_steps"] == \
        R.MOTIONLESS_HOLD_STEPS


def test_a_load_test_without_an_object_footprint_is_refused_before_the_lift():
    """A carried object without a registered footprint (prior_at_close raises) is refused geometry_unmeasured before
    the lift instead of crashing the episode."""
    ctx = make_ctx([part("C", move="free"), part("D", move="free", roles=("carried",))])

    def no_footprint(*args, **kwargs):
        raise ValueError("part D has no registered footprint")
    ctx.held_estimator = SimpleNamespace(prior_at_close=no_footprint)
    out = drive(R._load_test(ctx, grip="C", obj="D", attempt=1), ctx)
    assert (out["ok"], out["reason"]) == (False, "geometry_unmeasured") and ctx.memory["yielded"] == []


# =================================================================== blocker 2 (grip-part pointing), BR-1 .. BR-5
def cells_record(pid, xyz, cells, *, status="registered", provisional=False, text=None, sigma=1e-6):
    """A record with a 3-D point and HeightField region cells (the coincidence / overlap tests)."""
    base = part(pid, move="free", registered=status == "registered")
    return dataclasses.replace(base, text=text or base.text, xyz=np.asarray(xyz, dtype=float),
                               cov=np.eye(3) * sigma, region_cells=np.asarray(cells, dtype=int).reshape(-1, 2),
                               provisional=provisional, registration_step=1)


POT_CELLS = [(i, j) for i in range(10) for j in range(10)]              # 100 cells
LOOP_CELLS = [(i, j) for i in range(10, 13) for j in range(4, 7)]        # 9 cells beside them


def test_coincidence_is_the_same_point_or_a_shared_region():
    pot = cells_record("C", (0.03, 0.0, 1.05), POT_CELLS)
    same = cells_record("D", (0.03, 0.0, 1.05), POT_CELLS)
    assert R._coincides(same, pot)["iou"] == pytest.approx(1.0)
    moved = cells_record("D", (0.031, 0.0005, 1.0503), POT_CELLS[:60])               # IoU 0.6, 1.2 mm away
    assert R._coincides(moved, pot)["iou"] == pytest.approx(0.6)
    near = cells_record("D", (0.0302, 0.0, 1.05), LOOP_CELLS)                        # 0.2 mm, no overlap
    assert R._coincides(near, pot)["distance_m"] == pytest.approx(0.0002)
    # negatives: a separate part (IoU below 0.5, centimetres apart), records without points or cells
    loop = cells_record("D", (0.03, -0.07, 1.003), POT_CELLS[:40] + LOOP_CELLS)      # IoU 40 / 109
    assert R._region_iou(loop, pot) == pytest.approx(40 / 109) and R._coincides(loop, pot) is None
    assert R._coincides(dataclasses.replace(loop, region_cells=None), pot) is None
    assert R._coincides(part("D", registered=False), pot) is None
    assert R._region_iou(loop, dataclasses.replace(pot, region_cells=np.zeros((0, 2), int))) is None


def test_a_grip_record_that_coincides_with_its_object_hovers_from_its_start_view_estimate():
    """BR-1 [F, live 002 / 003]: D registered with C's region (overlap 1.0) put every hover over the pot centre; the
    hover now comes from D's start-view record, else its start-view seed lift; never the coincident record while an
    independent one exists."""
    pot = cells_record("C", (0.03, 0.0, 1.05), POT_CELLS)
    first = cells_record("D", (0.0297, -0.0723, 1.003), LOOP_CELLS, status="seeded", provisional=True)
    shared = cells_record("D", (0.03, 0.0, 1.05), POT_CELLS)
    ctx = make_ctx([pot, shared])
    ctx.memory["start_records"] = {"C": pot, "D": first}
    record, source = R._hover_reference(ctx, "D", "C")
    assert source == "independent:start_view" and record is first
    assert ctx.memory["shared_region"]["D"]["iou"] == pytest.approx(1.0)
    # the start record shares the region too -> the start-view lift of D's own plan seed
    ctx.memory["start_records"]["D"] = dataclasses.replace(first, region_cells=np.asarray(POT_CELLS))
    ctx.memory["seed_lift"] = {"D": (np.array([0.028, -0.075, 1.0]), np.eye(3) * 4e-4)}
    record, source = R._hover_reference(ctx, "D", "C")
    assert source == "independent:seed_lift" and record.status == "seeded" and record.region_cells is None
    assert np.allclose(record.xyz, [0.028, -0.075, 1.0])
    # a replan that gave the id to another part does not inherit its start record
    ctx.memory["start_records"]["D"] = dataclasses.replace(first, text="something else")
    assert R._hover_reference(ctx, "D", "C")[1] == "independent:seed_lift"
    # no independent estimate: the coincident record is kept and the coincidence logged
    ctx.memory.pop("seed_lift")
    record, source = R._hover_reference(ctx, "D", "C")
    assert record is shared and source == "registered" and ctx.memory["shared_region"]["D"]["independent"] is None
    # negatives: no object, the part is its own object, a distinct record
    assert R._hover_reference(ctx, "D")[1] == "registered" and R._hover_reference(ctx, "D", "D")[1] == "registered"
    ctx.scene.parts["D"] = cells_record("D", (0.03, -0.07, 1.003), LOOP_CELLS)
    assert R._hover_reference(ctx, "D", "C")[1] == "registered"


def band_ctx(parts=(), top_row=385):
    ctx = make_ctx(parts)
    band = np.zeros((512, 512), dtype=bool)
    band[top_row:, 100:412] = True
    ctx.cal.fingertip_band = band
    return ctx


def record_at_pixel(ctx, pid, uv, depth=0.2, sigma=1e-6):
    pose = R._pose(ctx)
    xyz = WristCamera(R._f_px(ctx)).backproject(pose, uv, depth)
    return cells_record(pid, xyz, LOOP_CELLS, sigma=sigma)


def built(ctx, factory):
    return factory(ctx.view)


def test_the_arrived_hover_is_framed_for_fine_pointing_by_a_guarded_shift_along_minus_r0():
    """BR-2 (BP-1 at the arrived hover, zero calls): the fine crop must end FT_PROBE_RADIUS_PX above the band; the
    excess / kappa is one guarded AnchoredMove along the horizontal -R0 (a WaypointReach would treat a few mm as
    reached), no pointing need, no raise while the shift is uncapped."""
    ctx = band_ctx()
    record = record_at_pixel(ctx, "D", (256.0, 340.0))
    info = drive(R._frame_for_pointing(ctx, "D", record), ctx)
    assert not any(isinstance(item, PointNeed) for item in ctx.memory["yielded"])
    moves = [built(ctx, f) for f in ctx.memory["yielded"]]
    shift = moves[0]
    assert type(shift).__name__ == "AnchoredMove" and shift.name == "frame_shift" and shift.stop_on_contact
    expected = planner.pointing_frame(record, np.asarray(ctx.view.position), R._pose(ctx), f_px=R._f_px(ctx),
                                      band_top_row=385.0)
    assert info["applied"] and info["before"]["excess_px"] == pytest.approx(expected["excess_px"])
    assert shift.travel_m == pytest.approx(expected["excess_px"] / expected["kappa_px_per_m"], rel=1e-6)
    r0 = np.asarray(ctx.view.R)[:2, 0]
    assert np.dot(np.asarray(shift.axis)[:2], r0 / np.linalg.norm(r0)) == pytest.approx(-1.0, abs=1e-6)
    assert "raise" not in info and len(moves) == 2                                  # the shift and its B4 settle
    # framed already: nothing moves
    ctx = band_ctx()
    info = drive(R._frame_for_pointing(ctx, "D", record_at_pixel(ctx, "D", (256.0, 250.0))), ctx)
    assert info == {"applied": False, "ok": True, "before": info["before"]} and ctx.memory["yielded"] == []
    # no band or no point: nothing to frame
    ctx = make_ctx()
    assert drive(R._frame_for_pointing(ctx, "D", record), ctx)["reason"] == "no_band"


def test_a_capped_framing_shift_rises_to_the_first_framing_height_within_the_stretch_cap(monkeypatch):
    """BP-1's second remedy at the arrived hover: the shift needed exceeds M2_SEGMENT_M, so after the capped shift
    the hover rises (HeightField-cell steps) to the first height that frames the part, within the stretch cap."""
    ctx = band_ctx()
    ctx.scene.hf = SimpleNamespace(cell_m=0.005)
    z0 = float(ctx.view.position[2])

    def frame(record, here, pose, *, f_px, band_top_row, pointing="auto"):
        rise = float(here[2]) - z0
        ok = rise >= 0.03 - 1e-9
        return {"ok": ok, "mode": "fine", "row": 400.0, "sigma_uv_px": 5.0, "excess_px": 0.0 if ok else 190.0,
                "kappa_px_per_m": 1668.0, "margin_px": 69.0, "crop_bottom_row": 464.0, "band_top_row": band_top_row}
    monkeypatch.setattr(R.planner, "pointing_frame", frame)
    record = record_at_pixel(ctx, "D", (256.0, 400.0))
    info = drive(R._frame_for_pointing(ctx, "D", record), ctx)
    moves = [built(ctx, f) for f in ctx.memory["yielded"]]
    assert moves[0].name == "frame_shift" and moves[0].travel_m == pytest.approx(planner.M2_SEGMENT_M)
    lifts = [m for m in moves if getattr(m, "name", "") == "frame_raise"]
    assert len(lifts) == 1 and np.allclose(lifts[0].axis, R.WORLD_UP) and lifts[0].travel_m == pytest.approx(0.03)
    assert info["shift"]["needed_m"] > planner.M2_SEGMENT_M and info["raise"]["reason"] == "scripted"
    # the stretch cap bounds the rise (a mount whose limit leaves no room: no raise at all)
    ctx = band_ctx()
    ctx.scene.hf = SimpleNamespace(cell_m=0.005)
    ctx.cal.mount = {"shoulder_xyz": [-0.6, 0.0, 0.9], "frontal_stretch_limit_m": 0.1}
    info = drive(R._frame_for_pointing(ctx, "D", record), ctx)
    assert [m.name for m in (built(ctx, f) for f in ctx.memory["yielded"])].count("frame_raise") == 0
    assert info["raise"]["raised_m"] == 0.0


def test_only_a_grip_part_of_another_part_gets_the_pointing_frame(monkeypatch):
    """BP-1 / BR-2 scope: hover_for(pointing="auto") and the arrival framing for a grip part P of object Q != P
    (the lens's pairs); a part gripped as itself keeps P-2 only."""
    seen = []

    def hover_for(record, hf, robot, pose, **kw):
        seen.append(kw.get("pointing"))
        return {"hover_grip_xyz": None, "reason": "geometry_unmeasured"}
    monkeypatch.setattr(R.planner, "hover_for", hover_for)
    ctx = make_ctx([part("A"), part("C", move="free"), part("D", move="free")])
    drive(R._grip_hover(ctx, "A", purpose="turn", obj="A"), ctx)
    drive(R._grip_hover(ctx, "C", purpose="carry", obj="D"), ctx)
    assert seen == [None, "auto"]
    assert R._frames_grip_part("C", "D") and not R._frames_grip_part("A", "A") and not R._frames_grip_part("A", None)


def scripted_round(monkeypatch, results, calls):
    """grasp_point_round stand-in: each call pops a result and, for a failure, notes ``rounds`` failed rounds."""
    def round_(ctx, pid, **kw):
        calls.append(kw)
        spec = results.pop(0)
        for _ in range(spec.get("rounds", 0)):
            R._note_failed_round(ctx, pid, kw.get("obj"), reason="pointed_not_grasp_point", answer=("fine", 24),
                                 uv=(250.0, 330.0), iou=spec.get("iou"), text="the pointed point lies on part C")
        return {"record": spec.get("record"), "reason": spec.get("reason"), "rounds": spec.get("rounds", 0),
                "calls": spec.get("rounds", 0), "seed": None, "stages": [], "rejected": [], "reasks": []}
        yield
    monkeypatch.setattr(R, "grasp_point_round", round_)


def escalation_ctx(monkeypatch):
    pot = cells_record("C", (0.03, 0.0, 1.05), POT_CELLS)
    loop = cells_record("D", (0.03, -0.07, 1.003), LOOP_CELLS)
    ctx = make_ctx([pot, loop])
    hovers = []

    def hover_target(ctx, record, *, skip=(), pointing=None):
        hovers.append((record.pid, pointing))
        return {"hover_grip_xyz": [float(record.xyz[0]), float(record.xyz[1]), 1.15], "clearance_m": 0.05}
    monkeypatch.setattr(R, "_hover_target", hover_target)
    monkeypatch.setattr(R, "_arrive", scripted({"ok": True, "reason": "travel_reached"}))
    monkeypatch.setattr(R, "_frame_for_pointing", scripted({"applied": False, "ok": True}))
    framed = []
    monkeypatch.setattr(R, "_frame_object", scripted(lambda ctx, record: framed.append(record.pid) or {"fits": True}))
    monkeypatch.setattr(R, "_plan", lambda ctx, record, **kw: contact_plan(record.pid))
    monkeypatch.setattr(R, "_restage_seed", scripted((None, 0)))
    return ctx, hovers, framed


def test_two_failed_rounds_escalate_once_per_step_to_an_object_framing_survey(monkeypatch):
    """BR-4 [F, live 003: three loops of ~165 steps and 3 calls, no cap]: after 2 failed rounds of the grip part at
    its object, one survey hovers from the OBJECT's record, frames its region (zero calls), then one coarse-first
    pointing round with no re-ask; at most once per plan step, then the failure goes to repair with the measured
    facts (failed rounds, the overlap flag, the escalation)."""
    ctx, hovers, framed = escalation_ctx(monkeypatch)
    calls = []
    scripted_round(monkeypatch, [{"rounds": 2, "reason": "pointed_not_grasp_point", "iou": 0.6},
                                 {"rounds": 1, "reason": "pointed_not_grasp_point", "iou": 0.3}], calls)
    ctx.memory["step_scope"] = 1
    out = drive(R._grip_hover(ctx, "D", purpose="carry", obj="C"), ctx)
    assert [c.get("coarse_first", False) for c in calls] == [False, True] and calls[1]["reasks"] == 0
    assert calls[1]["previous"] == "the pointed point lies on part C"
    assert hovers == [("D", "auto"), ("C", None)] and framed == ["C"]
    assert out["escalation"]["ran"] and out["reason"] == "pointed_not_grasp_point"
    assert out["pointing"]["failed_rounds"] == 3 and out["pointing"]["escalations"] == 1
    assert out["pointing"]["iou_obj_max"] == pytest.approx(0.6)
    text = R._pointing_failure_text(ctx, "D", "C", out)
    assert text["text_items"] == [["failed pointing rounds", 3, "count"]]
    assert text["text_flags"] == ["pointed point on part C itself", "object survey ran"]
    # the same plan step again: no second escalation
    calls.clear()
    scripted_round(monkeypatch, [{"rounds": 2, "reason": "pointed_not_grasp_point"}], calls)
    out = drive(R._grip_hover(ctx, "D", purpose="carry", obj="C"), ctx)
    assert len(calls) == 1 and "escalation" not in out
    # a new plan step escalates again
    ctx.memory["step_scope"] = 2
    calls.clear()
    scripted_round(monkeypatch, [{"rounds": 2, "reason": "pointed_not_grasp_point"},
                                 {"record": cells_record("D", (0.03, -0.072, 1.004), LOOP_CELLS)}], calls)
    out = drive(R._grip_hover(ctx, "D", purpose="carry", obj="C"), ctx)
    assert len(calls) == 2 and out["record"] is not None and out["reason"] is None
    assert out["two_stage"]["taken"] and hovers[-1] == ("D", None)       # the part's own hover after the object's


def test_no_escalation_for_one_failed_round_a_part_gripped_as_itself_or_a_held_hand(monkeypatch):
    ctx, hovers, framed = escalation_ctx(monkeypatch)
    calls = []
    scripted_round(monkeypatch, [{"rounds": 1, "reason": "pointed_not_grasp_point"}], calls)
    drive(R._grip_hover(ctx, "D", purpose="carry", obj="C"), ctx)
    assert len(calls) == 1 and framed == []
    ctx, hovers, framed = escalation_ctx(monkeypatch)
    scripted_round(monkeypatch, [{"rounds": 3, "reason": "pointed_not_grasp_point"}], calls)
    drive(R._grip_hover(ctx, "C", purpose="turn", obj="C"), ctx)
    assert framed == []
    entry = {"escalations": 0, "failed_rounds": 1, "answers": [["fine", 24], ["fine", 24]]}
    assert R._should_escalate(entry)                                             # the same answer id twice
    assert not R._should_escalate(dict(entry, answers=[["fine", 24], ["fine", 32]]))
    assert not R._should_escalate(dict(entry, failed_rounds=2, escalations=1))


def reask_ctx(monkeypatch, *, registered_cells, marker=False):
    pot = cells_record("C", (0.03, 0.0, 1.05), POT_CELLS)
    ctx = make_ctx([pot, cells_record("D", (0.03, -0.07, 1.003), LOOP_CELLS)])
    ctx.pointing_marker = marker
    rec = cells_record("D", (0.03, -0.01, 1.04), registered_cells)
    monkeypatch.setattr(R, "_probe_and_register", scripted(lambda ctx, pid, seed, label: {
        "record": rec, "failure": None, "label": label, "probe": {}}))
    monkeypatch.setattr(R, "_pointed_visibility", scripted(lambda ctx, pid, seed, **kw: (seed, None)))
    checks = iter([{"ok": False, "reason": "pointed_not_grasp_point",
                    "values": {"above_m": [0.0009, 0.03], "margin_m": 0.009}, "thresholds": []},
                   {"ok": False, "reason": "pointed_not_grasp_point", "values": {}, "thresholds": []}])
    monkeypatch.setattr(R.planner, "check_point_answer", lambda *a, **k: next(checks))
    monkeypatch.setattr(R, "_inputs", lambda ctx, **kw: None)
    monkeypatch.setattr(R, "_check_jaw", lambda ctx, record, **kw: (None, None))
    ctx.scene.predicted_uv = lambda pid, view, camera="wrist": (np.array([260.0, 300.0]), 6.0)
    return ctx


def test_a_reask_on_the_object_itself_goes_coarse_on_the_full_view_with_the_measured_text(monkeypatch):
    """BR-3 [F, live: each re-ask resent a nearly identical crop and got the identical answer, 4/4]: the rejected
    point's registered region overlaps the object's at IoU >= 0.2 -> coarse on the full view, text "the pointed point
    lies on part C itself"; never the same fine crop again; no marker until the client draws one."""
    ctx = reask_ctx(monkeypatch, registered_cells=POT_CELLS[:50])                    # IoU 0.5 with the pot
    needs = []
    out = drive(R.grasp_point_round(ctx, "D", purpose="carry", obj="C"), ctx,
                points=lambda need: (needs.append(need), answer(need, (262.0, 302.0)))[1])
    assert [n.kind for n in needs] == ["point_fine", "point_coarse", "point_fine"]
    assert needs[0].center_uv == (260.0, 300.0) and needs[1].previous_rejected == REASK_TEXT.format(obj="C")
    assert all(n.marker_uv is None for n in needs)
    assert out["reasks"][0]["mode"] == "coarse_on_object" and out["rejected"][0]["iou_obj"] == pytest.approx(0.5)
    entry = R._pointing_entry(ctx, "D", "C")
    assert entry["failed_rounds"] == 2 and entry["last_text"].startswith("the pointed point lies on part C")


REASK_TEXT = R.REASK_ON_OBJECT_TEXT


def test_a_reask_off_the_object_recentres_the_fine_crop_on_the_rejected_pixel_and_rings_it(monkeypatch):
    ctx = reask_ctx(monkeypatch, registered_cells=LOOP_CELLS, marker=True)            # IoU 0 with the pot
    needs = []
    replies = iter([(300.0, 350.0), (301.0, 349.0)])
    drive(R.grasp_point_round(ctx, "D", purpose="carry", obj="C"), ctx,
          points=lambda need: (needs.append(need), answer(need, next(replies)))[1])
    assert [n.kind for n in needs] == ["point_fine", "point_fine"]
    assert needs[1].center_uv == pytest.approx((300.0, 350.0), abs=1e-6)            # re-centred, not the prediction
    assert needs[1].marker_uv == pytest.approx((300.0, 350.0), abs=1e-6) and needs[0].marker_uv is None
    assert "0.1 cm above a finger landing" in needs[1].previous_rejected
    # the rejected pixel is the crop centre already (within the answer's quantization): the full view instead
    ctx = reask_ctx(monkeypatch, registered_cells=LOOP_CELLS)
    needs.clear()
    drive(R.grasp_point_round(ctx, "D", purpose="carry", obj="C"), ctx,
          points=lambda need: (needs.append(need), answer(need, (262.0, 303.0)))[1])
    assert [n.kind for n in needs] == ["point_fine", "point_coarse", "point_fine"]
    # the ringed text with the client's marker on an on-object re-ask
    ctx = reask_ctx(monkeypatch, registered_cells=POT_CELLS[:50], marker=True)
    needs.clear()
    drive(R.grasp_point_round(ctx, "D", purpose="carry", obj="C"), ctx,
          points=lambda need: (needs.append(need), answer(need, (262.0, 302.0)))[1])
    assert needs[1].previous_rejected == R.REASK_ON_OBJECT_RINGED_TEXT.format(obj="C")
    assert len(needs[1].previous_rejected) <= kernel.PREVIOUS_REJECTED_MAX_CHARS and needs[1].marker_uv is not None


def refine_infeasible(pid="C", reason="footprint_blocked", *, refine=True):
    cell = {"ij": [76, 148], "xyz": [0.04, -0.101, 1.039], "z": 1.039, "sigma": 0.0049, "source": "kappa",
            "unconfirmed": True}
    values = {"refine_cells": refine, "blocking_cells": [cell] if refine else []}
    rejection = planner.CandidateRejection("top_down", 0.0, "height_stop:P7", reason, values, ())
    return planner.Infeasible(pid, reason, (rejection,), (kernel.th_value("P7_footprint_margin_m", 0.009, "ONLINE",
                                                                          "P7"),), refine_cells=refine)


def test_refine_cells_remeasures_the_blocking_cells_once_and_replans(monkeypatch):
    """BR-5: an Infeasible whose P5 / P6 / P7 refusal rests only on unconfirmed cells (BP-2 refine_cells) gets one
    ParallaxProbe of those cells and one re-plan; a second refine_cells refusal fails with its own reason."""
    refined = []
    monkeypatch.setattr(R, "_refine_cells", scripted(lambda ctx, pid, cells: refined.append(cells) or {"ok": True}))
    GraspScript(monkeypatch, plans=[refine_infeasible(), contact_plan("C", move="free")], classes=["expected"])
    ctx = make_ctx()
    outcome = drive(R.grasp(ctx, "C", obj_pid="D", purpose="carry"), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "hold_pass")
    assert len(refined) == 1 and refined[0][0]["ij"] == [76, 148]
    assert outcome.evidence["refine_cells"][0]["reason_before"] == "footprint_blocked"
    refined.clear()
    GraspScript(monkeypatch, plans=[refine_infeasible(), refine_infeasible()], classes=[])
    outcome = drive(R.grasp(make_ctx(), "C", obj_pid="D", purpose="carry"), make_ctx())
    assert (outcome.status, outcome.reason) == ("failed", "footprint_blocked") and len(refined) == 1
    # negatives: no refine_cells -> no probe; the helper reads the planner's cells
    refined.clear()
    GraspScript(monkeypatch, plans=[refine_infeasible(refine=False)], classes=[])
    outcome = drive(R.grasp(make_ctx(), "C", obj_pid="D", purpose="carry"), make_ctx())
    assert outcome.reason == "footprint_blocked" and refined == []
    assert R._cell_refinement(refine_infeasible(refine=False)) == [] and R._cell_refinement(contact_plan()) == []


def test_refine_cells_probes_the_cells_pixels_off_the_band(monkeypatch):
    ctx = band_ctx()
    pose = R._pose(ctx)
    camera = WristCamera(R._f_px(ctx))
    visible = camera.backproject(pose, (250.0, 300.0), 0.2)
    under_band = camera.backproject(pose, (250.0, 450.0), 0.2)
    behind = pose.camera - 0.1 * np.asarray(pose.R)[:, 2]
    seen = {}

    def run_probe(ctx, make_recorder):
        ctx.survey_recorder = lambda view, robot, **kw: seen.update(kw) or SimpleNamespace(result=lambda cam: "s")
        recorder = make_recorder(ctx.view)
        return {"reason": "probe_complete", "status": "succeeded"}, recorder, [{"reason": "probe_complete"}]
        yield
    monkeypatch.setattr(R, "_run_probe", run_probe)
    ctx.scene.absorb_survey = lambda survey, view: 3
    cells = [{"ij": [1, 1], "xyz": list(visible)}, {"ij": [1, 2], "xyz": list(under_band)},
             {"ij": [1, 3], "xyz": list(behind)}]
    info = drive(R._refine_cells(ctx, "C", cells), ctx)
    assert info["ok"] and info["targets"] == 1 and info["cells_changed"] == 3
    assert [k for k, _ in seen["seeds"]] == ["refine:0"] and seen["seeds"][0][1] == pytest.approx((250.0, 300.0))
    assert seen["relief"][0][0] == "relief:refine:0" and seen["exclude_mask"] is ctx.cal.fingertip_band
    assert drive(R._refine_cells(ctx, "C", cells[1:]), ctx)["reason"] == "cells_out_of_view"


# ======================================================================= blocker 1 (effect visibility), BR-1 .. BR-4
class ScriptedWatch:
    """EffectWatch stand-in for the confirmation loop: confirm() returns the scripted reasons in order."""

    VERDICT = {"effect_confirmed": "met", "effect_region_unchanged": "violated", "effect_confirming": "pending",
               "effect_unverifiable": "pending"}

    def __init__(self, reasons):
        self.reasons = list(reasons)
        self.calls = []
        self.region = effect_mask()
        self.false_rate, self.sigma_px, self.exclude = 0.0, 0.4, ()

    def stays_check(self, projector_mask=None):
        self.stays_mask = projector_mask
        return lambda view: ObsReport("region_stays", "met", 1.0, None, {}, (), 1)

    def confirm(self, view, hidden_now=None, context=None):
        self.calls.append((hidden_now, dict(context or {})))
        reason = self.reasons.pop(0) if self.reasons else "effect_unverifiable"
        share = {"effect_confirmed": 0.21, "effect_confirming": 0.21, "effect_region_unchanged": 0.0}.get(reason)
        return ObsReport("effect", self.VERDICT[reason], None, reason,
                         {"visible_share": 0.72 if share is not None else 0.3, "share": share,
                          "reference": {"source": "noise_meter_start"}},
                         (kernel.th_value("O1_fraction", 0.02, "DD", "O1"),), len(self.calls))


def lifts_of(ctx):
    return [built(ctx, f) for f in ctx.memory["yielded"] if built(ctx, f).name == "effect_confirm_lift"]


def test_effect_confirm_meets_on_static_frames_without_lifting_when_the_region_is_visible():
    ctx = make_ctx()
    watch = ScriptedWatch(["effect_confirming", "effect_confirming", "effect_confirmed"])
    out = drive(R._effect_confirm(ctx, watch), ctx)
    assert (out["verdict"], out["reason"], out["frames"]) == ("met", "effect_confirmed", 3)
    assert out["flags"] == ["effect_confirmed_after_release"] and lifts_of(ctx) == []
    holds = [built(ctx, f) for f in ctx.memory["yielded"]]
    assert [h.name for h in holds] == ["effect_confirm_hold"] * 3 and watch.calls[0][0] is None
    assert out["visible_share"] == pytest.approx(0.72) and out["share"] == pytest.approx(0.21)


def test_effect_confirm_violated_needs_o1_consecutive_unchanged_frames():
    ctx = make_ctx()
    watch = ScriptedWatch(["effect_region_unchanged", "effect_confirming", "effect_region_unchanged",
                           "effect_region_unchanged", "effect_region_unchanged"])
    out = drive(R._effect_confirm(ctx, watch), ctx)
    assert (out["verdict"], out["frames"]) == ("violated", 5) and out["flags"] == ["effect_region_unchanged"]


def test_effect_confirm_lifts_in_1_cm_guarded_steps_while_hidden_up_to_8_cm():
    """Blocker-1 spec BR-2 [F, smokes 2 / 3: the S-16 masks needed +4.4 / +3.0 cm after the release]: an
    unverifiable frame lifts the open hand 1 cm (AnchoredMove, stop on contact); still hidden after 8 cm ->
    pending effect_unverifiable."""
    ctx = make_ctx()
    out = drive(R._effect_confirm(ctx, ScriptedWatch(["effect_unverifiable"] * 20)), ctx)
    lifts = lifts_of(ctx)
    assert len(lifts) == round(R.POST_CONFIRM_LIFT_MAX_M / R.POST_CONFIRM_LIFT_STEP_M)
    assert all(np.allclose(m.axis, R.WORLD_UP) and m.stop_on_contact and m.travel_m == pytest.approx(0.01)
               for m in lifts)
    assert (out["verdict"], out["reason"]) == ("pending", "effect_unverifiable")
    assert out["flags"] == ["effect_unverifiable"] and len(out["levels"]) == 9
    # visible after two lifts -> met there
    ctx = make_ctx()
    out = drive(R._effect_confirm(ctx, ScriptedWatch(["effect_unverifiable"] * 2 + ["effect_confirming"] * 2
                                                     + ["effect_confirmed"])), ctx)
    assert out["verdict"] == "met" and len(lifts_of(ctx)) == 2
    # a lift that meets a contact is judged once more there, then stops
    ctx = make_ctx()
    out = drive(R._effect_confirm(ctx, ScriptedWatch(["effect_unverifiable"] * 5)), ctx,
                factory_reply=lambda f: {"status": "failed", "reason": "contact_stall", "evidence": {}, "actions": 3}
                if built(ctx, f).name == "effect_confirm_lift" else {"status": "succeeded", "reason": "done",
                                                                     "evidence": {}, "actions": 1})
    assert out["lift_stopped"] == "contact_stall" and len(lifts_of(ctx)) == 1 and out["verdict"] == "pending"
    # the resume pre-check never lifts
    ctx = make_ctx()
    out = drive(R._effect_confirm(ctx, ScriptedWatch(["effect_unverifiable"] * 5), lift_max_m=0.0), ctx)
    assert lifts_of(ctx) == [] and out["verdict"] == "pending"
    # a hand that still grips the part is never opened and never lifted by the confirmation
    ctx = make_ctx()
    ctx.memory["gripped"] = {"pid": "A", "obj": "A"}
    out = drive(R._effect_confirm(ctx, ScriptedWatch(["effect_unverifiable"] * 5)), ctx)
    holds = [built(ctx, f) for f in ctx.memory["yielded"]]
    assert lifts_of(ctx) == [] and {h.gripper for h in holds} == {"close"} and out["verdict"] == "pending"


def confirmed_post(verdict, lift=0.03):
    return {"spring_back": "met", "persists": verdict,
            "confirm": {"verdict": verdict, "reason": {"met": "effect_confirmed", "violated": "effect_region_unchanged",
                                                       "pending": "effect_unverifiable"}[verdict],
                        "lift_m": lift, "frames": 7, "visible_share": 0.84, "share": 0.27,
                        "flags": [R.EFFECT_CONFIRM_FLAGS[verdict]], "levels": [], "report": None}}


def test_an_effect_seen_only_after_the_release_succeeds_turned_by_effect(monkeypatch):
    """Blocker-1 spec BR-2: no onset in the twist (the yaw budget ended it), the post-release confirmation is met ->
    succeeded turned_by_effect, flag effect_confirmed_after_release, theta_e null, the stays entry on the effect."""
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[("yaw_budget", 1.68)],
                        effect_region=effect_mask(), post=confirmed_post("met"))
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("effect", "B", None))), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_by_effect")
    assert outcome.evidence["effect_confirmed_after_release"] is True and outcome.evidence["theta_e_rad"] is None
    assert outcome.evidence["effect_confirm"]["lift_m"] == pytest.approx(0.03)
    assert outcome.evidence["effect_confirm"]["visible_share"] == pytest.approx(0.84)
    assert "effect confirmed after release" in outcome.evidence_text and "post-release lift 3.0 cm" in \
        outcome.evidence_text
    assert ("stays", True) in script.log and script.names()[-4:] == ["release", "post_check", "stays", "keepout"]


@pytest.mark.parametrize("stop, verdict, reason, flag", [
    ("yaw_budget", "pending", "effect_not_observed", "effect_unverifiable"),
    ("end_stop", "violated", "effect_not_observed", "effect_region_unchanged"),
    ("axis_tug", "pending", "axis_tug", "effect_unverifiable"),
])
def test_an_effect_not_confirmed_after_the_release_fails_with_the_confirmation_flag(monkeypatch, stop, verdict,
                                                                                      reason, flag):
    hinges = [(stop, 1.2)] + ([(stop, 1.3)] if stop == "axis_tug" else [])
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=hinges, effect_region=effect_mask(),
                        post=confirmed_post(verdict))
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("effect", "B", None))), ctx)
    assert (outcome.status, outcome.reason) == ("failed", reason)
    assert outcome.evidence[flag] is True and flag.replace("_", " ") in outcome.evidence_text
    assert ("stays", True) not in script.log


def test_an_onset_in_the_twist_is_still_confirmed_after_the_release(monkeypatch):
    TurnScript(monkeypatch, segments=["free_sense"], hinges=[("effect_margin_reached", 0.4)],
               effect_region=effect_mask(), post=confirmed_post("violated"))
    outcome = drive(R.turn(make_ctx(first_sense=-1), turn_step(Until("effect", "B", None))), make_ctx())
    assert (outcome.status, outcome.reason) == ("failed", "effect_not_observed") and outcome.evidence["persists"] is False
    TurnScript(monkeypatch, segments=["free_sense"], hinges=[("effect_margin_reached", 0.4)],
               effect_region=effect_mask(), post=confirmed_post("pending"))
    ctx = make_ctx(first_sense=-1)
    outcome = drive(R.turn(ctx, turn_step(Until("effect", "B", None))), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_by_effect")
    assert "effect_confirm_pending" in outcome.evidence["flags"]
    # a matched stop criterion is not re-judged by the effect
    script = TurnScript(monkeypatch, segments=["free_sense"], hinges=[("end_stop", 1.0)], effect_region=effect_mask(),
                        post=confirmed_post("violated"))
    outcome = drive(R.turn(make_ctx(first_sense=-1), effect_turn_step()), make_ctx())
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_to_stop")


def test_the_turn_watch_takes_the_meter_reference_and_keeps_it_for_stays_and_resume(monkeypatch):
    """Blocker-1 spec BR-3 / BO 1: the EffectWatch is built on the MotionNoiseMeter's start frame (before the
    approach, the hand clear) with the obj / grip regions as excludes; it is kept per turned part, and the stays
    check of a failed step (BX skip_step) uses it: the effect part's region is never taken again after the grasp."""
    asked = []
    TurnScript(monkeypatch, segments=["free_sense"], hinges=[("yaw_budget", 1.6)], effect_region=effect_mask(),
               post=UNCHANGED_AFTER_RELEASE)
    original = R._external_region

    def region(ctx, pid):
        asked.append((pid, ctx.holding is None and not ctx.memory.get("gripped")))
        return original(ctx, pid)
    monkeypatch.setattr(R, "_external_region", region)
    ctx = make_ctx(first_sense=-1)
    step = turn_step(Until("effect", "B", None))
    drive(R.turn(ctx, step), ctx)
    stored = ctx.memory["effect_ref"]["A"]
    assert stored["pid"] == "B" and stored["watch"].reference_source == "noise_meter_start"
    assert [pid for pid, _ in asked].count("B") == 1
    asked.clear()
    key, check = R.stays_check_for(ctx, step)
    assert key == "stays:turned:A" and check.kind == "effect" and asked == []
    assert R.stays_check_for(ctx, turn_step(Until("effect", "E", None))) is None or asked == [("E", True)]


def test_a_resumed_turn_confirms_the_effect_on_the_stored_reference_before_any_twist(monkeypatch):
    """Blocker-1 spec BR-4: the resume pre-check uses confirm() on a fresh watch of the stored reference (static
    frames, no lift): met -> succeeded turned_by_effect, flag effect_held_before_twist, no grasp, the stays entry."""
    script = TurnScript(monkeypatch, segments=[], hinges=[], effect_region=effect_mask())
    watch = ScriptedWatch(["effect_confirming", "effect_confirming", "effect_confirmed"])
    stored = ScriptedWatch([])
    monkeypatch.setattr(R, "_fresh_watch", lambda ctx, s: watch)
    ctx = make_ctx([part("A")])
    ctx.memory["turned"] = {"A": {"pid": "A", "free_sense": -1, "theta_part_rad": 1.6, "theta_net_rad": 1.9}}
    ctx.memory["effect_ref"] = {"A": {"pid": "B", "text": R._text_of(ctx, "A"), "watch": stored, "step": 3}}
    outcome = drive(R.turn(ctx, effect_turn_step()), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "turned_by_effect")
    assert outcome.evidence["effect_held_before_twist"] and script.log == []
    assert [built(ctx, f).name for f in ctx.memory["yielded"]] == ["effect_confirm_hold"] * 3
    assert ctx.stays.keys() == ["stays:turned:A"]
    # a stored watch of another effect part is not used
    assert R._stored_effect(ctx, turn_step(Until("effect", "E", None))) is None


def test_the_phase_validity_logs_the_scene_hand_mask_rule():
    from gpt_field.agent_speedup_v2_observers import EffectWatch
    ctx = make_ctx()
    ctx.scene.mask_log = {"hand_swept": {"model": "palm_and_fingers", "rule": "local", "margin_px": 25}}
    effect = EffectWatch(region=effect_mask(), false_rate=0.0, sigma_px=0.4)
    validity = R._sense_validity(ctx, effect, sense=-1, steps=40, exclude=[], phase="free")
    assert validity["hand_mask"] == {"model": "palm_and_fingers", "rule": "local", "margin_px": 25}
    json.dumps(kernel.jsonable(validity), allow_nan=False)


def test_fake_carry_chain_with_the_pointing_frame_completes(monkeypatch):
    """Blocker-2 BP-1 / BR-2 on the fake: the grip part's pointing hover is framed (hover_for pointing="auto"), and
    the carry still completes (with repairs: the framed view's measurement of the thin part needs retries on the
    fake, see the unframed regression above)."""
    fake_body_start_region(monkeypatch)
    world = fakes.FakeTwoViewSim(fakes.free_object_with_thin_part_scene(), prompt=fakes.TASK2_PROMPT)
    policy, client = fake_policy(world, CARRY_PLAN)
    run_world(policy, world, 2400, stop=lambda p, w: p.ctx.stays.keys() == ["stays:supported_on:D:B"])
    relations = [(r["relation"], r["status"], r["reason"]) for r in policy.outcomes]
    assert relations[-1] == ("supported_on", "succeeded", "placed"), relations
    held = [r for r in policy.outcomes if r["relation"] == "held"]
    framing = held[0]["evidence"]["hover"]["hover"]["framing"]
    assert framing is not None and framing["ok"] and framing["mode"] in ("fine", "coarse")
    for record in policy.outcomes:
        assert record["reason"] in kernel.REASONS and len(record["evidence_text"]) <= kernel.EVIDENCE_TEXT_MAX_CHARS


# ============================================================ grasp-refusal decision J (PC-1, GC-1 handling, CJ-1)
REAL_CLOSE_AND_CLASSIFY = R.close_and_classify
FAKE_GRIPPER = GripperCalibration(fakes.START_GAP_MM, fakes.CLOSE_CURVE_MM[1:], 0.002, "online")


def gate_for(xy, jaw_xy, *, r1_mm=20.0, r2_mm=20.0, span_hi_mm=27.3):
    """A planner checks["gate"] (planner.close_gate's keys) evaluated at ``xy``."""
    return {"fingers": {"finger1": {"F_m": (r1_mm - 1.0) / 1000.0, "R_m": r1_mm / 1000.0},
                        "finger2": {"F_m": (r2_mm - 1.0) / 1000.0, "R_m": r2_mm / 1000.0}},
            "grip_xy": [float(v) for v in xy], "jaw_xy": [float(v) for v in jaw_xy], "span_hi_mm": span_hi_mm,
            "tol_mm": 1.0, "pad_face_offset_mm": robot_facts.PAD_SEPARATION_OFFSET_MM / 2.0}


def gated_plan(view, *, offset_xy=(0.0, 0.0), marginal=False, pid="A", yaw=None, **gate_kw):
    """contact_plan along the hand's own jaw, its checks evaluated at the hand's xy + ``offset_xy``."""
    jaw = np.asarray(view.R[:2, 1], dtype=float)
    yaw = math.atan2(float(jaw[1]), float(jaw[0])) if yaw is None else yaw
    plan = contact_plan(pid, yaw=yaw, marginal=marginal)
    xy = np.asarray(view.position[:2], dtype=float) + np.asarray(offset_xy, dtype=float)
    return dataclasses.replace(plan, grip_point_xyz=(float(xy[0]), float(xy[1]), 0.955),
                               checks={"gate": gate_for(xy, (math.cos(yaw), math.sin(yaw)), **gate_kw)})


def close_world(bodies=()):
    """A top-down hand 5 cm above the support (jaw R1 = world -y) over the given boxes (tests' gc geometry)."""
    return fakes.FakeTwoViewSim(fakes.FakeScene(list(bodies)), eef=(-0.10, 0.10, 0.95), start_tilt_deg=0.0,
                                images=False)


def run_primitive(world, primitive, max_steps=200):
    view, actions = StepView(world.observation()), []
    while not primitive.done and len(actions) < max_steps:
        action = primitive.next_action(view)
        actions.append(action)
        after = StepView(world.step(action))
        primitive.observe_transition(view, action, after)
        view = after
    assert primitive.done
    return primitive.outcome(), actions


def on_world(world):
    """drive() hooks that build each yielded primitive on ``world``'s current view and run it there to the end."""
    ran = {}

    def primitive_for(factory):
        primitive = factory(StepView(world.observation()))
        ran["outcome"], ran["actions"] = run_primitive(world, primitive)
        ran["primitive"] = primitive
        return primitive
    return ran, primitive_for, (lambda factory: ran["outcome"])


def bar_body(width_m=0.0243):
    return fakes.FakeBody("bar", "aabb", (-0.10, 0.10, 0.9), (0.03, width_m, 0.07))


def wall_body(s_in_m, s_out_m, finger):
    side = 1.0 if finger == 1 else -1.0                       # finger 1 on +jaw = world -y here
    return fakes.FakeBody(f"wall{finger}", "aabb", (-0.10, 0.10 - side * (s_in_m + s_out_m) / 2.0, 0.9),
                          (0.03, s_out_m - s_in_m, 0.07))


def close_ctx(world):
    ctx = make_ctx([part("A")])
    ctx.view = StepView(world.observation())
    ctx.cal.gripper = FAKE_GRIPPER
    return ctx


def no_re_evaluation(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("PC-1 re-evaluated a hand within the envelope")
    monkeypatch.setattr(R.planner, "evaluate_candidate", refuse)


def test_pc1_within_the_envelope_closes_with_the_v1_action_stream(monkeypatch):
    """PC-1: a hand within e_lat of the gate's grip point leaves the plan and the close unchanged: the close is
    GripperSettlePrimitive("close", GRASP_SETTLE)'s action stream (GC-1 monitor-only on a clean plan), then
    classify_close and G13 as before; the envelope and the guard are logged as "contact" events."""
    no_re_evaluation(monkeypatch)
    world = close_world([bar_body()])
    ctx = close_ctx(world)
    plan = gated_plan(ctx.view, offset_xy=(0.003, -0.004))                  # 4 mm < e_lat 8.5 mm
    ran, primitive_for, reply = on_world(world)
    closed = drive(R.close_and_classify(ctx, plan, part("A")), ctx, primitive_for=primitive_for, factory_reply=reply)
    twin = close_world([bar_body()])
    _, plain = run_primitive(twin, R.GripperSettlePrimitive("close", StepView(twin.observation()),
                                                            config=R.GRASP_SETTLE, name="grasp_close"))
    assert ran["actions"] == plain and isinstance(ran["primitive"], primitives.GuardedClose)
    assert not ran["primitive"].active                                      # clean plan: monitor-only
    assert closed["present"] and not closed.get("aborted") and closed["plan"] is plan
    assert closed["gap_mm"] == pytest.approx(24.3, abs=1.5) and closed["width"] is not None
    assert closed["envelope"]["ok"] and not closed["envelope"]["re_evaluated"]
    phases = [p.get("phase") for name, p in ctx.events.rows if name == "contact"]
    assert phases == ["close_envelope", "close_guard"]


def test_gc1_an_early_finger_contact_aborts_the_close_and_opens_before_any_classification(monkeypatch):
    """Safety rule: never close when a finger meets an unexpected surface before the part. A marginal plan (GC-1
    active) whose finger 1 meets a wall at 25 mm against R1 = 15 mm: the close is abandoned and the gripper opened
    (the part is never clamped), close_and_classify returns aborted early_finger_contact with the close height."""
    no_re_evaluation(monkeypatch)
    world = close_world([bar_body(), wall_body(0.015, 0.025, 1)])
    ctx = close_ctx(world)
    plan = gated_plan(ctx.view, marginal=True, r1_mm=15.0)
    ran, primitive_for, reply = on_world(world)
    closed = drive(R.close_and_classify(ctx, plan, part("A")), ctx, primitive_for=primitive_for, factory_reply=reply)
    assert ran["primitive"].active and closed["aborted"] == "early_finger_contact" and not closed["present"]
    assert closed["guard"]["verdict"][1]["finger"] == 1 and closed["width"] is None
    assert closed["close_grip_z"] == pytest.approx(0.95)
    assert not [c for c in world.contact_log if c["kind"] == "clamp"] and world.gap_mm > 70.0
    # the same close on a clean plan only records what it would have done (and clamps the part)
    world = close_world([bar_body(), wall_body(0.015, 0.025, 1)])
    ctx = close_ctx(world)
    ran, primitive_for, reply = on_world(world)
    closed = drive(R.close_and_classify(ctx, gated_plan(ctx.view, r1_mm=15.0), part("A")), ctx,
                   primitive_for=primitive_for, factory_reply=reply)
    assert not closed.get("aborted") and closed["guard"]["would_abort"]


def test_gc1_a_preemption_during_the_abort_open_still_abandons_the_close(monkeypatch):
    """Reviewer fix (decision J abort handling): a bus preemption delivered while an active guard is opening after
    its verdict (the policy's §C3.3 synthetic outcome: reason "preempted:...", no primitive evidence) must not be
    classified as a close: the half-open hand on an unexpected surface is no grasp. close_and_classify returns the
    guard's abort reason with the guard's evidence, never classify_close / G13."""
    no_re_evaluation(monkeypatch)
    world = close_world([bar_body(), wall_body(0.015, 0.025, 1)])
    ctx = close_ctx(world)
    plan = gated_plan(ctx.view, marginal=True, r1_mm=15.0)
    ran = {}

    def primitive_for(factory):
        primitive = factory(StepView(world.observation()))
        view = StepView(world.observation())
        while primitive.opening is None:                    # run to the verdict, then one abort-open action
            action = primitive.next_action(view)
            after = StepView(world.step(action))
            primitive.observe_transition(view, action, after)
            view = after
        action = primitive.next_action(view)
        primitive.observe_transition(view, action, StepView(world.step(action)))
        assert not primitive.done
        ran["primitive"] = primitive
        return primitive

    def preempted(factory):
        actions = int(ran["primitive"].actions_executed)
        return {"primitive": "grasp_close", "status": "failed", "reason": "preempted:keepout_contact",
                "evidence": {"invariant": "keepout:B", "report": {}, "actions": actions}, "actions": actions}
    monkeypatch.setattr(R, "classify_close", lambda *a, **k: pytest.fail("a preempted abort was classified"))
    closed = drive(R.close_and_classify(ctx, plan, part("A")), ctx, primitive_for=primitive_for,
                   factory_reply=preempted)
    assert closed["aborted"] == "early_finger_contact" and not closed["present"] and closed["width"] is None
    assert closed["guard"]["verdict"][0] == "early_finger_contact" and closed["guard"]["active"]
    assert closed["close_grip_z"] == pytest.approx(0.95)


def test_pc1_a_drifted_hand_is_re_evaluated_where_it_is_and_never_closes_when_refused(monkeypatch):
    """PC-1: a hand 20 mm across the jaw from the gate's point (e_lat 8.5 mm) re-runs the same (family, yaw, depth
    rule) at the realized xy with the plan's inputs; a refusal there returns pose_out_of_envelope with no close
    action; a ContactPlan there replaces the plan (its gate), the plan's grip point kept."""
    ctx = make_ctx([part("A")])
    view = ctx.view
    jaw = np.asarray(view.R[:2, 1], dtype=float)
    across = np.array([-jaw[1], jaw[0]])
    plan = gated_plan(view, offset_xy=-0.020 * across)
    calls = []

    def refused(record, hf, inputs, family, yaw, keepouts, **kw):
        calls.append(dict(kw, family=family, yaw=yaw))
        return planner.CandidateRejection(family, yaw, "height_stop:P6", "column_blocked", {}, ())
    monkeypatch.setattr(R.planner, "evaluate_candidate", refused)
    closed = drive(R.close_and_classify(ctx, plan, part("A"), purpose="turn", obj="A"), ctx)
    assert closed["aborted"] == "pose_out_of_envelope" and not closed["present"]
    assert not any(callable(item) for item in ctx.memory.get("yielded", []))   # no close action at all
    env = closed["envelope"]
    assert env["re_evaluated"] and not env["ok"] and env["dv_m"] == pytest.approx(0.020, abs=1e-9)
    assert env["rejection"] == {"rule": "height_stop:P6", "reason": "column_blocked"}
    (call,) = calls
    assert call["family"] == plan.family and call["yaw"] == pytest.approx(plan.yaw_rad)
    assert call["depth_rules"] == (plan.depth_rule,) and call["purpose"] == "turn"
    np.testing.assert_allclose(call["grip_xy"], view.position[:2])
    # feasible where the hand is: the re-evaluated plan (and its gate) is closed on, the grip point kept
    again = dataclasses.replace(gated_plan(view), score=2.0)
    monkeypatch.setattr(R.planner, "evaluate_candidate", lambda *a, **kw: again)
    closed = drive(R.close_and_classify(ctx, plan, part("A"), purpose="turn", obj="A"), ctx)
    assert closed["envelope"]["re_evaluated"] and closed["envelope"]["ok"]
    assert closed["plan"].score == 2.0 and closed["plan"].grip_point_xyz == plan.grip_point_xyz
    assert closed["plan"].checks["gate"] == again.checks["gate"]


def test_grasp_pc1_drift_excludes_the_candidate_counts_retries_and_never_closes(monkeypatch):
    """Grasp loop: every close refused by PC-1 (pose_out_of_envelope) returns to the hover, excludes the candidate
    and counts a B2 retry; after B2 the grasp fails pose_out_of_envelope, its evidence text naming the offsets."""
    ctx = make_ctx()
    jaw = np.asarray(ctx.view.R[:2, 1], dtype=float)
    yaws = [math.atan2(jaw[1], jaw[0]) + k * kernel.YAW_GRID_RAD for k in range(4)]
    plans = [gated_plan(ctx.view, offset_xy=-0.020 * np.array([-math.sin(y), math.cos(y)]), yaw=y) for y in yaws]
    script = GraspScript(monkeypatch, plans=plans, classes=["expected"] * 4)
    monkeypatch.setattr(R, "close_and_classify", REAL_CLOSE_AND_CLASSIFY)
    monkeypatch.setattr(R.planner, "evaluate_candidate", lambda record, hf, inputs, family, yaw, keepouts, **kw:
                        planner.CandidateRejection(family, yaw, "height_stop:P6", "column_blocked", {}, ()))
    outcome = drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "pose_out_of_envelope")
    assert outcome.local_retries == R.B2_LOCAL_RETRIES + 1
    assert not any(callable(item) for item in ctx.memory["yielded"])          # no close action was ever issued
    assert script.excludes[1:] == [[p.candidate_key() for p in plans[:k]] for k in range(1, R.B2_LOCAL_RETRIES + 1)]
    assert script.names().count("return_to_hover") == R.B2_LOCAL_RETRIES + 1
    assert [a["recovery"] for a in outcome.evidence["attempts"]] == ["pose_out_of_envelope"] * 3
    assert "hand offset across jaw 2.0 cm" in outcome.evidence_text


def touch_guard(finger=1, s_mm=25.0, limit_mm=15.0):
    return {"active": True, "verdict": ["early_finger_contact", {"finger": finger, "s_mm": s_mm,
                                                                 "limit_mm": limit_mm, "k": 6}],
            "touch": {str(finger): {"k": 6, "s_mm": s_mm, "deficit_mm": 4.6}}}


def test_grasp_early_finger_contact_excludes_the_candidate_and_its_twin_and_keeps_out_the_touch(monkeypatch):
    """GC-1 early_finger_contact: back to the hover, the candidate AND its twin excluded, a grasp-scoped clear_of
    keep-out over the touched cell that every later ``_plan`` of this grasp receives (never the scene's), and no
    retry counted; the next candidate is then held."""
    ctx = make_ctx()
    ctx.scene.hf = kernel.HeightField.empty(ctx.view.position[:2], half_extent_m=0.2)
    first = gated_plan(ctx.view, marginal=True)
    second = gated_plan(ctx.view, yaw=first.yaw_rad + kernel.YAW_GRID_RAD)
    abort = {"aborted": "early_finger_contact", "guard": touch_guard(), "close_grip_z": 0.944}
    script = GraspScript(monkeypatch, plans=[first, second], classes=["expected"] * 2, closes=[abort, (True, False)])
    outcome = drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("succeeded", "held_for_turn") and outcome.local_retries == 0
    assert script.excludes[1] == [first.candidate_key(), R._twin(first)]
    assert script.keepouts[0] == [] and len(script.keepouts[1]) == 1
    keepout = script.keepouts[1][0]
    assert isinstance(keepout, kernel.KeepOut) and keepout.kind == "clear_of"
    assert keepout.z_high == pytest.approx(0.944 - ROBOT.pad_span_m[1])        # the pad bottom at the close height
    assert keepout.z_low == pytest.approx(part().support_z.value)              # the record's measured support
    # one cell along the jaw ending at the touch, on finger 1's side (+jaw) of the gate's grip point
    jaw = np.array([math.cos(first.yaw_rad), math.sin(first.yaw_rad)])
    along = (keepout.polygon_xy - np.asarray(first.checks["gate"]["grip_xy"])) @ jaw
    assert along.max() == pytest.approx(0.025) and along.min() == pytest.approx(0.025 - ctx.scene.hf.cell_m)
    assert not ctx.scene.added_keepouts and not ctx.scene.keepouts           # grasp-scoped only
    assert script.close_kwargs[0]["extra_keepouts"] == [] and len(script.close_kwargs[1]["extra_keepouts"]) == 1
    assert outcome.evidence["attempts"][0]["recovery"] == "early_finger_contact"


def test_grasp_span_inconsistent_excludes_the_twin_and_counts_a_b2_retry(monkeypatch):
    ctx = make_ctx()
    yaws = [-math.pi / 2 + k * kernel.YAW_GRID_RAD for k in range(4)]
    plans = [gated_plan(ctx.view, marginal=True, yaw=y) for y in yaws]
    span = {"aborted": "span_inconsistent", "guard": {"active": True, "verdict": [
        "span_inconsistent", {"span_mm": 40.2, "span_hi_mm": 35.7, "k": 7}]}, "close_grip_z": 0.944}
    script = GraspScript(monkeypatch, plans=plans, classes=["expected"] * 4, closes=[span] * 4)
    outcome = drive(R.grasp(ctx, "A", obj_pid="A", purpose="turn"), ctx)
    assert (outcome.status, outcome.reason) == ("failed", "no_feasible_candidate")
    assert outcome.local_retries == R.B2_LOCAL_RETRIES + 1 and outcome.evidence["next_grip"]
    assert outcome.evidence["flags"] == ["span_inconsistent"]
    assert script.excludes[1] == [plans[0].candidate_key(), R._twin(plans[0])]
    assert script.names().count("close") == R.B2_LOCAL_RETRIES + 1 and all(k == [] for k in script.keepouts)
    assert "finger span 4.0 cm" in outcome.evidence_text and "span inconsistent" in outcome.evidence_text


def test_cj1_the_point_check_jaw_of_an_infeasible_part_is_the_representative_rejection(monkeypatch):
    """CJ-1: no feasible plan -> the jaw of planner.representative_rejection (the depth rejection with the highest
    by_rule progress; orient / width / part / axis rejections never), labelled would_plan; none -> (None, reason)."""
    def rejection(yaw, rule, progress):
        return planner.CandidateRejection("top_down", yaw, rule, "column_blocked",
                                          {"by_rule": {"height_stop": {"progress": progress}}}, ())
    rejections = (rejection(0.1, "orient:P12", 0.99), rejection(0.4, "height_stop:P5", 0.6),
                  rejection(-1.2, "touch_stop:P6", 0.8), rejection(2.0, "width:P2", 0.95))
    result = planner.Infeasible("C", "column_blocked", rejections, ())
    monkeypatch.setattr(R, "_plan", lambda ctx, record, **kw: result)
    ctx = make_ctx()
    record = part("C", move="free", elongated=False)
    assert R._check_jaw(ctx, record, kind="grasp_point", purpose="carry", obj="D") == (-1.2, "would_plan")
    only_width = planner.Infeasible("C", "geometry_unmeasured", (rejection(2.0, "width:P2", 0.95),), ())
    monkeypatch.setattr(R, "_plan", lambda ctx, record, **kw: only_width)
    assert R._check_jaw(ctx, record, kind="grasp_point", purpose="carry", obj="D") == (None, "geometry_unmeasured")


# --------------------------------------------------- the same rules with the real planner (a lever on a disk)
def lever_ctx(*, view_xy=(0.0, 0.004), obstacle=None):
    """The planner tests' lever scene in a relation context: a 24.3 mm ridge along x (top 0.955) over a 47.6 mm disk
    (top 0.918) on the 0.900 support; the hand straight down at 1.10 m with its jaw along -y; height_stop forced
    (its target 0.939 lands the finger columns on the disk). ``obstacle`` = (polygon, top z) stamped as measured."""
    hf = kernel.HeightField.empty((0.0, 0.0), half_extent_m=0.3)
    hf.z[:], hf.sigma[:], hf.known[:], hf.source[:], hf.stamp[:] = 0.900, 0.001, True, kernel.hf_code("kappa"), 0

    def stamp(poly, z):
        cells = hf.cells_in_polygon(poly)
        hf.z[cells[:, 0], cells[:, 1]] = z
        return cells
    a = np.arange(64) * 2.0 * math.pi / 64
    stamp(0.0476 * np.column_stack([np.cos(a), np.sin(a)]), 0.918)
    ridge = kernel.rect_polygon((0.0, 0.004), (1.0, 0.0), 0.0475, 0.01215)
    cells = stamp(ridge, 0.955)
    if obstacle is not None:
        stamp(*obstacle)
    widths = []
    for k in range(12):
        psi = k * kernel.YAW_GRID_RAD
        s, c = abs(math.sin(psi)), abs(math.cos(psi))
        widths.append((psi, Meas(min(0.0243 / s if s > 1e-9 else math.inf, 0.095 / c if c > 1e-9 else math.inf),
                                 0.001, "kappa")))
    record = PartRecord(pid="A", text="part A", move_prior="hinged", roles=frozenset({"grip"}),
                        xyz=np.array([0.010, 0.004, 0.955]), cov=np.eye(3) * 0.002 ** 2,
                        top_z=Meas(0.955, 0.0025, "kappa"), support_z=Meas(0.918, 0.001, "kappa"), region_xy=ridge,
                        region_cells=cells, width_by_yaw=tuple(widths), elongation=(np.array([1.0, 0.0]), 0.9),
                        circle=(np.array([0.0, 0.0]), 0.0476, 0.5), status="registered", registration_step=0,
                        sigma_h_m=0.0025)
    ctx = make_ctx([record], depth_rule="height_stop")
    ctx.scene.hf = hf
    jaw, approach = np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    q = matrix_to_quaternion_xyzw(np.column_stack([np.cross(jaw, approach), jaw, approach]))
    v = np.asarray(q[:3], float)
    rotvec = v / float(np.linalg.norm(v)) * (2.0 * math.atan2(float(np.linalg.norm(v)), float(q[3])))
    state = (*view_xy, 1.10, *rotvec, 0.03995, -0.03995)
    blank = fakes._blank_url()
    ctx.view = StepView(PolicyObservation(blank, blank, tuple(float(x) for x in state), "synthetic"))
    return ctx, record


def test_pc1_with_the_real_planner_a_real_obstacle_where_the_hand_drifted_still_refuses():
    """Negative (no false admission): the plan at the lever is feasible; the hand drifted 20 mm along the lever
    (across the jaw) over clear disk is re-evaluated there and still closes; drifted onto a real obstacle beside the
    part (a 0.935 block under finger 2's column, above the 0.933 landing limit, outside the plan's own inflated
    columns) it is refused (pose_out_of_envelope, no close)."""
    block = (kernel.rect_polygon((0.032, 0.004 + 0.047), (1.0, 0.0), 0.012, 0.016), 0.935)
    ctx, record = lever_ctx(obstacle=block)
    plan = R._plan(ctx, record, purpose="turn", obj="A")
    assert isinstance(plan, planner.ContactPlan) and plan.depth_rule == "height_stop", plan
    assert plan.grip_point_xyz[:2] == pytest.approx((0.0, 0.004), abs=1e-9)
    for drift_x, obstacle, ok in ((0.020, None, True), (0.020, block, False)):
        ctx, record = lever_ctx(view_xy=(drift_x, 0.004), obstacle=obstacle)
        env = R._realized_envelope(ctx, plan, record, purpose="turn", obj="A")
        assert env["re_evaluated"] and env["ok"] is ok, (drift_x, env.get("rejection"))
        if ok:
            assert env["plan"].grip_point_xyz == plan.grip_point_xyz
        else:
            closed = drive(R.close_and_classify(ctx, plan, record, purpose="turn", obj="A"), ctx)
            assert closed["aborted"] == "pose_out_of_envelope"
            assert not any(callable(item) for item in ctx.memory.get("yielded", []))


def test_a_touch_keepout_refuses_the_candidates_whose_sweep_crosses_it_in_the_next_plan():
    """The GC-1 touch keep-out is an obstacle the next ``_plan`` of the grasp sees: the lever plan is feasible
    without it and refused (P5, keep-out cells are never skipped) with a keep-out where finger 1 touched at 20 mm."""
    ctx, record = lever_ctx()
    plan = R._plan(ctx, record, purpose="turn", obj="A")
    assert isinstance(plan, planner.ContactPlan)
    keepout = R._touch_keepout(ctx, plan, touch_guard(1, 20.0, 15.0), record, grip_z=plan.target_grip_z)
    assert keepout.z_high == pytest.approx(plan.target_grip_z - ROBOT.pad_span_m[1])
    inputs, keepouts, kwargs = R._plan_inputs(ctx, record, purpose="turn", obj="A", extra_keepouts=[keepout])
    assert keepouts[-1] is keepout
    again = planner.evaluate_candidate(record, ctx.scene.hf, inputs, plan.family, plan.yaw_rad, keepouts,
                                       depth_rules=(plan.depth_rule,), **kwargs)
    assert isinstance(again, planner.CandidateRejection), again
    clear = planner.evaluate_candidate(record, ctx.scene.hf, inputs, plan.family, plan.yaw_rad, keepouts[:-1],
                                       depth_rules=(plan.depth_rule,), **kwargs)
    assert isinstance(clear, planner.ContactPlan)
