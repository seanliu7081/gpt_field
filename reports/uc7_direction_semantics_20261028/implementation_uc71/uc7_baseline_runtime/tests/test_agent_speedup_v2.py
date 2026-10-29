"""T1 tests of the agent_SpeedUp_v2 policy and StepRunner (contract §C13, §C3; spec §4.2, §4.3, §4.11, §3.7).

No network, no simulator, no model: every client is the real v2 client over ``ScriptedV2Transport`` (offline mock)
and the world is ``FakeTwoViewSim`` (physics only, ``images=False``). The relation layer (BR, Wave 2) is replaced by
``StandIn`` below, written against the contract's BR API (§C10: ``RelationContext``, ``calibrate_plus``,
``register_regions``, ``CONTROLLERS``, ``release``, ``evidence_text``, ``CalibratePlus``); its controllers yield
the contract's Needs and real primitives and return scripted outcomes. The end-to-end test with the real relations
(§C13 test 7) skips until BR lands.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from dataclasses import dataclass, field
import inspect
import json

import numpy as np
import pytest

import agent_speedup_v2_fakes as fakes
from gpt_field import agent_speedup_v2 as policy_module
from gpt_field import agent_speedup_v2_thresholds as th
from gpt_field import agent_speedup_v2_types as kernel
from gpt_field.agent_speedup_v2_client import AgentSpeedUpV2Client, FixedAnswers, ScriptedV2Transport
from gpt_field.agent_speedup_v2_primitives import CalibratePlus
from gpt_field.agent_speedup_v2_tasks import list_v2_tasks, resolve_v2_task
from gpt_field.contact_skills import HoldPrimitive, Primitive, StepView

LIBERO = "/workspace/heading_flow_pub/third_party/LIBERO"
POLICY_PATH = th.PACKAGE_DIR / "agent_speedup_v2.py"
TASK2 = resolve_v2_task("libero_10_t2").prompt
OTHER = next(spec.prompt for spec in list_v2_tasks() if spec.prompt != TASK2)
AgentSpeedUpV2Policy, StepRunner = policy_module.AgentSpeedUpV2Policy, policy_module.StepRunner

# §3.9 task-2 example (test fixture only; the policy module holds no plan or task text)
PARTS = [
    {"id": "A", "text": "stove knob: lever on top of the knob", "move": "hinged", "w": 31, "x": 30},
    {"id": "B", "text": "stove: ring burner plate", "move": "fixed", "w": 47, "x": 39},
    {"id": "C", "text": "moka pot: black handle loop", "move": "free", "w": 59, "x": 36},
    {"id": "D", "text": "moka pot: whole pot", "move": "free", "w": 60, "x": 44},
    {"id": "E", "text": "frying pan: long handle", "move": "free", "w": 43, "x": 35}]
TURN = {"do": "turn", "obj": "A", "grip": ["A"], "ref": None, "sense": "any", "keep": [],
        "until": [{"o": "effect", "p": "B", "txt": "burner rings turn red"}, {"o": "stop", "p": None, "txt": None}]}
PLACE = {"do": "place_on", "obj": "D", "grip": ["C"], "ref": "B", "sense": "any",
         "keep": [{"c": "upright", "p": "D"}, {"c": "clear_of", "p": "E"}], "until": []}
HOLD_D = {"do": "hold", "obj": "D", "grip": ["C"], "ref": None, "sense": "any", "keep": [], "until": []}
SCENE_TEXT = "stove off; black knob with lever below the stove; moka pot at top edge"


def plan_json(steps=(TURN, PLACE), *, hold_true=(), parts=PARTS, scene=SCENE_TEXT):
    return {"scene": scene, "parts": copy.deepcopy(list(parts)), "hold_true": copy.deepcopy(list(hold_true)),
            "steps": copy.deepcopy(list(steps))}


TASK2_PLAN = plan_json()


def repair(action, *, part=None, step=None, why="scripted repair"):
    return {"why": why, "action": action, "part": part, "step": copy.deepcopy(step)}


HOLD_STILL = repair("hold_still")


# ------------------------------------------------------------------------------------------- the stand-in BR
def outcome(relation, status, reason, pids=None, *, resume=None, steps=0):
    return kernel.RelationOutcome(relation, status, reason, dict(pids or {}), {"stand_in": True},
                                  [kernel.th_value("STANDIN_check", 1.0, "DD", "stand-in comparison")],
                                  f"{reason}: stand-in", 0, steps, resume=resume)


@dataclass
class StandInContext:
    """The contract §C10 RelationContext fields and methods."""
    robot: object
    scene: object
    monitor: object
    stays: object
    events: object
    view: object = None
    step: int = 0
    cal: object = None
    held_estimator: object = None
    holding: object = None
    grasp_quaternion: object = None
    hold_gap_ref_mm: object = None
    plan: object = None
    roles: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)
    active_relation: object = None
    pending_outcomes: list = field(default_factory=list)
    consecutive_rejections: object = None

    def record(self, item):
        self.pending_outcomes.append(item)
        return item

    def note_rejection(self, kind):
        self.consecutive_rejections = (self.consecutive_rejections or 0) + 1

    def note_accepted(self, kind):
        self.consecutive_rejections = 0


class StandInExternal:
    def image_right_world_xy(self):
        return np.array([0.0, 1.0])            # world +y is image right: k = 2 at the task-2 start (spec §3.6)


class StandInCal:
    quarter_turns = 2
    mount = {"frontal": False}

    def __init__(self):
        self.external = StandInExternal()
        self.sigma_px, self.false_change = {}, {}

    def summary(self):
        return {"stand_in": True, "quarter_turns": self.quarter_turns}


@dataclass
class HeldStub:
    pid: str
    grip_pid: str


class StaysFlag:
    """A controllable stays check (verdict met / violated / pending)."""

    def __init__(self, verdict="met"):
        self.verdict, self.calls = verdict, 0

    def __call__(self, view):
        self.calls += 1
        return kernel.ObsReport("stand_in_stays", self.verdict, None,
                                "kept_relation_lost" if self.verdict == "violated" else None, {"calls": self.calls},
                                (kernel.th_value("STANDIN_stays", 1.0, "DD", "stand-in stays"),), self.calls)


class FlagObserver:
    name = "flag"

    def __init__(self):
        self.verdict, self.contexts = "met", []

    def start(self, view):
        pass

    def update(self, before, action, after, context=None):
        self.contexts.append(dict(context or {}))
        return kernel.ObsReport("flag", self.verdict, None, "flag_violated" if self.verdict == "violated" else None,
                                {}, (), len(self.contexts))


class ScriptedMove(Primitive):
    """Test primitive: ``steps`` actions, the first half down by ``dz`` and the rest back up (net zero)."""

    def __init__(self, view, *, steps, dz, name="scripted_move"):
        super().__init__(name)
        self.n, self.dz = int(steps), float(dz)
        if self.n == 0:
            self._finish("succeeded", "goal_reached", {})

    def _next(self, view):
        down = self.actions_executed < self.n // 2
        return (0.0, 0.0, self.dz if down else -self.dz, 0.0, 0.0, 0.0, -1.0)

    def _observe(self, before, action, after):
        if self.actions_executed >= self.n:
            self._finish("succeeded", "goal_reached", {"z": float(after.position[2])})
        return {}


SUCCESS = {"turn": ("turned", "turned_by_effect"), "place_on": ("supported_on", "placed"),
           "hold": ("held", "hold_pass")}


class NetMove(Primitive):
    """Test primitive: ``steps`` actions all along z by ``dz`` (a net displacement of the hand)."""

    def __init__(self, view, *, steps, dz, name="net_move"):
        super().__init__(name)
        self.n, self.dz = int(steps), float(dz)

    def _next(self, view):
        return (0.0, 0.0, self.dz, 0.0, 0.0, 0.0, -1.0)

    def _observe(self, before, action, after):
        if self.actions_executed >= self.n:
            self._finish("succeeded", "goal_reached", {"z": float(after.position[2])})
        return {}


class StandIn:
    """Stand-in relation layer. ``script[do]`` = per-attempt specs {"status", "reason", "resume", "points",
    "moves", "hold", "evidence", "checks", "net_move"} (popped per call; an empty list = the default success);
    ``points[do]`` = [(kind, pid|None)]; ``stays[do]`` = a StaysFlag registered on success as
    stays:<relation>:<obj>. ``evidence`` is merged into the outcome's evidence; ``checks`` = [(pid, uv, ok)] emits one
    relation-layer "pointing" check event each (contract §C18.1 payload) after the move; ``net_move`` = steps of a
    NetMove after the checks (the hand leaves the pose the checks were made at)."""

    RELATIONS_VERSION = "stand-in-relations"
    RelationContext = StandInContext
    CalibratePlus = CalibratePlus

    def __init__(self, *, real_calibration=False, script=None, points=None, stays=None, moves=4):
        self.real_calibration = real_calibration
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.points = {"turn": [("point_fine", None)], "place_on": [("point_fine", None)], "hold": []}
        self.points.update(points or {})
        self.stays = dict(stays or {})
        self.moves = moves
        self.log, self.sent = [], []
        self.CONTROLLERS = {do: self._controller(do) for do in SUCCESS}
        self.CONTROLLERS["release"] = self.release

    def evidence_text(self, item):
        return f"{item.reason}: part {item.pids.get('obj')}"[:200]

    def calibrate_plus(self, ctx):
        self.log.append(("calibrate_plus",))
        if self.real_calibration:
            reply = yield (lambda view: CalibratePlus(view, robot=ctx.robot))
        else:
            reply = yield (lambda view: HoldPrimitive(view, steps=2, gripper="open", name="stand_in_calibration"))
        self.sent.append(reply)
        ctx.cal = StandInCal()
        ctx.record(outcome("calibrated", "succeeded", "calibrated_online"))
        return ctx.cal

    def register_regions(self, ctx, plan, roles):
        named = tuple(sorted(pid for pid, r in roles.items() if r))
        self.log.append(("register_regions", named))
        out = []
        for pid in named:
            part = ctx.scene.parts.get(pid)
            if part is None:
                continue
            region = bool(set(roles[pid]) & {"carried", "ref", "effect", "keepout", "hold_true"})
            changes = {"xyz": np.array([-0.2, 0.02, 0.95]), "sigma_h_m": 0.003, "registration_step": int(ctx.step)}
            changes.update({"status": "registered"} if region else {"provisional": True, "status": "seeded"})
            ctx.scene.parts[pid] = dataclasses.replace(part, **changes)
            out.append(ctx.record(outcome("registered", "succeeded", "registered", {"obj": pid})))
        return out
        yield                                   # a generator, like the real controller

    def release(self, ctx, step):
        self.log.append(("release", None if step is None else step.i))
        reply = yield (lambda view: HoldPrimitive(view, steps=2, gripper="open", name="stand_in_release"))
        self.sent.append(reply)
        ctx.holding = None
        return ctx.record(outcome("released", "succeeded", "released"))

    def _controller(self, do):
        def controller(ctx, step):
            self.log.append(("controller", do, step.i, step.obj, tuple(step.grip), step.attempt, step.resume))
            queue = self.script.get(do, [])
            spec = queue.pop(0) if queue else {}
            uv = None
            for kind, pid in spec.get("points", self.points.get(do, [])):
                pid = pid or step.grip[0]
                need = kernel.PointNeed(kind, pid, ctx.plan.part(pid).text, "wrist",
                                        (256.0, 256.0) if kind == "point_fine" else None, ctx.cal.quarter_turns,
                                        "grasp_point")
                answer = yield need
                self.sent.append(answer)
                if answer.reply is not None and answer.reply.uv_upright is not None:
                    uv = answer.reply.uv_upright
            dz = -0.2 if uv is not None and uv[0] > 256.0 else -0.1
            moves = spec.get("moves", self.moves)
            reply = yield (lambda view: ScriptedMove(view, steps=moves, dz=dz))
            self.sent.append(reply)
            for k, (pid, check_uv, ok) in enumerate(spec.get("checks", ())):
                ctx.events.emit("pointing", kind="check", pid=pid, camera="wrist", purpose="grasp_point",
                                answer_id=None, uv_upright=list(check_uv), quantization_px=8.0, round=k,
                                check={"ok": bool(ok), "reason": None if ok else "pointed_not_grasp_point",
                                       "values": {}}, source="relations")
            if spec.get("net_move"):
                yield (lambda view: NetMove(view, steps=spec["net_move"], dz=-0.3))
            relation, success = SUCCESS[do]
            status = spec.get("status", "succeeded")
            result = outcome(relation, status, spec.get("reason", success),
                             {"obj": step.obj, "grip": list(step.grip), "ref": step.ref}, resume=spec.get("resume"))
            result.evidence.update(copy.deepcopy(spec.get("evidence") or {}))
            if status == "succeeded":
                if do in self.stays:
                    ctx.stays.add(f"stays:{relation}:{step.obj}", self.stays[do], step_i=step.i, kind="step")
                if do == "hold" or spec.get("hold"):
                    ctx.holding = HeldStub(step.obj, step.grip[0])
                elif do == "place_on":
                    ctx.holding = None
            ctx.record(result)
            return result
        return controller

    def controller_calls(self, do=None):
        return [row for row in self.log if row[0] == "controller" and (do is None or row[1] == do)]


# ------------------------------------------------------------------------------------------------- drivers
@pytest.fixture(scope="module")
def scene():
    return fakes.task2_like_scene()


def make_world(scene, sentence=TASK2):
    return fakes.FakeTwoViewSim(scene, prompt=sentence, images=False)


def make_client(answers, sentence=TASK2):
    transport = ScriptedV2Transport(answers)
    return AgentSpeedUpV2Client(task_prompt=sentence, transport=transport), transport


def answers(*, plans=(TASK2_PLAN,), fine=(36,), coarse=(28,), repairs=(HOLD_STILL,)):
    return FixedAnswers(plans=plans, points_fine=fine, points_coarse=coarse, repairs=repairs)


def build(scene, layer, *, sentence=TASK2, scripted=None, **policy_kwargs):
    client, transport = make_client(scripted or answers(), sentence)
    policy = AgentSpeedUpV2Policy(client, relations=layer, **policy_kwargs)
    return policy, transport, make_world(scene, sentence)


def drive(policy, world, steps, *, stop=None):
    """act -> world step -> observe_transition; returns the decisions."""
    observation, decisions = world.observation(), []
    for _ in range(steps):
        decision = policy.act(observation)
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        decisions.append(decision)
        observation = after
        if stop is not None and stop(policy, decision):
            break
    return decisions


def events(decisions, name=None):
    rows = [row for d in decisions for row in d.diagnostics["v2_events"]]
    return [row for row in rows if name is None or row["event"] == name]


def recorded(decisions):
    return [row for d in decisions for row in d.diagnostics["outcomes"]]


def relations_of(decisions):
    return [(row["relation"], row["status"], row["reason"]) for row in recorded(decisions)]


def outcome_seen(relation, reason=None):
    def stop(policy, decision):
        return any(row["relation"] == relation and (reason is None or row["reason"] == reason)
                   for row in decision.diagnostics["outcomes"])
    return stop


def needs_of(transport):
    return list(transport.kinds)


def transport_of(policy):
    return policy.client.transport


# ------------------------------------------------------------------------------- 1. the runner contract
DIAGNOSTIC_KEYS = {"step", "controller", "macro", "primitive", "execution_kind", "macro_step", "refreshed", "routing",
                   "api_calls_in_act", "api_latency_s", "api_request_key", "api_request_log", "pointing", "outcomes",
                   "outcome", "history_stats", "calibration_probe", "api_calls", "v2_events", "relation",
                   "observers", "parts_uv", "holding", "finger_gap_mm", "pending_need"}


def test_runner_contract_invariants_and_json_safe_diagnostics(scene):
    policy, _, world = build(scene, StandIn())
    assert policy.controller == "agent_relation" == policy_module.CONTROLLER
    observation = world.observation()
    decision = policy.act(observation)
    with pytest.raises(RuntimeError):
        policy.act(observation)                                  # act twice without observe_transition
    after = world.step(decision.action)
    with pytest.raises(ValueError):
        policy.observe_transition(after, decision, after)        # wrong before-observation
    with pytest.raises(ValueError):
        policy.observe_transition(observation, policy_module.Decision(decision.action, {}), after)
    feedback = policy.observe_transition(observation, decision, after)
    assert feedback["type"] == "agent_transition" and json.dumps(feedback, allow_nan=False)
    with pytest.raises(ValueError):
        policy.act(observation)                                  # not the observation of the last transition
    decisions = [decision] + drive(policy, world, 60)
    for d in decisions:
        action = np.asarray(d.action, float)
        assert action.shape == (7,) and np.isfinite(action).all() and np.all(np.abs(action) <= 1.0)
        assert DIAGNOSTIC_KEYS <= set(d.diagnostics), DIAGNOSTIC_KEYS - set(d.diagnostics)
        assert d.diagnostics["controller"] == "agent_relation"
        json.dumps(d.diagnostics, allow_nan=False)
    stats = policy.agent_stats()
    json.dumps(stats, allow_nan=False)
    assert {"policy_version", "step", "phase", "calibration", "counts", "plan", "steps", "parts", "holding", "ledger",
            "last_outcome", "rulings", "mount", "versions"} <= set(stats)
    assert {"calls_by_kind", "relations_by_status", "local_retries", "repairs", "reviews", "rejections",
            "preemptions"} <= set(stats["counts"])
    assert policy.relation_plan == TASK2_PLAN and policy.calibration == {"stand_in": True, "quarter_turns": 2}


def test_no_task_input_reaches_the_policy_and_the_module_passes_t0():
    parameters = list(inspect.signature(AgentSpeedUpV2Policy.__init__).parameters)
    assert parameters[:2] == ["self", "client"]
    taskish = set(th.TASK_TEXT_IDENTIFIERS) | {"task", "task_name", "suite"}
    for target in (AgentSpeedUpV2Policy.__init__, StepRunner.__init__, policy_module.NeedFulfiller.__init__):
        assert not set(inspect.signature(target).parameters) & taskish
    vocab = th.lint_vocabulary(LIBERO)["vocabulary"]
    assert th.lint_module(POLICY_PATH, vocab) == []                       # allow_task_text=False
    assert th.literal_audit(POLICY_PATH) == []
    tree = ast.parse(POLICY_PATH.read_text())
    assert th.firewall_violations(tree, "agent_speedup_v2") == []
    assert "agent_speedup_v2.py" in th.CONTROL_MODULES
    labels = th.module_labels(POLICY_PATH)
    assert labels and all(key in th.constants_register() for key in labels)
    source = POLICY_PATH.read_text()
    for spec in list_v2_tasks():
        assert spec.prompt not in source
    assert "start_poses" not in source and "measured_tasks" not in source


def test_the_client_sentence_must_be_the_episode_sentence(scene):
    client, transport = make_client(answers(), TASK2)
    policy = AgentSpeedUpV2Policy(client, relations=StandIn())
    with pytest.raises(ValueError):                       # client.check_observation at the first act (spec §4.2)
        policy.act(make_world(scene, OTHER).observation())
    assert transport.kinds == []


def test_restated_v1_constants_equal_their_sources():
    from gpt_field import agent_speedup_v1 as v1
    assert policy_module.B1_MAX_CALLS_PER_ACT == v1.MAX_DECISIONS_PER_ACT
    assert policy_module.REJECTION_STORM_LIMIT == v1.REJECTION_STORM_LIMIT
    assert policy_module.MAX_TRANSITIONS_PER_ACT == v1.MAX_TRANSITIONS_PER_ACT
    assert policy_module.LEDGER_LIMIT == v1.LEDGER_LIMIT
    assert policy_module.OUTCOME_HISTORY_LIMIT == v1.OUTCOME_HISTORY_LIMIT
    from gpt_field.mark_prompting import FINGERTIP_EDGE
    assert [FINGERTIP_EDGE[k] for k in range(4)] == list(kernel.FINGERTIP_EDGES)   # the policy indexes it by k


def test_overrides_and_stand_in_layers_need_an_offline_mock_client(scene):
    class LiveLike:
        offline_mock = False
    with pytest.raises(ValueError):
        AgentSpeedUpV2Policy(LiveLike(), overrides={"first_sense": -1}, relations=StandIn())
    with pytest.raises(ValueError):
        AgentSpeedUpV2Policy(LiveLike(), relations=StandIn())
    client, _ = make_client(answers())
    with pytest.raises(ValueError):
        AgentSpeedUpV2Policy(client, overrides={"first_sense": -1, "speed": 2}, relations=StandIn())
    policy = AgentSpeedUpV2Policy(client, overrides={"first_sense": -1, "depth_rule": "touch_stop",
                                                     "forced_yaw_rad": 0.1}, relations=StandIn())
    assert policy.ctx.overrides == {"first_sense": -1, "depth_rule": "touch_stop", "forced_yaw_rad": 0.1}
    assert AgentSpeedUpV2Policy(client, overrides={}, relations=StandIn()).ctx.overrides == {}


def test_the_runner_factory_takes_only_the_client(scene):
    client, _ = make_client(answers())
    if policy_module.relations is None:                       # BR not landed: the default layer is refused clearly
        with pytest.raises(RuntimeError, match="relation layer"):
            AgentSpeedUpV2Policy(client)
    else:
        policy = AgentSpeedUpV2Policy(client)                 # runner: policy_factory(client), no task key
        assert isinstance(policy._runner, StepRunner) and policy.controller == "agent_relation"


class PointingPrimitive(ScriptedMove):
    @property
    def pointing_request(self):
        return {"stage": "coarse"}


def test_a_primitive_that_requests_pointing_is_refused(scene):
    def episode(ctx):
        yield lambda view: PointingPrimitive(view, steps=2, dz=-0.1)
        yield from hold_forever()
    policy, _, world = build(scene, StandIn(), episode_factory=episode)
    with pytest.raises(RuntimeError, match="never"):
        policy.act(world.observation())


def test_event_sink_accepts_only_registry_events_and_keeps_colliding_keys():
    sink = policy_module.EventSink(lambda: 7)
    sink.emit("registration", policy_step=3, pid="A", where=np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        sink.emit("not_an_event")
    assert sink.drain() == [{"event": "registration", "policy_step": 7, "payload_policy_step": 3, "pid": "A",
                             "where": [1.0, None]}] and sink.drain() == []


def test_calibration_probe_and_calibration_without_layer_exports(scene):
    class Bare(StandIn):
        CalibratePlus = None                                  # a layer that does not re-export the class

        def calibrate_plus(self, ctx):
            reply = yield (lambda view: CalibratePlus(view, robot=ctx.robot))
            self.sent.append(reply)
            ctx.record(outcome("calibrated", "succeeded", "calibrated_online"))
            return StandInCal()                               # returned, not stored on ctx (§C10 return value)
    policy, _, world = build(scene, Bare())
    decisions = drive(policy, world, 200, stop=outcome_seen("calibrated"))
    assert decisions[0].diagnostics["calibration_probe"] is True
    drive(policy, world, 2)
    assert policy.calibration == {"stand_in": True, "quarter_turns": 2} and policy.ctx.cal is not None
    assert policy_module._is_calibration(None, None) is False


# ------------------------------------------------------------------------- 2. B1, the storm, the call path
def point_need(kind="point_fine"):
    return kernel.PointNeed(kind, "A", "the scripted part", "wrist", (256.0, 256.0) if kind == "point_fine" else None,
                            0, "grasp_point")


def hold_forever():
    while True:
        yield lambda view: HoldPrimitive(view, steps=5, gripper="open", name="test_hold")


def test_b1_three_point_needs_two_calls_then_the_third_next_act(scene):
    replies = []

    def episode(ctx):
        for _ in range(3):
            replies.append((yield point_need()))
        yield from hold_forever()
    policy, transport, world = build(scene, StandIn(), episode_factory=episode)
    decisions = drive(policy, world, 4)
    per_act = [d.diagnostics["api_calls_in_act"] for d in decisions]
    assert per_act[:3] == [["point_fine", "point_fine"], ["point_fine"], []]
    assert decisions[0].diagnostics["pending_need"] == "point_fine" and decisions[0].diagnostics["primitive"] == "hold"
    assert decisions[1].diagnostics["primitive"] == "test_hold"
    assert len(replies) == 3 and all(isinstance(r, kernel.PointAnswer) and not r.schema_error for r in replies)
    assert needs_of(transport) == ["point_fine"] * 3
    assert policy.agent_stats()["counts"]["hold_actions_by_cause"] == {"need_deferred": 1}


def test_rejection_storm_one_call_per_act_until_an_accepted_answer(scene):
    def episode(ctx):
        for _ in range(3):
            ctx.note_rejection("plan")
        for _ in range(3):
            yield point_need()
        yield lambda view: HoldPrimitive(view, steps=1, gripper="open", name="test_hold")
        ctx.note_accepted("point")
        for _ in range(3):
            yield point_need()
        yield from hold_forever()
    policy, _, world = build(scene, StandIn(), episode_factory=episode)
    decisions = drive(policy, world, 6)
    assert [len(d.diagnostics["api_calls_in_act"]) for d in decisions] == [1, 1, 1, 2, 1, 0]
    assert decisions[0].diagnostics["call_limit"]["value"] == 1.0
    assert decisions[5].diagnostics["call_limit"]["value"] == 2.0


def test_schema_errors_are_rejections_and_trigger_the_storm(scene):
    replies = []

    def episode(ctx):
        for _ in range(5):
            replies.append((yield point_need()))
        yield from hold_forever()
    scripted = answers(fine=({"point_id": "not a number"},))
    policy, _, world = build(scene, StandIn(), episode_factory=episode, scripted=scripted)
    decisions = drive(policy, world, 4)
    assert all(r.schema_error and r.reply is None for r in replies)
    assert [len(d.diagnostics["api_calls_in_act"]) for d in decisions] == [2, 1, 1, 1]
    assert policy.ctx.consecutive_rejections == 5 and policy.fulfiller.errors["schema_error"] == 5
    assert all(call["ok"] is False for d in decisions for call in d.diagnostics["api_calls"])


class GuardedClient:
    """Wraps the real client: a request while the policy's primitive is pending or not done raises (§C13 test 3)."""

    def __init__(self, inner):
        self.inner, self.policy, self.kinds = inner, None, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _check(self, kind):
        primitive = self.policy._primitive
        if primitive is not None and (primitive.pending or not primitive.done):
            raise AssertionError(f"model call ({kind}) inside primitive {primitive.name}")
        self.kinds.append(kind)

    def request_plan(self, *args, **kwargs):
        self._check("plan")
        return self.inner.request_plan(*args, **kwargs)

    def request_point(self, *args, **kwargs):
        self._check("point")
        return self.inner.request_point(*args, **kwargs)

    def request_repair(self, *args, **kwargs):
        self._check("repair")
        return self.inner.request_repair(*args, **kwargs)


def test_no_model_call_inside_a_primitive_or_a_step(scene):
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
    inner, transport = make_client(answers(repairs=(repair("retry"),)))
    guarded = GuardedClient(inner)
    policy = AgentSpeedUpV2Policy(guarded, relations=layer)
    guarded.policy = policy
    decisions = drive(policy, make_world(scene), 400, stop=outcome_seen("supported_on"))
    assert transport.kinds == ["plan", "point_fine", "repair", "point_fine", "point_fine"]
    assert guarded.kinds == ["plan", "point", "repair", "point", "point"]
    for d in decisions:                                      # calls only in acts that reached a Need boundary
        if d.diagnostics["api_calls_in_act"]:
            assert d.diagnostics["refreshed"] is True
    with pytest.raises(RuntimeError):                        # the fulfiller refuses while a primitive is active
        policy._primitive = ScriptedMove(policy.ctx.view, steps=3, dz=-0.1)
        policy.fulfiller.fulfil(point_need(), policy.ctx.view, [])


def test_outcome_drain_numbers_steps_and_calibration_probe(scene):
    layer = StandIn(real_calibration=True)
    policy, _, world = build(scene, layer)
    decisions = drive(policy, world, 600, stop=outcome_seen("supported_on"))
    records = recorded(decisions)
    assert [r["n"] for r in records] == list(range(1, len(records) + 1))
    assert [(r["relation"], r["reason"]) for r in records] == (
        [("calibrated", "calibrated_online")] + [("registered", "registered")] * 5
        + [("turned", "turned_by_effect"), ("supported_on", "placed")])
    assert [r["plan_step"] for r in records] == [None] * 6 + [0, 1]
    for d in decisions:
        for r in d.diagnostics["outcomes"]:
            assert r["step"] == d.diagnostics["step"]
        assert d.diagnostics["calibration_probe"] == (d.diagnostics["primitive"] == "calibrate_plus")
    probes = [d.diagnostics["calibration_probe"] for d in decisions]
    assert probes[0] and sum(probes) >= 150 and not any(probes[sum(probes):])
    assert len(policy.outcomes) == len(records) and policy.ledger[-1]["relation"] == "supported_on"


# ------------------------------------------------------------------------------------- 3. the StepRunner
def test_expansion_of_the_task2_plan_registers_before_any_grasp(scene):
    layer = StandIn()
    policy, _, world = build(scene, layer)
    decisions = drive(policy, world, 300, stop=outcome_seen("supported_on"))
    order = [row[0] if row[0] != "controller" else row[1] for row in layer.log]
    assert order == ["calibrate_plus", "register_regions", "turn", "place_on"]
    assert layer.log[1] == ("register_regions", ("A", "B", "C", "D", "E"))
    assert policy.ctx.roles == {"A": {"grip"}, "B": {"effect", "ref"}, "C": {"grip"}, "D": {"carried"},
                                "E": {"keepout"}}
    assert "clear_of:E" in policy.ctx.scene.keepouts                   # §C3.6: every clear_of part has a KeepOut
    plan_meta = transport_of(policy).metas[0]                          # k computed at the plan view (spec §3.6)
    assert plan_meta["quarter_turns"] == 2 and plan_meta["fingertip_edge"] == "top"
    assert layer.controller_calls("turn")[0][2:6] == (0, "A", ("A",), 1)
    assert layer.controller_calls("place_on")[0][2:6] == (1, "D", ("C",), 1)
    names = [row["event"] for row in events(decisions)]
    for name in ("plan_validation", "relation_plan", "step_start", "step_end"):
        assert name in names
    assert names.index("relation_plan") < names.index("step_start")
    starts = events(decisions, "step_start")
    assert [(e["i"], e["do"], e["obj"], e["grip"]) for e in starts] == [(0, "turn", "A", ["A"]),
                                                                        (1, "place_on", "D", ["C"])]
    ends = events(decisions, "step_end")
    assert [(e["status"], e["reason"]) for e in ends] == [("done", "turned_by_effect"), ("done", "placed")]
    assert ends[0]["calls"] == {"point_fine": 1} and ends[0]["steps"] >= 4
    macros = [d.diagnostics["macro"] for d in decisions]
    assert "turned(A)" in macros and "supported_on(D)" in macros
    relation = next(d.diagnostics["relation"] for d in decisions if d.diagnostics["macro"] == "supported_on(D)")
    assert relation["step_i"] == 1 and relation["pids"] == {"obj": "D", "grip": ["C"], "ref": "B"}


def test_implicit_release_before_a_step_that_needs_a_free_hand(scene):
    layer = StandIn()
    scripted = answers(plans=(plan_json((HOLD_D, TURN, PLACE)),), fine=(36,))
    policy, _, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 400, stop=outcome_seen("supported_on"))
    order = [row[0] if row[0] != "controller" else row[1] for row in layer.log]
    assert order == ["calibrate_plus", "register_regions", "hold", "release", "turn", "place_on"]
    layer = StandIn()                                         # holding the step's own grip part: no release
    scripted = answers(plans=(plan_json((HOLD_D, PLACE)),))
    policy, _, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 400, stop=outcome_seen("supported_on"))
    assert [row[0] if row[0] != "controller" else row[1] for row in layer.log] == [
        "calibrate_plus", "register_regions", "hold", "place_on"]
    layer = StandIn()                                         # a second hold needs an empty hand
    scripted = answers(plans=(plan_json((HOLD_D, HOLD_D), parts=PARTS[2:4]),))   # unused_object: only the pot's parts
    policy, _, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 400, stop=lambda p, d: len(layer.controller_calls("hold")) == 2 and any(
        r["relation"] == "held" for r in d.diagnostics["outcomes"]))
    assert [row[0] if row[0] != "controller" else row[1] for row in layer.log] == [
        "calibrate_plus", "register_regions", "hold", "release", "hold"]


def test_plan_reask_then_a_valid_plan(scene):
    bad = plan_json(({**TURN, "until": []}, PLACE))                       # turn_needs_criterion
    scripted = answers(plans=(bad, TASK2_PLAN))
    policy, transport, world = build(scene, StandIn(), scripted=scripted)
    decisions = drive(policy, world, 200, stop=outcome_seen("supported_on"))
    validations = events(decisions, "plan_validation")
    assert [(v["ok"], v["reask"], v["final"]) for v in validations] == [(False, False, False), (True, True, True)]
    assert validations[0]["issues"][0]["code"] == "turn_needs_criterion"
    assert transport.metas[1]["previous_rejected"].startswith("turn_needs_criterion")
    assert transport.kinds[:2] == ["plan", "plan"] and policy.ctx.consecutive_rejections == 0


def test_second_invalid_plan_records_invalid_plan_and_holds_forever(scene):
    bad = plan_json(({**TURN, "until": []},))
    scripted = answers(plans=(bad, {"scene": "x", "parts": [], "hold_true": [], "steps": []}))   # schema error
    layer = StandIn()
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 120)
    assert relations_of(decisions)[-1] == ("plan", "rejected", "invalid_plan")
    invalid = [r for r in recorded(decisions) if r["reason"] == "invalid_plan"][0]
    assert invalid["thresholds"][0]["name"] == "PLAN_reasks" and invalid["thresholds"][0]["label"] == "DD"
    validations = events(decisions, "plan_validation")
    assert [(v["ok"], v["final"]) for v in validations] == [(False, False), (False, True)]
    assert validations[1]["issues"][0]["code"] == "schema_error"
    assert transport.kinds == ["plan", "plan"] and layer.controller_calls() == []
    tail = decisions[-60:]
    assert all(d.diagnostics["primitive"] == "hold_recheck" for d in tail)
    assert policy.relation_plan is None and not policy._gen_done


def test_nominal_task2_like_plan_needs_at_most_four_calls(scene):
    # coarse first for C (decision §2.7: coarse-first pointing when sigma_uv > 54 px), fine for A and C
    layer = StandIn(points={"place_on": [("point_coarse", None), ("point_fine", None)]})
    policy, transport, world = build(scene, layer)
    decisions = drive(policy, world, 500, stop=outcome_seen("supported_on"))
    assert transport.kinds == ["plan", "point_fine", "point_coarse", "point_fine"]
    assert len(transport.kinds) <= 4
    assert max(len(d.diagnostics["api_calls_in_act"]) for d in decisions) <= policy_module.B1_MAX_CALLS_PER_ACT
    assert decisions[-1].diagnostics["outcome"]["relation"] == "supported_on"
    stats = policy.agent_stats()
    assert stats["counts"]["hold_actions"] == 0               # no idle step on the nominal path (AC-4 budget)
    assert stats["step_budget"]["steps"] == len(decisions)
    assert set(stats["step_budget"]["by_phase"]) <= {"calibrate", "plan", "register", "step", "hold"}
    # The episode continues (no LIBERO success in the fake; the diagnostic chain runs with stop_on_success=False):
    # spec §3.7 then sends ONE review, exactly one hold cycle after the last relation ended, and nothing after its
    # hold_still. So such a run makes 5 calls (C19's 3-4 + the review) and emits review and model_claim events.
    placed_at = decisions[-1].diagnostics["step"]
    later = drive(policy, world, 300)
    assert transport.kinds == ["plan", "point_fine", "point_coarse", "point_fine", "review"]
    review_at = next(d.diagnostics["step"] for d in later if d.diagnostics["api_calls_in_act"] == ["review"])
    assert review_at - placed_at == policy_module.HOLD_RECHECK_STEPS
    assert [c["claim"] for c in events(later, "model_claim")] == ["hold_still"]
    assert all(d.diagnostics["primitive"] == "hold_recheck" for d in later[-40:])


# --------------------------------------------------------------------------------------- 4. repairs
def test_repair_loop_retry_resumes_from_the_failed_outcome(scene):
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip", "resume": {"theta_part_rad": 0.2}}]})
    scripted = answers(repairs=(repair("retry"),))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 400, stop=outcome_seen("supported_on"))
    assert transport.kinds == ["plan", "point_fine", "repair", "point_fine", "point_fine"]
    meta = transport.metas[2]
    assert meta["kind"] == "repair" and meta["review"] is False
    request = json.loads(transport.calls[2]["input"][0]["content"][0]["text"])
    assert set(request["last_outcome"]) == {"reason", "evidence_text", "thresholds"}           # [C-45]
    assert request["last_outcome"]["reason"] == "turn_slip"
    assert request["plan"]["steps"][0]["status"] == "failed" and request["plan"]["steps"][0]["reason"] == "turn_slip"
    turns = layer.controller_calls("turn")
    assert [(row[5], row[6]) for row in turns] == [(1, None), (2, {"theta_part_rad": 0.2})]
    rep = events(decisions, "repair")[0]
    assert rep["valid"] and rep["applied"]["action"] == "retry" and rep["applied"]["status"] == "pending"
    assert policy.agent_stats()["counts"]["repairs"] == 1


def test_invalid_repair_twice_records_repair_invalid_and_holds_still(scene):
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
    scripted = answers(repairs=(repair("use_part"),))                        # part_required, twice
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 200)
    assert transport.kinds == ["plan", "point_fine", "repair", "repair"]
    assert ("plan", "rejected", "repair_invalid") in relations_of(decisions)
    reps = events(decisions, "repair")
    assert [(r["valid"], r["reask"]) for r in reps] == [(False, False), (False, True)]
    assert reps[-1]["applied"]["action"] == "hold_still" and reps[-1]["issues"][0]["code"] == "part_required"
    assert layer.controller_calls("place_on") == []           # hold_still: the remaining steps are not started
    assert all(d.diagnostics["primitive"] == "hold_recheck" for d in decisions[-30:])
    assert [s["status"] for s in policy.agent_stats()["steps"]] == ["failed", "pending"]


@pytest.mark.parametrize("case", ["repoint", "use_part", "insert_step", "skip_step", "swap_next", "replan",
                                  "hold_still"])
def test_each_repair_action(scene, case):
    turn_fails = {"turn": [{"status": "failed", "reason": "pointed_not_grasp_point"}]}
    if case == "repoint":
        layer, rep = StandIn(script=turn_fails), repair("repoint", part="A")
    elif case == "use_part":
        layer = StandIn(script={"place_on": [{"status": "failed", "reason": "no_feasible_candidate"}]})
        rep = repair("use_part", part="D")
    elif case == "insert_step":
        layer, rep = StandIn(script=turn_fails), repair("insert_step", step=HOLD_D)
    elif case == "replan":
        layer, rep = StandIn(script=turn_fails), repair("replan")
    else:
        layer, rep = StandIn(script=turn_fails), repair(case)
    replanned = plan_json(({**TURN, "until": [{"o": "stop", "p": None, "txt": None}]}, PLACE))
    scripted = answers(plans=(TASK2_PLAN, replanned), repairs=(rep,))
    policy, transport, world = build(scene, layer, scripted=scripted)
    stop = {"hold_still": None, "swap_next": outcome_seen("turned", "turned_by_effect")}.get(
        case, outcome_seen("supported_on"))
    decisions = drive(policy, world, 500, stop=stop)
    calls = [(row[1], row[2], row[4], row[5]) for row in layer.controller_calls()]
    applied = events(decisions, "repair")[0]["applied"]
    assert applied["action"] == case
    if case == "repoint":
        assert ("register_regions", ("A",)) in layer.log                   # registration repeats for A
        assert calls[:2] == [("turn", 0, ("A",), 1), ("turn", 0, ("A",), 2)]
    elif case == "use_part":
        assert calls[1:] == [("place_on", 1, ("C",), 1), ("place_on", 1, ("D", "C"), 2)]
        assert "grip" in policy.ctx.roles["D"]
    elif case == "insert_step":
        assert calls[:2] == [("turn", 0, ("A",), 1), ("hold", 0, ("C",), 1)]
        assert calls[2] == ("turn", 1, ("A",), 2)                         # after the implicit release
        assert ("release", None) in layer.log
        assert [s["do"] for s in policy.agent_stats()["steps"]] == ["hold", "turn", "place_on"]
    elif case == "skip_step":
        assert calls == [("turn", 0, ("A",), 1), ("place_on", 1, ("C",), 1)]
        assert [s["status"] for s in policy.agent_stats()["steps"]] == ["skipped", "done"]
    elif case == "swap_next":
        assert calls == [("turn", 0, ("A",), 1), ("place_on", 0, ("C",), 1), ("turn", 1, ("A",), 2)]
        assert [step["do"] for step in policy.relation_plan["steps"]] == ["place_on", "turn"]
    elif case == "replan":
        assert transport.kinds[:4] == ["plan", "point_fine", "repair", "replan"]
        assert transport.metas[3]["replan"] is True
        plans = events(decisions, "relation_plan")
        assert [p["source"] for p in plans] == ["plan", "replan"]
        assert policy.relation_plan["steps"][0]["until"] == [{"o": "stop", "p": None, "txt": None}]
        assert calls[1] == ("turn", 0, ("A",), 1)                          # the new plan's state, first attempt
    elif case == "hold_still":
        assert calls == [("turn", 0, ("A",), 1)]
        assert all(d.diagnostics["primitive"] == "hold_recheck" for d in decisions[-40:])
        assert not events(decisions, "model_claim")                       # only a review's hold_still is a claim


def test_reserved_relation_is_recorded_by_the_runner_and_escalated(scene):
    parts = [PARTS[0], PARTS[1], {"id": "F", "text": "box: sliding tray front", "move": "sliding", "w": 10, "x": 10}]
    slide = {"do": "slide", "obj": "F", "grip": [], "ref": None, "sense": "open", "keep": [],
             "until": [{"o": "stop", "p": None, "txt": None}]}
    scripted = answers(plans=(plan_json((slide, TURN), parts=parts),), repairs=(repair("skip_step"),))
    layer = StandIn()
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 300, stop=outcome_seen("turned"))
    assert ("slide", "failed", "relation_not_implemented") in relations_of(decisions)
    assert transport.kinds[:2] == ["plan", "repair"]
    assert [s["status"] for s in policy.agent_stats()["steps"]] == ["skipped", "done"]


# ------------------------------------------------------------------------------------------ 5. stays
def test_a_lost_stays_relation_reopens_its_step_after_the_current_relation(scene):
    flag = StaysFlag("met")
    layer = StandIn(stays={"turn": flag})
    policy, transport, world = build(scene, layer)
    observation = world.observation()
    decisions = []
    for _ in range(600):
        decision = policy.act(observation)
        decisions.append(decision)
        if decision.diagnostics["macro"] == "supported_on(D)":
            flag.verdict = "violated"                     # lost while the next relation runs
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
        if len(layer.controller_calls("turn")) == 2:
            flag.verdict = "met"
        if sum(r["relation"] == "turned" and r["status"] == "succeeded" for r in recorded(decisions)) == 2:
            break
    order = [(row[1], row[5]) for row in layer.controller_calls()]
    assert order == [("turn", 1), ("place_on", 1), ("turn", 2)]
    lost = [r for r in recorded(decisions) if r["reason"] == "kept_relation_lost"]
    assert len(lost) == 1 and lost[0]["relation"] == "turned" and lost[0]["plan_step"] == 0   # the lost step
    checks = events(decisions, "stays_check")
    assert any(c["key"] == "stays:turned:A" and c["verdict"] == "violated" for c in checks)
    assert "repair" not in transport.kinds                                   # a reopen asks nothing


class HoldTrueRunner(StepRunner):
    """A StepRunner whose already-true claims are checked by a controllable flag (the real check needs the scene's
    external region masks, the start baseline and the calibration)."""

    def __init__(self, ctx, *, relations=None, flag=None):
        super().__init__(ctx, relations=relations)
        self.flag = flag

    def hold_true_check(self, claim):
        return self.flag, None


def test_a_lost_already_true_claim_escalates_at_once(scene):
    flag = StaysFlag("met")
    layer = StandIn()
    # the task-8 pattern (spec §3.3): the burner B is already on by the knob A, and no turn step is written
    scripted = answers(plans=(plan_json((PLACE,), hold_true=({"rel": "effect", "a": "B", "b": "A"},)),),
                       repairs=(HOLD_STILL,))
    client, transport = make_client(scripted)
    policy = AgentSpeedUpV2Policy(client, relations=layer,
                                  episode_factory=lambda ctx: HoldTrueRunner(ctx, relations=layer, flag=flag))
    world = make_world(scene)
    observation, decisions = world.observation(), []
    for _ in range(300):
        decision = policy.act(observation)
        decisions.append(decision)
        if decision.diagnostics["macro"] == "supported_on(D)":
            flag.verdict = "violated"
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
    claims = events(decisions, "model_claim")
    assert claims[0]["claim"] == "hold_true:effect" and claims[0]["unverified"] is True
    assert claims[0]["detail"]["checked"] is True
    assert "hold_true:A" in policy.ctx.scene.keepouts                          # clear_of(b)
    assert transport.kinds == ["plan", "point_fine", "repair"]               # once, not at every boundary
    request = json.loads(transport.calls[2]["input"][0]["content"][0]["text"])
    assert request["request"] == "repair" and request["last_outcome"]["reason"] == "kept_relation_lost"
    lost = [r for r in recorded(decisions) if r["reason"] == "kept_relation_lost"]
    assert len(lost) == 1 and lost[0]["relation"] == "keeps"
    assert not events(decisions, "review")                                   # a violated claim blocks the review


# ------------------------------------------------------------------------------------------ 6. reviews
def test_review_rules_bc_r4(scene):
    flags = {"turn": StaysFlag("met"), "place_on": StaysFlag("met")}
    layer = StandIn(stays=flags)
    reviews = (repair("retry"),                                # no step named: step_required (BC-R4) -> re-ask
               repair("retry", step=TURN),                     # names the turn step -> rerun it
               repair("swap_next"),                            # second review: no failed step -> no_next_step
               HOLD_STILL)                                     # -> an unverified claim; the episode continues
    scripted = answers(repairs=reviews)
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 700)
    assert transport.kinds.count("review") == 4 and "repair" not in transport.kinds
    rows = events(decisions, "review")
    assert [(r["valid"], r["reask"]) for r in rows] == [(False, False), (True, True), (False, False), (True, True)]
    assert rows[0]["issues"][0]["code"] == "step_required"
    assert rows[1]["applied"] == {"action": "retry", "part": None, "step_i": 0, "status": "pending"}
    assert rows[2]["issues"][0]["code"] == "no_next_step"
    assert [(row[1], row[5]) for row in layer.controller_calls()] == [("turn", 1), ("place_on", 1), ("turn", 2)]
    claims = [c for c in events(decisions, "model_claim") if c["claim"] == "hold_still"]
    assert len(claims) == 1 and claims[0]["unverified"] is True
    assert policy.agent_stats()["counts"]["reviews"] == policy_module.B3_MAX_REVIEWS == 2
    assert not policy._gen_done and all(d.diagnostics["primitive"] == "hold_recheck" for d in decisions[-40:])
    assert all(m["review"] for m in transport.metas if m["kind"] == "review")


def test_reviews_wait_for_every_stays_check_to_hold_and_insert_step_appends(scene):
    flag = StaysFlag("pending")
    layer = StandIn(stays={"turn": flag})
    reviews = (repair("insert_step", step=HOLD_D), HOLD_STILL)
    policy, transport, world = build(scene, layer, scripted=answers(repairs=reviews))
    decisions = drive(policy, world, 260)
    assert "review" not in transport.kinds                                  # a pending (unknown) check blocks it
    flag.verdict = "met"
    decisions += drive(policy, world, 400, stop=lambda p, d: len(layer.controller_calls("hold")) == 1 and any(
        r["relation"] == "held" for r in d.diagnostics["outcomes"]))
    assert transport.kinds.count("review") == 1
    assert [s["do"] for s in policy.agent_stats()["steps"]] == ["turn", "place_on", "hold"]
    assert events(decisions, "review")[0]["applied"]["step_i"] == 2


def test_skip_step_becomes_stays_only_when_its_own_check_holds(scene):
    layer = StandIn()
    policy, _, world = build(scene, layer)
    drive(policy, world, 300, stop=outcome_seen("supported_on"))
    runner = policy._runner
    done = runner.states[0]
    runner._owner["stays:turned:A"] = done
    runner.last_stays = {"stays:turned:A": StaysFlag("met")(None)}
    applied = list(runner.apply(kernel.RepairAnswer("x", "skip_step", None, None), done, review=False, outcome=None))
    assert applied == [] and done.status == "stays"
    runner.last_stays = {"stays:turned:A": StaysFlag("violated")(None)}
    list(runner.apply(kernel.RepairAnswer("x", "skip_step", None, None), done, review=False, outcome=None))
    assert done.status == "skipped"


def test_the_already_true_check_compares_the_claimed_region_with_the_start_baseline(scene):
    from gpt_field.agent_speedup_v2_observers import RegionStaysCheck
    from gpt_field.observation import PolicyObservation
    policy, _, world = build(scene, StandIn())
    drive(policy, world, 3)
    runner, ctx = policy._runner, policy.ctx
    claim = kernel.HoldTrue("on", "D", "B")
    assert runner.hold_true_check(claim) == (None, "region_unavailable")      # no external region measured
    mask = np.zeros((512, 512), bool)
    mask[100:220, 140:260] = True
    ctx.scene.region_mask = lambda pid, camera, view, **kwargs: mask if (pid, camera) == ("D", "external") else None
    grey = np.full((512, 512, 3), 128, np.uint8)
    ctx.scene.baselines["start"] = {"external": grey, "wrist": grey, "pose": tuple(world.state)}
    check, why = runner.hold_true_check(claim)
    assert why is None and isinstance(check, RegionStaysCheck)
    assert check(StepView(world.observation())).verdict == "met"        # unchanged (blank 128 grey)
    moved = np.full((512, 512, 3), 128, np.uint8)
    moved[..., 0] = 40
    changed = PolicyObservation(fakes._png_data_url(moved), world.observation().wrist_image, tuple(world.state), TASK2)
    report = check(StepView(changed))
    assert report.verdict == "violated" and report.reason == "kept_relation_lost"
    ctx.cal = None
    assert runner.hold_true_check(claim) == (None, "calibration_unavailable")


# ----------------------------------------------------- 6b. already-true claims, preconditions, escalation order
HOLD_A = {"do": "hold", "obj": "A", "grip": ["A"], "ref": None, "sense": "any", "keep": [], "until": []}
RELEASE_A = {"do": "release", "obj": "A", "grip": [], "ref": None, "sense": "any", "keep": [], "until": []}


def test_hold_true_keepout_is_clear_of_b_only_and_never_fences_a_step_part(scene):
    claims = ({"rel": "effect", "a": "B", "b": None},          # b null: no keep-out at all (spec §3.3 clear_of(b))
              {"rel": "on", "a": "D", "b": "B"},               # b = the place_on destination: skipped, logged
              {"rel": "on", "a": "C", "b": "E"})               # b of no step: hold_true keep-out of b
    policy, _, world = build(scene, StandIn(), scripted=answers(plans=(plan_json(hold_true=claims),)))
    decisions = drive(policy, world, 30)
    keepouts = sorted(policy.ctx.scene.keepouts)
    assert keepouts == ["clear_of:E", "hold_true:E"]
    details = {c["detail"]["key"]: c["detail"] for c in events(decisions, "model_claim")}
    assert details["hold_true:effect:B:-"]["keepout_skipped"] == "b_is_null"
    assert details["hold_true:on:D:B"]["keepout_skipped"] == "part_of_a_step_to_run"
    assert details["hold_true:on:D:B"]["keepout"] is False
    assert details["hold_true:on:C:E"]["keepout"] is True and details["hold_true:on:C:E"]["keepout_skipped"] is None


def test_a_second_invalid_replan_holds_still_and_starts_no_new_step(scene):
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
    bad = plan_json(({**TURN, "until": []}, PLACE))                        # turn_needs_criterion, twice
    scripted = answers(plans=(TASK2_PLAN, bad, bad), repairs=(repair("replan"),))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 300)
    assert transport.kinds == ["plan", "point_fine", "repair", "replan", "replan"]
    assert ("plan", "rejected", "invalid_plan") in relations_of(decisions)
    assert events(decisions, "repair")[0]["applied"]["replan"] == "invalid_plan"
    assert [row[1] for row in layer.controller_calls()] == ["turn"]          # place_on never starts
    assert [s["status"] for s in policy.agent_stats()["steps"]] == ["failed", "pending"]
    assert policy._runner.halted and all(d.diagnostics["primitive"] == "hold_recheck" for d in decisions[-40:])


def test_a_repoint_while_holding_waits_for_an_empty_hand_and_never_runs_unregistered(scene):
    # hold A, then turn A until effect(B): the turn fails while A is held; repoint B cannot register with a full
    # hand, so the rerun fails its precondition (recorded, escalated); an inserted release empties the hand, and B
    # registers at the start of the next attempt, before the controller runs
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
    scripted = answers(plans=(plan_json((HOLD_A, TURN), parts=PARTS[:2]),),
                       repairs=(repair("repoint", part="B"), repair("insert_step", step=RELEASE_A)))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 500, stop=lambda p, d: len(layer.controller_calls("turn")) == 2 and any(
        r["relation"] == "turned" and r["status"] == "succeeded" for r in d.diagnostics["outcomes"]))
    reps = events(decisions, "repair")
    assert reps[0]["applied"]["action"] == "repoint" and reps[0]["applied"]["registration"] == "waits_for_empty_hand"
    unmet = [r for r in recorded(decisions) if r["reason"] == "precondition_unmet"]
    assert len(unmet) == 1 and unmet[0]["relation"] == "turned" and unmet[0]["plan_step"] == 1
    assert unmet[0]["evidence"]["unmet"] == {"B": "region_registration"} and unmet[0]["evidence"]["holding"] == "A"
    assert reps[1]["request"]["last_outcome"]["reason"] == "precondition_unmet"
    order = [row[0] if row[0] != "controller" else row[1] for row in layer.log]
    assert order == ["calibrate_plus", "register_regions", "hold", "turn", "release", "register_regions", "turn"]
    assert layer.log[5] == ("register_regions", ("B",))
    assert policy.ctx.scene.parts["B"].status == "registered"
    assert transport.kinds == ["plan", "point_fine", "repair", "repair", "point_fine"]


class FailE(StandIn):
    """register_regions: the clear_of part E fails its registration (part_not_found), the rest as the stand-in."""

    def register_regions(self, ctx, plan, roles):
        out = yield from super().register_regions(ctx, plan, roles)
        if "E" in ctx.scene.parts and roles.get("E"):
            ctx.scene.parts["E"] = dataclasses.replace(ctx.scene.parts["E"], status="failed",
                                                       reason="part_not_found")
        return out


def test_a_missing_clear_of_keepout_is_retried_then_fails_the_step(scene):
    layer = FailE()
    policy, transport, world = build(scene, layer)
    decisions = drive(policy, world, 300, stop=lambda p, d: bool(events([d], "repair")))
    assert "clear_of:E" not in policy.ctx.scene.keepouts
    assert [row[1] for row in layer.controller_calls()] == ["turn"]          # place_on never ran without it
    assert ("register_regions", ("E",)) in layer.log                        # one retry with the empty hand
    failed = [r for r in recorded(decisions) if r["status"] != "succeeded"]
    assert [(r["relation"], r["reason"], r["plan_step"]) for r in failed] == [("supported_on", "part_not_found", 1)]
    assert failed[0]["evidence"]["unmet"] == {"E": "keepout"}
    assert transport.kinds == ["plan", "point_fine", "repair"]
    assert events(decisions, "step_end")[-1]["reason"] == "part_not_found"


def test_a_lost_claim_and_a_failed_step_at_one_boundary_ask_one_repair(scene):
    flag = StaysFlag("met")
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
    plan = plan_json((TURN, PLACE), hold_true=({"rel": "on", "a": "E", "b": None},))
    client, transport = make_client(answers(plans=(plan,), repairs=(repair("retry", step=TURN), repair("retry"))))
    policy = AgentSpeedUpV2Policy(client, relations=layer,
                                  episode_factory=lambda ctx: HoldTrueRunner(ctx, relations=layer, flag=flag))
    world = make_world(scene)
    observation, decisions = world.observation(), []
    for _ in range(400):
        decision = policy.act(observation)
        decisions.append(decision)
        if decision.diagnostics["macro"] == "turned(A)":
            flag.verdict = "violated"
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
        if any(r["relation"] == "supported_on" for r in decision.diagnostics["outcomes"]):
            break
    assert transport.kinds.count("repair") == 1                    # the claim's repair named and reset the turn
    request = json.loads(transport.calls[2]["input"][0]["content"][0]["text"])
    assert request["last_outcome"]["reason"] == "kept_relation_lost"
    assert request["plan"]["steps"][0]["status"] == "failed"           # the status block shows the failure too
    assert [(row[1], row[5]) for row in layer.controller_calls()] == [("turn", 1), ("turn", 2), ("place_on", 1)]


class StaysHook(StandIn):
    """A layer with the optional skip_step hook stays_check_for(ctx, step) -> (key, check) | None."""

    def __init__(self, verdict, **kwargs):
        super().__init__(**kwargs)
        self.verdict, self.asked = verdict, []

    def stays_check_for(self, ctx, step):
        self.asked.append((step.i, step.do, step.obj))
        return f"stays:{SUCCESS[step.do][0]}:{step.obj}", StaysFlag(self.verdict)


@pytest.mark.parametrize("verdict", ["met", "violated"])
def test_skip_step_of_a_failed_step_becomes_stays_when_its_observer_holds_now(scene, verdict):
    layer = StaysHook(verdict, script={"turn": [{"status": "failed", "reason": "effect_not_observed"}]})
    policy, _, world = build(scene, layer, scripted=answers(repairs=(repair("skip_step"),)))
    decisions = drive(policy, world, 400, stop=outcome_seen("supported_on"))
    assert layer.asked == [(0, "turn", "A")]
    status = [s["status"] for s in policy.agent_stats()["steps"]]
    if verdict == "met":
        assert status == ["stays", "done"] and "stays:turned:A" in policy.ctx.stays.keys()
        assert policy.ctx.stays.entry("stays:turned:A")["step_i"] == 0
    else:
        assert status == ["skipped", "done"] and "stays:turned:A" not in policy.ctx.stays.keys()
    assert any(c["key"] == "stays:turned:A" and c["verdict"] == verdict and c.get("skip_step")
               for c in events(decisions, "stays_check"))


class LateCheckRunner(StepRunner):
    """An already-true claim whose check cannot be built until ``available`` (the region is not measured yet)."""

    def __init__(self, ctx, *, relations=None):
        super().__init__(ctx, relations=relations)
        self.available, self.flag = False, StaysFlag("met")

    def hold_true_check(self, claim):
        return (self.flag, None) if self.available else (None, "region_unavailable")


def test_an_unchecked_claim_is_retried_and_blocks_the_review_until_checked(scene):
    layer = StandIn()
    plan = plan_json((TURN,), hold_true=({"rel": "on", "a": "E", "b": None},), parts=[PARTS[0], PARTS[1], PARTS[4]])
    client, transport = make_client(answers(plans=(plan,)))
    made = []

    def factory(ctx):
        made.append(LateCheckRunner(ctx, relations=layer))
        return made[-1]
    policy = AgentSpeedUpV2Policy(client, relations=layer, episode_factory=factory)
    world = make_world(scene)
    decisions = drive(policy, world, 200)
    assert transport.kinds == ["plan", "point_fine"]                         # no review over an unchecked claim
    pending = [c for c in events(decisions, "stays_check") if c["key"] == "hold_true:on:E:-"]
    assert len(pending) >= 3 and all(c["verdict"] == "pending" and c["checked"] is False for c in pending)
    assert "hold_true:on:E:-" not in policy.ctx.stays.keys()
    made[-1].available = True                                                # the region is measured now
    later = drive(policy, world, 100)
    assert "hold_true:on:E:-" in policy.ctx.stays.keys()
    assert transport.kinds == ["plan", "point_fine", "review"]
    checks = [c for c in events(later, "stays_check") if c["key"] == "hold_true:on:E:-"]
    assert checks and checks[-1]["verdict"] == "met"


class HeldScopeFlag(StandIn):
    """hold registers a preempting held-scope observer (as grasp registers HoldFloorInvariant, §C10.1)."""

    def __init__(self, flag, **kwargs):
        super().__init__(**kwargs)
        self.flag = flag
        base = self.CONTROLLERS["hold"]

        def hold(ctx, step):
            result = yield from base(ctx, step)
            ctx.monitor.add("hold_floor", self.flag, view=ctx.view, preempt=True, scope="held")
            return result
        self.CONTROLLERS["hold"] = hold


def test_a_preemption_of_the_idle_hold_is_recorded_escalated_once_and_does_not_churn(scene):
    flag = FlagObserver()
    layer = HeldScopeFlag(flag, script={"place_on": [{"status": "failed", "reason": "no_feasible_candidate"}]})
    scripted = answers(plans=(plan_json((HOLD_D, PLACE)),), repairs=(HOLD_STILL,))
    policy, transport, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 200)
    assert transport.kinds == ["plan", "point_fine", "repair"] and policy.holding == "D"
    flag.verdict = "violated"                                    # e.g. hold_lost while holding still
    later = drive(policy, world, 80)
    assert policy.agent_stats()["counts"]["preemptions"] == 1    # removed from the bus after its delivery
    assert "hold_floor" not in policy.ctx.monitor.keys()
    lost = [r for r in recorded(later) if r["status"] == "failed"]
    assert [(r["relation"], r["reason"]) for r in lost] == [("keeps", "kept_relation_lost")]
    assert lost[0]["evidence"]["invariant"] == "hold_floor" and lost[0]["evidence"]["removed_from_bus"] is True
    assert transport.kinds == ["plan", "point_fine", "repair", "repair"]
    request = json.loads(transport.calls[3]["input"][0]["content"][0]["text"])
    assert request["last_outcome"]["reason"] == "kept_relation_lost" and request["holding"] == "D"
    assert all(d.diagnostics["primitive"] == "hold_recheck" for d in later[-40:])


def test_observer_events_carry_the_step_of_the_executed_action(scene):
    flag = FlagObserver()

    def episode(ctx):
        ctx.monitor.add("flag", flag, view=ctx.view, preempt=False)
        yield from hold_forever()
    policy, _, world = build(scene, StandIn(), episode_factory=episode)
    observation, decisions = world.observation(), []
    for k in range(4):
        decision = policy.act(observation)
        decisions.append(decision)
        flag.verdict = "violated" if k == 1 else "met"
        after = world.step(decision.action)
        feedback = policy.observe_transition(observation, decision, after)
        assert feedback["step"] == k
        observation = after
    decisions.append(policy.act(observation))
    assert [(e["policy_step"], e["verdict"]) for e in events(decisions, "observer")] == [
        (0, "met"), (1, "violated"), (2, "met")]


class CalibrationClose(StandIn):
    """calibrate_plus yields a close stage that is not the CalibratePlus primitive itself."""

    def calibrate_plus(self, ctx):
        reply = yield (lambda view: HoldPrimitive(view, steps=3, gripper="close", name="calibration_close"))
        self.sent.append(reply)
        ctx.cal = StandInCal()
        ctx.record(outcome("calibrated", "succeeded", "calibrated_online"))
        return ctx.cal


def test_calibration_probe_covers_every_stage_of_the_calibrate_phase(scene):
    policy, _, world = build(scene, CalibrationClose())
    decisions = drive(policy, world, 200, stop=outcome_seen("turned"))
    first = decisions[:3]
    assert all(d.diagnostics["primitive"] == "calibration_close" and d.action[6] > 0 for d in first)
    assert all(d.diagnostics["calibration_probe"] for d in first)              # kept out of first_close_event
    assert not any(d.diagnostics["calibration_probe"] for d in decisions[3:])


def test_status_blocks_show_the_grip_in_force_and_plan_reasks_carry_the_parse_error(scene):
    layer = StandIn(script={"place_on": [{"status": "failed", "reason": "no_feasible_candidate"}] * 2})
    scripted = answers(repairs=(repair("use_part", part="D"), HOLD_STILL))
    policy, transport, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 400)
    repairs = [json.loads(c["input"][0]["content"][0]["text"]) for c, k in zip(transport.calls, transport.kinds)
               if k == "repair"]
    assert [r["plan"]["steps"][1]["grip"] for r in repairs] == [["C"], ["D", "C"]]
    # a structural parse error: its text is the re-ask's error text (spec §3.4)
    policy, _, world = build(scene, StandIn())
    drive(policy, world, 1)
    runner = StepRunner(policy.ctx, relations=StandIn())
    asking = runner.obtain_plan("plan")
    need = next(asking)
    call = kernel.CallRecord("plan", 0.0, None, None, False, "parse_error")
    text = "steps[0].do: not one of the allowed values"
    second = asking.send(kernel.PlanAnswer(need, {}, None, text, False, call, policy.ctx.view))
    assert second.reask_error == text


class ScopeAudit(StandIn):
    """Every StepRunner-called controller leaves a relation-scope observer behind; each call records the
    relation-scope keys it finds when it starts (§C10.1: scope relation is cleared on every controller return)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.found = []
        for do, base in list(self.CONTROLLERS.items()):
            self.CONTROLLERS[do] = self._audited(do, base)

    def _audited(self, name, base):
        def run(ctx, *args):
            self.found.append((name, tuple(ctx.monitor.keys("relation"))))
            result = yield from base(ctx, *args)
            ctx.monitor.add(f"left_by_{name}_{len(self.found)}", FlagObserver(), view=ctx.view, preempt=False)
            return result
        return run

    def calibrate_plus(self, ctx):
        return (yield from self._audited("calibrate_plus", super().calibrate_plus)(ctx))

    def register_regions(self, ctx, plan, roles):
        return (yield from self._audited("register_regions", super().register_regions)(ctx, plan, roles))

    def release(self, ctx, step):
        return (yield from self._audited("release", super().release)(ctx, step))


def test_the_relation_scope_is_cleared_after_every_controller_the_runner_calls(scene):
    layer = ScopeAudit()
    policy, _, world = build(scene, layer, scripted=answers(plans=(plan_json((HOLD_D, TURN, PLACE)),)))
    drive(policy, world, 500, stop=outcome_seen("supported_on"))
    assert [name for name, _ in layer.found] == ["calibrate_plus", "register_regions", "hold", "release", "turn",
                                                 "place_on"]
    assert all(keys == () for _, keys in layer.found)
    assert policy.ctx.monitor.keys("relation") == []


# --------------------------------------------------------------------------------------- 7. preemption
def preempting_episode(replies, flag, steps):
    def episode(ctx):
        ctx.monitor.add("flag", flag, view=ctx.view, preempt=True)
        replies.append((yield lambda view: ScriptedMove(view, steps=steps, dz=-0.1)))
        yield from hold_forever()
    return episode


def test_preemption_mid_primitive_drops_it(scene):
    replies, flag = [], FlagObserver()
    policy, _, world = build(scene, StandIn(), episode_factory=preempting_episode(replies, flag, 10))
    observation = world.observation()
    for k in range(5):
        decision = policy.act(observation)
        flag.verdict = "violated" if k == 2 else "met"     # violated on one transition only (the bus re-arms a
        after = world.step(decision.action)                # violation that persists, by design)
        policy.observe_transition(observation, decision, after)
        observation = after
    sent, primitive = replies[0]
    assert sent["reason"] == "preempted:flag_violated" and sent["status"] == "failed"
    assert sent["evidence"]["invariant"] == "flag" and sent["actions"] == 3 and primitive.name == "scripted_move"
    assert not primitive.done and policy._primitive.name == "test_hold"
    assert policy.agent_stats()["counts"]["preemptions"] == 1
    assert flag.contexts[0]["primitive"] == "scripted_move" and "step" in flag.contexts[0]
    assert flag.contexts[0]["commanded_reorientation"] is False


def test_preemption_on_the_finishing_transition_replaces_the_outcome(scene):
    replies, flag = [], FlagObserver()
    policy, _, world = build(scene, StandIn(), episode_factory=preempting_episode(replies, flag, 3))
    observation = world.observation()
    decisions = []
    for k in range(5):
        decision = policy.act(observation)
        decisions.append(decision)
        flag.verdict = "violated" if k == 2 else "met"
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
    sent, primitive = replies[0]
    assert primitive.done and primitive.outcome()["reason"] == "goal_reached"
    assert sent["reason"] == "preempted:flag_violated" and sent["actions"] == 3
    assert events(decisions, "invariant")[0]["key"] == "flag"


def test_preemption_while_at_a_need_skips_the_next_factory(scene):
    replies, flag, built = [], FlagObserver(), []

    def episode(ctx):
        ctx.monitor.add("flag", flag, view=ctx.view, preempt=True)
        for _ in range(3):
            yield point_need()

        def factory(view):
            built.append(view)
            return ScriptedMove(view, steps=3, dz=-0.1)
        replies.append((yield factory))
        yield from hold_forever()
    policy, _, world = build(scene, StandIn(), episode_factory=episode)
    observation = world.observation()
    for k in range(3):
        decision = policy.act(observation)
        flag.verdict = "violated" if k == 0 else "met"
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
    sent, primitive = replies[0]
    assert primitive is None and built == []
    assert sent["primitive"] is None and sent["reason"] == "preempted:flag_violated" and sent["actions"] == 0
    assert policy.ctx.monitor.peek_preemption() is None                    # delivered once, then removed


# --------------------------------------------------------------------- 8. invariance (TLR-4) and AC-5
def comparable(decision):
    d = copy.deepcopy(decision.diagnostics)
    for key in ("api_request_key", "api_request_log", "history_stats", "api_latency_s", "api_calls"):
        d.pop(key, None)
    return d


def test_tlr4_two_task_sentences_give_identical_action_traces(scene):
    traces = []
    for sentence in (TASK2, OTHER):
        layer = StandIn(real_calibration=True, script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
        policy, transport, world = build(scene, layer, sentence=sentence,
                                         scripted=answers(repairs=(repair("retry"),)))
        decisions = drive(policy, world, 700)
        payload_tasks = {json.loads(call["input"][0]["content"][0]["text"])["task"] for call in transport.calls}
        assert payload_tasks == {sentence}                                  # the sentence reaches only the payloads
        traces.append(([tuple(d.action) for d in decisions], [comparable(d) for d in decisions],
                       list(transport.kinds)))
    assert traces[0][0] == traces[1][0]
    assert traces[0][1] == traces[1][1]
    assert traces[0][2] == traces[1][2]


def test_ac5_no_model_number_or_text_reaches_control(scene):
    # (a) model free text (scene, part texts, criterion text, repair why) does not change a single action
    texts = [p | {"text": f"part {p['id']} as described {k}"} for k, p in enumerate(PARTS)]
    other_plan = plan_json(({**TURN, "until": [{**TURN["until"][0], "txt": "it changes"}, TURN["until"][1]]}, PLACE),
                           parts=texts, scene="a different description")
    traces = []
    for plan, why in ((TASK2_PLAN, "first reason"), (other_plan, "another reason")):
        layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
        policy, _, world = build(scene, layer, scripted=answers(plans=(plan,), repairs=(repair("retry", why=why),)))
        traces.append([tuple(d.action) for d in drive(policy, world, 400, stop=outcome_seen("supported_on"))])
        for item in layer.sent:                                  # what the relation layer receives from the model
            if isinstance(item, kernel.PointAnswer):
                assert item.reply.answer_id is None or 0 <= item.reply.answer_id <= 64
                assert set(dataclasses.asdict(item.reply)) == {"stage", "camera", "answer_id", "uv_upright",
                                                               "quantization_px", "origin", "crop_px", "upsample",
                                                               "quarter_turns"}
            else:
                assert isinstance(item, tuple) and isinstance(item[0], dict)          # primitive outcomes only
        for part in policy._runner.plan.parts:
            assert 0 <= part.w_cell <= 64 and 0 <= part.x_cell <= 64
    assert traces[0] == traces[1]
    # (b) a number smuggled into a plan or a point answer is a schema error: it never reaches the generator
    smuggled = plan_json() | {"offset_m": 0.05}
    replies = []

    def episode(ctx):
        replies.append((yield kernel.PlanNeed("plan", 0, kernel.FINGERTIP_EDGES[0])))
        replies.append((yield point_need()))
        yield from hold_forever()
    scripted = answers(plans=(smuggled,), fine=({"point_id": 5, "dx_px": 3},))
    policy, _, world = build(scene, StandIn(), episode_factory=episode, scripted=scripted)
    drive(policy, world, 2)
    plan_answer, point_answer = replies
    assert plan_answer.schema_error and plan_answer.plan is None and plan_answer.raw is None
    assert point_answer.schema_error and point_answer.reply is None


def test_outcomes_are_json_safe_with_labelled_thresholds(scene):
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
    policy, _, world = build(scene, layer, scripted=answers(repairs=(repair("use_part"),)))
    decisions = drive(policy, world, 200)
    records = recorded(decisions)
    assert {"invalid_plan", "repair_invalid"} & {r["reason"] for r in records}
    for record in records:
        assert record["reason"] in kernel.REASONS and record["status"] in kernel.OUTCOME_STATUSES
        assert set(record) >= {"n", "step", "skill", "relation", "status", "reason", "evidence", "thresholds",
                               "evidence_text", "pids", "plan_step", "local_retries", "steps", "calls",
                               "model_claims"}
        assert record["skill"] == record["relation"] and len(record["evidence_text"]) <= 200
        for item in record["thresholds"]:
            assert item["label"] in th.LABEL_CLASSES and item["derivation"].strip() and item["name"]
        json.dumps(record, allow_nan=False)
    json.dumps(policy.agent_stats(), allow_nan=False)


# ------------------------------------------------------------- 9. end to end with the real relations (BR)
AC4_STEP_CEILING = 1300          # spec AC-4; the step budget is being re-budgeted (decision §7 risk 6)


def test_end_to_end_task2_fake_with_the_real_relations():
    relations = pytest.importorskip("gpt_field.agent_speedup_v2_relations")
    assert relations is not None
    world = fakes.FakeTwoViewSim(fakes.task2_like_scene(), prompt=TASK2)
    client = AgentSpeedUpV2Client(task_prompt=TASK2, transport=ScriptedV2Transport(fakes.FakeOracleAnswers(world)))
    policy = AgentSpeedUpV2Policy(client)
    policy.reset()
    observation, steps = world.observation(), 0
    for steps in range(1, AC4_STEP_CEILING + 1):
        decision = policy.act(observation)
        after = world.step(decision.action)
        policy.observe_transition(observation, decision, after)
        observation = after
        if world.goal_satisfied():
            break
    assert world.goal_satisfied() and steps <= AC4_STEP_CEILING                # §C13 test 7, AC-4 ceiling
    assert len(client.transport.kinds) <= 4                                    # plan + points (decision §2.7: 3-4)
    sequence = [r["relation"] for r in policy.outcomes if r["status"] == "succeeded"]
    assert sequence[0] == "calibrated" and "registered" in sequence[1:sequence.index("held")]
    # registered outcomes may come between the relations (decision §2.6: grasp(C, obj=D) runs survey_hover(D) while
    # D is partial; place_on's destination survey); the relations themselves keep their order
    relations = [name for name in sequence if name != "registered"]
    distinct = [name for k, name in enumerate(relations) if k == 0 or relations[k - 1] != name]
    assert distinct[:4] == ["calibrated", "held", "turned", "held"]             # supported_on may be cut off


# ------------------------------------------- 10. handle lens BX-1 / BX-2, BI review BRR-2 / F5 / F7, BC-1 wiring
def request_text(call):
    return json.loads(call["input"][0]["content"][0]["text"])


def repair_requests(transport):
    return [(request_text(call), meta) for call, kind, meta in zip(transport.calls, transport.kinds, transport.metas)
            if kind in ("repair", "review")]


def applied_actions(decisions):
    return [(r["valid"], (r["applied"] or {}).get("action")) for r in events(decisions, "repair")]


def test_the_fulfiller_sends_the_pointing_purpose(scene):
    """BC-1: the PointNeed's purpose reaches the pointing request (and its meta)."""
    policy, transport, world = build(scene, StandIn())
    drive(policy, world, 400, stop=outcome_seen("supported_on"))
    pointing = [(request_text(call), meta) for call, meta in zip(transport.calls, transport.metas)
                if meta["kind"].startswith("point_")]
    assert pointing and all(text["purpose"] == "grasp_point" == meta["purpose"] for text, meta in pointing)


def test_a_third_repoint_of_one_part_in_one_step_is_refused_and_reasked_with_the_way_out(scene):
    """BX-1: after REPOINT_LIMIT (2) failed repoints of the same part within one step, a further repoint answer is
    not applied: the repair is re-asked once carrying the issue that names use_part, replan and skip_step."""
    assert policy_module.REPOINT_LIMIT == 2 and policy_module.LABELS["REPOINT_LIMIT"][0] == "DD"
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "pointed_not_grasp_point"}] * 3})
    scripted = answers(repairs=(repair("repoint", part="A"), repair("repoint", part="A"), repair("repoint", part="A"),
                                repair("skip_step")))
    policy, transport, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 700, stop=outcome_seen("supported_on"))
    assert applied_actions(decisions) == [(True, "repoint"), (True, "repoint"), (False, None), (True, "skip_step")]
    refused = events(decisions, "repair")[2]
    text = "repoint of part A failed 2 times at this object; choose use_part, replan or skip_step"
    assert [(i["code"], i["pid"], i["text"]) for i in refused["issues"]] == [("repoint_limit", "A", text)]
    sent = repair_requests(transport)
    assert [s["previous_answer_rejected"] for s, _ in sent] == [None, None, None, "repoint_limit: " + text]
    assert [row[5] for row in layer.controller_calls("turn")] == [1, 2, 3]       # no fourth run of the turn
    assert [row for row in layer.log if row == ("register_regions", ("A",))] == [("register_regions", ("A",))] * 2
    assert [s["status"] for s in policy.agent_stats()["steps"]] == ["skipped", "done"]


def test_the_repoint_limit_counts_one_grip_part_within_one_step(scene):
    """Negative BX-1: a repoint of a part that is no grip part of the step (B, the effect region), a repoint of a
    grip part in another step, and a repoint after a successful grasp of the part are applied."""
    fails = [{"status": "failed", "reason": "pointed_not_grasp_point"}] * 3
    layer = StandIn(script={"turn": fails, "place_on": fails[:2]})
    scripted = answers(repairs=(repair("repoint", part="A"), repair("repoint", part="A"), repair("repoint", part="B"),
                                repair("repoint", part="C"), repair("repoint", part="C")))
    policy, _, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 1200, stop=outcome_seen("supported_on", "placed"))
    assert applied_actions(decisions) == [(True, "repoint")] * 5
    assert [(row[1], row[5]) for row in layer.controller_calls()] == [("turn", 1), ("turn", 2), ("turn", 3),
                                                                     ("turn", 4), ("place_on", 1), ("place_on", 2),
                                                                     ("place_on", 3)]
    # a successful grasp of A (a "held" outcome of the step) clears its count: the third repoint is applied
    runner = policy._runner
    state = runner.states[0]
    runner._repoints[(id(state), "A")] = policy_module.REPOINT_LIMIT
    answer = kernel.RepairAnswer("again", "repoint", "A", None)
    assert [i.code for i in runner.runner_issues(answer, state, None)] == ["repoint_limit"]
    runner.observe_outcome(outcome("held", "succeeded", "held_for_turn", {"obj": "A", "grip": ["A"]}), state)
    assert runner.runner_issues(answer, state, None) == []


def test_a_retry_of_a_repeated_motionless_refusal_is_refused_and_reasked(scene):
    """BRR-2 (BX half): a retry answered for a failure the relation layer flagged ``motionless_repeat`` (the same
    refusal without motion again from an unchanged hand state) is not executed; the repair is re-asked once with the
    retry_unchanged issue. A first motionless failure (no repeat yet) may still be retried."""
    repeated = {"status": "failed", "reason": "reach_limit", "evidence": {"motionless": True, "motionless_repeat": 2}}
    layer = StandIn(script={"turn": [repeated]})
    policy, transport, world = build(scene, layer, scripted=answers(repairs=(repair("retry"), repair("skip_step"))))
    decisions = drive(policy, world, 500, stop=outcome_seen("supported_on"))
    assert applied_actions(decisions) == [(False, None), (True, "skip_step")]
    issue = events(decisions, "repair")[0]["issues"][0]
    assert issue["code"] == "retry_unchanged" and "2" in issue["text"]
    sent = repair_requests(transport)
    assert sent[0][0]["previous_answer_rejected"] is None
    assert sent[1][0]["previous_answer_rejected"] == "retry_unchanged: " + issue["text"]
    assert len(layer.controller_calls("turn")) == 1
    first = {"status": "failed", "reason": "reach_limit", "evidence": {"motionless": True}}
    layer = StandIn(script={"turn": [first]})
    policy, _, world = build(scene, layer, scripted=answers(repairs=(repair("retry"),)))
    decisions = drive(policy, world, 500, stop=outcome_seen("supported_on"))
    assert applied_actions(decisions) == [(True, "retry")] and len(layer.controller_calls("turn")) == 2


def test_a_failed_precondition_names_the_unmet_part_in_the_evidence_text(scene):
    """F5: the outcome of an unmet StepRunner precondition says which part is unmet and why, so the model is not
    told the step's own part failed (the repair payload carries only the text)."""
    layer = FailE()
    policy, transport, world = build(scene, layer)
    decisions = drive(policy, world, 300, stop=lambda p, d: bool(events([d], "repair")))
    failed = [r for r in recorded(decisions) if r["status"] != "succeeded"][0]
    assert failed["reason"] == "part_not_found" and failed["pids"]["obj"] == "D"
    text = failed["evidence_text"]
    assert text == "place_on of part D not started: part E needs a keep-out, registration failed part_not_found"
    assert repair_requests(transport)[0][0]["last_outcome"]["evidence_text"] == text
    # a registration that waits for an empty hand says so (the repoint-while-holding path)
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
    scripted = answers(plans=(plan_json((HOLD_A, TURN), parts=PARTS[:2]),),
                       repairs=(repair("repoint", part="B"), HOLD_STILL))
    policy, _, world = build(scene, layer, scripted=scripted)
    decisions = drive(policy, world, 400)
    unmet = [r for r in recorded(decisions) if r["reason"] == "precondition_unmet"][0]
    assert unmet["evidence_text"] == ("turn of part A not started: part B needs a region registration, waits for "
                                      "an empty hand (holding part A)")       # B is no grip part: no BX-2 fact


class DestinationAudit(StandIn):
    """Region parts register with an empty hand; each place_on call records (held pid, its ref's
    registered_empty_hand) when it starts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.destination = []
        base = self.CONTROLLERS["place_on"]

        def audited(ctx, step):
            part = ctx.scene.parts.get(step.ref)
            self.destination.append((getattr(ctx.holding, "pid", None), bool(part.registered_empty_hand)))
            return (yield from base(ctx, step))
        self.CONTROLLERS["place_on"] = audited

    def register_regions(self, ctx, plan, roles):
        out = yield from super().register_regions(ctx, plan, roles)
        for pid, part in list(ctx.scene.parts.items()):
            if part.status == "registered":
                ctx.scene.parts[pid] = dataclasses.replace(part, registered_empty_hand=True)
        return out


@pytest.mark.parametrize("reason, resurvey", [("no_support_within_bound", True),
                                              ("support_above_destination", False)])
def test_a_place_on_retry_after_no_support_within_bound_releases_and_resurveys(scene, reason, resurvey):
    """F7 / [C-32]: a repair that reruns a place_on failed no_support_within_bound runs the StepRunner's implicit
    release first and leaves the destination for an empty-hand re-survey (place_on step 1); another failure reason
    keeps the held grip part and the destination's registration."""
    layer = DestinationAudit(script={"place_on": [{"status": "failed", "reason": reason}]})
    scripted = answers(plans=(plan_json((HOLD_D, PLACE)),), repairs=(repair("retry"),))
    policy, _, world = build(scene, layer, scripted=scripted)
    drive(policy, world, 500, stop=outcome_seen("supported_on", "placed"))
    order = [row[0] if row[0] != "controller" else row[1] for row in layer.log]
    if resurvey:
        assert order == ["calibrate_plus", "register_regions", "hold", "place_on", "release", "place_on"]
        assert layer.destination == [("D", True), (None, False)]
    else:
        assert order == ["calibrate_plus", "register_regions", "hold", "place_on", "place_on"]
        assert layer.destination == [("D", True), ("D", True)]


class RegionCells(StandIn):
    """register_regions also gives the parts in ``cells`` their HeightField cells (region_cells)."""

    def __init__(self, cells, **kwargs):
        super().__init__(**kwargs)
        self.cells = {pid: np.asarray(value, dtype=int) for pid, value in cells.items()}

    def register_regions(self, ctx, plan, roles):
        out = yield from super().register_regions(ctx, plan, roles)
        for pid, value in self.cells.items():
            if pid in ctx.scene.parts and ctx.scene.parts[pid].xyz is not None:
                ctx.scene.parts[pid] = dataclasses.replace(ctx.scene.parts[pid], region_cells=value)
        return out


SAME = [[i, j] for i in range(10, 14) for j in range(20, 24)]
APART = [[i, j] for i in range(40, 44) for j in range(20, 24)]


def test_repair_evidence_carries_rounds_repoints_the_shared_region_and_the_ringed_point(scene):
    """BX-2: the repair of a failed grasp step adds the StepRunner's measured facts to the evidence text: the
    rejected pointing rounds of the part in this step, its failed repoints, the region it shares with the step's
    object (IoU >= OBJECT_OVERLAP_REPORT_IOU), and the rejected pointed pixel ringed on the repair's wrist image (only
    while the hand is still at the pose of that check)."""
    checks = [("C", (300.0, 260.0), False), ("C", (290.0, 250.0), False)]
    fail = {"status": "failed", "reason": "pointed_not_grasp_point", "checks": checks}
    layer = RegionCells({"C": SAME, "D": SAME}, script={"place_on": [fail, fail]})
    policy, transport, world = build(scene, layer, scripted=answers(repairs=(repair("repoint", part="C"),
                                                                             HOLD_STILL)))
    decisions = drive(policy, world, 900, stop=lambda p, d: len(repair_requests(transport)) >= 2)
    sent = repair_requests(transport)
    first, second = (text["last_outcome"]["evidence_text"] for text, _ in sent[:2])
    assert first == ("pointed_not_grasp_point: stand-in; part C: 2 pointings rejected, region overlaps part D "
                     "(IoU 1.00); rejected point ringed in image 2")             # the layer's head first
    # gap round RL-C1: the rounds are this attempt's (2), not the step's running total; the repoints stay per step
    assert second == ("pointed_not_grasp_point: stand-in; part C: 2 pointings rejected, 1 repoint failed, region "
                      "overlaps part D (IoU 1.00); rejected point ringed in image 2")
    assert len(first) <= kernel.EVIDENCE_TEXT_MAX_CHARS and len(second) <= kernel.EVIDENCE_TEXT_MAX_CHARS
    assert all(meta["marker_drawn"] is True and meta["marker_uv"] == [290.0, 250.0] for _, meta in sent[:2])
    failed = [r for r in recorded(decisions) if r["relation"] == "supported_on"][0]
    assert failed["evidence"]["runner_facts"]["rejected_rounds"] == {"C": 2}
    assert failed["evidence"]["runner_facts"]["shared_region"] == {"part": "C", "obj": "D", "iou": 1.0}
    assert failed["evidence_text"] == first
    # negative: disjoint regions add no overlap; a hand that left the check pose gets no ring and no ring claim
    moved = dict(fail, net_move=6)
    layer = RegionCells({"C": SAME, "D": APART}, script={"place_on": [moved]})
    policy, transport, world = build(scene, layer)
    drive(policy, world, 600, stop=lambda p, d: bool(events([d], "repair")))
    (text, meta), = repair_requests(transport)
    evidence = text["last_outcome"]["evidence_text"]
    assert evidence == "pointed_not_grasp_point: stand-in; part C: 2 pointings rejected"
    assert meta["marker_uv"] is None and meta["marker_drawn"] is False
    # nothing measured (no pointing rounds, no region of the grip part): the layer's text is sent unchanged
    layer = StandIn(script={"turn": [{"status": "failed", "reason": "turn_slip"}]})
    policy, transport, world = build(scene, layer)
    drive(policy, world, 400, stop=lambda p, d: bool(events([d], "repair")))
    assert repair_requests(transport)[0][0]["last_outcome"]["evidence_text"] == "turn_slip: stand-in"
