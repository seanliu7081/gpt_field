"""Executable skill prerequisites for the user-check framework.

These adapters consume the same measured scene and controller predicates used
by execution. They contain no task names, evaluator state or inferred success.
Observation is an action with prerequisites; requesting it is not evidence that
it happened. The legacy v2 modules are deliberately not modified.
"""

from __future__ import annotations

from dataclasses import replace

from . import agent_speedup_v2_planner as planner
from . import agent_speedup_v2_relations as relations
from .agent_speedup_v2_types import jsonable
from .agent_speedup_v2_user_check_spatial import ground_region, region_precondition


_RELATION = {"hold": "held", "turn": "turned", "place_on": "supported_on", "release": "released"}


def destination_readiness(ctx, record) -> dict:
    """One definition of usable destination for preflight, survey and place_on.

    In particular a registered top height alone is insufficient: aim_target
    consumes _region_cells, so we evaluate that exact geometry conversion here.
    An empty explicit cell array can still be useful if its polygon rasterizes
    to cells, exactly as in the planner.
    """
    pid = getattr(record, "pid", None)
    unmet = []
    if record is None:
        return {"ready": False, "unmet": [{"predicate": "known_destination", "part": None,
                                             "value": False}],
                "evidence": {"destination": None}, "can_observe": False}
    is_region = pid in getattr(ctx.scene, "user_check_regions", {})
    region = ground_region(ctx, pid) if is_region else None
    if is_region:
        record = ctx.scene.parts[pid]
    registered = relations._registered(record)
    empty_registration = bool(getattr(record, "registered_empty_hand", False))
    measured_top = getattr(record, "top_z", None) is not None
    cell_count = len(planner._region_cells(record, ctx.scene.hf))
    pre = region_precondition(ctx, record) if is_region else relations._destination_check(ctx, record)
    for name, value in (
        ("destination_registered", registered),
        ("destination_registered_empty_hand", empty_registration),
        ("destination_top_measured", measured_top),
        ("destination_cells_nonempty", cell_count > 0),
        ("destination_support_distinguishable", bool(pre.get("ok"))),
    ):
        if not value:
            unmet.append({"predicate": name, "part": pid, "value": value})
    if unmet and ctx.holding is not None and not is_region:
        unmet.append({"predicate": "empty_hand_for_observation", "part": pid, "value": False,
                      "holding": getattr(ctx.holding, "pid", None)})
    evidence = {"destination": pid, "status": record.status, "registered": registered,
                "registered_empty_hand": empty_registration,
                "registration_step": getattr(record, "registration_step", None),
                "top_z": None if record.top_z is None else jsonable(record.top_z),
                "region_cells": cell_count, "destination_precondition": jsonable(pre),
                "entity_kind": "region" if is_region else "object", "region": region}
    return {"ready": not unmet, "unmet": unmet, "evidence": evidence,
            "can_observe": ctx.holding is None or is_region}


def destination_check(ctx, record) -> dict:
    """Native relation-check shape, backed by the shared readiness predicate."""
    is_region = record.pid in getattr(ctx.scene, "user_check_regions", {})
    pre = dict(region_precondition(ctx, record) if is_region else relations._destination_check(ctx, record))
    readiness = destination_readiness(ctx, record)
    return dict(pre, ok=readiness["ready"], reason=None if readiness["ready"] else "geometry_unmeasured",
                values={**pre.get("values", {}), "region_cells": readiness["evidence"]["region_cells"],
                        "registered_empty_hand": readiness["evidence"]["registered_empty_hand"]},
                unmet=readiness["unmet"])


def destination_need(ctx):
    """survey_hover's success predicate, identical to execution's predicate."""
    return lambda record: destination_readiness(ctx, record)["ready"]


def skill_readiness(ctx, step) -> dict:
    """Check stateful prerequisites without moving, observing or changing state.

    Existing grasp and turn controllers own their detailed contact measurements;
    requiring these to exist already would prevent their normal observation path.
    """
    parts = ctx.scene.parts
    held = ctx.holding
    held_pid = getattr(held, "pid", None)
    held_grip = getattr(held, "grip_pid", None)
    grips = list(step.grip) or [step.obj]
    unmet = []
    evidence = {"skill": step.do, "holding": held_pid, "held_grip": held_grip}
    if step.do == "release":
        if held is not None and held_pid != step.obj:
            unmet.append({"predicate": "held_object_matches", "part": step.obj,
                          "value": False, "holding": held_pid})
    else:
        for pid in dict.fromkeys([step.obj, *grips]):
            if pid not in parts:
                unmet.append({"predicate": "known_part", "part": pid, "value": False})
        if held is not None and (held_pid != step.obj or held_grip not in grips):
            unmet.append({"predicate": "hand_empty_or_matching_grip", "part": step.obj,
                          "value": False, "holding": held_pid, "held_grip": held_grip})
    can_observe = False
    if step.do == "place_on":
        destination = destination_readiness(ctx, parts.get(step.ref))
        unmet.extend(dict(item, part=item.get("part") or step.ref) for item in destination["unmet"])
        evidence["destination"] = destination["evidence"]
        can_observe = destination["can_observe"] and not destination["ready"]
    return {"ready": not unmet, "unmet": unmet, "evidence": evidence,
            "can_observe": can_observe}


def _blocked(ctx, relation, pids, readiness, *, start, observation=None):
    feedback = {"status": "blocked", "executed": False, "unmet_conditions": readiness["unmet"],
                "evidence": readiness["evidence"], "can_observe": readiness["can_observe"]}
    if observation is not None:
        feedback["observation"] = observation
    evidence = {"execution_feedback": feedback}
    return relations._outcome(ctx, relation, "failed", "precondition_unmet", pids=pids,
                              evidence=evidence, start=start)


def ensure_ready(ctx, step, *, relation_layer=None):
    """Return None to execute, or a recorded blocked outcome to recover from.

    A destination may get one controller-owned survey while the hand is empty.
    The survey's own bounded lower-hover refinement is retained. Readiness is
    always checked afterwards, even if an observation reports success.
    """
    start = int(ctx.step)
    readiness = skill_readiness(ctx, step)
    observation = None
    if readiness["ready"]:
        return None
    destination_only = all(item["predicate"].startswith("destination_") for item in readiness["unmet"])
    if step.do == "place_on" and readiness["can_observe"] and destination_only:
        if step.ref in getattr(ctx.scene, "user_check_regions", {}):
            surveyed = yield from observe_region(ctx, step.ref, relation_layer=relation_layer)
        else:
            surveyed = yield from (relation_layer or relations).survey_hover(
                ctx, step.ref, purpose="skill_precondition", need=destination_need(ctx))
        observation = {"status": surveyed.status, "reason": surveyed.reason,
                       "evidence": jsonable(surveyed.evidence)}
        readiness = skill_readiness(ctx, step)
        if surveyed.status == "succeeded" and readiness["ready"]:
            return None
        if readiness["ready"]:
            readiness = dict(readiness, ready=False, unmet=[{
                "predicate": "observation_completed", "part": step.ref, "value": False,
                "reason": surveyed.reason}])
    return _blocked(ctx, _RELATION.get(step.do, step.do),
                    relations._pids(step.obj, list(step.grip), step.ref), readiness,
                    start=start, observation=observation)


def observe_region(ctx, pid, *, relation_layer=None):
    """Explicit observation; a held object is never implicitly released.

    Refusing observation leaves the old record intact. Repoint requests fresh
    localization before the survey; the old seed must not pull the hover back
    to the location the model is trying to correct. A destination-role record
    must pass the same geometry check used by place_on before it counts as ready.
    """
    start = int(ctx.step)
    record = ctx.scene.parts.get(pid)
    if pid in getattr(ctx.scene, "user_check_regions", {}):
        return (yield from observe_spatial_region(ctx, pid, relation_layer=relation_layer))
    unmet = []
    if record is None:
        unmet.append({"predicate": "known_part", "part": pid, "value": False})
    if ctx.holding is not None:
        unmet.append({"predicate": "empty_hand_for_observation", "part": pid, "value": False,
                      "holding": getattr(ctx.holding, "pid", None)})
    if unmet:
        readiness = {"ready": False, "unmet": unmet, "can_observe": False,
                     "evidence": {"part": pid, "observation_started": False}}
        return _blocked(ctx, "registered", relations._pids(pid), readiness, start=start)
    is_destination = "ref" in getattr(record, "roles", ())
    layer = relation_layer or relations
    seed = yield from layer._coarse_seed(ctx, pid, purpose="region_seed", previous="model_requested_repoint")
    if seed is None:
        return relations._outcome(ctx, "registered", "failed", "part_not_found", pids=relations._pids(pid),
            evidence={"phase": "model_observation", "execution_feedback": {
                "status": "failed", "executed": True, "unmet_conditions": [
                    {"predicate": "visible_part_seed", "part": pid, "value": False}],
                "observed_part": pid, "geometry_updated": False, "geometry_invalidated": False,
                "source": "model_pointing"}}, start=start)
    # The hand prerequisite has passed and new localization is available. Only
    # now invalidate geometry. drop() retains old plan cells, so explicitly
    # replace the seed list: _seed_lift chooses its FIRST wrist seed.
    ctx.scene.drop(pid)
    ctx.scene.parts[pid] = replace(ctx.scene.parts[pid], seeds=[seed])
    for key in ("seed_lift", "start_records", "shared_region"):
        ctx.memory.get(key, {}).pop(pid, None)
    # Fresh localization supersedes the old independent starting point too.
    getattr(ctx.scene, "_start_record", {}).pop(pid, None)
    layer._seed_lift(ctx, pid, ctx.view)
    ctx.scene.register(pid, ctx.view, seed=seed, empty_hand=True, allow_partial=True)
    need = destination_need(ctx) if is_destination else None
    result = yield from layer.survey_hover(ctx, pid, purpose="model_observation", need=need)
    current = ctx.scene.parts.get(pid)
    readiness = destination_readiness(ctx, current) if is_destination else {
        "ready": relations._registered(current), "unmet": [], "can_observe": True}
    verified = result.status == "succeeded" and readiness["ready"]
    result.evidence["execution_feedback"] = {
        "status": "succeeded" if verified else "pending_verification" if result.status == "succeeded" else "failed",
        "executed": True, "unmet_conditions": readiness["unmet"], "observed_part": pid,
        "geometry_ready": readiness["ready"], "geometry_invalidated": True,
        "geometry_updated": relations._registered(current), "source": "measured_survey",
    }
    return result


def observe_spatial_region(ctx, pid, *, relation_layer=None):
    """Refresh physical dependencies, then derive a surface region from observations.

    Occupied-hand requests may use already observed geometry. If new support or
    anchor data is needed, return the actual dependency; no queued registration
    is misreported as completed and the held object is not implicitly released.
    """
    start = int(ctx.step)
    layer = relation_layer or relations
    result = ground_region(ctx, pid)
    observations = []
    if not result["ready"] and ctx.holding is None:
        spec = ctx.scene.user_check_regions[pid]
        dependencies = [p for p in (spec.get("support"), spec.get("anchor"))
                        if p and p != "observed_plane"]
        for dependency in dict.fromkeys(dependencies):
            measured = yield from layer.survey_hover(ctx, dependency, purpose="region_dependency")
            observations.append({"part": dependency, "status": measured.status, "reason": measured.reason})
        # Existing scene plane fills use visible calibrated observations only.
        # A missing plane requires a physical survey, never object segmentation
        # of the table region itself.
        try:
            ctx.scene.fill_unknown(ctx.view)
        except (ValueError, AttributeError):
            pass
        result = ground_region(ctx, pid)
    unmet = [] if result["ready"] else [{"predicate": "observed_spatial_region", "part": pid,
        "value": False, "reason": result.get("reason"), "dependency": result.get("dependency")}]
    if unmet and ctx.holding is not None:
        unmet.append({"predicate": "fresh_dependency_geometry", "part": result.get("dependency"),
                      "value": False, "holding": getattr(ctx.holding, "pid", None),
                      "remedy": "reveal_observed_support_or_put_down_then_observe_dependency"})
    evidence = {"spatial_region": result, "dependency_observations": observations,
        "execution_feedback": {"status": "succeeded" if result["ready"] else "blocked",
            "executed": bool(observations), "unmet_conditions": unmet,
            "geometry_ready": bool(result["ready"]), "geometry_updated": bool(result["ready"]),
            "observed_part": pid, "source": "observed_support_region"}}
    return relations._outcome(ctx, "registered", "succeeded" if result["ready"] else "failed",
        "registered" if result["ready"] else "geometry_unmeasured", pids=relations._pids(pid),
        evidence=evidence, start=start)
