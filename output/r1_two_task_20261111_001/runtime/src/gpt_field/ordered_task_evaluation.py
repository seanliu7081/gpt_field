"""Evaluator-only ordered composite-task progress; never policy observation data.

Promoted from the frozen custom task's ordered_success.py. Progress is advanced
only by the UC worker after actual policy steps, never by reset settling or
post-terminal release/settle diagnostics. LIBERO raw and physical goals remain
available independently of the extra chronological completion requirement.
"""
from __future__ import annotations

ORDERED_KIND = "drawer_bowl_moka_v1"
ORDERED_EVALUATOR_VERSION = "drawer-bowl-moka-ordered-evaluator-v1"

BOWL = "akita_black_bowl_1"
CABINET = "wooden_cabinet_1"
STOVE = "flat_stove_1"
POT = "moka_pot_1"

FINAL_KEYS = ("bowl_on_cabinet_top", "drawer_closed", "stove_on", "moka_on_stove")
PREDICATE_KEYS = (*FINAL_KEYS, "drawer_open", "bowl_inside_drawer", "stove_off")
STAGES = (
    "drawer_opened",
    "bowl_placed_on_cabinet_top_while_drawer_open",
    "drawer_closed_with_bowl_on_top",
    "stove_turned_on_after_drawer_stage",
    "moka_placed_on_stove_after_turn_on",
)


def _values(predicates):
    missing = set(PREDICATE_KEYS) - predicates.keys()
    if missing:
        raise ValueError("Missing task predicates: " + ", ".join(sorted(missing)))
    return {key: bool(predicates[key]) for key in PREDICATE_KEYS}


class OrderedProgress:
    """Finite-state order checker driven by consecutive predicate observations.

    Each stage requires a false-to-true edge after the previous stage. At most
    one stage advances per observation. An early/out-of-order action does not
    permanently fail an episode: repeating it after its prerequisites can pass.
    Completed stages remain recorded, but success also requires all final goals
    to hold now. Restoring an arbitrary final state cannot establish an episode.
    """

    def __init__(self):
        self.invalidate("not_initialized")

    def invalidate(self, reason="external_state_change"):
        self.stage = 0
        self.observation_index = 0
        self.previous = None
        self.events = []
        self.initial_state_valid = False
        self.invalid_reason = reason

    def reset(self, predicates):
        self.invalidate("initial_predicates_do_not_match_task")
        current = _values(predicates)
        self.previous = current
        self.initial_state_valid = bool(
            current["drawer_closed"]
            and not current["drawer_open"]
            and current["bowl_inside_drawer"]
            and not current["bowl_on_cabinet_top"]
            and current["stove_off"]
            and not current["stove_on"]
            and not current["moka_on_stove"]
        )
        if self.initial_state_valid:
            self.invalid_reason = None
        return self.snapshot()

    @property
    def ordered_complete(self):
        return self.initial_state_valid and self.stage == len(STAGES)

    def observe(self, predicates):
        current = _values(predicates)
        previous = self.previous
        self.observation_index += 1
        advances = False
        if self.initial_state_valid and previous is not None:
            def rose(key):
                return not previous[key] and current[key]

            if self.stage == 0:
                advances = rose("drawer_open")
            elif self.stage == 1:
                advances = rose("bowl_on_cabinet_top") and current["drawer_open"]
            elif self.stage == 2:
                advances = rose("drawer_closed") and current["bowl_on_cabinet_top"]
            elif self.stage == 3:
                advances = (rose("stove_on") and current["drawer_closed"]
                            and current["bowl_on_cabinet_top"])
            elif self.stage == 4:
                advances = rose("moka_on_stove") and all(current[key] for key in FINAL_KEYS)
        if advances:
            self.events.append({"stage": STAGES[self.stage],
                                "observation_index": self.observation_index})
            self.stage += 1
        self.previous = current
        return self.snapshot()

    def snapshot(self):
        return {
            "kind": "ordered_custom_task_progress_v1",
            "initial_state_valid": self.initial_state_valid,
            "invalid_reason": self.invalid_reason,
            "completed_stage_count": self.stage,
            "next_stage": STAGES[self.stage] if self.stage < len(STAGES) else None,
            "ordered_complete": self.ordered_complete,
            "observation_index": self.observation_index,
            "events": [dict(event) for event in self.events],
        }


def drawer_bowl_moka_predicates(env):
    """Read stock predicates at the worker's actual step endpoint."""
    evaluate = env.env._eval_predicate
    return {
        "bowl_on_cabinet_top": bool(evaluate(["on", BOWL, CABINET + "_top_side"])),
        "drawer_closed": bool(evaluate(["close", CABINET + "_top_region"])),
        "moka_on_stove": bool(evaluate(["on", POT, STOVE + "_cook_region"])),
        "drawer_open": bool(evaluate(["open", CABINET + "_top_region"])),
        "bowl_inside_drawer": bool(evaluate(["in", BOWL, CABINET + "_top_region"])),
        "stove_off": bool(evaluate(["turnoff", STOVE])),
        # LIBERO's turn predicates also update burner visualization state.
        # Match normal update_state() last, including intermediate knob angles.
        "stove_on": bool(evaluate(["turnon", STOVE])),
    }


def ordered_snapshot(progress, task_key):
    return {**progress.snapshot(), "enabled": True, "task_key": task_key,
            "version": ORDERED_EVALUATOR_VERSION}


def qualify_status(status, progress, task_key, *, stop_on_success):
    """Keep raw LIBERO status and qualify only task completion/termination."""
    result = dict(status)
    raw = result.get("libero_success") is True
    success = raw and progress.ordered_complete
    result.update(success_qualification="ordered_custom_goal",
                  ordered_task=ordered_snapshot(progress, task_key),
                  ordered_success=success, success=success,
                  ordered_collision_free_success=(success and not
                      (result.get("collision") or {}).get("detected", False)),
                  terminated=bool((success and stop_on_success) or result.get("horizon_reached")))
    return result


def qualify_truth(result, progress, task_key):
    """Freeze chronological evidence; diagnostics never create ordered success.

    Preserve raw physical truth separately. The qualified strict conjunction is
    consumed by the existing physical_summary hold-only exception, ensuring a
    release/settle diagnostic cannot rescue an incomplete sequence.
    """
    result = dict(result)
    complete = progress.ordered_complete
    physical = result.get("strict_conjunction") is True
    result.update(raw_strict_conjunction=physical,
                  strict_conjunction=physical and complete,
                  ordered_task=ordered_snapshot(progress, task_key),
                  ordered_success=bool(complete and result.get("raw_conjunction") is True),
                  ordered_completion_checks={"sequence_complete": complete},
                  ordered_completion_why_not=[] if complete else ["ordered_sequence_incomplete"])
    return result
