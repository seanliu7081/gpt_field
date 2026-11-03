"""Task-local, evaluator-only order checking for the drawer/bowl/stove task.

Only steps through ``OrderedTaskEnv.step`` advance the sequence. The wrapped
LIBERO reward/done tuple is returned unchanged, and ``raw_goal_success`` exposes
its unordered final-state goal separately. Consumers of this task must use
``check_success`` for the ordered task, not the raw step reward/done value.

This module is not integrated with the shared UC7.1 runner. It supplies no
privileged predicate/progress data to a policy observation. Direct mutation via
``env.env`` or ``env.sim`` is an evaluator bypass: callers must invalidate the
tracker and restore the initial state before calling ``reset_progress`` again.
The wrapper's own reset/state-setting methods invalidate progress automatically.
"""

from __future__ import annotations


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


class OrderedTaskEnv:
    """Forwarding wrapper for a settled raw LIBERO OffScreenRenderEnv.

    Construct after restoring/settling the initial state, then call
    ``reset_progress()``. Public ``goal_predicates`` and ``ordered_progress`` are
    evaluator diagnostics; their values must not be added to policy input.
    """

    def __init__(self, raw_env):
        self._raw_env = raw_env
        self._progress = OrderedProgress()

    def __getattr__(self, name):
        return getattr(self._raw_env, name)

    def goal_predicates(self):
        evaluate = self._raw_env.env._eval_predicate
        return {
            "bowl_on_cabinet_top": bool(evaluate(["on", BOWL, CABINET + "_top_side"])),
            "drawer_closed": bool(evaluate(["close", CABINET + "_top_region"])),
            "moka_on_stove": bool(evaluate(["on", POT, STOVE + "_cook_region"])),
            "drawer_open": bool(evaluate(["open", CABINET + "_top_region"])),
            "bowl_inside_drawer": bool(evaluate(["in", BOWL, CABINET + "_top_region"])),
            "stove_off": bool(evaluate(["turnoff", STOVE])),
            # LIBERO's turn predicates also update its burner visualization.
            # Match its usual update_state() last, including intermediate angles.
            "stove_on": bool(evaluate(["turnon", STOVE])),
        }

    def raw_goal_success(self):
        return bool(self._raw_env.check_success())

    def check_success(self):
        predicates = self.goal_predicates()
        return bool(self._progress.ordered_complete
                    and all(predicates[key] for key in FINAL_KEYS)
                    and self.raw_goal_success())

    def ordered_progress(self):
        result = self._progress.snapshot()
        result["final_predicates"] = self.goal_predicates()
        result["raw_goal_success"] = self.raw_goal_success()
        result["ordered_success"] = self.check_success()
        return result

    def reset_progress(self):
        return self._progress.reset(self.goal_predicates())

    def invalidate_progress(self, reason="external_state_change"):
        self._progress.invalidate(reason)

    def step(self, action):
        result = self._raw_env.step(action)
        self._progress.observe(self.goal_predicates())
        return result

    def reset(self, *args, **kwargs):
        self.invalidate_progress("reset_requires_reset_progress")
        return self._raw_env.reset(*args, **kwargs)

    def set_state(self, *args, **kwargs):
        self.invalidate_progress("set_state_requires_reset_progress")
        return self._raw_env.set_state(*args, **kwargs)

    def set_init_state(self, *args, **kwargs):
        self.invalidate_progress("set_init_state_requires_reset_progress")
        return self._raw_env.set_init_state(*args, **kwargs)

    def regenerate_obs_from_state(self, *args, **kwargs):
        self.invalidate_progress("regenerate_obs_from_state_requires_reset_progress")
        return self._raw_env.regenerate_obs_from_state(*args, **kwargs)

    def reset_from_xml_string(self, *args, **kwargs):
        self.invalidate_progress("reset_from_xml_string_requires_reset_progress")
        return self._raw_env.reset_from_xml_string(*args, **kwargs)
