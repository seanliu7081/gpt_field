"""Task-local behavioral checks; run with Python, no simulator/model needed."""

import unittest

from ordered_success import FINAL_KEYS, OrderedProgress, OrderedTaskEnv


INITIAL = {
    "bowl_on_cabinet_top": False, "drawer_closed": True,
    "stove_on": False, "moka_on_stove": False, "drawer_open": False,
    "bowl_inside_drawer": True, "stove_off": True,
}


def changed(state, **changes):
    return dict(state, **changes)


OPEN = changed(INITIAL, drawer_closed=False, drawer_open=True)
BOWL_TOP = changed(OPEN, bowl_inside_drawer=False, bowl_on_cabinet_top=True)
CLOSED = changed(BOWL_TOP, drawer_open=False, drawer_closed=True)
STOVE_ON = changed(CLOSED, stove_off=False, stove_on=True)
FINAL = changed(STOVE_ON, moka_on_stove=True)
ORDERED_TRACE = (OPEN, BOWL_TOP, CLOSED, STOVE_ON, FINAL)


class FakeRawEnv:
    def __init__(self):
        self.env = self
        self.sim = object()
        self.state = dict(INITIAL)
        self.step_result = ({"rgb": "unchanged"}, 1, True, {"raw": "unchanged"})

    def _eval_predicate(self, predicate):
        op, obj, *target = predicate
        key = {"close": "drawer_closed", "open": "drawer_open",
               "turnon": "stove_on", "turnoff": "stove_off",
               "in": "bowl_inside_drawer"}.get(op)
        if op == "on":
            key = "bowl_on_cabinet_top" if obj.startswith("akita") else "moka_on_stove"
        return self.state[key]

    def check_success(self):
        return all(self.state[key] for key in FINAL_KEYS)

    def step(self, state):
        self.state = state
        return self.step_result

    def reset(self):
        self.state = dict(INITIAL)
        return self.state

    def set_state(self, state):
        self.state = state

    set_init_state = set_state
    regenerate_obs_from_state = set_state
    reset_from_xml_string = set_state


class OrderedTaskTests(unittest.TestCase):
    def tracker(self):
        tracker = OrderedProgress()
        self.assertTrue(tracker.reset(INITIAL)["initial_state_valid"])
        return tracker

    def test_normal_sequence(self):
        tracker = self.tracker()
        for i, state in enumerate(ORDERED_TRACE, 1):
            self.assertEqual(tracker.observe(state)["completed_stage_count"], i)
        self.assertTrue(tracker.ordered_complete)

    def test_final_state_cannot_initialize_or_advance_sequence(self):
        tracker = OrderedProgress()
        tracker.reset(FINAL)
        for state in ORDERED_TRACE:
            tracker.observe(state)
        self.assertFalse(tracker.ordered_complete)
        tracker = self.tracker()
        tracker.observe(FINAL)
        self.assertEqual(tracker.stage, 0)

    def test_early_stove_requires_off_then_on_after_drawer_stage(self):
        tracker = self.tracker()
        tracker.observe(changed(INITIAL, stove_on=True, stove_off=False))
        for state in (OPEN, BOWL_TOP, CLOSED, FINAL):
            tracker.observe(changed(state, stove_on=True, stove_off=False))
        self.assertEqual(tracker.stage, 3)
        tracker.observe(changed(CLOSED, moka_on_stove=False))
        tracker.observe(STOVE_ON)
        tracker.observe(FINAL)
        self.assertTrue(tracker.ordered_complete)

    def test_early_pot_requires_remove_and_replace_after_turn_on(self):
        tracker = self.tracker()
        tracker.observe(changed(INITIAL, moka_on_stove=True))
        for state in ORDERED_TRACE:
            tracker.observe(changed(state, moka_on_stove=True))
        self.assertEqual(tracker.stage, 4)
        tracker.observe(STOVE_ON)
        tracker.observe(FINAL)
        self.assertTrue(tracker.ordered_complete)

    def test_bowl_placement_while_closed_does_not_advance(self):
        tracker = self.tracker()
        tracker.observe(OPEN)
        tracker.observe(changed(INITIAL, bowl_inside_drawer=False))
        tracker.observe(CLOSED)
        self.assertEqual(tracker.stage, 1)
        tracker.observe(OPEN)
        tracker.observe(BOWL_TOP)
        self.assertEqual(tracker.stage, 2)

    def test_early_close_is_recoverable_without_reset(self):
        tracker = self.tracker()
        for state in (OPEN, INITIAL, OPEN, BOWL_TOP, CLOSED, STOVE_ON, FINAL):
            tracker.observe(state)
        self.assertTrue(tracker.ordered_complete)

    def test_one_stage_per_observation(self):
        tracker = self.tracker()
        tracker.observe(changed(FINAL, drawer_open=True, drawer_closed=False))
        self.assertEqual(tracker.stage, 1)
        tracker.observe(changed(FINAL, drawer_open=True, drawer_closed=False))
        self.assertEqual(tracker.stage, 1)

    def test_wrapper_preserves_step_tuple_and_separates_success(self):
        raw = FakeRawEnv()
        env = OrderedTaskEnv(raw)
        env.reset_progress()
        self.assertIs(env.step(FINAL), raw.step_result)
        self.assertTrue(env.raw_goal_success())
        self.assertFalse(env.check_success())
        self.assertIs(env.env, raw.env)
        self.assertIs(env.sim, raw.sim)

    def test_all_final_predicates_must_still_hold(self):
        env = OrderedTaskEnv(FakeRawEnv())
        env.reset_progress()
        for state in ORDERED_TRACE:
            env.step(state)
        self.assertTrue(env.check_success())
        env.step(changed(FINAL, stove_on=False, stove_off=True))
        self.assertFalse(env.check_success())
        env.step(FINAL)
        self.assertTrue(env.check_success())

    def test_state_replacement_invalidates_progress(self):
        for method in ("reset", "set_state", "set_init_state", "regenerate_obs_from_state", "reset_from_xml_string"):
            with self.subTest(method=method):
                env = OrderedTaskEnv(FakeRawEnv())
                env.reset_progress()
                for state in ORDERED_TRACE:
                    env.step(state)
                self.assertTrue(env.check_success())
                if method == "reset":
                    getattr(env, method)()
                else:
                    getattr(env, method)(INITIAL)
                for state in ORDERED_TRACE:
                    env.step(state)
                self.assertFalse(env.check_success())
                env.reset()
                env.reset_progress()
                for state in ORDERED_TRACE:
                    env.step(state)
                self.assertTrue(env.check_success())


if __name__ == "__main__":
    unittest.main(verbosity=2)
