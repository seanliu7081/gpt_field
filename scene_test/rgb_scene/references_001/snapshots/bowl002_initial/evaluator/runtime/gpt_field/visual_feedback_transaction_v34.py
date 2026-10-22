"""Assemble the final transition diagnostic once after all observers run.

The inherited observer chain updates control state directly through monitors,
guards and executors. Its repeated feedback dictionaries are diagnostic-only;
each wrapper overwrites the same returned row with a later snapshot. Defer
those intermediate snapshots, but execute every observer in its original order.
No snapshot is cached across observations or model requests.
"""
from __future__ import annotations

import copy
import time


class FinalTransitionFeedbackMixinV34:
    def _observable_feedback(self):
        if getattr(self, '_transition_feedback_deferred', False):
            return {}
        return super()._observable_feedback()

    def observe_transition(self, before, decision, after):
        if getattr(self, '_transition_feedback_deferred', False):
            raise RuntimeError('Nested transition observation is not supported')
        old_feedback = self.feedback
        self._transition_feedback_deferred = True
        try:
            row = super().observe_transition(before, decision, after)
        except BaseException:
            self.feedback = old_feedback
            raise
        finally:
            self._transition_feedback_deferred = False
        started = time.monotonic()
        # V18's reference summary precedes V24's witness query in the inherited
        # rendering chain. Resolve this current-observation dependency first:
        # otherwise one render would expose an empty cache in reference_repair
        # and the populated witness later in the very same feedback row.
        bank = getattr(self, 'reference_candidates', None)
        monitor = getattr(self, 'monitor', None)
        state = getattr(monitor, 'state', None)
        if state is not None and callable(getattr(bank, 'raw_support_evidence', None)):
            bank.raw_support_evidence(state, monitor)
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
