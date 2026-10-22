"""Review declining wrist visibility during continuous Field execution."""
from __future__ import annotations

import copy
import time

from .observation import decode_image
from .visual_field_policy_v35 import VisualFieldPolicyV35
from .visual_intent_watch_v36 import WristIntentWatchV36
from .visual_intent_replan_v36 import ordinary_replan_previews


class VisualFieldPolicyV36(VisualFieldPolicyV35):
    def reset(self):
        self.intent_watch = WristIntentWatchV36()
        self.ordinary_replans = None
        super().reset()

    def _observable_feedback(self):
        if getattr(self, '_transition_feedback_deferred', False):
            return {}
        result = super()._observable_feedback()
        from .visual_field_subscription_v36 import geometry_semantics
        result['geometry_semantics'] = geometry_semantics()
        result['ongoing_objective_review'] = self.intent_watch.summary()
        result['ordinary_replan_previews'] = copy.deepcopy(self.ordinary_replans)
        return result

    def _refresh(self, obs):
        started = time.monotonic()
        # Current intent review is sent while useful observations still exist.
        # Preview only on such a handoff; ordinary requests incur no extra search.
        self.ordinary_replans = (ordinary_replan_previews(self, obs)
            if self.intent_watch.summary()['review_required'] else None)
        preparation_s = time.monotonic() - started
        latency = super()._refresh(obs)
        started = time.monotonic()
        # Acknowledge only AFTER the response consumed the event. The watcher
        # owns no physical ledger and cannot renew any guard allowance.
        self.intent_watch.begin_intent(self.intent_refreshes)
        self.ordinary_replans = None
        self.feedback['ongoing_objective_review'] = self.intent_watch.summary()
        self.feedback['ordinary_replan_previews'] = None
        elapsed = preparation_s + time.monotonic() - started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def observe_transition(self, before, decision, after):
        # All original tracking, learned response, geometry and debt observers
        # finish first. Update only two presentation fields on the final row;
        # do not rebuild the full feedback or observe a transition twice.
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        prediction = self.prediction or {}
        external = (prediction.get('visual_references') or {}).get('external') or {}
        selection = decision.diagnostics.get('selection') or {}
        acknowledged = (
            decision.diagnostics.get('controller') == 'visual_field'
            and selection.get('accepted') is True
            and prediction.get('phase') in ('approach', 'align')
            and prediction.get('target_contact') == 'none'
            and prediction.get('contact_alignment') is None
            and external.get('moving_kind') == 'tool'
            and not self.target_contact.summary().get('carried_payload', {}).get('active'))
        # The monitor attaches current-point availability/normalized coordinates
        # to the underlying track summary. Reuse that already-rendered snapshot.
        track = (row.get('visual') or {}).get('tracking', {}).get('wrist_target')
        report = self.intent_watch.observe(before.state, decision.action, after.state,
            wrist_summary=self.visual_goal_field.wrist_navigation.summary(),
            wrist_track_summary=track,
            image_size_wh=decode_image(after.wrist_image).size,
            intent_id=self.intent_refreshes, target_id=prediction.get('target_id'),
            phase=prediction.get('phase'), acknowledged=acknowledged)
        if report['new_event']:
            event = report['event']
            # Existing physical refusals/arrival handoffs have precedence. This
            # new reason requests an earlier model decision, not a failed action.
            if not self.refresh_reason:
                self.refresh_reason = event['reason']
            self.events.append({'step': self.step_index - 1, 'reason': event['reason'],
                'event_id': event['event_id'], 'task_success_certified': False,
                'automatic_goal_retarget': False, 'physical_allowance_replenished': False})
        self.ordinary_replans = None
        self.feedback['ongoing_objective_review'] = report
        self.feedback['ordinary_replan_previews'] = None
        row['ongoing_objective_review'] = copy.deepcopy(report)
        row['ordinary_replan_previews'] = None
        self.local_total_s += time.monotonic() - started
        return row

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v36'
        return decision
