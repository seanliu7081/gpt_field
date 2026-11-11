"""Actionable review of ongoing Field objectives; unchanged native wire."""
from __future__ import annotations

from .visual_field_subscription_v35 import (
    VisualFieldSubscriptionClientV35, output_schema, HISTORY_VERSION, FEEDBACK_VERSION,
    shared_guidance as guidance_v35, geometry_semantics as semantics_v35,
)

VERSION = 'visual-field-rgb-joints-v36-ongoing-objective-review-v1'
OBJECTIVE_REVIEW_GUIDANCE = (
    'ONGOING OBJECTIVE REVIEW. ongoing_objective_review reports measured progress and '
    'the evidence for requesting an early decision on an unfinished Field intent. '
    'Read its current status, event and measured/predicted evidence; archived events are historical. '
    'A review requests an explicit decision; it does not certify collision, forbid every '
    'other heading, complete the subgoal or refund travel. Distinguish a current image '
    'track, a prospectively validated motion direction, and physical action admission. '
    'A tracked target with too few motion samples has no learned descent direction yet. '
    'A rank-one response only supports its observed subspace: vertical evidence does not '
    'validate lateral motion. Do not wait in zero-action loops for a learned direction '
    'that requires actual motion to become observable.\n'
    'At this paid decision, inspect current RGB and the remaining physical limits. '
    'If an appropriate CURRENT checked task offer exists, explicitly select its exact '
    'contract, including phase and subgoal_update; do not choose recovery merely because '
    'the previous phase was recover. An offer remains subject to actual execution checks. '
    'If no suitable offer exists but RGB and current geometric hypotheses justify a '
    'different noncontact task move, use an ordinary visual_field response with '
    'field_proposal_id=null and an explicit field_intent. To change an unfinished goal '
    'or its motion_axes, use subgoal_update=replace with the SAME active subgoal_id and '
    'a concrete evidence reason; retain ignores a newly written goal offset. Choose a '
    'useful bounded task direction and span supported by current evidence, not the '
    'longest possible descent. Such ordinary task motion may supply future samples, '
    'but it is not prevalidated or an extra calibration allowance. All route, actual '
    'command, obstacle, unknown-space, contact and shared/heading-debt checks still apply.\n'
    'ordinary_replan_previews, when present, lists current ordinary Field alternatives '
    'with full-route and first-command preview results. An accepted row supports only '
    'feasibility under current hypotheses; it predicts neither visual improvement nor '
    'physical collision-free motion. Compare its direction with current RGB and the '
    'task objective, then explicitly submit the ordinary field_intent and matching '
    'motion_context with field_proposal_id=null and the required replace update. '
    'These rows are not selectable field_proposal_id offers. Changed references or '
    'geometry invalidate their preview; actual admission rechecks every constraint, '
    'and no travel allowance is renewed.\n'
    'Preserve healthy external and wrist references and the intended desired bearing '
    'through a metric-goal review. Actual tracking loss requires explicit repair from '
    'current RGB: when visible, choose a stable observed interior target feature and '
    'bounded ROI rather than a clipped image edge or invented image center. Keep its '
    'physical identity distinct from desired_target_uv, which is a declared bearing '
    'goal, not an observed feature, target pose or grasp/pinch location. External tool '
    'anatomy, rigid transit proxy and the true hand/finger clearance extent remain '
    'distinct. Do not enlarge the hand extent merely to include a remote reference. '
    'A justified reference repair and ordinary Field intent may share this response; '
    'a justified same-ID geometry correction additionally needs scene_revisions and '
    'its actual evidence. Exact field_proposal_id selection cannot combine those '
    'changes; changed bindings need an ordinary response or later fresh offers. '
    'Do not move the desired UV merely to erase observed worsening; change it only '
    'when the intended task bearing changes and explain that reason. '
    'Reseeding, changing desired bearing, replacing a goal or revising geometry grants '
    'no clearance, predictive validation, contact permission or travel refund. If '
    'neither a supported task move nor a currently admitted recovery is justified, '
    'report the specific missing evidence or exhausted limit honestly with the '
    'existing zero-action branch. Holding is not progress, success or new evidence.'
)


def shared_guidance(task, demonstration=False):
    return guidance_v35(task, demonstration) + '\n\n' + OBJECTIVE_REVIEW_GUIDANCE


def geometry_semantics():
    result = semantics_v35()
    result['version'] = 'ongoing-field-objective-review-v36'
    result['ongoing_objective_review_contract'] = {
        'feedback_key': 'ongoing_objective_review',
        'source': 'acknowledged_measured_transition_observer',
        'review_requires_explicit_model_decision': True,
        'ordinary_field_replacement_remains_available_subject_to_admission': True,
        'reference_repair_and_ordinary_field_may_share_response': True,
        'exact_offer_cannot_change_reference_or_scene_bindings': True,
        'ordinary_replan_previews_require_explicit_ordinary_field_response': True,
        'ordinary_replan_preview_certifies_visual_improvement_or_clearance': False,
        'changed_bindings_preserve_preview_authority': False,
        'unobserved_heading_is_predictively_validated': False,
        'declared_desired_bearing_is_grasp_or_target_pose': False,
        'review_is_collision_or_clearance_evidence': False,
        'review_completes_goal_or_changes_budget': False,
        'automatic_action_or_stage_change': False,
        'wire_schema_changed': False,
    }
    return result


class VisualFieldSubscriptionClientV36(VisualFieldSubscriptionClientV35):
    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format']['name'] = 'visual_field_rgb_joints_v36'
        return payload
