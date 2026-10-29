"""Same-wire task control with compact, role-specific current guidance."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v29 import (
    VisualFieldSubscriptionClientV29, output_schema, geometry_semantics as semantics_v29,
    FEEDBACK_VERSION, CANDIDATE_LABEL, compact_older_contexts, RECOVERY_ROUTES,
)

VERSION = 'visual-field-rgb-joints-v30-typed-reference-target-approach-v1'
HISTORY_VERSION = 'visual-v30-current-contract-recent-two-turns-v1'


def geometry_semantics():
    result = semantics_v29()
    result['version'] = 'typed-reference-and-role-specific-target-approach-v30'
    result['projected_warning_transit'].update(
        warned_object_model_clearance_must_not_decrease=False,
        non_target_warned_object_model_clearance_must_not_decrease=True,
        explicit_target_approach_exception=True,
        boundary_evidence_must_be_ambiguous_not_resolved_closing=False,
        non_target_boundary_must_be_ambiguous_not_resolved_closing=True)
    result['constraint_aware_tracking'].update(
        warned_object_model_gap_nondecrease_retained=False,
        non_target_model_gap_nondecrease_retained=True,
        explicit_target_approach_exception=True)
    result['target_approach_execution'] = {
        'scope': 'normal_empty_tool_noncontact_field_approach_or_align',
        'translation_mode': 'transit', 'purpose': 'task_translation',
        'target_contact': 'none', 'current_exact_role_binding_required': True,
        'target_only_modeled_gap_may_decrease': True,
        'positive_full_margin_model_clearance_and_outside_contact_band_required': True,
        'current_causal_same_heading_response_required': True,
        'all_non_target_checks_unchanged': True,
        'target_contact_or_collision_exemption': False,
        'history_or_physical_debt_reset': False, 'clearance_certified': False,
    }
    result['paid_handoff_reference_review'].update(
        dispatch='typed_current_reference_and_control_events',
        stop_reason_string_membership_required=False,
        extraction_status=['not_requested', 'ineligible', 'extracted_none', 'available'],
        actual_observation_binding_required=True,
        automatic_reference_promotion=False)
    result['guidance_presentation'] = 'One current contract replaces accumulated superseded prose; schema and raw history unchanged.'
    return result


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError('V30 evaluation is zero-shot')
    return f'''Complete this task: {task.prompt}
Use only the two RGB views, measured world EEF xyz/axis-angle/two fingers, joints and acknowledged history. No demonstration, depth, camera calibration, object truth or evaluator feedback. Visual geometry is a hypothesis; neither local arrival nor appearance proves task completion.

CONTROL. Choose exactly one nonnull branch: visual_field/field_intent for bulk translation; direct/direct_actions for necessary contact, articulation or reorientation; gripper/gripper_intent for explicit open/close. All other branches are null. Never select direct while leaving direct_actions null. Direct has exactly FIVE normalized 7D [xyz,rotation xyz,grip] actions in [-1,1]: translation1=.05 m, rotation1=.5 rad, world axes, grip +close/-open/0hold,20 Hz. Actual motion differs. Accepted Direct stays unchanged; a rejected block is discarded. Free Direct allows <=2 mm commands, <=5 mm with supported response; nearby risk can refuse. Direct release has zero XYZ. Gripper holds measured pose for10..30 actions in grasp/release; settling fingers alone is not grasp evidence. If all choices are blocked, valid five-zero Direct is an honest hold; do not fake opening or rotate merely to repair tracking.

FIELD. The local33^3 grid at.005 m stores weighted goal/risk cost, not measured SDF or certified clearance. Goal/region offsets use world axes relative to THIS EEF. Goal components stay within+/- .08 m; orientation_command=hold, grip held. motion_axes xyz|xy|z requires zero offsets on locked axes; release Field is z only. Prefer useful5–8 cm free-space goals/max_actions80–120; allowed10..120, max_travel_m<=.16, command_limit_m .012–.02 free/.003–.006 near. Runtime caps can be smaller. Placement uses align/xy then release/z, followed by model-selected opening after visible support. Keep useful fixed goals through reference repair. field.axis_tracking_constraint may omit conflicting locked-axis recentering inside the existing route corridor; exact selected commands still pass current checks and arrival uses actual measured position.

SCENE. scene_inventory lists all visible nonrobot objects, including target, fixtures and distractors, with stable physical IDs, current external/wrist boxes, approximate world-axis center_offset_m/half_extents_m/uncertainty_m. The12-entry output bound is not a required object count. Null geometry retains remembered world geometry; omission/occlusion never deletes it. New image-only objects remain unsupported risk. Do not alter, rename, shrink or delete a hypothesis to evade rejection; reference-only repair should retain unchanged geometry as null. Inventory is the sole object source: do not emit legacy obstacles/nearby_obstacles/obstacle_assessment/target_geometry. unknown_regions are genuinely unlocalized/unseen space with independent IDs, never a renamed remembered object. Lack of calibrated depth alone does not turn a visible object into unknown space.

Nominal cores constrain known tool parts; eligible uncertainty adds traversal cost. Unknown/unobserved regions stay hard/full-margin. uncertainty_m is a geometric margin, not a statistical standard deviation: exact swept tool/payload unions use the full margin once. Full world/projected geometry persists; partial tracked texture is not whole-object visibility. extent_visibility=boundary_limited takes precedence over legacy current_roi_uv. Coverage checks the actual candidate sweep. Unobservable scene execution is capped at15 mm travel/5 mm command. Whole-tool ROI must cover hand/open fingers/carried extent, not just a texture patch. All model clearance remains conditional, not a physical certificate.

TARGET AND CONTACT. target_contact=none forbids contact; fingers permits pads/fingers only, never palm/hand; payload_support permits an independently tracked payload to meet a different destination during release, exempting no robot part. Destination is none until support approach. Target geometry is required for approach/closure; honest unavailable geometry permits bounded observation/recovery/opening. Carried geometry persists until observed separation, not merely opening. transit/separate requires intended_contact_id=null; interaction requires exact target_id. Stable IDs remain unchanged when an obstacle becomes the task target.

For normal empty-tool Field approach/align, translation_mode=transit, purpose=task_translation and target_contact=none, the intended target's modeled gap MAY DECREASE when the exact current full-margin sweep stays positively separated, outside the contact band, with current causal same-heading response. Read translation_guard.target_approach_execution. This target-only rule supersedes blanket target-never-approach/nondecreasing-target-gap guidance: every NON-TARGET check remains unchanged; it grants no contact, clearance certificate, erased warning or fresh allowance. Target-role changes require explicit current grounding, not relabeling a distractor.

Inside max(30 mm,target uncertainty), approaching the target core needs current two-view contact descriptors. Choose an actual graspable patch/edge/handle within jaw span, not simply its center. contact_alignment only Field approach/align: seed target_contact_point_uv/tool_pinch_point_uv in BOTH views, or retain with both null on the same tracks. External/wrist target IDs match, moving_kind=tool. Pinch means actual open-pad midpoint, never image center; coupled xyz can resolve parallax. Fingers/orientation changes invalidate descriptors. Other phases use null. Missing descriptors require honest handoff; away/lateral recovery remains possible. Strong measured misalignment blocks closure; unavailable alignment alone is inconclusive. Use needed Direct5 terminal correction, then explicit grip. Tiny/noisy lift without co-motion is inconclusive; if images/fingers still support grasp, extend one bounded useful test. Visible miss/slip requires a revised grasp. Transport requires independent payload/tool co-motion.

REFERENCES. UV origin top-left, u right/v down, [0,1]; seed boxes are ordered and contain their points. Select physical distal hand/tool texture, not fixed base/proximal arm. External moving tool/payload differs from stationary anchor; payload needs an independent tool reference. Field anchor target_id matches navigation target; Direct/gripper may use an independent anchor. Wrist tracks the interaction target when visible; do not invent coordinates. Desired external offset is moving minus anchor; wrist bearing is target position. Choose anatomical offset consistently when changing controlled feature. retain keeps live identities with seed coordinates null; it cannot resurrect a lost track.

reference_repair uses typed evidence events, including controlled-reference loss, target-reference loss, unreliable response, a control-risk handoff and static-support review. extraction_status distinguishes not_requested, ineligible, extracted_none and available; these are not interchangeable. A lost stationary anchor can make candidate extraction ineligible even with paid motion history. Weak response alone does not prove wrong identity, but actual loss or unchanged declared support despite visible motion calls for review. Inspect the current labeled board and original RGB. Candidates are texture hypotheses from already acknowledged history (at most32 frames), never certified distal-tool identity; no extra movement is performed to present them. Choose a physically correct candidate with a fresh whole-tool ROI, or explicitly ground another visible feature. Do not blindly repeat the same point after an evidence failure.

To select a candidate, reference_candidate_id is nonnull and external update=seed/moving_kind=tool; moving_point_uv,moving_roi_uv,tool_point_uv,tool_roi_uv are ALL null. Supply anchor/ROI and whole-tool ROI containing the selected candidate. Current-observation resolution supplies exact coordinates; stale/unknown IDs fail. Null candidate ID uses ordinary explicit coordinates. Causal appearance maintenance, board selection, a new ID/ROI or measurement epoch provides no permission, visual validation, attachment proof or budget refill. Geometry/ROI re-expression is not robot motion; prior physical evidence remains while fresh image differences use compatible measurement epochs.

ADMISSION AND RECOVERY. Read actual reasons/caps and current checked offers. A normal/transit empty-tool fixed-goal Field uses complete current model geometry and causal image/robot response. For non-target warnings, modeled clearance must not decrease and resolved closing/contradictory/lost/stale evidence still stops. Ambiguous projected overlap is not collision proof, and zero apparent worsening is not clearance. Unvalidated heading acquisition spends the existing8 mm heading/20 mm shared allowance and intent remaining; cold commands start bounded and may grow to3 mm as measured response supports them. A separate validated_intent_execution state requires fresh held-out per-object response and responsive robot/current exact constraints; its supported travel is counted under fixed intent limits while old acquisition debt stays. Loss falls back only to old remaining allowance. Neither state automatically renews a goal.

Only weak image-response allowance exhaustion as the sole blocker can qualify for proprioceptive_transit: reliable tool/anchor, empty tool, target_contact=none, transit|separate, outside contact band, no payload/refinement/goal revision. Exact full-margin candidates use measured robot state toward the fixed world goal, <=5 mm commands, original intent limits; no image-validation claim or debt reset. Read field.proprioceptive_transit_admission. Lost/contradicted/domain-shifted evidence does not qualify.

Every response requires recovery_route=normal|reverse_observed|initial_uncertainty_egress. Direct/gripper specify normal without erasing a retained Field route. Non-normal routes require Field phase=recover/translation_mode=separate; phase/purpose alone never dispatches them. reverse_observed selects the current reverse_observed_worsening_motion offer; copy its exact checked goal/tolerance/limits (normally<=8 mm goal,.003 m tolerance). The whole short route remains inside its original8 mm route/per-direction and20 mm shared allowance, commands<=3 mm. The4 mm observation is telemetry, not new credit; current further worsening, unavailable boundary/reference, unknown/contact/domain failures still stop it. A refused reverse is not an initial-egress route.

initial_uncertainty_egress selects only that current offer: Field recover/separate, purpose=reference_repair, target_contact=none, empty tool. Copy offered goal/tolerance/command/travel exactly. This is bounded model-geometry exposure from an initial uncertain halo with clear nominal cores, not permission to enter new uncertainty or contact. Total contextual travel<=40 mm includes earlier uncertain debt; an initial<=4 mm probe requires current causal reference/contradiction checks to continue. A current selected reference may receive one<=4 mm confirmation within the same total, never renewed by reseeding. Positive image-gap improvement is not a clearance certificate or mandatory fresh credit. Unsupported/contradictory geometry and actual unavailable reference evidence remain stops. Offers are current previews, not guarantees after a changed model response.

PROGRESS AND HISTORY. subgoal_update new makes a unique world goal/ID/phase/target/axes; retain keeps those and the Field route, ignoring fresh relative offsets. Direct/gripper repair retains the active ID. replace uses that ID and a meaningful evidence-based reason; new cannot discard unfinished work. finish requires measured local arrival; none/null ID does not clear active work. Metric improvement with image-objective worsening calls for reviewing grounded feature/offset and replacing a contradicted goal. Arrival at an egress/recovery goal prompts task resumption, not completion. Latest feedback is complete; older engineering feedback is explicitly summarized, while raw assistant outputs and two recent RGB/state/joint observations remain. Do not treat old support as fresh evidence or repeat failed calibration/zero-action loops as progress.'''


class VisualFieldSubscriptionClientV30(VisualFieldSubscriptionClientV29):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v30', schema=output_schema())
        for message in payload['input']:
            record = _text_record(message)
            if record and record.get('type') == 'visual_history_summary':
                record['protocol'] = HISTORY_VERSION
                message['content'][0]['text'] = _canonical(record)
        return payload

    def history_stats(self):
        result = super().history_stats()
        result['protocol'] = HISTORY_VERSION
        return result
