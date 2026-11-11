"""V21: concise generic guidance; the V18 wire and V20 transport stay exact."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v20 import (
    VisualFieldSubscriptionClientV20, output_schema, geometry_semantics as semantics_v20,
    CANDIDATE_LABEL,
)
from .visual_feedback_compact_v21 import compact_payload_feedback

VERSION = 'visual-field-rgb-joints-v21-causal-appearance-progress-v1'
HISTORY_VERSION = 'visual-v21-compact-causal-recent-two-turns-v1'


def geometry_semantics():
    result = semantics_v20()
    result['version'] = 'candidate-bound-causal-appearance-progress-v21'
    result['presentation_only_compaction'] = True
    result['reference_maintenance'] = 'Causal appearance updates do not validate response, change identity, or reset travel debt.'
    result['task_progress'] = 'Metric arrival and declared image progress are separate observations, never task success.'
    return result


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError('V21 evaluation is zero-shot')
    return f'''Complete this task: {task.prompt}
Inputs: two RGB views, measured [world EEF xyz, axis-angle xyz, two fingers], joints and acknowledged history. No depth, camera calibration, simulator object poses, evaluator signals or demo. Visual geometry is hypothesis, never metric certainty or completion.

CONTROL. Exactly one branch nonnull, others null: visual_field/field_intent (bulk translation), direct/direct_actions (contact/articulation/reorientation), gripper/gripper_intent. Direct: exactly FIVE normalized 7D [xyz,rotation xyz,grip] in [-1,1]; translation 1=.05 m, rotation 1=.5 rad world-axis; grip +close/-open/0hold. 20 Hz; actual motion may differ. Accepted Direct is unchanged; checks can discard a block. Free Direct: <=2 mm, <=5 mm with supported response; nearby risk can refuse. Direct release must have zero XYZ throughout. Do not use yaw or endless zero holds to repair weak tracking. Blocked: five-zero Direct is hold, not progress or permission to open. Gripper: explicit open/close, 10..30 held-pose actions in grasp/release; settled fingers do not prove grasp.

FIELD. 33^3/.005 m grid: weighted goal cost plus model geometry/risk, not measured SDF or clearance. Goal/region offsets use world axes from THIS measured EEF; goal +/-0.08 m; hold orientation/grip. motion_axes xyz|xy|z locks other goal-offset coordinates to zero; release Field is z only. Prefer useful 5–8 cm free-space goals/max_actions 80–120; allowed 10..120/max_travel_m<=.16. command_limit_m .012–.02 free, .003–.006 near; checks may stop early. Retain a useful fixed world goal through reference repair. Place: Field align/xy, release/z, explicit open after visible support. Centering/arrival/stalled descent proves neither grasp readiness nor task completion.

SCENE. scene_inventory lists ALL visible nonrobot objects (target/distractors/fixtures), regardless of relevance; <=12 is an output limit, not object count. Keep physical IDs across roles, whole-object external/wrist ROI where visible, approximate center_offset_m/half_extents_m/uncertainty_m. Visible geometry revisions anchor to measured EEF. Null geometry retains memory; new image-only objects remain unsupported risk. Omission/occlusion never deletes. Never shrink, rename, delete or move hypotheses merely to evade a refusal. Inventory alone supplies geometry: no field obstacles/nearby_obstacles/obstacle_assessment/target_geometry. unknown_regions are genuinely unseen/unlocalized space with different IDs, not reclassified remembered objects. Missing calibrated depth alone does not make every visible object unknown.

Nominal cores constrain known tool parts; supported uncertainty adds cost. Unknown/unobserved regions remain hard/full-margin. uncertainty_m is a geometric margin, NOT a calibrated statistical standard deviation. Exact current full-uncertainty tool/payload swept unions determine model-near status; do not apply another arbitrary multiple of that margin. Model separation is never physical clearance certification. Retain full world/projected extents; partial texture does not reveal whole-object extent. extent_visibility=boundary_limited takes precedence even if legacy current_roi_uv exists. Coverage uses full uncertain candidate sweeps. Genuine near, unknown, lost or conflicting evidence still constrains motion; unobservable scene motion is capped to 15 mm travel/5 mm command.

CONTACT. target_contact is none|fingers|payload_support, independently constrained from distractors. none permits no target contact; fingers permits finger/pad contact, NEVER hand/palm contact. payload_support permits the independently tracked carried object on a different destination during release; it exempts no robot part. Destination stays none until support approach. Unavailable target geometry allows bounded observation/recovery/opening, not approach or closure. Carried constraints persist until observed separation, not merely opening. Whole-tool ROI covers hand/open fingers/carried extent, not a texture patch. motion_context translation_mode=transit|separate has intended_contact_id=null; interaction names exactly target_id. Reversal/separate is not a collision exemption.

Approaching target core within max(30 mm,target uncertainty) needs current two-view contact descriptors. Select an actual graspable patch/edge/handle within jaw span, not necessarily object center. contact_alignment only Field approach/align: seed both views' target_contact_point_uv/tool_pinch_point_uv; retain both null on the same tracks. Match external/wrist target IDs; moving_kind=tool. Pinch is actual open-pad midpoint, not image center. Coupled xyz can resolve parallax. Fingers/orientation changes invalidate descriptors. Other phases require null alignment. Missing descriptors are null/handoff; away/lateral recovery remains possible. Strong measured misalignment blocks closure; unavailable alignment alone is inconclusive. Use needed Direct5 terminal pose correction before explicit grip. A very small/noisy lift without co-motion is inconclusive: extend one bounded useful test if images/fingers still support grasp; visible miss/slip calls for a changed grasp. Field transport requires independent payload/tool co-motion.

REFERENCES. UV origin is top-left; u right/v down, [0,1]. Seed ROIs are ordered positive boxes containing their points. Track a physical distal tool feature, not fixed base/proximal arm. External moving tool/payload differs from stationary anchor; payload requires independent tool reference. Field external target_id matches navigation target; Direct/gripper may use independent anchor. Wrist always observes the interaction target when visible; never invent its center. Desired external offset=moving-anchor; wrist bearing=target position. Update anatomical offset for a new controlled feature. retain reuses actual live tracks with identities, seed coordinates null; it cannot resurrect a lost correspondence.

Weak response alone does not mean wrong identity. local_reference_maintenance uses causal RGB; no visual validation or travel replenishment. reference_support/reference_repair review with unchanged support despite other motion requires semantic review even if reference_health says weakly_observable. Inspect current candidate board/RGB and fresh whole-tool ROI. Candidates are retrospective texture hypotheses, not certified tool identity. To choose one, reference_candidate_id is nonnull, external update=seed/moving_kind=tool, and moving_point_uv/moving_roi_uv/tool_point_uv/tool_roi_uv are ALL null; explicitly supply anchor/ROI and whole-tool ROI containing the candidate. Resolution supplies exact current coordinates; unknown/stale IDs are invalid. Null candidate ID uses ordinary explicit seed coordinates. Selection/renaming/reseeding resets no shared/directional debt or attachment/clearance proof.

RECOVERY/PROGRESS. Read reasons/caps/recovery_options; proposed segments require actual rechecks. Preserve world subgoal unless evidence warrants explicit replacement. A directional_retreat is only the checked reversal of observed worsening, explicit separate mode, empty tool/no contact, fresh full-margin scene proof, <=3 mm commands, existing shared/heading allowance and a 4 mm observed-improvement checkpoint. For an offered reversal, use its exact goal_offset_m direction with the offered 8 mm goal and goal_tolerance_m=.003; a long diagonal goal may leave the admissible cone. Prior worsening remains. If task_progress shows metric improvement but image-objective worsening beyond noise, review grounding/anatomical offset and explicitly replace contradicted goals. Reference changes archive progress intervals rather than reset evidence.

Only weak image-response allowance exhaustion as the sole blocker may qualify for proprioceptive_transit: reliable current tool/anchor, empty tool, target_contact=none, translation_mode=transit|separate, outside contact band, no payload/contact refinement/goal revision. The executor checks the exact current full-margin candidate, <=5 mm command, original intent budgets; it tracks the existing world goal using measured robot state without image goal correction. No visual validation or debt reset. Lost/contradicted/stale/changed references do not qualify. Read field.proprioceptive_transit_admission.

LEDGER/HISTORY. subgoal_update new creates a unique ID/world goal/phase/target/axes; Field retain requires the same ID/phase/target/axes and ignores fresh relative offsets. Direct/gripper repair retains active ID across phase changes. replace uses the active ID plus evidence-based reason; new cannot discard an unfinished goal. finish requires measured local arrival; none/null ID does not clear an active goal. Latest feedback is self-contained; older snapshots may be exact patches against it. Presentation tables/references encode exact repeated fields; local traces keep descriptor samples/derived boxes. Raw RGB/state/model outputs stay intact. Old support is not new evidence; local goals are not task success.'''


class VisualFieldSubscriptionClientV21(VisualFieldSubscriptionClientV20):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v21', schema=output_schema())
        for message in payload['input']:
            record = _text_record(message)
            if record and record.get('type') == 'visual_history_summary':
                record['protocol'] = HISTORY_VERSION
                message['content'][0]['text'] = _canonical(record)
        return compact_payload_feedback(payload)

    def history_stats(self):
        result = super().history_stats()
        result['protocol'] = HISTORY_VERSION
        return result
