"""Same V13 wire contract with actionable recovery guidance and compact feedback."""
from __future__ import annotations

import copy
import json
from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v13 import VisualFieldSubscriptionClientV13, output_schema, geometry_semantics
from .visual_feedback_compact_v14 import compact_payload_feedback
from .visual_reference_overlay import LABEL, reference_diagnostic_image

VERSION = 'visual-field-rgb-joints-v14-evidence-producing-recovery-v1'
HISTORY_VERSION = 'visual-v14-actionable-recovery-recent-two-turns-v1'
DIAGNOSTIC_LABEL = ('Current external RGB reference diagnostic: M=declared controlled feature, '
    'T=stationary anchor, G=independent tool. Labels identify tracker claims, not verified object '
    'identity or a proven wrong reference. Inspect the actual distal gripper/payload and original '
    'views. Static support or weak motion can be ambiguous; only observed changed-direction '
    'response can resolve it. Reseeding, renaming or adjusting a box never renews recovery debt.')


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError('Visual v14 is zero-shot Hybrid only')
    return f"""Task: {task.prompt}. ZERO-SHOT, no demonstration. Complete the full task using current external/wrist RGB, measured EEF XYZ/axis-angle/two-finger state and seven arm joint positions/velocities. No depth, environment calibration, object truth, contacts, reward or evaluator success is supplied. Avoid unintended robot/payload collision and obstacle displacement. Local arrival, tracking, settling and model judgement do not establish task completion. Return strict JSON; no completion flag.

ROUTING. Select exactly one controller and populate only its branch. visual_field performs bulk translation, bounded approach and useful test motion with local measured feedback. direct supplies exactly FIVE unchanged normalized [dx,dy,dz,rx,ry,rz,grip] commands in[-1,1]: translation1=0.05m, rotation1=0.5rad world delta, grip+1 closes/-1 opens/0 holds intent;20Hz, actual motion may be smaller. Use Direct for needed contact correction, articulation or task reorientation. Current free-image Direct translation allows0.002m/action, up to0.005m only with supported response; near-risk geometry may refuse it. gripper supplies explicit open|close,10..30 held-pose actions, phases grasp/release only. Unsafe selections are refused; model waiting advances no physics. Never use stationary Direct holds as new directional evidence or yaw solely for weak tracking.

FIELD. Goal offsets and all box centers are world-axis metres from the SAME CURRENT measured EEF. goal_offset_m is within±0.08m per axis. A33³ grid at0.005m represents goal-directed path cost through hypotheses, not measured SDF, calibrated probability or certified clearance. orientation_command and gripper_command are hold. motion_axes=xyz permits spatial detours; xy requires zero goal z and holds height; z requires zero goal x/y and holds XY, so cannot detour sideways. Phases: approach/transport/align/release/recover; release is z-only. max_actions10..120 and max_travel_m≤0.16. For visibly clear task travel choose a useful5–8cm goal and80..120 actions; local limits stop reuse early. command_limit_m0.012..0.02 free,0.003..0.006 near interaction. Use measured robot-response feedback; commanded and measured displacement differ.

GEOMETRY. Stable obstacle IDs, positive half_extents_m and honest uncertainty_m describe approximate RGB world-axis boxes. Preserve static world locations as EEF moves. obstacles are visibly localized objects; nominal known-robot/tool/payload-inflated cores stay hard, with separate nonnegative uncertainty traversal cost when current whole-tool and every named-neighbor ROI support assessment. Missing/unobservable/inconsistent support keeps full hard inflation. unknown_regions are truly unseen/occluded/unlocalized regions, fully hard inflated; lacking calibrated depth alone does not make visible objects unknown. Preserve all hypotheses, revising only from visual evidence; never delete, shrink or reclassify to evade refusal.

MOTION CONTEXT. Give actual purpose, current whole-tool external ROI including fingers/payload, and relevant nearby-object ROIs. These differ from small texture patches. present requires neighbors, none_relevant requires empty neighbors after reviewing the route; every listed Field neighbor needs a matching obstacle/unknown box. unobservable limits Field travel0.015m and commands0.005m. transit/separate require intended_contact_id=null; interaction requires exact top-level target_id. Maintain physical IDs across neighbor/target/contact/payload roles. Projected overlap is ambiguity, not proof of collision or clearance. Read current translation_guard, reference_support and recovery_options; old snapshots are not new evidence.

RECOVERY. If normal progress lacks permission, consider an offered nominally checked bounded Field recovery with phase=recover, translation_mode=separate and explicit subgoal revision/retention. Candidates in recovery_options are proposals, not authorization: actual_segment_must_be_rechecked applies to the real first segment. Choose a useful changed direction, with a goal beyond arrival tolerance; the finite allowance bounds MEASURED travel, not the proposed goal distance. Use new only with no unfinished goal, otherwise explicit replace with the same ID when changing direction/axes. The local guard may limit the segment and collect actual response. All hard geometry, palm and attachment constraints remain. Read recovery_experiments and the shared remaining measured allowance: renaming a goal/object, reseeding points or changing an ROI NEVER renews debt. Evidence comes from executed measured motion and current images, not static holds or model repetition. Static-looking support is not automatically a wrong landmark; inspect the M/T/G diagnostic and use purposeful changed-direction evidence. Only seed a different physical feature when current imagery supports it. Do not repeat the same refused corridor or submit idle blocks to manufacture evidence. If no executable option is available, describe the unresolved constraint honestly rather than claiming completion or clearance.

REFERENCES. UV origin top-left,u right,v down. Seed actual visible points; nullable ordered positive-area[u0,v0,u1,v1] ROIs contain their points. Choose a rigid distal tool feature, not the fixed base/background. Retain exact identities/moving_kind with seed=null; use live tracked coordinates, not new guesses. Retain cannot recover lost tracks. External moving tool/payload and stationary anchor are distinct; payload needs an independent tool point. Field external target_id equals top-level target_id; Direct/gripper may use a distinct stationary anchor. Wrist always matches the top-level target; keep it when visible, null when unavailable, never invent an image-center target. External goal_offset_uv is desired moving projection minus anchor; wrist desired_target_uv is desired target bearing. Preserve independent payload/tool observation across stage changes. Weak response is not itself slip, semantic mismatch or permission to rotate.

TARGET/PART CONTACT. Every response supplies target_geometry independently of other obstacle lists: exact target_id, WHOLE target center_offset_m/half_extents_m/uncertainty_m and allowed_contact=none|fingers|payload_support. Visible targets require a grounded approximate box for approach/interaction/closure. Null permits only finite observation/hold/recovery or held opening; it never erases remembered geometry or authorizes approach/closure. Small target texture ROI is not whole-object extent. none permits no intended target contact; fingers exempts only known fingers/pads, NEVER hand/palm. payload_support permits only the independently observed carried object to meet a DIFFERENT destination during release/interaction, exempts no robot part. Preserve carried-object constraints through uncertain attachment or changing destination; opening alone does not prove separation.

CONTACT ALIGNMENT. Select a local graspable patch/edge/handle within jaw span, not whole-object centering. A coupled xyz Field approach may avoid high-plane parallax errors. Optional contact_alignment is only Field approach/align: seed exact target_id and BOTH view pairs{{target_contact_point_uv,tool_pinch_point_uv}}; pinch is the actual open-pad midpoint, not wrist image center. Existing external moving_kind=tool and both target references match. Use visible points compatible with tracked regions. Local measured image response permits bounded Field correction. Retain uses both descriptor views=null and the same live references; changed fingers/orientation invalidate its domain. Null is honest when unavailable; do not force far/occluded descriptors. Recover/transport/release require null so correction cannot pull recovery back toward contact. Strong observed mismatch forbids closure; unavailable alignment alone is inconclusive. No automatic closing.

SUBGOALS/STAGES. new stores a unique world goal/phase/target/axes. Field retain requires the SAME ID/phase/target/axes; a new relative proposal is logged but ignored. Direct/gripper repair retains the active ID despite temporary phase/target changes. replace changes an unfinished goal using SAME ID and an evidence-based reason; new cannot discard it. finish is Direct/gripper only after measured arrival; none has null ID and never clears a goal. Read subgoal_ledger instead of repeatedly adding lift clearance. Stalled descent is not grasp readiness. Use current two-view jaw/feature evidence and needed Direct5 contact correction before explicit closure. Then obtain useful independent payload/tool co-motion; Field transport requires observed attachment. Tiny/noisy inconclusive tests may be extended once if images/fingers support grasp; visible miss/slip needs changed grasp geometry. Place using payload/support error and grasp offset: Field align/xy first, held-XY release/z next, explicit opening from observed support. Direct release has zero XYZ in all5 commands. Reassess separation.

HISTORY. Two recent observation turns plus observable memory remain. Raw RGB, robot state and assistant outputs stay exact; full diagnostics remain local. Feedback references share repeated facts inside the request; current guard, recovery permissions, target, attachment and world-subgoal facts are authoritative. Decision-time snapshots retain actual veto reasons/budgets without repeating full current geometry. Keep reasoning concise and select useful executable progress."""


def _append_v14_diagnostic(payload, observation, context):
    feedback = (context or {}).get('visual_feedback') or {}
    needed = any(isinstance(feedback.get(key), dict) and feedback[key].get('diagnostic_required') is True
                 for key in ('reference_support','recovery_options'))
    if not needed:
        return payload
    for message in reversed(payload.get('input', [])):
        record = _text_record(message)
        if message.get('role')=='user' and record and record.get('type')=='observation':
            content=message['content']
            if any(item.get('text') in (LABEL,DIAGNOSTIC_LABEL) for item in content):
                return payload
            # Only an internal rendering switch: the actual policy feedback and
            # its repair_required flag are never changed or forwarded as true.
            render_context=copy.deepcopy(context)
            visual=render_context.setdefault('visual_feedback',{}).setdefault('visual',{})
            visual.setdefault('reference_health',{})['repair_required']=True
            diagnostic=reference_diagnostic_image(observation,render_context)
            if diagnostic is not None:
                content.extend([{'type':'input_text','text':DIAGNOSTIC_LABEL},
                                {'type':'input_image','image_url':diagnostic}])
            return payload
    return payload


class VisualFieldSubscriptionClientV14(VisualFieldSubscriptionClientV13):
    history_protocol=HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION+'-hybrid'

    def build_payload(self,observation,grid=None,mode='hybrid'):
        payload=super().build_payload(observation,grid,mode)
        payload['instructions']=shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v14',schema=output_schema())
        for message in payload['input']:
            record=_text_record(message)
            if record and record.get('type')=='visual_history_summary':
                record['protocol']=HISTORY_VERSION
                message['content'][0]['text']=_canonical(record)
        payload=_append_v14_diagnostic(payload,observation,self.observable_context)
        return compact_payload_feedback(payload)

    def history_stats(self):
        result=super().history_stats();result['protocol']=HISTORY_VERSION;return result
