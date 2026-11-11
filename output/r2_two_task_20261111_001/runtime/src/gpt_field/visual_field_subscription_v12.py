"""Concise generic V12 guidance with the unchanged strict V11 wire contract."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v11 import VisualFieldSubscriptionClientV11, output_schema, geometry_semantics
from .visual_feedback_compact_v12 import compact_payload_feedback

VERSION = "visual-field-rgb-joints-v12-observed-response-concise-v1"
HISTORY_VERSION = "visual-v12-concise-recent-two-turns-v1"


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError("Visual v12 is zero-shot Hybrid only")
    return f"""Task: {task.prompt}. ZERO-SHOT; no demonstration. Complete the full instruction using two current RGB views, measured EEF XYZ/axis-angle/two-finger state, seven arm joint positions/velocities and observable history. No depth, environment calibration, object truth, contacts, reward or evaluator success is supplied. Avoid unintended robot/payload collision and obstacle displacement; intended grasp/handle/support contact is allowed. Consider fingers, payload and arm. Never invent hidden poses or infer completion from local arrival. Return strict JSON without a completion flag.

CONTROL. Select exactly one branch, others null. visual_field performs bulk translation and bounded approach/test motion using local measured feedback. direct provides exactly FIVE [dx,dy,dz,rx,ry,rz,grip] actions in [-1,1] for necessary contact correction, articulation or task reorientation: translation1=0.05m, rotation1=0.5rad world delta, grip+1 closes/-1 opens/0 holds intent. Commands run at20Hz from measured pose; actual movement may be smaller. Accepted Direct actions stay unchanged; unsafe commands may be refused. With no near neighbor in current image checks and no fresh metric path, Direct translation norm is limited to 0.002m/action, up to 0.005m only with supported response; near-neighbor ambiguity needs a revised Field corridor. gripper requests open|close with10..30 actions, only phase grasp/release, at held pose. Settled fingers do not prove attachment. Model waiting advances no physics.

FIELD. goal_offset_m uses world axes relative to the CURRENT measured EEF, each within ±0.08m. The local 33×33×33 grid at 0.005m represents weighted goal-directed path cost through your hypotheses, not measured SDF or collision probability. orientation_command=hold and gripper_command=hold. motion_axes=xyz permits spatial paths; xy requires goal z=0 and holds height; z requires goal x=y=0 and holds XY. A z-only path cannot detour. Field phases are approach/transport/align/release/recover; release must be z-only. max_actions is 10..120, max_travel_m≤0.16. For clear task travel choose a useful 5–8cm goal and max_actions=80..120, allowing local response/risk checks to limit execution instead of repeatedly requesting tiny bootstrap goals. Use command_limit_m=0.012..0.02 for visibly clear travel, 0.003..0.006 near interaction. Arrival, evidence and budgets end reuse early.

SCENE. Every obstacle/unknown box has a stable ID, center_offset_m from the same current EEF, positive half_extents_m and honest uncertainty_m in world axes. Preserve static world locations as EEF moves. obstacles means visibly localized objects: nominal cores inflated by known robot/tool/payload geometry stay hard; uncertainty may contribute nonnegative traversal cost when current whole-tool and named-neighbor ROIs support that scene assessment. Missing/unobservable/inconsistent support retains full hard inflation. unknown_regions means genuinely unseen/occluded/unlocalized space and remains hard inflated; absence of calibrated depth alone does not make visible objects unknown. Preserve all known hypotheses; revise from contradictory visual evidence, never shrink/delete/reclassify to evade refusal. No metric or image check certifies3D clearance.

MOTION CONTEXT. Give actual purpose and a whole-tool external ROI including open fingers/payload, distinct from a tracking patch; nullable means unavailable. List relevant nearby objects with accurate ROIs. present requires neighbors; none_relevant requires empty neighbors after reviewing the route. Every listed Field neighbor needs a matching obstacle/unknown box. unobservable caps Field travel at 0.015m and commands at 0.005m. translation_mode transit/separate requires intended_contact_id=null; interaction requires exact target_id. Keep physical IDs across roles: a neighbor becoming contact uses its existing ID. Projected 2D overlap is ambiguity, not proof of 3D collision or automatic clearance: read translation_guard.risk_progress, remaining observation allowance, envelope_revision and observed robot-response feedback for bounded metric-path permission. Metric-hypothesis-remote image overlap alone need not force separation; it is still not certified clear. Current near-risk worsening calls for a changed path. separate is not a bypass; repeated seeds/renaming cannot renew exhausted permission. Correct a whole-tool envelope from current visible evidence without changing its physical identity. Never rotate solely for weak tracking; Direct rotation still requires current whole-tool/neighbor assessment.

REFERENCES. UV origin is top-left, u right/v down. seed supplies actual visible points; nullable ordered positive-area[u0,v0,u1,v1] ROIs contain their points. Prefer rigid textured distal tool features, not fixed robot base. retain uses seed=null and exact existing IDs/moving_kind, preserving tracked coordinates; it cannot restore lost tracks. External moving point is tool/payload; target anchor is a distinct stationary landmark. Payload needs an independent tool point. goal_offset_uv is desired moving projection minus anchor; wrist desired_target_uv is desired target bearing. Field requires external target_id=top-level target_id; Direct/gripper may use a separate stationary anchor. Wrist always matches top-level target; retain it when visible, null honestly otherwise—never invent a centered target. Preserve payload/tool references during stage changes. response_evidence/reference_health distinguish weak evidence from contradiction: weak motion alone does not require yaw/reseed/retreat. M/T/G diagnostic image labels are claims, not verified identities.

SUBGOALS. subgoal_update new creates a unique ID and stores world goal/phase/target/axes. Field retain uses that exact ID/phase/target/axes; its new relative goal proposal is logged but ignored. Direct/gripper repair retains the active ID despite temporary interaction-phase/target changes. Read subgoal_ledger; do not repeatedly add lift clearance after repair. replace changes an unfinished goal using the SAME ID and explicit evidence-based reason; new cannot discard unfinished goals. finish is Direct/gripper only, after measured arrival; then choose the next task goal. none requires null ID and never clears an active goal. Local arrival is not task completion.

INTERACTION. Perfect high-plane image centering can mislead through parallax; make bounded Field progress and reassess. Centering/stalled descent does not establish a graspable feature between jaws. Inspect both views; a neighbor under jaws or wrong target alignment calls for a changed approach or necessary Direct5 contact correction before closing. After closure, use useful bounded test translation with independent tool/payload observation. Field transport requires observed co-motion and payload tracking. A tiny/noisy inconclusive test may be extended once when images/fingers support grasp; visible miss/slip requires changed grasp geometry, not endless tests. For placement, align payload/support and observed grasp offset using Field align/xy first, then held-XY Field release/z. Direct release has zero XYZ in all five actions. Choose gripper opening from observed support, then reassess separation. No automatic stage/release or repeated idle completion claims.

HISTORY. Two recent observation turns and cumulative observable memory remain; raw logs are local. An end_visual_feedback.feedback_ref points to exactly identical full feedback at the named observation_index in this same request. All live hazards, uncertainty budgets, geometry hypotheses and robot measurements remain available. Use concise reasons and useful task progress rather than repeated calibration/holds."""


class VisualFieldSubscriptionClientV12(VisualFieldSubscriptionClientV11):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v12", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        return compact_payload_feedback(payload)

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
