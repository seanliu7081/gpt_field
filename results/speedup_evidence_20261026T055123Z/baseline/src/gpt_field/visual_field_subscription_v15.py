"""V14 wire with persistent recovery guidance and a current observed-motion image."""
from __future__ import annotations

from .astra import _canonical
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v13 import VisualFieldSubscriptionClientV13, output_schema, geometry_semantics
from .visual_field_subscription_v14 import (
    shared_guidance as guidance_v14, _append_v14_diagnostic, DIAGNOSTIC_LABEL,
)
from .visual_feedback_compact_v14 import compact_payload_feedback
from .visual_reference_overlay import LABEL

VERSION = "visual-field-rgb-joints-v15-persistent-recovery-motion-evidence-v1"
HISTORY_VERSION = "visual-v15-observed-motion-recent-two-turns-v1"
MOTION_LABEL = ("Current repair-handoff RGB motion diagnostic: the marked descriptor and tool ROI "
    "are declared tracking hypotheses. Shown changes come from recent acknowledged frames and may "
    "include shadows, background or proximal arm. They are NOT distal-tool segmentation, verified "
    "identity, automatic reference promotion or clearance. Use both original views to select a "
    "physically visible distal feature; changing a reference never replenishes motion allowance.")


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError("Visual v15 is zero-shot Hybrid only")
    text = guidance_v14(task)
    text = text.replace("Select exactly one controller and populate only its branch.",
        "Select exactly one controller: visual_field requires field_intent NON-NULL; direct requires "
        "direct_actions NON-NULL with exactly five actions; gripper requires gripper_intent NON-NULL. "
        "The other two branches MUST be null. Never return a selected controller with a null branch.")
    begin, end = text.index("RECOVERY."), text.index("\n\nREFERENCES.")
    text = text[:begin] + """RECOVERY. Read current translation_guard, reference_support, recovery_options and motion_evidence. Side-effect-free guard preview filters nominal candidates without spending allowance; they remain proposals, and the actual planned segment is rechecked. Select a useful changed direction with Field phase=recover, translation_mode=separate, and a goal beyond arrival tolerance. Use offered recommended_local_max_travel_m to avoid an unusable tiny remainder; it never overrides guard debt or promises future permission. Finite allowance bounds MEASURED travel, not goal distance. Use new only without an unfinished goal; otherwise replace with the same active ID and an evidence-based reason when changing axes/direction. Hard scene, palm and attachment constraints remain. Shared AND same-direction recovery debt persist across reseeding, IDs, goals or ROIs; a new texture does not prove new directional evidence. Do not repeat a refused corridor or idle to manufacture evidence. The current repair-handoff motion image shows actual changed pixels and descriptor bounds, not semantic segmentation: changes may be background, shadows or proximal arm. Inspect both raw views to locate a real distal feature. Static-looking support is inconclusive; meaningful executed changed-direction motion can resolve it. If genuinely blocked with no useful executable correction, return a VALID direct branch of exactly five [0,0,0,0,0,0,0] actions, field_intent=null and gripper_intent=null, and explain the constraint. Never select a null controller branch, invent gripper opening, or claim completion/clearance. This hold provides no directional evidence and does not automatically terminate the episode.""" + text[end:]
    return text


def _motion_image_needed(context):
    feedback = (context or {}).get("visual_feedback") or {}
    if any(isinstance(feedback.get(key), dict) and feedback[key].get("diagnostic_required") is True
           for key in ("reference_support", "recovery_options")):
        return True
    visual = feedback.get("visual") or {}
    return (visual.get("reference_health") or {}).get("repair_required") is True


def _replace_current_diagnostic(payload, image_url):
    for message in reversed(payload.get("input", [])):
        record = _text_record(message)
        if message.get("role") != "user" or not record or record.get("type") != "observation":
            continue
        # Remove only an explicitly labelled derived image, never either raw view.
        content, kept, skip_derived = message["content"], [], False
        for part in content:
            if part.get("type") == "input_text" and part.get("text") in (LABEL, DIAGNOSTIC_LABEL, MOTION_LABEL):
                skip_derived = True
                continue
            if skip_derived and part.get("type") == "input_image":
                skip_derived = False
                continue
            skip_derived = False
            kept.append(part)
        kept.extend([{"type": "input_text", "text": MOTION_LABEL},
                     {"type": "input_image", "image_url": image_url}])
        message["content"] = kept
        return payload
    return payload


class VisualFieldSubscriptionClientV15(VisualFieldSubscriptionClientV13):
    history_protocol = HISTORY_VERSION
    motion_diagnostic_provider = None

    @property
    def prompt_version(self):
        return VERSION + "-hybrid"

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = shared_guidance(self.task_spec)
        payload["text"]["format"].update(name="visual_field_rgb_joints_v15", schema=output_schema())
        for message in payload["input"]:
            record = _text_record(message)
            if record and record.get("type") == "visual_history_summary":
                record["protocol"] = HISTORY_VERSION
                message["content"][0]["text"] = _canonical(record)
        diagnostic = None
        if _motion_image_needed(self.observable_context) and callable(self.motion_diagnostic_provider):
            # Provider checks that its acknowledged frame exactly matches this
            # observation. No image cache can silently roll over to another turn.
            diagnostic = self.motion_diagnostic_provider(observation)
        if isinstance(diagnostic, str) and diagnostic.startswith("data:image/"):
            payload = _replace_current_diagnostic(payload, diagnostic)
        else:
            payload = _append_v14_diagnostic(payload, observation, self.observable_context)
        return compact_payload_feedback(payload)

    def history_stats(self):
        result = super().history_stats()
        result["protocol"] = HISTORY_VERSION
        return result
