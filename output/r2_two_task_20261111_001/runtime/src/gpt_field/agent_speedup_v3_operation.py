"""Persistent manipulation state shared by every v3 execution route.

Preparing a command can capture an RGB reference, but only an acknowledged
action changes the commanded-gripper lifecycle. Finger settling starts an
uncertain payload probe; accepted two-view material motion is still required
before an interaction is supported. Neither result declares task success.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .agent_speedup_v2_user_check_fragments import (
    GAP_STABLE_RANGE_MM, GAP_STABLE_SAMPLES, MIN_GRIPPER_STEPS,
)
from .agent_speedup_v3_interaction import InteractionEvidence

OPERATION_VERSION = "agent-speedup-v3-persistent-operation-2"


class OperationRuntime:
    def __init__(self, geometry, *, monitor=None, initial_gripper="open"):
        if initial_gripper not in ("open", "close"):
            raise ValueError("initial gripper must be explicit")
        self.geometry = geometry
        self.monitor = monitor or InteractionEvidence(geometry)
        self.last_command = initial_gripper
        self.operations = {}
        self._bound_id = None
        self._serial = 0
        self._pending = None
        self._observed_step = None
        self._last_result = None
        self._reference_compute_s = 0.
        self.events = []
        self.counts = {"acknowledged_actions": 0, "references_started": 0, "references_prepared": 0,
                       "attachment_probes": 0, "release_probes": 0,
                       "hold_supported": 0, "release_supported": 0,
                       "confirmation_rejections": 0, "recovery_claims": 0,
                       "recovery_rejections": 0, "raw_hold_supported_samples": 0,
                       "raw_release_supported_samples": 0, "active_probe_actions": 0}

    @property
    def bound_object_id(self):
        return self._bound_id

    @property
    def operation_id(self):
        record = self.operations.get(self._bound_id)
        return record["operation_id"] if record else None

    def _event(self, kind, step, **detail):
        self.events.append({"kind": kind, "step": int(step), "operation_id": self.operation_id,
                            "object_id": self._bound_id, **deepcopy(detail)})

    def _object(self, ident):
        entity = self.geometry.entities.get(ident)
        return ident if entity and entity.get("role") in ("object", "graspable_part") else None

    def _bind(self, ident, step):
        previous = self.operations.get(self._bound_id)
        if ident not in self.operations or self.operations[ident]["phase"] == "released_supported":
            self._serial += 1
            self.operations[ident] = {
                "operation_id": f"operation-{self._serial}", "object_id": ident,
                "created_step": int(step), "phase": "approach", "last_step": int(step),
                "action_steps": 0, "command_steps": 0, "gaps_mm": [],
                "close_reference_step": None, "release_reference_step": None,
                "probe_revision": None, "probe_window": None, "confirmations": {}, "last_evidence": None,
                "recovery_limit": 2, "recovery_attempts": [],
            }
        changed = self._bound_id != ident
        self._bound_id = ident
        if changed:
            if previous and previous["phase"] == "released_supported":
                self.monitor.active = None
            self._event("operation_bound", step)
        return self.operations[ident]

    def _command(self, command):
        if command == "keep":
            return self.last_command
        if command not in ("open", "close"):
            raise ValueError("command must be open, close or keep")
        return command

    def before_action(self, view, context, command, step):
        """Prepare a reference without claiming that a physical action ran."""
        command, step = self._command(command), int(step)
        context = context or {}
        held = self._object(context.get("held_object_id"))
        target = self._object(context.get("target_id"))
        prior = self.operations.get(self._bound_id)
        grasp_target = (target or held) if context.get("intent") == "grasp" else None
        active_grasp = bool(prior and prior["close_reference_step"] is not None
                            and prior["phase"] != "released_supported")
        # An approach can establish semantic identity, but cannot replace an
        # active held/releasing object merely because the route target changes.
        candidate = grasp_target or (target if context.get("intent") == "approach" else None)
        if candidate and (prior is None or (candidate != self._bound_id and prior["phase"] in ("approach", "released_supported"))):
            self._bind(candidate, step)
        ident = self._bound_id
        if command == "close":
            if grasp_target is not None:
                ident = grasp_target
            elif not active_grasp:
                # Closing an empty hand for pushing or staging is tool posture,
                # even if a scene target or model-held claim is present.
                self._pending = None
                return {"prepared": False, "reason": "gripper_posture_without_grasp_interaction",
                        "object_id": ident}
            elif prior["release_reference_step"] is not None:
                # A posture change cannot silently turn unresolved release
                # evidence into a new grasp. Keep its existing payload history.
                self._pending = None
                return {"prepared": False, "reason": "unresolved_release_reference_retained",
                        "object_id": ident}
        elif command == "open":
            # Release uses the object bound at closing, never the destination.
            if not active_grasp:
                self._pending = None
                return {"prepared": False, "reason": "gripper_posture_without_grasp_interaction",
                        "object_id": ident}
            ident = self._bound_id
        if ident is None or self._object(ident) is None:
            self._pending = None
            return {"prepared": False, "reason": "explicit_manipulated_object_required"}
        record = self.operations.get(ident)
        if (prior and ident != self._bound_id and prior["phase"] not in ("approach", "released_supported")):
            self._pending = None
            return {"prepared": False, "reason": "active_manipulated_object_binding_conflict",
                    "object_id": self._bound_id, "requested_object_id": ident}
        needs_reference = command != self.last_command
        needs_reference |= bool(command == "close" and record is not None
                                and (record["close_reference_step"] is None or record["phase"] == "released_supported"))
        if not needs_reference:
            # A gripper-entry rejection may be followed by an ordinary pose
            # hold with the old command. Its unissued reference is discarded.
            self._pending = None
            return {"prepared": False, "reason": "existing_command_reference_retained", "object_id": ident}
        kind = "hold" if command == "close" else "release"
        token = (step, command, ident)
        if self._pending and self._pending["token"] == token:
            return {"prepared": True, "object_id": ident, "kind": kind, "reference_step": step}
        if (kind == "release" and record is not None and self.last_command == "close"
                and record["probe_revision"] is not None
                and self.geometry.entities[ident].get("attachment") != "held"):
            # A new model shape can replace attachment metadata while the last
            # actually acknowledged command is still settled close. Preserve
            # the possible payload until the physical release is observed.
            # This restores an uncertain extent, never a verified attachment.
            probe = self.geometry.begin_attachment_probe(ident, view, step)
            record["probe_revision"] = self.geometry.entities[ident].get("geometry_revision", 0)
            record["confirmations"].pop("hold", None)
            record["phase"] = "hold_probe"
            self.counts["attachment_probes"] += 1
            self._event("payload_extent_retained_before_release", step, result=probe)
        temporary = type(self.monitor)(self.geometry)
        temporary.begin(ident, view, step, kind=kind)
        self._reference_compute_s += temporary.compute_s
        self.counts["references_prepared"] += 1
        if kind == "release":
            probe = self.geometry.begin_release_probe(ident, view, step)
        else:
            probe = None
        self._pending = {"token": token, "tracker": temporary, "kind": kind,
                         "target_id": ident, "command": command, "step": step,
                         "release_probe": deepcopy(probe)}
        return {"prepared": True, "object_id": ident, "kind": kind, "reference_step": step}

    def _commit_reference(self, before, context, command, before_step):
        # Revalidate the actual acknowledged intent as well as its command.
        # A prepared grasp reference must not survive an unissued route change
        # into an empty-hand posture action at the same step.
        self.before_action(before, context, command, before_step)
        pending = self._pending
        if not pending or pending["command"] != command:
            return
        record = self._bind(pending["target_id"], before_step)
        self.monitor.adopt_reference(pending["tracker"])
        kind = pending["kind"]
        record.update(phase="closing" if kind == "hold" else "opening", command_steps=0, gaps_mm=[])
        record["close_reference_step" if kind == "hold" else "release_reference_step"] = before_step
        if kind == "hold":
            record["release_reference_step"] = None
            record["probe_revision"] = None
            record["probe_window"] = None
            record["confirmations"] = {}
        else:
            record["confirmations"].pop("release", None)
            self.counts["release_probes"] += 1
        self.counts["references_started"] += 1
        self._event("interaction_reference_committed", before_step, interaction=kind)
        self._pending = None

    def observe_transition(self, before, after, context, command, step):
        """Acknowledge once, then consume current evidence before reassessment."""
        command, step = self._command(command), int(step)
        if self._observed_step == step:
            return deepcopy(self._last_result)
        if self._observed_step is not None and step < self._observed_step:
            raise ValueError("operation transition steps must be monotonic")
        self._commit_reference(before, context or {}, command, step-1)
        self.last_command = command
        # Geometry transition is itself idempotent; this covers ordinary/native
        # routes while preserving the child-before-reassessment ordering.
        observe = getattr(self.geometry, "observe_transition", None)
        if callable(observe):
            observe(before, after, step, command)
        self._observed_step = step
        self.counts["acknowledged_actions"] += 1
        record = self.operations.get(self._bound_id)
        if record is not None:
            record["last_step"] = step
            record["action_steps"] += 1
            record["command_steps"] += 1
            record["gaps_mm"] = (record["gaps_mm"]+[float(after.gap_mm)])[-GAP_STABLE_SAMPLES:]
            stable = (len(record["gaps_mm"]) == GAP_STABLE_SAMPLES and
                      max(record["gaps_mm"])-min(record["gaps_mm"]) <= GAP_STABLE_RANGE_MM)
            entity = self.geometry.entities.get(self._bound_id, {})
            revision = entity.get("geometry_revision", 0)
            if (command == "close" and record["close_reference_step"] is not None
                    and record["release_reference_step"] is None and record["phase"] != "released_supported"
                    and record["command_steps"] >= MIN_GRIPPER_STEPS["close"] and stable
                    and (record["probe_revision"] != revision or entity.get("attachment") != "held")):
                result = self.geometry.begin_attachment_probe(self._bound_id, after, step)
                record["probe_revision"] = revision
                record["confirmations"].pop("hold", None)
                record["phase"] = "hold_probe"
                self.counts["attachment_probes"] += 1
                self._event("conditional_payload_probe", step, result=result,
                            gripper_settled=True, grasp_verified=False)
            if entity.get("attachment_probe_unconfirmed"):
                self.counts["active_probe_actions"] += 1
        self.monitor.observe(after, step)
        if record is not None:
            kind = "hold" if command == "close" else "release"
            record["last_evidence"] = self.evidence(kind, after, step)
        self._last_result = self.snapshot()
        return deepcopy(self._last_result)

    def evidence(self, kind, view, step, target_id=None):
        if kind not in ("hold", "release"):
            raise ValueError("interaction kind must be hold or release")
        step = int(step)
        ident = self._bound_id if target_id is None else target_id
        record = self.operations.get(ident)
        unknown = {"version": OPERATION_VERSION, "status": "unknown", "kind": kind,
                   "target_id": ident, "operation_id": record["operation_id"] if record else None,
                   "observed_at_step": step, "geometry_confirmation": {"accepted": False},
                   "task_success": "not_available_to_policy", "attachment_verified": False}
        if record is None or ident != self._bound_id:
            return {**unknown, "reason": "current_manipulated_object_binding_required"}
        reference = record["close_reference_step" if kind == "hold" else "release_reference_step"]
        if reference is None:
            return {**unknown, "reason": "matching_physical_command_not_acknowledged"}
        if kind == "hold" and record["release_reference_step"] is not None:
            return {**unknown, "reason": "release_command_revoked_prior_hold_evidence"}
        raw = self.monitor.evidence(kind, ident, view, step, reference)
        result = {**deepcopy(raw), "operation_id": record["operation_id"],
                  "interaction_phase": record["phase"], "visual_status": raw.get("status"),
                  "geometry_confirmation": {"accepted": False}}
        if raw.get("status") != "supported":
            return result
        if record.get("last_raw_supported_"+kind) != step:
            record["last_raw_supported_"+kind] = step
            self.counts["raw_"+kind+"_supported_samples"] += 1
        if kind == "hold" and record["probe_revision"] is None:
            return {**result, "status": "unknown", "reason": "gripper_settling_required_before_payload_probe"}
        confirmation = record["confirmations"].get(kind)
        entity = self.geometry.entities.get(ident, {})
        usable = bool(confirmation and (kind == "release" or (
            entity.get("attachment") == "held" and not entity.get("stale")
            and not entity.get("release_pending") and not entity.get("attachment_probe_unconfirmed"))))
        if not usable:
            confirmation = self.geometry.confirm_attachment_evidence(raw, view, step)
            if confirmation.get("accepted") is True:
                record["confirmations"][kind] = deepcopy(confirmation)
                if kind == "hold":
                    record["probe_revision"] = self.geometry.entities.get(ident, {}).get("geometry_revision", 0)
                record["phase"] = "held_supported" if kind == "hold" else "released_supported"
                self.counts[kind+"_supported"] += 1
                self._event("interaction_supported", step, interaction=kind,
                            evidence=raw, geometry_confirmation=confirmation)
            else:
                # Repeated reads of the same sample must not inflate failure
                # counters or conceal the geometry module's rejection reason.
                marker = (kind, step, confirmation.get("reason"))
                if record.get("last_rejection") != marker:
                    record["last_rejection"] = marker
                    self.counts["confirmation_rejections"] += 1
                    self._event("interaction_confirmation_rejected", step,
                                interaction=kind, geometry_confirmation=confirmation)
                return {**result, "status": "unknown", "reason": "geometry_confirmation_rejected",
                        "geometry_confirmation": deepcopy(confirmation)}
        return {**result, "status": "supported", "interaction_phase": record["phase"],
                "geometry_confirmation": deepcopy(confirmation)}

    def probe_context(self, context, step, max_steps):
        """Authorize one finite dual-hypothesis lift; never authorize transfer.

        Semantic lift verification and geometry_context for an ordinary grasp
        motion share this deadline. Private fields cannot come from the model.
        A continuation can shorten, but cannot renew, the original deadline.
        """
        result = deepcopy(context)
        record = self.operations.get(self._bound_id)
        entity = self.geometry.entities.get(self._bound_id, {})
        if (record is None or self.last_command != "close" or record["probe_revision"] is None
                or context.get("held_object_id") != self._bound_id
                or not entity.get("attachment_probe_unconfirmed")
                or not entity.get("attachment_probe_shape_usable")):
            return result
        requested = max(0, min(120, int(max_steps)))
        if requested == 0:
            return result
        step = int(step)
        if record["probe_window"] is None:
            record["probe_window"] = {"start_step": step, "deadline_step": step+requested}
        else:
            record["probe_window"]["deadline_step"] = min(record["probe_window"]["deadline_step"], step+requested)
        window = record["probe_window"]
        if step >= window["deadline_step"]:
            return result
        result.update(attachment_probe_target_id=self._bound_id, attachment_probe_authorized=True,
                      attachment_probe_start_step=window["start_step"],
                      attachment_probe_max_steps=window["deadline_step"]-window["start_step"])
        return result

    def geometry_context(self, context, step, max_steps, *, allow_grasp_probe=False):
        """Derive a private planning context from acknowledged interaction state.

        A null public held ID remains unknown. For a finite post-close grasp
        motion only, bind the possible payload in the geometry copy so both
        stationary and carried hypotheses are checked. No extent, uncertainty,
        free-region hypothesis or public task intent changes here.
        """
        result = deepcopy(context)
        record = self.operations.get(self._bound_id)
        if record is None:
            return result
        released = (self.last_command == "open" and record["phase"] == "released_supported"
                    and record["confirmations"].get("release", {}).get("accepted") is True)
        if released and result.get("held_object_id") == self._bound_id:
            # The world-space released extent stays in geometry. Only the old
            # carried-ID declaration is obsolete after accepted actual release.
            result["held_object_id"] = None
            for key in ("attachment_probe_target_id", "attachment_probe_authorized",
                        "attachment_probe_start_step", "attachment_probe_max_steps"):
                result.pop(key, None)
        # A semantic program already owns this same finite probe window. Do not
        # replace its stage deadline with an internal child chunk's deadline.
        if result.get("attachment_probe_authorized") is True:
            return result
        if (not allow_grasp_probe or result.get("intent") != "grasp"
                or result.get("target_id") != self._bound_id
                or result.get("held_object_id") not in (None, self._bound_id)
                or result.get("commanded_gripper") != "close" or result.get("gripper_action")
                or self.last_command != "close" or record["close_reference_step"] is None
                or record["command_steps"] < MIN_GRIPPER_STEPS["close"]
                or len(record["gaps_mm"]) != GAP_STABLE_SAMPLES
                or max(record["gaps_mm"])-min(record["gaps_mm"]) > GAP_STABLE_RANGE_MM):
            return result
        possible_payload = {**result, "held_object_id": self._bound_id}
        authorized = self.probe_context(possible_payload, step, max_steps)
        return authorized if authorized.get("attachment_probe_authorized") is True else result

    def claim_recovery(self, reason, evidence_key, step, max_attempts=2, object_id=None):
        ident = object_id or self._bound_id
        if self._object(ident) is None:
            return {"accepted": False, "reason": "explicit_manipulated_object_required", "remaining": 0}
        record = self.operations.get(ident) or self._bind(ident, step)
        requested = max(0, min(2, int(max_attempts)))
        record["recovery_limit"] = min(record["recovery_limit"], requested)
        token = hashlib.sha256(json.dumps(evidence_key, sort_keys=True, default=str).encode()).hexdigest()
        attempts = record["recovery_attempts"]
        remaining = max(0, record["recovery_limit"]-len(attempts))
        if remaining == 0 or any(item["evidence_key"] == token for item in attempts):
            self.counts["recovery_rejections"] += 1
            return {"accepted": False, "reason": "recovery_budget_exhausted" if remaining == 0 else "unchanged_evidence_already_attempted",
                    "operation_id": record["operation_id"], "remaining": remaining}
        item = {"reason": str(reason), "evidence_key": token, "step": int(step), "attempt": len(attempts)+1}
        attempts.append(item)
        self.counts["recovery_claims"] += 1
        self._event("local_recovery_claimed", step, recovery=item)
        return {"accepted": True, "operation_id": record["operation_id"], "attempt": len(attempts),
                "remaining": max(0, record["recovery_limit"]-len(attempts))}

    def snapshot(self):
        record = self.operations.get(self._bound_id)
        current = None
        if record:
            current = {key: deepcopy(record.get(key)) for key in (
                "operation_id", "object_id", "phase", "created_step", "last_step", "action_steps",
                "close_reference_step", "release_reference_step", "probe_window", "recovery_limit", "recovery_attempts", "last_evidence")}
        return {"version": OPERATION_VERSION, "bound_object_id": self._bound_id,
                "last_command": self.last_command, "current": current, "counts": dict(self.counts),
                "recent_events": deepcopy(self.events[-4:]), "task_success": "not_available_to_policy"}

    def performance_stats(self):
        timings = self.monitor.performance_stats()
        timings["visual_interaction_s"] = timings.get("visual_interaction_s", 0.)+self._reference_compute_s
        return {**timings, "interaction_reference_preparation_s": self._reference_compute_s,
                "operation_runtime_counts": dict(self.counts),
                "operation_count": len(self.operations),
                "interaction_evidence_events": deepcopy([event for event in self.events
                    if event["kind"] in ("interaction_supported", "interaction_confirmation_rejected")][-20:])}
