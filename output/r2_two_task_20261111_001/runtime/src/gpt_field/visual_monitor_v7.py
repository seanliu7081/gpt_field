"""Current controlled-reference evidence and persistent local repair state.

V6 numerical feature identities and attachment checks are retained. V7 limits
unvalidated travel by observed local correspondence, not model-request count.
Only supplied, bounded image seeds are considered; no whole-scene search, object
truth or camera calibration is used.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .visual_monitor_v4 import _orientation_distance
from .visual_monitor_v5 import _sample, _ncc
from .visual_monitor_v6 import RoleAwareVisualMonitorV6


class RoleAwareVisualMonitorV7(RoleAwareVisualMonitorV6):
    recent_window_size = 8
    reference_travel_limit_m = .015

    def __init__(self):
        super().__init__()
        self._recent_motion = deque(maxlen=self.recent_window_size)
        self._reference_records = []
        self._active_reference = None
        self._seed_reused_footprint = False
        self._recovery_motion = deque(maxlen=self.recent_window_size)
        self._recovery_supported_transitions = 0

    @staticmethod
    def _fingerprint(track):
        if track is None or not track.reliable:
            return None
        offsets = np.stack(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5)), axis=-1)
        return {"point": track.position.copy(), "scale": track.scale.copy(), "points": track.points.copy(),
                "templates": [_sample(track.previous_image, point+track.offsets) for point in track.points],
                "center_template": _sample(track.previous_image, track.position+offsets)}

    @staticmethod
    def _same_local_correspondence(first, second):
        """Match existing local texture, independently of model-authored IDs.

        Merely overlapping ROIs is insufficient: require matching descriptors
        at the same local image locations. A moving foreground that occludes an
        old rejected background may therefore establish a new correspondence.
        """
        if first is None or second is None:
            return False
        if (np.linalg.norm(first["point"]-second["point"]) <= 5
                and _ncc(first["center_template"], second["center_template"]) >= .8):
            return True
        points_a, points_b = first["points"], second["points"]
        if not len(points_a) or not len(points_b):
            return False
        distances = np.linalg.norm(points_a[:, None]-points_b[None, :], axis=-1)
        matched_a, matched_b = set(), set()
        for a, b in zip(*np.where(distances <= 3.)):
            if _ncc(first["templates"][a], second["templates"][b]) >= .8:
                matched_a.add(int(a)); matched_b.add(int(b))
        return min(len(matched_a), len(matched_b)) >= max(3, int(np.ceil(.35*min(len(points_a), len(points_b)))))

    def _snapshot_active_reference(self):
        if self._active_reference is not None:
            current = self._fingerprint(self.tracks.get("moving"))
            if current is not None:
                self._active_reference["fingerprint"] = current

    def configure(self, images, references, state8, *, intent_id=None):
        self._snapshot_active_reference()
        previous = self._active_reference
        old_epoch = self.epoch
        super().configure(images, references, state8, intent_id=intent_id)
        if self.epoch != old_epoch:
            self._recent_motion.clear()
            self._reset_recovery()
        ext = references.get("external")
        self._seed_reused_footprint = False
        if ext is None:
            self._active_reference = None
        elif ext["update"] == "retain":
            # An intent or failed retain never issues a new reference budget.
            self._active_reference = previous
        else:
            fingerprint = self._fingerprint(self.tracks.get("moving"))
            matches = [record for record in self._reference_records
                       if any(self._same_local_correspondence(old, fingerprint) for old in
                              (record.get("rejected_fingerprint"), record["fingerprint"]))]
            if matches:
                self._active_reference = max(matches, key=lambda r: (bool(r["blocked_reason"]), r["unvalidated_travel_m"]))
                self._seed_reused_footprint = True
            elif fingerprint is not None:
                self._active_reference = {"id": len(self._reference_records)+1,
                    "fingerprint": fingerprint, "unvalidated_travel_m": 0., "blocked_reason": None,
                    "validated_transitions": 0}
                self._reference_records.append(self._active_reference)
            else:
                self._active_reference = previous
        if self._active_reference is not None and self._active_reference["blocked_reason"]:
            self.status = self._active_reference["blocked_reason"]
        return self.summary()

    def _reset_recovery(self):
        self._recovery_motion.clear()
        self._recovery_supported_transitions = 0

    def _block_reference(self, reason):
        self._active_reference["blocked_reason"] = reason
        self._active_reference["rejected_fingerprint"] = self._fingerprint(self.tracks.get("moving"))
        self._reset_recovery()
        # A blocked correspondence may become informative after a purposeful
        # Direct interaction, but all predictive rescue evidence must be new.
        # This does not reset its debt, feature identity or attachment state.
        self.response = self._new_model()
        self.validation_count = self.bad_predictions = 0
        self._held_out_motion = []

    def _recent_metrics(self, rows=None):
        rows = list(self._recent_motion if rows is None else rows)
        if not rows:
            return {"transitions": 0, "measured_path_m": 0., "measured_net_translation_m": 0.,
                    "feature_displacement_px": 0., "motion_threshold_px": .35,
                    "predictive_response_validated": False}
        eef = np.array([r[0] for r in rows])
        motion = np.array([r[1] for r in rows])
        image_positions = np.vstack((np.zeros(2), np.cumsum(motion, axis=0)))
        extent = float(np.max(np.linalg.norm(image_positions[:, None]-image_positions[None, :], axis=-1)))
        threshold = max(.35, 3*max(r[2] for r in rows))
        return {"transitions": len(rows), "measured_path_m": float(np.linalg.norm(eef, axis=1).sum()),
                "measured_net_translation_m": float(np.linalg.norm(eef.sum(axis=0))),
                "feature_displacement_px": extent, "motion_threshold_px": threshold,
                "predictive_response_validated": False}

    def _recent_motion_supported(self, rows=None):
        metrics = self._recent_metrics(rows)
        return (metrics["transitions"] >= 3
                and metrics["feature_displacement_px"] >= metrics["motion_threshold_px"])

    @property
    def response_ready(self):
        if self._active_reference is not None and self._active_reference["blocked_reason"]:
            return False
        return bool(self._recent_motion_supported() and super().response_ready)

    def observe(self, images, state8, action):
        state = np.asarray(state8, float)
        delta = state[:3]-self.state[:3]
        distance = float(np.linalg.norm(delta))
        domain_changed = (_orientation_distance(state[3:6], self.state[3:6]) > .035
                          or np.max(np.abs(state[6:8]-self.state[6:8])) > .002)
        previous = {k: self.tracks[k].position.copy() for k in ("moving", "target")
                    if self.tracks.get(k) is not None and self.tracks[k].reliable}
        if domain_changed:
            self._recent_motion.clear()
            self._reset_recovery()
        super().observe(images, state8, action)
        row = None
        if not domain_changed and len(previous) == 2 and self._external_reliable():
            moving, target = self.tracks["moving"], self.tracks["target"]
            motion = (moving.position-previous["moving"])-(target.position-previous["target"])
            noise = float(np.hypot(self._noise(moving), self._noise(target)))
            # Keep actual zero-motion observations too: old moving evidence
            # must age out when the camera feature becomes stationary.
            row = (delta.copy(), motion.copy(), noise)
            self._recent_motion.append(row)
        elif not self._external_reliable():
            self._recent_motion.clear()
            self._reset_recovery()
        reference = self._active_reference
        if reference is not None:
            if reference["blocked_reason"] and row is not None:
                self._recovery_motion.append(row)
                # Seed/retain and a later good-looking image do not rehabilitate
                # the old reference. Require three NEW moving, predictive
                # transitions after an independent fit/held-out reset. Payload
                # references additionally need independent tool co-motion.
                payload = (self.refs.get("external") or {}).get("moving_kind") == "payload"
                supported = (distance > .0001 and np.linalg.norm(row[1]) > row[2]
                    and self._recent_motion_supported(self._recovery_motion)
                    and super().response_ready and (not payload or self.attachment_evidence))
                self._recovery_supported_transitions = self._recovery_supported_transitions+1 if supported else 0
                if self._recovery_supported_transitions >= 3:
                    reference["blocked_reason"] = None
                    reference["recoveries"] = reference.get("recoveries", 0)+1
            if self.response_ready and not domain_changed:
                reference["unvalidated_travel_m"] = 0.
                reference["validated_transitions"] += 1
            else:
                reference["unvalidated_travel_m"] += distance
            recent = self._recent_metrics()
            if not reference["blocked_reason"]:
                if (recent["transitions"] >= 3 and recent["measured_path_m"] >= .006
                        and recent["measured_net_translation_m"] >= .004
                        and recent["feature_displacement_px"] < recent["motion_threshold_px"]):
                    self._block_reference("controlled_reference_unresponsive_change_correspondence")
                elif reference["unvalidated_travel_m"] >= self.reference_travel_limit_m-.001:
                    self._block_reference("controlled_reference_budget_exhausted_change_correspondence")
            if reference["blocked_reason"]:
                self.status = reference["blocked_reason"]
            elif self.response_ready:
                self.status = "response_validated"
            elif self.status == "response_validated":
                self.status = "awaiting_recent_predictive_motion_evidence"
        self._snapshot_active_reference()
        return self.summary()

    def control_limit(self):
        if self._active_reference is not None and self._active_reference["blocked_reason"]:
            return 0., self._active_reference["blocked_reason"]
        if self.missing >= 2 or not self._external_reliable():
            return 0., "external_reference_unavailable"
        if self.status in ("visual_response_contradiction", "target_anchor_shifted"):
            return 0., self.status
        if self.response_ready:
            return None, None
        used = 0. if self._active_reference is None else self._active_reference["unvalidated_travel_m"]
        remaining = self.reference_travel_limit_m-used
        if remaining <= .001:
            return 0., "controlled_reference_budget_exhausted_change_correspondence"
        return min(.005, remaining), None

    def summary(self):
        result = super().summary()
        result["version"] = "role-aware-current-controlled-reference-v7"
        record = self._active_reference
        blocked = None if record is None else record["blocked_reason"]
        reliable = self._external_reliable()
        current = self.tracks.get("moving")
        point = (None if record is None else record["fingerprint"]["point"].tolist())
        scale = (record["fingerprint"]["scale"] if current is None and record is not None
                 else None if current is None else current.scale)
        rejected_uv = (current.uv.tolist() if current is not None else
                       None if point is None or scale is None else (np.asarray(point)/scale).tolist())
        used = 0. if record is None else record["unvalidated_travel_m"]
        metrics = self._recent_metrics()
        metrics["predictive_response_validated"] = self.response_ready
        repair = bool(blocked) or not reliable
        result["reference_health"] = {"status": blocked or ("validated" if self.response_ready else "observing" if reliable else "unavailable"),
            "repair_required": repair,
            "recommended_action": ("Select a different visible rigid landmark on the controlled tool or payload, with a bounded seed/ROI. The current local image correspondence did not support measured robot motion; retaining, renaming or reseeding the same features does not renew its allowance. Fresh measured predictive motion can rehabilitate this correspondence after a visibly justified Direct task interaction or informative view change; repeated static/probe requests are not progress."
                if blocked else "Explicitly seed a currently visible bounded correspondence, or choose a visually justified terminal interaction." if not reliable else
                "Continue useful task motion with the current correspondence; no separate calibration routine is required."),
            "rejected_current_uv": rejected_uv if repair else None,
            "correspondence_id": None if record is None else record["id"],
            "unvalidated_travel_m": used,
            "unvalidated_travel_remaining_m": max(0., self.reference_travel_limit_m-used),
            "recent_window": metrics, "seed_reused_rejected_footprint": bool(self._seed_reused_footprint and blocked),
            "fresh_recovery_supported_transitions": self._recovery_supported_transitions,
            "evidence_based_recoveries": 0 if record is None else record.get("recoveries", 0),
            "semantic_identity_certified": False}
        return result
