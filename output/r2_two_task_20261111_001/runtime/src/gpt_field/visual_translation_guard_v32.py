"""Separate rigid proxy motion from independently tracked actual hand bounds."""
from __future__ import annotations

import copy
import numpy as np

from .action import _rotvec_quaternion
from .visual_translation_guard_v30 import ProjectedTranslationGuardV30


class ProjectedTranslationGuardV32(ProjectedTranslationGuardV30):
    version = 'independent-hand-envelope-and-rigid-proxy-v32'

    def __init__(self):
        self.reference_role_context = None
        self._proxy_role_pending = None
        self.proxy_unmatched_travel_m = 0.
        super().__init__()

    @staticmethod
    def _proxy(monitor):
        return (((getattr(monitor, 'refs', {}) or {}).get('external') or {})
                .get('reference_role', getattr(monitor, 'reference_role', 'tool_anatomy'))
                == 'rigid_transit_proxy')

    @staticmethod
    def _tool(monitor):
        if ProjectedTranslationGuardV32._proxy(monitor):
            return getattr(monitor, 'tracks', {}).get('tool')
        return ProjectedTranslationGuardV30._tool(monitor)

    def configure(self, *args, **kwargs):
        self.reference_role_context = None
        return super().configure(*args, **kwargs)

    def bind_reference_role_context(self, prediction):
        """Current role claim only; no motion samples, template or debt changes."""
        value = prediction or {}
        intent = value.get('field_intent') or {}
        refs = (value.get('visual_references') or {}).get('external') or {}
        context = value.get('motion_context') or {}
        self.reference_role_context = {
            'controller': value.get('controller'), 'phase': value.get('phase'),
            'target_id': value.get('target_id'), 'target_contact': value.get('target_contact'),
            'contact_alignment': copy.deepcopy(value.get('contact_alignment')),
            'reference_role': refs.get('reference_role', 'tool_anatomy'),
            'moving_kind': refs.get('moving_kind'),
            'translation_mode': context.get('translation_mode'),
            'intended_contact_id': context.get('intended_contact_id'),
            'gripper_command': intent.get('gripper_command'),
            'orientation_target_xyzw': copy.deepcopy(intent.get('orientation_target_xyzw'))}
        return {'bound': True, 'physical_debt_modified': False, 'clearance_certified': False}

    def _proxy_role_reason(self, monitor, orientation_hold):
        c = self.reference_role_context or {}
        if not self._proxy(monitor):
            return 'translation_proxy_current_role_binding_mismatch'
        if (c.get('controller') != 'visual_field' or c.get('phase') not in ('approach', 'align', 'recover')
                or c.get('target_id') != self.target_id or c.get('target_contact') != 'none'
                or c.get('contact_alignment') is not None or c.get('moving_kind') != 'tool'
                or c.get('reference_role') != 'rigid_transit_proxy'
                or c.get('translation_mode') not in ('transit', 'separate')
                or c.get('intended_contact_id') is not None or c.get('gripper_command') != 'hold'
                or not orientation_hold):
            return 'translation_proxy_role_or_contact_ineligible'
        reason = getattr(monitor, 'proxy_support_reason', lambda: 'proxy_monitor_contract_unavailable')()
        if reason:
            return 'translation_' + reason
        if getattr(monitor, 'attachment_evidence', False):
            return 'translation_proxy_payload_evidence_ineligible'
        if (self.state is None or not np.array_equal(getattr(monitor, 'state', None), self.state)
                or self._tool(monitor) is not self.tool_track):
            return 'translation_proxy_current_binding_unavailable'
        try:
            q = np.asarray(c['orientation_target_xyzw'], float)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
                raise ValueError('orientation')
            q = q / np.linalg.norm(q)
            observed = _rotvec_quaternion(self.state[3:6])
            if 2*np.arccos(np.clip(abs(np.dot(q, observed)), 0., 1.)) > .035:
                return 'translation_proxy_orientation_target_changed'
        except (ValueError, TypeError, KeyError):
            return 'translation_proxy_orientation_target_unavailable'
        return None

    def assess(self, delta_xyz, **kwargs):
        delta = np.asarray(delta_xyz, float)
        monitor = kwargs.get('monitor')
        self._proxy_role_pending = None
        proxy = (self._proxy(monitor)
                 or (self.reference_role_context or {}).get('reference_role') == 'rigid_transit_proxy')
        reason = (self._proxy_role_reason(monitor, kwargs.get('orientation_hold', False))
                  if proxy and np.linalg.norm(delta) > 1e-12 else None)
        original = self.context_invalid
        if reason:
            # Run inherited rejection dispatch so every pending-admission
            # ledger is treated exactly as for other unavailable contexts.
            self.context_invalid = reason
        try:
            result = super().assess(delta_xyz, **kwargs)
        finally:
            self.context_invalid = original
        if proxy and result.get('accepted') and np.linalg.norm(delta) > 1e-12:
            self._proxy_role_pending = {'context': copy.deepcopy(self.reference_role_context),
                'state8': self.state.copy(), 'delta': delta.copy(),
                'proxy': getattr(monitor, 'tracks', {}).get('moving'), 'hand': self.tool_track}
        result.setdefault('diagnostics', {})['rigid_proxy_separation'] = self._role_summary()
        return result

    def observe(self, images, state8, action, *, monitor=None):
        pending = self._proxy_role_pending
        before = None if self.state is None else self.state.copy()
        nonzero = np.linalg.norm(np.asarray(action, float)[:3]) > 1e-12
        mismatch = bool(pending is not None and nonzero and (
            pending['context'] != self.reference_role_context
            or not np.array_equal(pending['state8'], self.state)
            or not np.allclose(pending['delta'], np.asarray(action, float)[:3]*.05, atol=1e-10, rtol=0)
            or pending['proxy'] is not getattr(monitor, 'tracks', {}).get('moving')
            or pending['hand'] is not self._tool(monitor)))
        super().observe(images, state8, action, monitor=monitor)
        self._proxy_role_pending = None
        if mismatch:
            self.domain_invalid = 'translation_proxy_acknowledgement_context_mismatch'
            if before is not None:
                self.proxy_unmatched_travel_m += float(np.linalg.norm(
                    np.asarray(state8, float)[:3]-before[:3]))
        return self.summary()

    def _supported_acknowledgement_extra_valid(self, pending):
        role = self._proxy_role_pending
        return (super()._supported_acknowledgement_extra_valid(pending)
                and (role is None or role['context'] == self.reference_role_context))

    def _role_summary(self):
        return {'context': copy.deepcopy(self.reference_role_context),
            'hand_envelope_transform_source': 'independent_actual_hand_track',
            'proxy_motion_copied_to_hand_envelope': False,
            'proxy_training_copied_to_hand_response': False,
            'unmatched_acknowledged_travel_m': self.proxy_unmatched_travel_m,
            'anatomical_reference_required_for_contact': True,
            'physical_allowance_replenished': False, 'clearance_certified': False}

    def summary(self):
        result = super().summary()
        result['rigid_proxy_separation'] = self._role_summary()
        return result
