"""Explicit rigid motion proxy, independent from anatomical hand evidence."""
from __future__ import annotations

import copy
import numpy as np

from .visual_monitor_v21 import RoleAwareVisualMonitorV21
from .visual_monitor_v4 import _orientation_distance
from .visual_reference_track_v32 import ExactDescriptorPartTrackV32


class RoleAwareVisualMonitorV32(RoleAwareVisualMonitorV21):
    feature_bundle_class = ExactDescriptorPartTrackV32

    def __init__(self):
        self.reference_role = 'tool_anatomy'
        self._proxy_contract_error = None
        self._proxy_origin = None
        super().__init__()

    def configure(self, images, references, state8, *, intent_id=None):
        ext = (references or {}).get('external') or {}
        role = ext.get('reference_role', 'tool_anatomy')
        if role not in ('tool_anatomy', 'rigid_transit_proxy'):
            raise ValueError('Unknown external reference role')
        previous_role = self.reference_role
        self._proxy_contract_error = None
        if ext.get('update') == 'retain' and role != previous_role:
            self._proxy_contract_error = 'reference_role_change_requires_seed'
        self.reference_role = role
        if role == 'rigid_transit_proxy':
            if ext.get('moving_kind') != 'tool':
                self._proxy_contract_error = 'proxy_requires_empty_tool_role'
            if ext.get('update') == 'seed':
                seed = ext.get('seed') or {}
                if seed.get('tool_point_uv') is None or seed.get('tool_roi_uv') is None:
                    self._proxy_contract_error = 'proxy_requires_independent_hand_seed'
                self._proxy_origin = np.asarray(state8, float).copy()
            elif self._proxy_origin is None:
                self._proxy_contract_error = 'proxy_retain_has_no_seed_domain'
        else:
            self._proxy_origin = None
        super().configure(images, references, state8, intent_id=intent_id)
        if self.refs.get('external') is not None:
            self.refs['external']['reference_role'] = role
        return self.summary()

    def proxy_support_reason(self):
        if self._proxy_contract_error:
            return self._proxy_contract_error
        if self.reference_role != 'rigid_transit_proxy':
            return None
        moving, hand = self.tracks.get('moving'), self.tracks.get('tool')
        if hand is None or not hand.reliable:
            return 'proxy_independent_hand_reference_unavailable'
        if moving is None or not moving.reliable:
            return 'proxy_motion_reference_unavailable'
        if moving is hand:
            return 'proxy_and_hand_reference_not_independent'
        # Separate constructors alone cannot turn the same image support into
        # an independent hand observer. Reuse the established overlap test.
        if not self._distinct(moving, hand):
            return 'proxy_and_hand_reference_not_independent'
        if (self._proxy_origin is None or self.state is None
                or _orientation_distance(self.state[3:6], self._proxy_origin[3:6]) > .035
                or np.max(abs(self.state[6:]-self._proxy_origin[6:])) > .002):
            return 'proxy_fixed_configuration_domain_changed'
        return None

    def _external_reliable(self):
        return bool(super()._external_reliable() and self.proxy_support_reason() is None)

    def control_limit(self):
        reason = self.proxy_support_reason()
        if reason:
            return 0., reason
        return super().control_limit()

    def planar_correction(self, goal_xyz):
        if self.reference_role == 'rigid_transit_proxy':
            return None
        return super().planar_correction(goal_xyz)

    def summary(self):
        result = super().summary()
        proxy = self.reference_role == 'rigid_transit_proxy'
        result['version'] = 'separate-proxy-and-anatomical-hand-reference-v32'
        result['reference_role'] = self.reference_role
        result['rigid_proxy'] = {
            'active': proxy, 'qualification_reason': self.proxy_support_reason(),
            'motion_reference': 'moving', 'hand_envelope_reference': 'tool' if proxy else 'moving',
            'proxy_transform_defines_hand_envelope': False,
            'anatomical_goal_or_contact_evidence': not proxy,
            'fixed_orientation_and_fingers_required': proxy,
            'semantic_rigidity_certified': False, 'physical_debt_modified': False}
        if proxy:
            result['external_goal_error_px'] = None
            result['same_goal_image_improvement_px'] = None
            result['proxy_relative_image_residual_is_not_anatomical_error'] = True
        return result
