"""Observation-bound Field offers, evaluated by the actual planner and tracker.

An offer is an explicit model choice of a fixed local goal. It neither executes
motion nor changes any scene estimate, visual identity, or physical allowance.
"""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np

from .action import _rotvec_quaternion
from .visual_field_policy import images


def observation_binding(observation):
    value = observation.to_dict() if hasattr(observation, 'to_dict') else observation
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


class FieldProposalBankV31:
    def __init__(self):
        self.binding = None
        self.offers = []
        self.rejected = []
        self.visual_report = None

    @staticmethod
    def _clone(policy):
        # Preserve identities of read-only visual evidence used in epoch keys.
        # Mutable planner, guard debt, pending actions and target geometry are
        # copied. Never deepcopy a bound policy callback and its RGB history.
        memo = {id(x): x for x in (policy.monitor, policy.robot_response,
                policy.inventory, policy.reference_candidates, policy.contact_alignment)}
        prospective_monitor = copy.copy(policy.monitor)
        # New/replace is a new explicit model intent. A retry may make the real
        # serial larger; only distinctness from all acknowledged old IDs is
        # relevant here. Physical ledgers are copied with their debt intact.
        prospective_monitor.intent_id = ('field-proposal-preview', policy.model_requests+1)
        memo[id(policy.monitor)] = prospective_monitor
        for track in policy.monitor.tracks.values():
            if track is not None:
                memo[id(track)] = track
        for row in policy.translation_guard.neighbors:
            for value in row.values():
                if hasattr(value, 'position') and hasattr(value, 'reliable'):
                    memo[id(value)] = value
        local = copy.copy(policy.local)
        guard = copy.copy(policy.translation_guard)
        # Checked-field/egress validity wrappers point back to their executor.
        # Register the prospective owners before descending into either graph;
        # otherwise a nested owner bypasses the callback exclusions below and
        # deepcopies the live policy, including its subscription transport lock.
        # It can also silently create a second executor with stub transports.
        local.__dict__, guard.__dict__ = {}, {}
        memo[id(policy.local)] = local
        memo[id(policy.translation_guard)] = guard
        values = {key: value for key, value in policy.local.__dict__.items()
                  if key not in ('scene_assessor', 'translation_guard')}
        local.__dict__.update(copy.deepcopy(values, memo))
        guard.__dict__.update(copy.deepcopy(policy.translation_guard.__dict__, memo))
        local.bind_translation_guard(guard, prospective_monitor, policy.robot_response)
        guard.bind_egress_runtime(policy.inventory, prospective_monitor)
        witness = copy.copy(policy.reference_candidates)
        witness.__dict__ = policy.reference_candidates.__dict__.copy()
        witness._support_cache = copy.deepcopy(getattr(witness, '_support_cache', None))
        guard.bind_reference_witness(witness)
        return local, guard

    @staticmethod
    def _intent(policy, state, offset, *, recovery=False):
        intent = copy.deepcopy(policy.local.intent)
        # Intent-relative unknown geometry must retain its world coordinates.
        shift = np.asarray(policy.local.origin)-state[:3]
        for box in intent['unknown_regions']:
            for key in ('lower_offset_m', 'upper_offset_m'):
                box[key] = (np.asarray(box[key])+shift).tolist()
        distance = float(np.linalg.norm(offset))
        intent.update(goal_offset_m=np.asarray(offset, dtype=float).tolist(), goal_tolerance_m=.003,
            max_travel_m=min(float(intent['max_travel_m']), distance+.010),
            command_limit_m=min(float(intent['command_limit_m']), .003),
            orientation_target_xyzw=_rotvec_quaternion(state[3:6]).tolist())
        if recovery:
            intent.update(motion_axes='xyz')
        return intent

    @staticmethod
    def _wire_intent(intent):
        from .visual_field_subscription_v30 import output_schema
        keys = output_schema()['properties']['field_intent']['anyOf'][0]['properties']
        value = {key: copy.deepcopy(intent[key]) for key in keys if key in intent}
        value['orientation_command'] = 'hold'
        value['unknown_regions'] = [{'id': box['id'],
            'center_offset_m': ((np.asarray(box['lower_offset_m'])+box['upper_offset_m'])/2).tolist(),
            'half_extents_m': ((np.asarray(box['upper_offset_m'])-box['lower_offset_m'])/2).tolist(),
            'uncertainty_m': box['uncertainty_m']} for box in intent['unknown_regions']]
        return value

    def _configure_preview_guard(self, guard, obs, context, state, prediction, monitor):
        guard.configure(images(obs), context, state, target_id=prediction['target_id'],
                        monitor=monitor)

    def _preview(self, policy, obs, offset, *, recovery=False):
        state = np.asarray(obs.state, float)
        offset = np.asarray(offset, float)
        if (offset.shape != (3,) or not np.isfinite(offset).all()
                or np.any(np.abs(offset) > .08) or np.linalg.norm(offset) < .006):
            return None, 'proposal_offset_outside_local_domain'
        local, guard = self._clone(policy)
        context = copy.deepcopy(guard.context)
        box, scale = guard.tool_box, guard.scale
        if box is None or scale is None:
            return None, 'proposal_current_tool_outline_unavailable'
        roi = np.asarray(box, float)/np.tile(scale, 2)
        if (roi.shape != (4,) or not np.isfinite(roi).all() or
                np.any(roi < 0) or np.any(roi > 1) or np.any(roi[:2] >= roi[2:])):
            return None, 'proposal_current_tool_outline_not_fully_in_view'
        context['tool_external_roi_uv'] = roi.tolist()
        context.update(translation_mode='separate' if recovery else 'transit',
                       intended_contact_id=None, purpose='task_translation')
        guard.context = context
        phase = 'recover' if recovery else policy.prediction['phase']
        prediction = copy.deepcopy(policy.prediction)
        prediction.update(controller='visual_field', phase=phase, recovery_route='normal',
                          motion_context=context, target_contact='none', contact_alignment=None)
        material = policy.inventory.materialize(state, prediction['target_id'], allowed_contact='none')
        context['nearby_obstacles'] = [{'id': r['id'], 'external_roi_uv': r['external_roi_uv']}
                                       for r in material['nearby_obstacles']]
        context['obstacle_assessment'] = 'present' if context['nearby_obstacles'] else 'none_relevant'
        intent = self._intent(policy, state, offset, recovery=recovery)
        local.contact_refinement_enabled = False
        local.proprioceptive_transit_request = None
        local.egress_context = {'controller': 'visual_field', 'phase': phase,
                               'purpose': 'task_translation', 'translation_mode': context['translation_mode']}
        guard.bind_target_approach_context({key: prediction[key]
                                           for key in ('controller', 'phase', 'target_id')})

        def assess(current, delta):
            current_material = policy.inventory.materialize(current, prediction['target_id'], allowed_contact='none')
            current_material['applied_metric_ids'] = [r['id'] for r in local.region_geometry]
            current_material['applied_image_ids'] = [r['id'] for r in guard.neighbors]
            current_material['geometrically_disjoint_ids'] = []
            return policy.scene_guard.assess(current, delta, materialization=current_material,
                field=local, monitor=local.visual_monitor, prediction=prediction,
                contact_alignment=policy.contact_alignment)

        local.scene_assessor = assess
        local.configure(state, intent, scene_assessment=context, inventory_materialization=material)
        self._configure_preview_guard(guard, obs, context, state, prediction, local.visual_monitor)
        binding = local.bind_recovery_route('normal', state)
        if not binding['accepted']:
            return None, binding['reason']
        # Offers use a checked straight subgoal in the learned local response
        # domain. A graph detour needs a different visual prediction model.
        if not local.checked.segment_valid(state[:3], local.goal):
            return None, 'proposal_full_straight_segment_blocked'
        contract = local._transit_reconciliation_contract(state, state[:3], local.goal)
        if contract.get('eligible') is not True:
            return None, 'proposal_full_margin_scene_or_contact_contract_unavailable'
        selected = local.select(state, obs.joint_position)
        if not selected['accepted']:
            return None, selected['status']
        delta = np.asarray(selected['action'][:3])*.05
        direction = np.asarray(offset, float)
        if (np.linalg.norm(delta) < 1e-9 or
                np.dot(delta, direction) < .985*np.linalg.norm(delta)*np.linalg.norm(direction)):
            return None, 'proposal_first_command_leaves_response_heading'
        return {'target_id': prediction['target_id'], 'phase': phase,
            'recovery_route': 'normal', 'field_intent': self._wire_intent(intent),
            'motion_context_overrides': {key: copy.deepcopy(context[key]) for key in
                ('translation_mode', 'intended_contact_id', 'purpose', 'tool_external_roi_uv')},
            'preview_action': selected['action'], 'preview_status': selected['status'],
            'full_route_geometry_sha256': contract.get('geometry_sha256'),
            'geometry_is_hypothesis': True, 'clearance_certified': False,
            'physical_debt_replenished': False, 'actual_selection_rechecks_all_constraints': True}, None

    def build(self, policy, obs, visual_report):
        self.binding = observation_binding(obs)
        self.offers, self.rejected = [], []
        self.visual_report = copy.deepcopy(visual_report)
        prediction = policy.prediction or {}
        if (policy.local.intent is None or policy.local.pending is not None
                or prediction.get('target_contact') != 'none'
                or policy.target_contact.summary().get('carried_payload', {}).get('active')):
            return self.summary()
        raw = []
        if prediction.get('phase') in ('approach', 'align'):
            raw.extend((row, False) for row in visual_report.get('candidates', []))
        # Ordinary recovery offers also carry the exact tested uncertainty and
        # all other intent parameters, avoiding an old-margin/new-margin split.
        for offer in (policy.recovery_options or {}).get('candidates', []):
            if offer.get('required_recovery_route', 'normal') != 'normal':
                continue
            direction = np.asarray(offer.get('direction_xyz'), float)
            if direction.shape != (3,) or np.linalg.norm(direction) < 1e-9:
                continue
            offset = offer.get('recommended_goal_offset_m', (direction*.008).tolist())
            raw.append(({'goal_offset_m': offset, 'source': 'checked_normal_recovery'}, True))
        for row, recovery in raw[:8]:
            offer, reason = self._preview(policy, obs, row['goal_offset_m'], recovery=recovery)
            if offer is None:
                self.rejected.append({'goal_offset_m': row['goal_offset_m'], 'reason': reason,
                                      'source': row.get('source', 'visual_goal_field')})
                continue
            offer['visual_prediction'] = copy.deepcopy(row) if not recovery else None
            offer['source'] = 'checked_normal_recovery' if recovery else 'visual_goal_field'
            offer['binding'] = {'observation_sha256': self.binding}
            digest = hashlib.sha256(json.dumps(offer, sort_keys=True).encode()).hexdigest()[:16]
            offer['proposal_id'] = 'field-'+digest
            active = policy.subgoals.active
            offer['subgoal_update'] = {
                'action': 'replace' if active and not active['local_complete'] else 'new',
                'subgoal_id': active['subgoal_id'] if active and not active['local_complete'] else 'visual-'+digest,
                'reason': ('Choose the current checked recovery Field with unchanged uncertainty.' if recovery else
                           'Choose the current measured-image-error Field subgoal in its supported motion subspace.')}
            offer['visual_references'] = copy.deepcopy(policy.prediction['visual_references'])
            for view in ('external', 'wrist'):
                reference = offer['visual_references'].get(view)
                if reference is not None:
                    reference.update(update='retain', seed=None)
            wrist = policy.monitor.tracks.get('wrist_target')
            if wrist is None or not wrist.reliable:
                offer['visual_references']['wrist'] = None
            offer['scene_declarations'] = []
            for item in policy.inventory.summary(obs.state)['entries']:
                views = item['views']
                external = views['external']['current_roi_uv']
                wrist = views['wrist']['current_roi_uv']
                if external is not None or wrist is not None:
                    offer['scene_declarations'].append({'id': item['id'],
                        'external_roi_uv': copy.deepcopy(external),
                        'wrist_roi_uv': copy.deepcopy(wrist), 'geometry': None})
            from .astra import SchemaError
            from .visual_field_subscription import _validate
            from .visual_field_subscription_v4 import _runtime_schema
            from .visual_field_subscription_v30 import output_schema
            schema = output_schema()['properties']
            try:
                _validate(offer['field_intent'], _runtime_schema(schema['field_intent']['anyOf'][0]))
                _validate(offer['scene_declarations'], _runtime_schema(schema['scene_inventory']))
            except SchemaError:
                self.rejected.append({'goal_offset_m': row['goal_offset_m'],
                    'reason': 'proposal_not_representable_in_native_wire_schema', 'source': offer['source']})
                continue
            self.offers.append(offer)
            if len(self.offers) >= 3:
                break
        return self.summary()

    def resolve(self, identity, observation):
        if self.binding != observation_binding(observation):
            raise ValueError('Field proposal observation changed')
        offer = next((row for row in self.offers if row['proposal_id'] == identity), None)
        if offer is None:
            raise ValueError('Unknown Field proposal')
        return copy.deepcopy(offer)

    def summary(self):
        return {'version': 'observation-bound-field-proposals-v31', 'binding': self.binding,
            'offers': copy.deepcopy(self.offers), 'rejected': copy.deepcopy(self.rejected),
            'visual_goal_field': copy.deepcopy(self.visual_report),
            'explicit_model_choice_required': True, 'automatic_action': False,
            'clearance_certified': False, 'physical_debt_replenished': False}
