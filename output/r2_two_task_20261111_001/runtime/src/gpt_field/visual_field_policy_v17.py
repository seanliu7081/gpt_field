"""Persistent scene support and candidate-sweep coverage for Field reuse."""
from __future__ import annotations

import copy
import time
import numpy as np

from .observation import decode_image
from .visual_field_policy import images
from .visual_field_policy_v13 import VisualFieldPolicyV13
from .visual_field_policy_v15 import VisualFieldPolicyV15, recovery_options
from .visual_field_v16 import InventoryTargetGeometryV16
from .visual_field_v17 import MetricFieldExecutorV17
from .visual_scene_inventory_v17 import VisualSceneInventoryV17
from .visual_scene_guard_v17 import SceneCoverageGuardV17
from .visual_translation_guard_v17 import ProjectedTranslationGuardV17
from .visual_rotation_guard_v8 import projected_rotation_risk
from .contact_geometry_v13 import _segment_intersection
from .robot_tool_geometry import world_aabbs


class VisualFieldPolicyV17(VisualFieldPolicyV15):
    def reset(self):
        super().reset()
        self.inventory = VisualSceneInventoryV17()
        self.scene_guard = SceneCoverageGuardV17()
        self.local = MetricFieldExecutorV17()
        self.target_contact = InventoryTargetGeometryV16()
        self.translation_guard = ProjectedTranslationGuardV17()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self.scene_materialization = None
        self.effective_motion_context = None

    def _observable_feedback(self):
        result = super()._observable_feedback()
        inventory = getattr(self, 'inventory', None)
        result['scene_inventory'] = None if inventory is None else inventory.summary()
        result['scene_coverage'] = copy.deepcopy(getattr(self.local, 'last_scene_admission', None))
        result['effective_motion_context'] = copy.deepcopy(getattr(self, 'effective_motion_context', None))
        return result

    def _materialize(self, state8):
        payload = self.target_contact.summary().get('carried_payload', {})
        controlled = payload.get('payload_id') if payload.get('active') else None
        material = self.inventory.materialize(state8, self.prediction['target_id'],
            allowed_contact=self.prediction['target_contact'], controlled_payload_id=controlled)
        material['applied_metric_ids'] = [r['id'] for r in getattr(self.local, 'region_geometry', [])]
        material['applied_image_ids'] = [r['id'] for r in self.translation_guard.neighbors]
        material['geometrically_disjoint_ids'] = []
        return material

    def _scene_assess(self, state8, delta_xyz):
        self.scene_materialization = self._materialize(state8)
        return self.scene_guard.assess(state8, delta_xyz, materialization=self.scene_materialization,
            field=self.local, monitor=self.monitor, prediction=self.prediction,
            contact_alignment=self.contact_alignment)

    def _screen_recovery_options(self, state8):
        """Offer only candidates admitted by current scene coverage as well.

        A fresh model declaration can still change admission; this is evidence
        about this observation, not permission to bypass the next actual check.
        """
        options = copy.deepcopy(self.recovery_options)
        accepted = []
        for candidate in options.get('candidates', []):
            proposal = candidate['preview_first_command_delta_m']
            check = self._scene_assess(state8, proposal)
            if check['accepted']:
                candidate['scene_coverage_preview'] = check
                accepted.append(candidate)
            else:
                options.setdefault('rejected_candidates', []).append({
                    'direction_xyz': candidate['direction_xyz'],
                    'reason': check['reason'], 'scene_coverage_preview': check})
        options['candidates'] = accepted
        options['scene_coverage_observation_step'] = self.step_index
        options['preview_scope'] += '; current candidate-sweep scene coverage and target handoff'
        if not accepted:
            options['reason'] = 'no_candidate_passes_current_geometry_translation_and_scene_admission'
        self.recovery_options = options

    def _refresh(self, obs):
        self.subgoals.observe(obs.state)
        self._flush()
        self.client.observable_context = {'visual_feedback': self._observable_feedback(),
            'refresh_reason': self.refresh_reason, 'failed_hypotheses': copy.deepcopy(self.failed_hypotheses[-6:]),
            'attempts_started': len(self.attempts), 'recovery_episodes_detected': self.recovery_episodes,
            'efficiency_goal': 'Reuse Field for useful travel with persistent whole-scene inventory and current contact identity.',
            'task_success_certified': False}
        snapshot = getattr(self.client, 'budget_snapshot', None)
        before_calls = None if snapshot is None else snapshot()['api_calls']
        started = time.monotonic()
        try:
            prediction = self.client.predict(obs.to_dict())
        finally:
            self.model_requests += 1 if before_calls is None else snapshot()['api_calls']-before_calls
        latency = time.monotonic()-started
        started = time.monotonic()
        self.intent_refreshes += 1
        self.phase_previous = self.prediction['phase'] if self.prediction else None
        self.prediction = prediction  # Includes exact unmodified raw_wire_prediction.
        self.chunk_index += 1
        self.block_start, self._block_effective = self.step_index, 0
        self.monitor.configure(images(obs), prediction['visual_references'], obs.state, intent_id=self.model_requests)
        self.inventory.configure(obs, prediction['scene_inventory'])
        self.initial_visual_error_px = self.monitor.last_error if self._anchor_is_interaction_target() else None
        self.visual_divergence_streak = 0
        self.refresh_reason = None
        self.rotation_screen = self.scene_execution_limits = None
        self.chunk, self.goal = [], None
        self._block_initial_error = self._best_goal_error = None
        # Only the legacy goal ledger sees an empty scene list. It validates
        # goal identity/axes; canonical complete geometry is applied below.
        goal_prediction = copy.deepcopy(prediction)
        if goal_prediction.get('field_intent') is not None:
            goal_prediction['field_intent']['obstacles'] = []
        update = self.subgoal_update_result = self.subgoals.apply(obs.state, goal_prediction)
        self.subgoal_rejection = None if update['accepted'] else update['status']
        self.field_configured = prediction['controller'] == 'visual_field' and update['accepted']
        material = self._materialize(obs.state)
        context = copy.deepcopy(prediction['motion_context'])
        context['nearby_obstacles'] = [{'id': r['id'], 'external_roi_uv': r['external_roi_uv']}
                                     for r in material['nearby_obstacles']]
        context['obstacle_assessment'] = ('unobservable' if context.get('tool_external_roi_uv') is None
            else 'present' if context['nearby_obstacles'] else 'none_relevant')
        self.effective_motion_context = context
        intent = prediction.get('field_intent') or {}
        self.target_contact.configure(obs.state, material['target_geometry'], target_id=prediction['target_id'],
            tool_radius_m=intent.get('tool_radius_m', .045), payload_observed=bool(self.monitor.attachment_evidence))
        target_record = next((r for r in material['objects'] if r['id'] == prediction['target_id']), None)
        current_target = bool(target_record and any(v.get('current_roi_uv') is not None or v.get('partial_current_support', False) for v in target_record['views'].values()))
        if not current_target:
            # Retained world geometry remains hard; missing visibility cannot
            # renew a target's intentional-contact exemption.
            self.target_contact.configure(obs.state, None, target_id=prediction['target_id'],
                tool_radius_m=intent.get('tool_radius_m', .045), payload_observed=bool(self.monitor.attachment_evidence))
        self.local.contact_geometry = self.target_contact
        if self.field_configured:
            effective = copy.deepcopy(update['field_intent'])
            if context['obstacle_assessment'] == 'unobservable' or not current_target:
                effective['max_travel_m'] = min(effective['max_travel_m'], .015)
                effective['command_limit_m'] = min(effective['command_limit_m'], .005)
                self.scene_execution_limits = {'reason': 'scene_or_target_image_support_unavailable',
                    'max_travel_m': effective['max_travel_m'], 'command_limit_m': effective['command_limit_m'],
                    'goal_and_regions_preserved': True, 'clearance_certified': False}
            self.local.configure(obs.state, effective, scene_assessment=context,
                inventory_materialization=material, **update['executor_context'])
            self.goal = self.local.goal.copy()
            self._block_initial_error = self._best_goal_error = float(np.linalg.norm(self.goal-obs.state[:3]))
        elif update['accepted']:
            if prediction['controller'] == 'gripper':
                self.gripper.configure(obs.state, prediction['gripper_intent'])
            else:
                self.chunk = copy.deepcopy(prediction['direct_actions'])
                self.rotation_screen = projected_rotation_risk(context, self.chunk, image_size=decode_image(obs.image).size)
                self.rotation_screen['observation_step'] = self.step_index
        descriptor = prediction.get('contact_alignment')
        closes = ((prediction['controller'] == 'gripper' and prediction['gripper_intent']['command'] == 'close')
                  or (prediction['controller'] == 'direct' and any(a[6] > 0 for a in self.chunk)))
        if descriptor is None and closes and self.contact_alignment.summary().get('target_id') == prediction['target_id']:
            descriptor = {'update':'retain','target_id':prediction['target_id'],'external':None,'wrist':None}
        self.contact_alignment.configure(descriptor, self.monitor, obs.state,
            tool_roi_uv=context.get('tool_external_roi_uv'))
        self.local.contact_refinement_enabled = bool(prediction.get('contact_alignment') and self.field_configured
                                                     and prediction['phase'] in ('approach','align'))
        self.subgoals.enable_contact_refinement(self.local.contact_refinement_enabled)
        self.last_contact_refinement = None
        self.translation_guard.configure(images(obs), context, obs.state,
            target_id=prediction['target_id'], monitor=self.monitor)
        self.reference_support.configure(obs, self.monitor)
        self.motion_diagnostic.configure(obs, self.monitor, context)
        self.recovery_options = recovery_options(self.local, self.translation_guard, obs.state,
            monitor=self.monitor, robot_response=self.robot_response)
        self._screen_recovery_options(obs.state)
        self.scene_materialization = self._materialize(obs.state)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-started
        self._refresh_local_s = elapsed
        self.local_total_s += elapsed
        return latency

    def _target_admission(self, obs, action):
        admission = super()._target_admission(obs, action)
        if not admission['accepted']:
            return admission
        # Direct/gripper never borrow stale Field regions. Screen current
        # persistent non-target boxes against the entire known tool sweep.
        material = self._materialize(obs.state)
        state, action = np.asarray(obs.state), np.asarray(action)
        angle = float(np.linalg.norm(action[3:6])*.5)
        stationary = np.linalg.norm(action[:3]) < 1e-12 and angle <= self.local.OSC_GOAL_REFRESH_RAD+1e-15
        if stationary and action[6] == 0:
            return admission
        coverage = self.scene_guard.assess(state, action[:3]*.05, materialization=material,
            field=self.local, monitor=self.monitor, prediction=self.prediction,
            contact_alignment=self.contact_alignment, requires_scene_coverage=True, action7=action)
        self.local.last_scene_admission = copy.deepcopy(coverage)
        if not coverage['accepted']:
            self.last_target_admission = coverage
            return coverage
        parts = world_aabbs(state, orientation_margin_rad=max(.04, angle), full_finger_range=True)
        end = state[:3]+action[:3]*.05
        collisions = []
        for obj in material['objects']:
            if obj['id'] in (self.prediction['target_id'], material['controlled_payload_excluded_id']):
                continue
            world = obj['world_geometry']
            if world is None:
                continue  # Current RGB guard/coverage owns unlocalized regions.
            center, half = np.asarray(world['center_xyz_m']), np.asarray(world['half_extents_m'])
            for part in parts:
                low, high = center-half-np.asarray(part['offset_max_xyz']), center+half-np.asarray(part['offset_min_xyz'])
                if _segment_intersection(state[:3], end, low, high) is not None:
                    collisions.append({'id':obj['id'], 'robot_part':part['name']})
        if collisions:
            admission = {'accepted':False,'need_refresh':True,'reason':'direct_or_gripper_inventory_core_intersection',
                         'diagnostics':{'intersections':collisions, 'clearance_certified':False}}
            self.last_target_admission = admission
        return admission

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics['policy_version'] = 'visual-field-policy-v17'
        if decision.diagnostics.get('refreshed'):
            decision.diagnostics['scene_materialization'] = copy.deepcopy(self.scene_materialization)
        return decision

    def observe_transition(self, before, decision, after):
        started = time.monotonic()
        self.inventory.observe(before, after)
        self.local_total_s += time.monotonic()-started
        row = super().observe_transition(before, decision, after)
        if self.refresh_reason:
            started = time.monotonic()
            self._screen_recovery_options(after.state)
            self.feedback = self._observable_feedback()
            row.update(copy.deepcopy(self.feedback))
            self.local_total_s += time.monotonic()-started
        return row
