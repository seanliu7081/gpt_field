"""Full persistent scene hypotheses in the existing weighted Field planner.

The wire limits bound a model update, not how many remembered objects exist.
No object is dropped or clipped to fit the old local-response schema.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import numpy as np

from .field import GridField
from .visual_field_v6 import _AxisConstrainedField
from .visual_field_v11 import ToolEnvelopeRiskPlanner, _bounds
from .visual_field_v13 import MetricFieldExecutorV13
from .contact_geometry_v13 import TargetContactGeometry, _boolean
from .visual_field_v4 import _vector, _number


class InventoryTargetGeometryV16(TargetContactGeometry):
    """Accept exact rebased persistent centres, while keeping all part rules."""
    def configure(self, state8, geometry, *, target_id, tool_radius_m=.045, payload_observed=False):
        if geometry is None:
            return super().configure(state8, None, target_id=target_id,
                tool_radius_m=tool_radius_m, payload_observed=payload_observed)
        state = _vector(state8, 8, "state8")
        radius = _number(tool_radius_m, 0., .5, "tool_radius_m")
        payload = _boolean(payload_observed, "payload_observed")
        keys = {"id", "center_offset_m", "half_extents_m", "uncertainty_m", "allowed_contact"}
        if not isinstance(geometry, dict) or set(geometry) != keys or geometry['id'] != target_id:
            raise ValueError("Inventory target geometry must have exact fields and identity")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("Inventory target needs nonempty identity")
        center = _vector(geometry['center_offset_m'], 3, 'center_offset_m')
        half = _vector(geometry['half_extents_m'], 3, 'half_extents_m')
        if np.any(half < .0001) or np.any(half > .5):
            raise ValueError("Target half extents outside observed-model contract")
        uncertainty = _number(geometry['uncertainty_m'], 0., .1, 'uncertainty_m')
        if geometry['allowed_contact'] not in ('none', 'fingers', 'payload_support'):
            raise ValueError('Invalid target contact permission')
        world = state[:3]+center
        if not np.isfinite(np.r_[world-half, world+half]).all():
            raise ValueError('Nonfinite inventory target world bounds')
        record = {'geometry': copy.deepcopy(geometry), 'origin_xyz': state[:3].tolist(),
            'lower_xyz': (world-half).tolist(), 'upper_xyz': (world+half).tolist(),
            'uncertainty_m': uncertainty}
        self._known[target_id] = record
        self.target_id, self.tool_radius_m = target_id, radius
        self.available, self.payload_observed = True, payload
        self._record, self._last_state = copy.deepcopy(record), state.copy()
        return self.configuration(state)


class MetricFieldExecutorV16(MetricFieldExecutorV13):
    def __init__(self):
        super().__init__()
        self.scene_assessor = None
        self.last_scene_admission = None
        self.inventory_applied_ids = []

    def configure(self, state8, intent, *, scene_assessment=None, subgoal_origin_xyz=None,
                  inventory_materialization=None):
        material = inventory_materialization or {'objects': [], 'obstacles': []}
        base = copy.deepcopy(intent)
        # Unknown spatial regions remain under the original strict validator.
        # Inventory geometry is validated at declaration and reconstructed below.
        base['obstacles'] = []
        context = copy.deepcopy(scene_assessment or {})
        context['nearby_obstacles'] = []
        if context.get('obstacle_assessment') != 'unobservable':
            context['obstacle_assessment'] = 'none_relevant'
        super().configure(state8, base, scene_assessment=context,
                          subgoal_origin_xyz=subgoal_origin_xyz)
        self.last_scene_admission = None
        self.inventory_applied_ids = []
        objects = {item['id']: item for item in material['objects']}
        points = self.grid.coordinates(self.origin)
        mask = np.zeros(self.grid.count, bool)
        offsets = [(part['name'], np.asarray(part['offset_min_xyz']),
                    np.asarray(part['offset_max_xyz'])) for part in self.tool_parts]
        radius = float(self.intent['tool_radius_m'])
        offsets.append(('model_declared_tool_or_payload_extent', np.full(3, -radius), np.full(3, radius)))
        represented = {region['id'] for region in self.region_geometry}
        for obstacle in material['obstacles']:
            identity = obstacle['id']
            if identity in represented:
                # Never silently choose between a named unknown and a known
                # object with the same identity; caller must resolve the schema.
                raise ValueError('Inventory identity also appears in another geometry branch: '+identity)
            item = objects[identity]
            world = item['world_geometry']
            center = np.asarray(world['center_xyz_m'], float)
            half = np.asarray(world['half_extents_m'], float)
            low, high = center-half, center+half
            sigma = self.intent['uncertainty_m']+float(world['uncertainty_m'])
            view = item['views']['external']
            current = view.get('current_roi_uv') is not None
            soft = bool(current and (scene_assessment or {}).get('tool_external_roi_uv') is not None)
            core_parts, halo_parts = [], []
            for name, part_low, part_high in offsets:
                core_low, core_high = low-part_high, high-part_low
                core = {'tool_part': name, 'lower_xyz': core_low.tolist(), 'upper_xyz': core_high.tolist()}
                halo = {'tool_part': name, 'lower_xyz': (core_low-sigma).tolist(),
                        'upper_xyz': (core_high+sigma).tolist()}
                core_parts.append(core); halo_parts.append(halo)
                selected = core if soft else halo
                lo, hi = np.asarray(selected['lower_xyz']), np.asarray(selected['upper_xyz'])
                self.boxes.append((lo, hi))
                mask |= np.all((points >= lo) & (points <= hi), axis=1)
            mode = 'persistent_parts_soft_uncertainty' if soft else 'persistent_parts_hard_unobserved_uncertainty'
            self.region_geometry.append({'id': identity, 'kind': 'obstacles', 'geometry_mode': mode,
                'nominal': {'lower_xyz': low.tolist(), 'upper_xyz': high.tolist()},
                'tool_core': _bounds(core_parts), 'full_halo': _bounds(halo_parts),
                'tool_core_parts': core_parts, 'full_halo_parts': halo_parts,
                'margins_m': {'tool_radius': radius, 'intent_uncertainty': self.intent['uncertainty_m'],
                    'region_uncertainty': world['uncertainty_m'], 'combined_uncertainty': sigma},
                'union_bounds_are_display_only': True, 'hard_collision_hypothesis': True,
                'certified_occupied': False})
            self.regions.append({'id': identity, 'kind': 'obstacles', 'geometry_mode': mode,
                'hard_part_count': len(core_parts), 'bounds_semantics': 'union_of_configuration_boxes'})
            if soft:
                self.risk_regions.append({'id': identity, 'parts': core_parts, 'sigma_m': max(sigma,.001)})
            self.inventory_applied_ids.append(identity)
            represented.add(identity)
        self.intent['obstacles'] = copy.deepcopy(material['obstacles'])
        underlying = self.field
        while isinstance(underlying, _AxisConstrainedField):
            underlying = underlying._obstacle_field
        blocked = set(underlying.prediction.blocked_indices) | set(np.flatnonzero(mask).tolist())
        prediction = replace(underlying.prediction, blocked_indices=tuple(sorted(blocked)),
            unknown_indices=tuple(i for i in underlying.prediction.unknown_indices if i not in blocked))
        self.field = GridField(self.grid, tuple(self.origin), prediction)
        if self.locked_axes:
            self.field = _AxisConstrainedField(self.field, self.locked_axes, self.origin)
            self.field = _AxisConstrainedField(self.field, self.locked_axes, self.subgoal_origin)
        self.planner = ToolEnvelopeRiskPlanner(self.boxes, self.risk_regions)
        return self.summary()

    def select(self, state8, joint_position, *, command_cap_m=None, measured_cap_m=None):
        count, rotation = self.orientation_goal_refresh_commands, self.last_orientation_command_rad.copy()
        result = super().select(state8, joint_position, command_cap_m=command_cap_m,
                                measured_cap_m=measured_cap_m)
        if result['accepted'] and self.scene_assessor is not None:
            admission = self.scene_assessor(state8, np.asarray(result['action'][:3])*.05)
            self.last_scene_admission = copy.deepcopy(admission)
            if not admission['accepted']:
                self.pending = None
                self.orientation_goal_refresh_commands = count
                self.last_orientation_command_rad = rotation
                return self._stop(admission['reason'], {'scene_coverage_admission': admission})
            result['diagnostics']['scene_coverage_admission'] = copy.deepcopy(admission)
        return result

    def summary(self):
        result = super().summary()
        result.update(version='persistent-inventory-field-v16',
            inventory_applied_ids=copy.deepcopy(getattr(self,'inventory_applied_ids',[])),
            scene_coverage_admission=copy.deepcopy(getattr(self,'last_scene_admission',None)))
        return result
