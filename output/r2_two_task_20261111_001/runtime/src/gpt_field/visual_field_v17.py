"""Weighted Field planning with V17 scene support and sweep admission."""
import copy
from dataclasses import replace
import numpy as np

from .field import GridField
from .visual_field_v6 import _AxisConstrainedField
from .visual_field_v11 import ToolEnvelopeRiskPlanner
from .visual_field_v16 import MetricFieldExecutorV16
from .visual_scene_guard_v17 import SceneCoverageGuardV17


class MetricFieldExecutorV17(MetricFieldExecutorV16):
    def configure(self, state8, intent, *, inventory_materialization=None, **kwargs):
        material = copy.deepcopy(inventory_materialization)
        self.partial_visibility_hard_region_ids = []
        for item in (material or {}).get('objects', []):
            view = item['views']['external']
            if view.get('extent_visibility') == 'boundary_limited' or view.get('partial_current_support'):
                self.partial_visibility_hard_region_ids.append(item['id'])
                # V16's geometry builder uses current_roi solely to choose
                # full uncertainty versus soft risk. Partial support never
                # earns the latter. Only this private compatibility copy is
                # changed; canonical full/visible projections stay intact.
                view['current_roi_uv'] = None
        super().configure(state8, intent, inventory_materialization=material, **kwargs)
        target_id = (self.target_configuration or {}).get('target_id')
        target = next((row for row in (material or {}).get('objects', [])
                       if row['id'] == target_id), None)
        self.partial_target_coverage_parts = []
        if target is not None and target_id in self.partial_visibility_hard_region_ids:
            # The primary target has its own per-part contact branch in V13.
            # Its partial-image coverage is an ADDITIONAL constraint, matching
            # actual admission even when model contact permission says fingers.
            # Preserve that permission and every existing obstacle/axis mask.
            state = np.asarray(state8, float)
            payload = SceneCoverageGuardV17._carried_bounds(state, self, self.MAX_DRIFT_RAD)
            bounds = SceneCoverageGuardV17._sweep_boxes(target, state,
                {'field_intent': self.intent}, self.MAX_DRIFT_RAD, payload)
            points = self.grid.coordinates(self.origin)
            mask = np.zeros(self.grid.count, bool)
            for part in bounds or []:
                low, high = part['lower_xyz'], part['upper_xyz']
                self.boxes.append((low, high))
                mask |= np.all((points >= low) & (points <= high), axis=1)
                self.partial_target_coverage_parts.append({
                    'tool_part': part['tool_part'], 'lower_xyz': low.tolist(), 'upper_xyz': high.tolist()})
            self.risk_regions = [r for r in self.risk_regions if r['id'] != target_id]
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

    def summary(self):
        result = super().summary()
        result.update(version='persistent-inventory-candidate-sweep-field-v17',
            scene_coverage_scope='actual proposed whole-tool sweep under full model uncertainty when visibility is partial or absent',
            partial_visibility_hard_region_ids=copy.deepcopy(getattr(self, 'partial_visibility_hard_region_ids', [])),
            partial_target_coverage_parts=copy.deepcopy(getattr(self, 'partial_target_coverage_parts', [])),
            clearance_certified=False)
        return result
