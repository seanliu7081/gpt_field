"""Exact continuous connectors for completely represented metric grid masks.

Grid masks continue to determine Dijkstra vertices and discrete edges. Where
every invalid base-grid vertex is explained by an explicit closed hard box,
continuous queries/segments can instead use those boxes directly. This avoids
an extra interpolation-cell-width collision inflation. Unknown boxes, robot
part boxes, axis corridors, scalar values and all action admission remain.
Scene boxes are model hypotheses, not measured free-space certificates.
"""
from __future__ import annotations

import copy
from itertools import product

import numpy as np

from .distance_planner import DistanceFieldPlanner
from .field import GridField, _GRID_EPS
from .schema import vector3
from .visual_field_v6 import _AxisConstrainedField
from .visual_field_v29 import MetricFieldExecutorV29


class ExactBoxConnectorFieldV31(GridField):
    """Unchanged discrete mask/costs with verified analytic connector validity."""

    def __init__(self, source, boxes):
        if type(source) is not GridField:
            raise ValueError("Only a known unwrapped GridField can replace mask interpolation")
        checked = []
        for lower, upper in boxes:
            low, high = np.asarray(lower, float), np.asarray(upper, float)
            if (low.shape != (3,) or high.shape != (3,) or
                    not np.isfinite(np.r_[low, high]).all() or np.any(low >= high)):
                raise ValueError("Connector hard boxes must have finite ordered bounds")
            low, high = low.copy(), high.copy()
            low.setflags(write=False); high.setflags(write=False)
            checked.append((low, high))
        invalid = np.flatnonzero(~source._valid.ravel())
        points = source.grid.coordinates(source.center)[invalid]
        covered = np.zeros(len(invalid), bool)
        for low, high in checked:
            # Exact inclusion, with no tolerance that can invent coverage for
            # a genuinely unexplained mask vertex outside the declared box.
            covered |= np.all((points >= low) & (points <= high), axis=1)
        if not np.all(covered):
            raise ValueError("Grid contains invalid vertices not represented by hard boxes")
        super().__init__(source.grid, source.center, source.prediction)
        object.__setattr__(self, "_connector_boxes", tuple(checked))
        object.__setattr__(self, "represented_invalid_vertex_count", int(len(invalid)))

    def _point(self, position):
        try:
            point = np.asarray(vector3(position, "position"))
        except (ValueError, TypeError):
            return None
        lattice = (point-self.lower)/self.grid.spacing_m
        if (np.any(lattice < -_GRID_EPS) or
                np.any(lattice > self.grid.size-1+_GRID_EPS)):
            return None
        if not DistanceFieldPlanner._avoids_boxes(point, point, self._connector_boxes)[0]:
            return None
        return point, np.clip(lattice, 0., self.grid.size-1.)

    def query(self, position):
        checked = self._point(position)
        if checked is None:
            return None
        _, lattice = checked
        axes = []
        for coordinate in lattice:
            nearest = int(round(float(coordinate)))
            if abs(float(coordinate)-nearest) <= _GRID_EPS:
                axes.append([(nearest, 1.)])
            else:
                low = int(np.floor(coordinate)); fraction = float(coordinate-low)
                axes.append([(low, 1.-fraction), (low+1, fraction)])
        value = sum(float(self._values[tuple(item[0] for item in corner)])
                    * np.prod([item[1] for item in corner]) for corner in product(*axes))
        return float(value) if np.isfinite(value) else None

    def segment_valid(self, start, end):
        first, last = self._point(start), self._point(end)
        if first is None or last is None:
            return False
        # The cube is convex, so endpoint bounds cover the full segment.
        # Closed slab intersection checks every crossing, including boxes
        # narrower than a lattice cell, grazing and an entirely masked cell.
        return bool(DistanceFieldPlanner._avoids_boxes(
            first[0], last[0], self._connector_boxes)[0])


def exact_box_connector_field(field, boxes):
    """Return a checked wrapper or the unchanged conservative legacy field."""
    original, layers = field, []
    while type(field) is _AxisConstrainedField:
        layers.append((field._locked_axes, field._axis_origin.copy()))
        field = field._obstacle_field
    report = {"version": "represented-mask-exact-connectors-v31", "enabled": False,
              "reason": "unrecognized_base_field", "grid_masks_changed": False,
              "grid_values_changed": False, "discrete_invalid_vertices_preserved": True,
              "continuous_connectors_rechecked_against_all_hard_boxes": True,
              "axis_layers_preserved": len(layers), "clearance_certified": False}
    if type(field) is not GridField:
        return original, report
    try:
        wrapped = ExactBoxConnectorFieldV31(field, boxes)
    except (ValueError, TypeError):
        report["reason"] = "hard_box_coverage_unproven"
        return original, report
    report.update(represented_invalid_vertex_count=wrapped.represented_invalid_vertex_count,
                  analytic_hard_box_count=len(wrapped._connector_boxes))
    for axes, origin in reversed(layers):
        wrapped = _AxisConstrainedField(wrapped, axes, origin)
    if (wrapped.prediction != original.prediction or
            not np.array_equal(wrapped._valid, original._valid)):
        report["reason"] = "original_discrete_mask_not_preserved"
        return original, report
    report.update(enabled=True, reason="all_base_invalid_vertices_have_explicit_hard_box_coverage")
    return wrapped, report


class MetricFieldExecutorV31(MetricFieldExecutorV29):
    """Keep inherited geometry/action checks; remove represented-stencil inflation."""

    def configure(self, *args, **kwargs):
        self.connector_validity = None
        super().configure(*args, **kwargs)
        self.field, self.connector_validity = exact_box_connector_field(self.field, self.boxes)
        self.planner.reset_cache()
        return self.summary()

    def summary(self):
        result = super().summary()
        result.update(version="metric-field-exact-continuous-connectors-v31",
                      connector_validity=copy.deepcopy(getattr(self, "connector_validity", None)))
        return result
