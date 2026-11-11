"""Deterministic presentation-only feedback compaction for subscription input.

Robot observations, images and assistant/model outputs are untouched. Complete
feedback remains in local traces; unknown/new diagnostic keys are preserved.
"""
from __future__ import annotations

import copy
import json
import math

from .astra import _canonical

VERSION = "observable-feedback-compact-v11-v1"


def _without(value, keys):
    return {key: copy.deepcopy(item) for key, item in value.items() if key not in keys}


def _clearance(point, box):
    """Signed point/AABB distance for the supplied hypothesis, not environment truth."""
    if point is None or not isinstance(box, dict):
        return None
    low, high = box.get("lower_xyz"), box.get("upper_xyz")
    if not isinstance(low, (list, tuple)) or not isinstance(high, (list, tuple)) or len(low) != 3 or len(high) != 3:
        return None
    delta = [max(low[k]-point[k], point[k]-high[k], 0.) for k in range(3)]
    if any(delta):
        return math.sqrt(sum(value*value for value in delta))
    return -min(min(point[k]-low[k], high[k]-point[k]) for k in range(3))


def _geometry_details(value, current_xyz=None):
    """Summarize robot-part unions without treating their display AABB as solid."""
    if isinstance(value, list):
        return [_geometry_details(item, current_xyz) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in ("translation_guard", "translation_admission"):
            result[key] = copy.deepcopy(item)  # Admission evidence remains complete.
        elif key == "robot_tool_envelope" and isinstance(item, dict):
            result[key] = copy.deepcopy(item)
            if isinstance(item.get("parts"), list):
                result[key]["parts"] = [_without(part, {
                    "min_xyz", "max_xyz", "offset_min_xyz", "offset_max_xyz", "max_radius_m"})
                    for part in item["parts"]]
                result[key]["part_coordinates_omitted"] = True
        elif key == "region_feasibility" and isinstance(item, list):
            regions = []
            for region in item:
                packed = copy.deepcopy(region)
                for label in ("tool_core", "full_halo"):
                    parts_key = label + "_parts"
                    parts = region.get(parts_key)
                    if not isinstance(parts, list) or not parts or not isinstance(region.get(label), dict):
                        continue
                    packed[label] = _without(region[label], {"lower_xyz", "upper_xyz"})
                    # Runtime containment and current_distance_to_union_m are
                    # already exact. Preserve those values; add missing point
                    # distances from the actual constituent union, not _bounds.
                    for prefix, point in (("current", current_xyz), ("goal", value.get("goal_xyz"))):
                        name = prefix + "_distance_to_union_m"
                        if point is not None and name not in packed[label]:
                            distances = [_clearance(point, part) for part in parts]
                            if all(distance is not None for distance in distances):
                                packed[label][name] = min(max(distance, 0.) for distance in distances)
                    packed[label]["part_count"] = len(parts)
                    packed.pop(parts_key, None)
                    packed["union_part_coordinates_omitted"] = True
                regions.append(packed)
            result[key] = regions
        else:
            result[key] = _geometry_details(item, current_xyz)
    return result


def compact_feedback(feedback, current_xyz=None):
    if not isinstance(feedback, dict):
        return copy.deepcopy(feedback)
    result = copy.deepcopy(feedback)
    if isinstance(result.get("controller_handoff"), dict):
        result["controller_handoff"].pop("decision_rule", None)  # Supplied once in instructions.
    if isinstance(result.get("geometry_semantics"), dict):
        result["geometry_semantics"] = {key: value for key, value in result["geometry_semantics"].items()
            if key in ("version", "unobservable_scene_limits", "clearance_certified")}
    field = result.get("field")
    if isinstance(field, dict):
        field.pop("limitations", None)
        field.pop("field_semantics", None)
        regions = field.get("region_feasibility")
        if isinstance(regions, list) and regions:
            compact = []
            for region in regions:
                item = copy.deepcopy(region)
                for name in ("tool_core", "full_halo"):
                    if isinstance(region.get(name), dict):
                        item[name] = _without(region[name], {"lower_xyz", "upper_xyz"})
                        # A V11 union may have free gaps inside its display
                        # bounds. Never calculate signed AABB clearance for it.
                        if not region.get("union_bounds_are_display_only"):
                            for label, point in (("current", current_xyz), ("goal", field.get("goal_xyz"))):
                                clearance = _clearance(point, region[name])
                                if clearance is not None:
                                    item[name][label+"_hypothesis_clearance_m"] = clearance
                compact.append(item)
            field["region_feasibility"] = compact
            field.pop("inflated_regions", None)  # Same IDs/core/halo and margins are above.
        if isinstance(field.get("risk_density"), dict):
            field["risk_density"] = _without(field["risk_density"], {"formula", "sigma"})
        if isinstance(field.get("orientation_hold"), dict):
            field["orientation_hold"] = _without(field["orientation_hold"], {
                "error_frame", "gain", "target_source", "target_xyzw", "osc_goal_refresh_epsilon_rad"})
    visual = result.get("visual")
    if isinstance(visual, dict):
        if isinstance(visual.get("attachment"), dict):
            visual["attachment"].pop("decision_contract", None)
        for value in (visual.get("response"), *(visual.get("tracking") or {}).values()):
            if isinstance(value, dict):
                for key in ("coordinate_semantics", "automatic_reseeding", "fit_samples", "fit_history", "design_matrix"):
                    value.pop(key, None)
    result = _geometry_details(result, current_xyz)
    result["feedback_representation"] = VERSION
    return result


def _record(value, state=None):
    if isinstance(value, list):
        return [_record(item, state) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    # The source states are measured observations retained exactly in the record.
    state = value.get("observation/state", value.get("after_state", state))
    xyz = None if state is None else state[:3]
    result = {}
    for key, item in value.items():
        if key in ("visual_feedback", "end_visual_feedback"):
            result[key] = compact_feedback(item, xyz)
        else:
            result[key] = _record(item, state)
    return result


def compact_payload_feedback(payload):
    result = copy.deepcopy(payload)
    for message in result.get("input", []):
        if message.get("role") != "user":
            continue
        for part in message.get("content", []):
            if part.get("type") != "input_text":
                continue
            try:
                record = json.loads(part["text"])
            except (ValueError, TypeError, KeyError):
                continue
            if not isinstance(record, dict) or record.get("type") not in (
                    "observation", "visual_execution_block", "visual_history_summary"):
                continue
            compact = _record(record)
            if compact != record:
                part["text"] = _canonical(compact)
    return result
