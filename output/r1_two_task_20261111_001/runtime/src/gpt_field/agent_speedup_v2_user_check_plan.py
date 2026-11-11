"""Lossless, variant-only plan extensions; baseline v2 keeps its own schema."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from .agent_speedup_v2_plan import PlanParseError, parse_plan as parse_native, plan_schema, validate_plan as validate_native
from .agent_speedup_v2_types import Plan

EXTENSIONS = ("regions", "placement_goals")


def native_plan(raw):
    return {key: copy.deepcopy(value) for key, value in raw.items() if key not in EXTENSIONS}


def validate_plan(plan):
    dependencies = {pid for region in plan.to_json().get("regions", ())
                    for pid in (region["support"], region["anchor"]) if pid is not None}
    return [issue for issue in validate_native(plan)
            if not (issue.code == "unused_object" and issue.pid in dependencies)]


@dataclass(frozen=True)
class ReviewedPlan(Plan):
    extensions: dict = field(default_factory=dict)

    def to_json(self):
        return {**super().to_json(), **copy.deepcopy(self.extensions)}


def parse_plan(raw, *, source="plan"):
    from .agent_speedup_v2_user_check_spatial import SPATIAL_PLAN_PROPERTIES
    original = parse_native(native_plan(raw), source=source)
    for key in EXTENSIONS:
        if key in raw:
            _conform(raw[key], SPATIAL_PLAN_PROPERTIES[key], key)
    return ReviewedPlan(original.scene, original.parts, original.hold_true, original.steps,
                        original.source, {k: copy.deepcopy(raw[k]) for k in EXTENSIONS if k in raw})


def _conform(value, schema, path):
    """The extension uses finite numbers, unsupported by the native conformer."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _conform(value, option, path)
                return
            except PlanParseError:
                pass
        raise PlanParseError(f"Invalid value for {path}")
    kind = schema.get("type")
    valid = {"object": lambda: isinstance(value, dict),
             "array": lambda: isinstance(value, list),
             "string": lambda: isinstance(value, str),
             "number": lambda: isinstance(value, (float,int)) and not isinstance(value,bool) and math.isfinite(value),
             "null": lambda: value is None}[kind]()
    if not valid or ("enum" in schema and value not in schema["enum"]):
        raise PlanParseError(f"Invalid {kind} for {path}")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise PlanParseError(f"Unexpected or missing fields in {path}")
        for key,item in value.items():
            _conform(item,schema["properties"][key],f"{path}.{key}")
    if kind == "array":
        if not schema.get("minItems",0) <= len(value) <= schema.get("maxItems",32):
            raise PlanParseError(f"Invalid array length for {path}")
        for index,item in enumerate(value):
            _conform(item,schema["items"],f"{path}[{index}]")


def with_extensions(plan, raw):
    """Native insert/swap operations rebuild Plan; retain the reviewed sidecar."""
    if isinstance(plan, ReviewedPlan):
        return plan
    result = plan.to_json()
    result.update({key: copy.deepcopy(raw[key]) for key in EXTENSIONS if key in raw})
    return parse_plan(result, source=plan.source)


def variant_plan_schema():
    from .agent_speedup_v2_user_check_spatial import SPATIAL_PLAN_PROPERTIES
    schema = copy.deepcopy(plan_schema())
    schema["properties"].update(copy.deepcopy(SPATIAL_PLAN_PROPERTIES))
    # Strict structured output requires every property. Empty arrays express
    # ordinary object destinations; old persisted native plans remain readable.
    schema["required"].extend(EXTENSIONS)
    return schema
