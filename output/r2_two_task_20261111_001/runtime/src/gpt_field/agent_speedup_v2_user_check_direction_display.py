"""Display-only direction diagnostics for uc7.1 review and trial logs.

These helpers never alter a plan, its validity, or its approval state. Direction
vectors are supplied by the shared spatial contract, not recomputed here.
"""
from __future__ import annotations

import re
from typing import Mapping

_WORLD_AXIS_WORDING = re.compile(r"\bworld\s*[+\-−]\s*[xy]\b", re.IGNORECASE)
_DIRECTION_LABELS = {
    "right_of": "image-right",
    "left_of": "image-left",
    "behind": "away along external view",
    "front_of": "toward external camera",
}


def world_axis_wording_diagnostics(plan: Mapping) -> list[dict]:
    """Locate horizontal world-axis wording in part text for highlighting only."""
    diagnostics = []
    for part in plan.get("parts", []):
        text = part.get("text", "")
        matches = [
            {"text": match.group(), "start": match.start(), "end": match.end()}
            for match in _WORLD_AXIS_WORDING.finditer(text)
        ]
        if matches:
            diagnostics.append({
                "part_id": part.get("id"), "text": text,
                "matches": matches, "display_only": True,
            })
    return diagnostics


def _component(value: float) -> str:
    number = float(value)
    if number == 0:
        return "0"
    return f"{number:+.6g}"


def region_direction_lines(contract: Mapping) -> list[str]:
    """Format the shared region_directions contract for human-readable logs."""
    lines = []
    for pid, region in contract.get("regions", {}).items():
        relation = region["relation"]
        prefix = f"{pid}: {relation} {region.get('anchor')}"
        direction = region.get("direction_world_xy")
        label = _DIRECTION_LABELS.get(relation, relation)
        if contract.get("available") and direction is not None:
            vector = ",".join(_component(value) for value in direction)
            lines.append(f"{prefix} → world ({vector}) [{label}]")
        else:
            reason = contract.get("reason") or "view_frame_unavailable"
            lines.append(f"{prefix} → unavailable ({reason}) [{label}]")
    return lines
