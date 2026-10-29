"""Offline display checks: frozen directions and non-blocking wording highlights."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from gpt_field.agent_speedup_v2_user_check import ReviewStore, plan_validation
from gpt_field.agent_speedup_v2_user_check_direction_display import (
    region_direction_lines,
    world_axis_wording_diagnostics,
)

HTML = Path(__file__).resolve().parents[1] / "src/gpt_field/agent_speedup_v2_user_check.html"


def _plan():
    return {
        "scene": "An object and a support surface.",
        "parts": [
            {"id": "A", "text": "object: body on world +x side", "move": "free", "w": 1, "x": 2},
            {"id": "B", "text": "support: upper surface", "move": "fixed", "w": 3, "x": 4},
        ],
        "hold_true": [],
        "steps": [{"do": "place_on", "obj": "A", "grip": [], "ref": "B", "sense": "any",
                   "keep": [{"c": "upright", "p": "A"}], "until": []}],
    }


def test_wording_diagnostics_find_horizontal_axes_without_mutating_plan():
    plan = {"parts": [
        {"id": "A", "text": "world +x, WORLD −y and world- x; world +z is vertical"},
        {"id": "B", "text": "world +xyz and underworld +x are not direction tokens"},
    ]}
    before = copy.deepcopy(plan)
    diagnostics = world_axis_wording_diagnostics(plan)
    assert len(diagnostics) == 1
    assert diagnostics[0]["part_id"] == "A"
    assert diagnostics[0]["display_only"] is True
    assert [match["text"] for match in diagnostics[0]["matches"]] == ["world +x", "WORLD −y", "world- x"]
    for match in diagnostics[0]["matches"]:
        assert diagnostics[0]["text"][match["start"]:match["end"]] == match["text"]
    assert plan == before


def test_diagnostic_does_not_turn_legacy_wording_into_plan_issue_or_approval_veto(tmp_path):
    plan = _plan()
    validation = plan_validation(plan)
    assert validation["valid"]
    assert world_axis_wording_diagnostics(plan)
    assert plan_validation(plan) == validation
    store = ReviewStore.create(
        tmp_path / "review", task_key="fixture", episode_index=0,
        task_prompt="Place the object on the support.", initial_observation={"offline": True},
        plan=plan, runtime_sha256={"fixture": "0" * 64}, preparation={},
    )
    approved = store.approve(expected_revision=0, reviewed_steps=[0])
    assert approved["status"] == "approved"
    assert approved["plan"] == plan


def test_direction_lines_use_contract_vectors_and_report_unavailable():
    contract = {"available": True, "regions": {
        "E": {"relation": "right_of", "anchor": "C", "direction_world_xy": [0.0, 1.0]},
        "F": {"relation": "front_of", "anchor": "B", "direction_world_xy": [1.0, -0.0]},
    }}
    before = copy.deepcopy(contract)
    assert region_direction_lines(contract) == [
        "E: right_of C → world (0,+1) [image-right]",
        "F: front_of B → world (+1,0) [toward external camera]",
    ]
    assert contract == before
    contract.update(available=False, reason="no_external_calibration")
    assert "unavailable (no_external_calibration)" in region_direction_lines(contract)[0]
    assert region_direction_lines({"available": True, "regions": {}}) == []


def _run_ui_helpers(script):
    node_binary = shutil.which("node")
    if node_binary is None:
        candidates = sorted(Path("/opt/nvm/versions/node").glob("*/bin/node"))
        node_binary = str(candidates[-1]) if candidates else None
    if node_binary is None:
        pytest.skip("Node is required for executing the browser's pure display helpers")
    text = HTML.read_text()
    # Evaluate exactly the helper functions shipped in the CSP-protected page.
    start = text.index("  // Display helpers only:")
    end = text.index("  const editable =", start)
    stub = """
const document = {createTextNode(text) {return {tag: 'text', text};}};
const node = (tag,text,className) => ({tag,text,className});
"""
    result = subprocess.run([node_binary, "-e", stub + text[start:end] + script],
                            check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def test_browser_display_recomputes_after_relation_and_anchor_edits():
    result = _run_ui_helpers("""
const frame={available:true,relation_directions:{right_of:[0,1],front_of:[1,0]}};
const region={id:'E',relation:'right_of',anchor:'C'};
const values=[regionDirectionLine(region,frame)];
region.relation='front_of';region.anchor='D';
values.push(regionDirectionLine(region,frame));
values.push(regionDirectionLine(region,{available:false,reason:'no_external_calibration'}));
region.relation='on';values.push(regionDirectionLine(region,frame));
console.log(JSON.stringify(values));
""")
    assert result == [
        "E: right_of C → world (0,+1) [image-right]",
        "E: front_of D → world (+1,0) [toward external camera]",
        "E: front_of D → unavailable (no_external_calibration) [toward external camera]",
        None,
    ]


def test_browser_highlights_text_nodes_without_html_interpolation():
    result = _run_ui_helpers("""
const parent={children:[],append(child){this.children.push(child);}};
appendPartWording(parent,'<img src=x> world +x and WORLD −y; world +z');
console.log(JSON.stringify(parent.children));
""")
    assert "".join(item["text"] for item in result) == "<img src=x> world +x and WORLD −y; world +z"
    assert [item["text"] for item in result if item["tag"] == "mark"] == ["world +x", "WORLD −y"]
    assert result[0] == {"tag": "text", "text": "<img src=x> "}


def test_ui_labels_and_notes_have_no_old_world_axis_mapping():
    text = HTML.read_text()
    labels = text.split("  const labels = {", 1)[1].split("  };", 1)[0]
    assert not re.search(r"world\s*[+\-−]\s*[xy]", labels, re.IGNORECASE)
    assert "follow world axes" not in text
    assert "image-right in external view" in labels
    assert "image-left in external view" in labels
    assert "Directions are frozen at review" in text
    assert "this highlighting does not block approval" in text
    assert 'regionDirectionLine(region, record.preparation?.view_frame)' in text
