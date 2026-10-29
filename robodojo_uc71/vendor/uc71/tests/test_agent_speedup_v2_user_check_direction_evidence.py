"""Development-only uc7.1 evidence checks; no simulator or inference is started.

Benchmark predicates and archived object positions below are test data only.
Nothing from this module is imported into policy/runtime requests.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gpt_field.astranomy_agentic_geometry import project_world_point
from gpt_field.contact_skills import StepView
from gpt_field.observation import PolicyObservation, decode_image, image_to_data_url
from gpt_field.robocasa_user_check_sim import RoboCasaConfig, RoboCasaWorker
from gpt_field.agent_speedup_v2_user_check_spatial import build_view_frame, validate_spatial_spec

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/uc7_direction_semantics_20261028"
T6 = ROOT / "runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6"
LIBERO = ROOT.parent / "heading_flow_pub/third_party/LIBERO/libero/libero"


def _require(path):
    if not path.exists():
        pytest.skip(f"Recorded development evidence is not installed: {path}")
    return path


def _rgb():
    y, x = np.indices((512, 512))
    return np.stack([x % 251, y % 241, (3*x + y) % 253], axis=-1).astype(np.uint8)


def _robot_obs():
    return {"robot0_eef_pos": [.1, -.1, .5], "robot0_eef_quat": [0, 0, 0, 1],
            "robot0_gripper_qpos": [.04, -.04], "robot0_joint_pos": [0.0]*7,
            "robot0_joint_vel": [0.0]*7}


def test_libero_actual_transport_then_stepview_is_one_vertical_flip():
    raw = _rgb()
    wrist = np.rot90(raw).copy()
    obs = PolicyObservation.from_libero(
        {**_robot_obs(), "agentview_image": raw, "robot0_eye_in_hand_image": wrist},
        "Offline image orientation fixture", image_size=512,
    )
    # Exercise the real encoder separately as well as the from_libero call site.
    assert obs.image == image_to_data_url(raw, 512)
    view = StepView(obs)
    np.testing.assert_array_equal(view.external_rgb_upright, raw[::-1, :])
    np.testing.assert_array_equal(view.wrist_rgb_upright, wrist[::-1, :])
    assert not np.array_equal(view.external_rgb_upright, raw[::-1, ::-1])


def test_robocasa_actual_response_then_stepview_does_not_add_a_second_flip(monkeypatch):
    from gpt_field import robocasa_user_check_geometry as geometry_module
    raw = _rgb()
    wrist = np.rot90(raw).copy()
    worker = RoboCasaWorker(RoboCasaConfig())
    # RoboSuite's opencv convention supplies A[::-1, :]. No simulator is built.
    native_obs = {**_robot_obs(), worker.config.external_camera + "_image": raw[::-1, :],
                  "robot0_eye_in_hand_image": wrist[::-1, :]}
    worker._fresh_observation = lambda: native_obs
    worker._contacts = lambda: {"count": 0, "records": []}
    worker.prompt = "Offline image orientation fixture"
    worker.base_controller = SimpleNamespace(qpos_index=[0, 1, 2])
    worker.torso_controller = SimpleNamespace(qpos_index=[3])
    worker.base_position = np.zeros(3)
    worker.torso_position = np.zeros(1)
    worker.env = SimpleNamespace(
        sim=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(4))),
        _check_success=lambda: False,
    )
    monkeypatch.setattr(geometry_module, "extract_robocasa_geometry", lambda *args, **kwargs: {})
    response = worker._response()
    obs = PolicyObservation.from_dict(response["observation"])
    view = StepView(obs)
    np.testing.assert_array_equal(view.external_rgb_upright, raw[::-1, :])
    np.testing.assert_array_equal(view.wrist_rgb_upright, wrist[::-1, :])
    assert not np.array_equal(view.external_rgb_upright, raw[::-1, ::-1])
    assert not response["status"]["terminated"]


def _recorded_geometry():
    requests = _require(T6 / "trial/requests")
    path = next(requests.glob("000001-*.request.json"))
    request = json.loads(path.read_text())
    return json.loads(request["payload"]["input"][0]["content"][0]["text"])["geometry"]


def _cell(uv):
    u, v = uv
    assert 0 <= u < 512 and 0 <= v < 512
    return int(v // 64) * 8 + int(u // 64) + 1


def audit_recorded_projection():
    camera = _recorded_geometry()["cameras"]["external"]
    physical = json.loads(_require(T6 / "physical_validation.json").read_text())
    positions = {row["object"]: row["settled_xyz_m"] for row in physical["objects"]}
    projections = {}
    for name in ("chocolate_pudding_1", "porcelain_mug_1", "plate_1"):
        uv = project_world_point(positions[name], camera)
        projections[name] = {"uv": uv, "cell": _cell(uv),
                             "mirrored_cell": _cell((511-uv[0], uv[1]))}
    row, column = divmod(61-1, 8)
    u, v = (column+.5)*64, (row+.5)*64
    cx, cy = camera["principal_point_px"]
    fx, fy = camera["focal_length_px"]
    rotation = np.asarray(camera["rotation_world_from_camera"])
    origin = np.asarray(camera["position_world_m"])
    ray = rotation @ np.array([(u-cx)/fx, -(v-cy)/fy, -1])
    table_point = origin + (.44-origin[2]) / ray[2] * ray
    distance = float(np.linalg.norm(table_point[:2] - np.asarray(positions["plate_1"])[:2]))
    return {"objects": projections, "plate_pointed_cell": 61,
            "plate_cell_center_xy": table_point[:2].tolist(),
            "plate_cell_center_distance_m": distance, "plate_radius_m": .069}


def test_recorded_projection_detects_mirror_using_settled_objects():
    result = audit_recorded_projection()
    pudding = result["objects"]["chocolate_pudding_1"]
    mug = result["objects"]["porcelain_mug_1"]
    assert (pudding["cell"], mug["cell"]) == (35, 38)
    assert (pudding["mirrored_cell"], mug["mirrored_cell"]) == (38, 35)
    assert result["objects"]["plate_1"]["cell"] == 53
    assert result["plate_cell_center_distance_m"] < result["plate_radius_m"]
    plan = json.loads((T6 / "trial/approved_plan.json").read_text())
    plate = next(part for part in plan["parts"] if part["id"] == "C")
    assert plate["x"] == 61


def test_recorded_review_external_image_matches_actual_stepview():
    initial = json.loads(_require(T6 / "review/initial_observation.json").read_text())
    review = json.loads((T6 / "review/review.json").read_text())
    displayed = next(image for image in review["preparation"]["review_images"]
                     if image["label"] == "external clean")
    view = StepView(PolicyObservation.from_dict(initial))
    np.testing.assert_array_equal(view.external_rgb_upright, np.asarray(decode_image(displayed["data_url"])))


def _bddl_documents():
    # Reuse only the archived read-only parser; it has no simulator imports.
    path = _require(REPORT / "work/semantics/bddl_parse.py")
    spec = importlib.util.spec_from_file_location("uc71_dev_bddl_parser", path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.L = str(_require(LIBERO / "bddl_files"))
    return parser.all_bddls()


def _flat(value):
    return value if isinstance(value, str) else "(" + " ".join(_flat(item) for item in value) + ")"


def _region_center(doc, full_name):
    for name, region in doc["regions"].items():
        if region["target"] + "_" + name == full_name:
            x0, y0, x1, y1 = region["ranges"][0]
            return np.array([(x0+x1)/2, (y0+y1)/2]) + np.asarray((doc["offset"] or (0,0,0))[:2])
    raise KeyError(full_name)


def _init_center(doc, obj):
    row = next(row for row in doc["init"] if row[0].lower() == "on" and row[1] == obj)
    return _region_center(doc, row[2])


def audit_bddl_conventions():
    """Recompute archived 41 decidable cases and all 31 anchored region signs."""
    docs = _bddl_documents()
    by_file = {doc["file"]: doc for doc in docs}
    archived = json.loads((REPORT / "work/semantics/verify_claim.json").read_text())
    direction_docs = {doc["file"] for doc in docs
                      if re.search(r"\b(left|right|front|back|behind|middle)\b", doc["language"].lower())}
    assert direction_docs == {row["file"] for row in archived}
    table = build_view_frame(_recorded_geometry())["relation_directions"]
    directions = {word: np.asarray(table[relation]) for word, relation in {
        "right": "right_of", "left": "left_of", "front": "front_of",
        "back": "behind", "behind": "behind"}.items()}
    asset = _require(LIBERO / "assets/turbosquid_objects/desk_caddy/desk_caddy.xml")
    sites = {site.attrib["name"].removesuffix("_contain_region"):
             np.array([float(value) for value in site.attrib["pos"].split()][:2])
             for site in ET.parse(asset).iter("site") if site.attrib.get("name", "").endswith("_contain_region")}
    checked = []
    for row in archived:
        doc = by_file[row["file"]]
        assert row["language"] == doc["language"]
        assert row["goal"] == _flat(doc["goal"])
        for case in row["checks"]:
            kind, word = case["kind"], case["word"]
            if word == "middle":
                ok = None
            elif kind == "select by direction":
                position = _init_center(doc, case["obj"])
                siblings = [name for name in doc["objects"] if name != case["obj"]
                            and name.rsplit("_",1)[0] == case["obj"].rsplit("_",1)[0]]
                assert siblings
                ok = all(float((position-_init_center(doc,name)) @ directions[word]) > 0 for name in siblings)
            elif kind == "side-of-anchor goal region":
                delta = _region_center(doc, case["region"]) - _region_center(doc, case["anchor_init"])
                ok = bool(delta @ directions[word] > 0)
            elif kind == "caddy compartment":
                yaw = doc["regions"]["desk_caddy_init_region"]["yaw"][0][0]
                rotation = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
                ok = bool((rotation @ sites[case["goal_site"]]) @ directions[word] > 0
                          and word == case["goal_site"])
            else:
                raise AssertionError(kind)
            checked.append({"file": doc["file"], "kind": kind, "word": word, "ok": ok})
    anchored = []
    for doc in docs:
        regions = doc["regions"]
        for name, region in regions.items():
            match = re.fullmatch(r"(\w+?)_(left|right|front|back)_region", name)
            if not match or "ranges" not in region:
                continue
            anchor, word = match.groups()
            anchor_name = anchor + "_init_region"
            if anchor_name in regions:
                anchor_region = regions[anchor_name]
                anchor_center = _region_center(doc, anchor_region["target"]+"_"+anchor_name)
            else:
                obj = next((row[1] for row in doc["init"] if row[0].lower() == "on"
                            and row[1].startswith(anchor)), None)
                if obj is None:
                    continue
                anchor_center = _init_center(doc, obj)
            delta = _region_center(doc, region["target"]+"_"+name) - anchor_center
            if np.linalg.norm(delta) < 1e-9:
                continue
            anchored.append({"file": doc["file"], "region": name, "word": word,
                             "delta": delta.tolist(), "ok": bool(delta @ directions[word] > 0)})
    return {"direction_bddl_files": len(direction_docs), "checks": checked,
            "decidable_passed": sum(case["ok"] is True for case in checked),
            "decidable_failed": sum(case["ok"] is False for case in checked),
            "unknown_middle": sum(case["ok"] is None for case in checked),
            "anchored_regions": anchored,
            "coverage_note": "Archived direction conventions only, not proof every language sentence matches its goal. Known LIBERO annotation inconsistencies remain."}


def test_bddl_conventions_from_actual_assets_agree_with_calibrated_directions():
    result = audit_bddl_conventions()
    assert result["direction_bddl_files"] == 42
    assert (result["decidable_passed"], result["decidable_failed"], result["unknown_middle"]) == (41, 0, 4)
    counts = Counter(case["kind"] for case in result["checks"] if case["ok"] is True)
    assert counts == {"select by direction": 21, "caddy compartment": 11, "side-of-anchor goal region": 9}
    assert len(result["anchored_regions"]) == 31
    assert all(case["ok"] for case in result["anchored_regions"])
    assert Counter(case["word"] for case in result["anchored_regions"]) == {"right": 23, "left": 5, "front": 3}


def audit_recorded_plan_validation():
    paths = sorted((ROOT / "runs").rglob("approved_plan.json"))
    if not paths:
        pytest.skip("No archived approved plans installed")
    return [{"path": str(path.relative_to(ROOT)),
             "issues": validate_spatial_spec(json.loads(path.read_text()))} for path in paths]


def test_all_recorded_approved_plans_still_pass_spatial_schema_validation():
    rows = audit_recorded_plan_validation()
    failures = [row for row in rows if row["issues"]]
    assert not failures, failures
