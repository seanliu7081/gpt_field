"""agent_SpeedUp_v2 diagnosis-only truth: pure helpers, AC-2 verdict logic, recorded verdict regressions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_truth as truth

ROOT = Path(__file__).resolve().parents[1]


def evaluation(strict, raw, failing=None):
    atoms = [{"atom": ["in", "obj_1", "region_1"], "strict": strict, "why_not": list(failing or []), "raw": raw}]
    return {"strict_conjunction": strict, "check_success": raw, "atoms": atoms}


def multi(raw, **failing):
    """An evaluation with one atom per keyword (atom name -> failing checks; [] = strict)."""
    atoms = [{"atom": name.split("__"), "strict": not checks, "why_not": list(checks), "raw": raw}
             for name, checks in failing.items()]
    return {"strict_conjunction": all(not c for c in failing.values()), "check_success": raw, "atoms": atoms}


def test_hull_helpers():
    square = truth.hull2d([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert len(square) == 4
    assert truth.hull_margin(square, (0.5, 0.5)) == pytest.approx(0.5)
    assert truth.hull_margin(square, (2.0, 0.5)) == pytest.approx(-1.0)
    assert truth.hull_margin([(0, 0), (1, 0)], (0.5, 1.0)) == pytest.approx(-1.0)
    assert truth.hull_margin([], (0, 0)) is None


def test_jsonable_and_rounding():
    value = truth.jsonable({"a": np.float32(1.5), "b": np.array([1, 2]), "c": (np.bool_(True),), "d": float("nan"),
                            "e": {3, 1}})
    assert value == {"a": 1.5, "b": [1, 2], "c": [True], "d": None, "e": [1, 3]}
    json.dumps(value, allow_nan=False)
    assert truth.r(np.array([0.123456, 2])) == [0.1235, 2.0] and truth.r(None) is None


def test_ac2_verdicts():
    held = evaluation(False, True, ["not_held", "at_rest"])
    resting = evaluation(True, True)
    perched = evaluation(False, True, ["not_held", "at_rest", "lowest_point_below_rim", "not_perched_on_rim"])
    s = truth.physical_summary(held, resting, raw_libero_success=True)
    assert s["verdict"] == "physical_success" and s["terminal_failure_is_hold_only"] and s["v2_success"]
    assert truth.physical_summary(perched, resting, raw_libero_success=True)["verdict"] == "lenient_success"
    assert truth.physical_summary(resting, None, raw_libero_success=True)["verdict"] == "physical_success"
    assert truth.physical_summary(held, held, raw_libero_success=True)["verdict"] == "lenient_success"
    assert truth.physical_summary(evaluation(False, False, ["supported_by_target"]), None,
                                  raw_libero_success=False)["verdict"] == "raw_failure"
    assert truth.physical_summary(resting, None, raw_libero_success=False)["verdict"] == "raw_failure_strict_holds"
    assert truth.physical_summary(None, None, raw_libero_success=True)["verdict"] == "truth_unavailable"
    rest_only = evaluation(False, True, ["at_rest"])     # moving but not held: not deferred to the settle
    assert not truth.physical_summary(rest_only, resting, raw_libero_success=True)["terminal_failure_is_hold_only"]
    tilted_in_hand = evaluation(False, True, ["not_held", "at_rest", "orientation_kept"])   # the hand's pose
    assert truth.physical_summary(tilted_in_hand, resting, raw_libero_success=True)["verdict"] == "physical_success"
    knocked_over = evaluation(False, True, ["orientation_kept"])      # unheld and reoriented: never deferred
    assert truth.physical_summary(knocked_over, resting, raw_libero_success=True)["verdict"] == "lenient_success"
    for summary in (s,):
        assert summary["diagnosis_only"] is True and summary["affects_completion"] is False


def test_held_only_is_per_atom():
    """TRUTH-2: an unheld moving atom is never deferred because some other atom is held."""
    assert not truth.held_only({"on a": ["at_rest"]})
    assert not truth.held_only({"in a": ["at_rest"], "in b": ["not_held"]})
    assert not truth.held_only({"on a": ["at_rest"], "turnon s": ["robot_not_touching_part"]})
    assert not truth.held_only({"turnon s": ["joint_at_rest"]})
    assert truth.held_only({"in a": ["not_held", "at_rest"], "turnon s": ["robot_not_touching_part",
                                                                        "joint_at_rest"]})
    assert not truth.held_only({}) and not truth.held_only(None)
    two = multi(True, on__a__p=["at_rest"], in__b__r=["not_held"])
    resting = multi(True, on__a__p=[], in__b__r=[])
    assert truth.physical_summary(two, resting, raw_libero_success=True)["verdict"] == "lenient_success"


def test_settle_needs_raw_success_too():
    """TRUTH-3: a deferred verdict needs LIBERO's raw success after the settle as well (strict is not a superset)."""
    held = evaluation(False, True, ["not_held", "at_rest"])
    strict_not_raw = evaluation(True, False)
    summary = truth.physical_summary(held, strict_not_raw, raw_libero_success=True)
    assert summary["verdict"] == "lenient_success" and summary["raw_libero_success_after_settle"] is False
    assert summary["terminal_failure_is_hold_only"] is True and summary["physical_success"] is False
    ok = truth.physical_summary(held, evaluation(True, True), raw_libero_success=True)
    assert ok["verdict"] == "physical_success" and ok["raw_libero_success_after_settle"] is True


def test_support_contacts_need_a_vertical_normal_and_orientation_rule():
    """TRUTH-7 support rule; TRUTH-1 orientation rule (fails closed without a reset reference)."""
    o = {"com": np.array([0.0, 0.0, 1.0]), "tilt_change": 10.0}
    below = {"pos": np.array([0.0, 0.0, 0.9]), "nz": 1.0}
    side = {"pos": np.array([0.0, 0.0, 0.9]), "nz": float(np.cos(np.radians(60)))}
    above = {"pos": np.array([0.0, 0.0, 0.999]), "nz": 1.0}
    assert truth._below_com(o, below) and not truth._below_com(o, side) and not truth._below_com(o, above)
    assert truth._orientation_kept(o)
    assert not truth._orientation_kept({**o, "tilt_change": truth.TOL["orientation_change_deg"] + 1})
    assert not truth._orientation_kept({**o, "tilt_change": None})
    checks = truth._with_orientation({"not_held": True}, o, "on_object")
    assert checks == {"not_held": True, "orientation_kept": True}
    assert "orientation_kept" not in truth._with_orientation({}, o, "articulation")
    assert "orientation_kept" not in truth._with_orientation({}, o, "inside_container")     # D1
    for kind in ("on_object", "on_fixture_region", "on_table_zone"):
        assert "orientation_kept" in truth._with_orientation({}, o, kind)
    assert "orientation_kept" in truth.HELD_CHECKS and "orientation_kept" not in truth.HOLD_CHECKS


RIM = 1.1199      # drawer_bowl top drawer: rim (lowest wall top) and floor of the M0 truth probes
FLOOR = 1.0641


def contact(z, nz=1.0, fn=1.0):
    return {"pos": np.array([0.0, 0.0, z]), "nz": nz, "fn": fn}


def placed(tilt_change, *, com_z=1.10, low=FLOOR, robot=(), rest=True):
    return {"com": np.array([0.0, 0.0, com_z]), "tilt_change": tilt_change, "low": low, "robot": list(robot),
            "rest": rest}


def inside(o, cavity, contents=(), outside=(), rim_kind="lowest_wall_top", footprint=0.0, com=0.01):
    return truth.inside_checks(o, cavity_contacts=cavity, contents_contacts=contents, outside_contacts=outside,
                               rim_z=RIM, rim_kind=rim_kind, footprint_margin=footprint, com_margin=com)


def test_d1_inside_ignores_orientation_but_not_the_perch():
    """D1 (2026-10-18): strict inside needs floor (or contents) support below the rim, not the reset orientation."""
    # reviewers' front-wall lean: tilt 56.6 deg, floor contact (z 1.0641) plus the front wall top (z 1.1207)
    lean = placed(56.6, com_z=1.115)
    checks, supports, deep = inside(lean, [contact(FLOOR), contact(1.1207, nz=0.6)])
    assert all(checks.values()) and "orientation_kept" not in checks and len(supports) == len(deep) == 1
    # run-003 corner perch: every contact at rim height, no floor contact (low 0.3 mm above the rim)
    perch = placed(0.7, com_z=1.150, low=RIM + 0.0003)
    checks, supports, deep = inside(perch, [contact(1.1202), contact(1.1207)], footprint=None)
    assert not checks["not_perched_on_rim"] and not checks["lowest_point_below_rim"] and supports and not deep
    # a lean on the rim that rests on the contents (an object below it in the cavity) is not a perch either
    on_contents = placed(40.0, com_z=1.12)
    checks, _, deep = inside(on_contents, [contact(1.1207, nz=0.5)], contents=[contact(1.09)])
    assert checks["not_perched_on_rim"] and checks["supported_by_container_or_contents"] and deep
    # a side contact below the rim is not support (normal 60 deg from vertical): still perched without a floor
    side_only = placed(10.0, com_z=1.13)
    checks, supports, _ = inside(side_only, [contact(1.1207), contact(1.10, nz=0.5)])
    assert not checks["not_perched_on_rim"]
    # a closed cavity (ceiling rim) is never perched; a held object fails on the hold
    assert inside(perch, [contact(1.1202)], rim_kind="ceiling", footprint=None)[0]["not_perched_on_rim"]
    held = inside(placed(5.0, robot=[{"other": "finger"}], rest=False), [contact(FLOOR)])[0]
    assert set(k for k, v in held.items() if not v) == {"not_held", "at_rest"}
    assert truth.rim_height_force_share([contact(FLOOR, fn=3.0), contact(1.1207, fn=1.0)], RIM) == 0.25
    assert truth.rim_height_force_share([], RIM) is None


def test_d1_on_requires_orientation():
    """D1: strict on (object / fixture region / table zone) needs the reset orientation: a tipped pot fails."""
    def on_checks(tilt):
        return truth._with_orientation({"not_held": True, "at_rest": True, "supported_by_target": True},
                                       placed(tilt), "on_fixture_region")
    assert all(on_checks(0.0).values())                                     # upright pot on the burner
    for tilt in (63.6, 90.0, 92.4, 179.9):                                  # tipped / on its side / upside down
        checks = on_checks(tilt)
        assert checks["orientation_kept"] is False and not all(checks.values())
    knocked_over = evaluation(False, True, ["orientation_kept"])
    assert truth.physical_summary(knocked_over, evaluation(True, True),
                                  raw_libero_success=True)["verdict"] == "lenient_success"


def test_d1_recorded_probe_cases():
    """scripts/agent_speedup_v2_truth_probes.py: the reviewers' probe cases in the real worker (teleports)."""
    path = ROOT / "runs/agent_speedup_v2_m0close_truth_probes/truth_probes.json"
    if not path.is_file():
        pytest.skip("M0 truth-probe record not present")
    data = json.loads(path.read_text())
    current = hashlib.sha256((ROOT / "src/gpt_field/agent_speedup_v2_truth.py").read_bytes()).hexdigest()
    assert data["truth_module_sha256"] == current, "stale probe record: rerun scripts/agent_speedup_v2_truth_probes.py"
    cases = {c["case"]: c for c in data["cases"]}
    expected = {"drawer:bowl_front_wall_lean": "physical_success", "drawer:bowl_corner_perch_run003": "lenient_success",
                "drawer:bowl_floor_centre": "physical_success", "t2:pot_upright_centred": "physical_success",
                "bowl:bowl_upright_on_plate": "physical_success"}
    for case, verdict in expected.items():
        assert cases[case]["summary"]["verdict"] == verdict, case
    for case in ("t2:pot_side_y90_centred", "t2:pot_upside_down_centred", "t2:pot_knob_side_dx-0.09",
                 "bowl:bowl_upside_down_on_plate"):
        summary = cases[case]["summary"]
        assert summary["physical_success"] is False, case
        assert {c for checks in summary["terminal_failing_checks"].values() for c in checks} == {"orientation_kept"}
    lean = cases["drawer:bowl_front_wall_lean"]["terminal"][0]["numbers"]
    assert lean["orientation_required"] is False and lean["tilt_change_from_reset_deg"] > 30
    assert lean["support_contacts_below_rim"] >= 1 and lean["rim_height_normal_force_share"] is not None
    perch = cases["drawer:bowl_corner_perch_run003"]["terminal"][0]["numbers"]
    assert perch["support_contacts_below_rim"] == 0 and perch["rim_height_normal_force_share"] == 1.0
    assert data["all_match"] is True


def test_release_settle_protocol():
    actions = truth.release_settle_actions()
    assert len(actions) == 10 + 8 + 20
    assert all(a[6] == -1.0 and len(a) == 7 for a in actions)
    assert [a[2] for a in actions[10:18]] == [0.5] * 8 and all(a[2] == 0 for a in actions[18:])


def test_parse_check_without_plan():
    key = {"goal_atoms": [["turnon", "flat_stove_1"], ["on", "moka_pot_1", "flat_stove_1_cook_region"]]}
    result = truth.parse_check(key)
    assert [a["expected_relation"] for a in result["atoms"]] == ["turn", "place_on_or_push_to"]
    assert result["plan_available"] is False and result["diagnosis_only"] is True


@pytest.mark.parametrize("name, verdict, failing", [
    ("drawer003", "lenient_success", {"lowest_point_below_rim", "not_perched_on_rim"}),
    ("bowl002", "physical_success", set()),
])
def test_recorded_truth_check_regressions(name, verdict, failing):
    """scripts/agent_speedup_v2_truth_check.py on the v1 runs (drawer 003 perch, bowl 002 held -> resting)."""
    path = ROOT / f"runs/agent_speedup_v2_m0close_truth_{name}/truth.json"
    if not path.is_file():
        pytest.skip("M0 truth-check record not present")
    data = json.loads(path.read_text())
    current = hashlib.sha256((ROOT / "src/gpt_field/agent_speedup_v2_truth.py").read_bytes()).hexdigest()
    assert data["truth_module_sha256"] == current, "stale truth record: rerun scripts/agent_speedup_v2_truth_check.py"
    assert data["replay"]["bitwise"] is True
    assert data["verdict"]["ac2"] == verdict and data["verdict"]["raw_libero_success_end"] is True
    end_failing = {c for checks in (data["verdict"]["failing_strict_checks_end"] or {}).values() for c in checks}
    assert failing <= end_failing
    if verdict == "physical_success":
        assert end_failing <= set(truth.HELD_CHECKS)
