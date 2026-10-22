"""Ensure before/after plots preserve paired queries and SELF-prior semantics."""
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("v2_plot_results",Path(__file__).resolve().parents[1]/"plot_results.py")
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


def record(case="a",method="astra",valid=None):
    accepted=valid is not None
    return {"snapshot":plot.SNAPSHOTS[0],"method":method,"case_id":case,"seed":17,
            "start_q":[0]*7,"goal_q":[.1]*7,
            "eef_path_world_m":[[0,0,1],[.1,0,1]] if accepted else [],
            "truth":{"valid_at_all_samples":valid} if accepted else None}


def test_pairing_rejects_changed_goal_and_missing_queries():
    a=record();b=record();assert plot.assert_same_queries([a],[b])["paired_queries"]==1
    b["goal_q"][0]=.2
    with pytest.raises(ValueError,match="endpoint mismatch"):plot.assert_same_queries([a],[b])
    with pytest.raises(ValueError,match="same snapshot"):plot.assert_same_queries([a],[])


def test_duplicate_seed_record_rejected():
    with pytest.raises(ValueError,match="Duplicate"):plot.indexed_records([record(),record()])


def test_prior_is_separate_from_inferred_free_and_missing_provenance_rejected():
    d={"method":"depthpro","field_points":{"points":100,"free":90,"occupied":2,"unknown":8},
       "field_support_layers":{"self_prior_free":70,"ray_inferred_free":20}}
    result=plot.initial_states(d,revised=True)
    assert result["field_free"]==20 and result["self_prior"]==70
    assert plot.initial_states(d,revised=False)["self_prior"]==0
    del d["field_support_layers"]
    with pytest.raises(ValueError,match="SELF-prior"):plot.initial_states(d,revised=True)


def test_prior_cannot_overcount_final_free_classifications():
    d={"method":"depthpro","field_points":{"points":100,"free":90,"occupied":2,"unknown":8},
       "field_support_layers":{"self_prior_free":91}}
    with pytest.raises(ValueError,match="subset"):plot.initial_states(d,revised=True)


def test_failure_example_precedes_success_and_remains_paired():
    records=[]
    for case in ("a","b"):
        for method in (*plot.METHODS,"oracle"):
            records.append(record(case,method,False if case=="b" and method=="astra" else True))
    selected,reason=plot.selected_pair(records,plot.SNAPSHOTS[0])
    assert all(r["case_id"]=="b" for r in selected.values())
    assert "invalid" in reason


def test_refusals_not_counted_as_collision_free_paths():
    rows=[record("a",valid=None),record("b",valid=True),record("c",valid=False)]
    counts=plot.outcomes(rows,plot.SNAPSHOTS[0],"astra")
    assert counts=={"queries":3,"returned":2,"valid":1,"invalid":1,"unscored":0,"refused":1}
