"""Plot checks focus on faithful path/geometry presentation, not pixel snapshots."""
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("plot_planning", Path(__file__).resolve().parents[1]/"plot_planning.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def calibration():
    return {"K": [[10.,0.,5.],[0.,10.,5.],[0.,0.,1.]],
            "T_camera_cv_from_world": np.eye(4).tolist(), "width":11,"height":11}


def test_projection_uses_upright_cv_and_excludes_behind_camera():
    uv, depth, visible = plot.project_world([[0,0,1],[.2,.3,1],[0,0,-1],[2,0,1]], calibration())
    np.testing.assert_allclose(uv[:2], [[5,5],[7,8]])
    assert np.isnan(uv[2]).all()
    np.testing.assert_array_equal(visible, [True,True,False,False])
    np.testing.assert_allclose(depth, [1,1,-1,1])


def test_refused_query_never_draws_endpoint_connection():
    fig, ax = plt.subplots()
    result = plot.draw_recorded_path(ax, [], calibration(), accepted=False, color="red")
    assert result["drawn_segments"] == 0
    assert not ax.collections
    with pytest.raises(ValueError, match="Refused query"):
        plot.draw_recorded_path(ax, [[0,0,1],[.2,0,1]], calibration(), accepted=False,color="red")
    plt.close(fig)


def test_path_breaks_at_behind_camera_point_instead_of_bridging():
    fig, ax = plt.subplots()
    result = plot.draw_recorded_path(ax, [[0,0,1],[.1,0,1],[0,0,-1],[.3,0,1]], calibration(),accepted=True,color="red")
    assert result["drawn_segments"] == 1
    np.testing.assert_allclose(ax.collections[0].get_segments()[0], [[5,5],[6,5]])
    plt.close(fig)


def test_geometry_display_sampling_preserves_rare_collision_and_unknown_states():
    states = np.array([0]*10000+[1]+[2]*5)
    indices = plot.sample_indices(states, limit=300)
    assert 10000 in indices
    assert np.sum(states[indices] == 2) == 5
    assert len(indices) <= 300
    np.testing.assert_array_equal(indices, plot.sample_indices(states, limit=300))


def test_diagnostics_count_offscreen_points_in_full_statistics():
    fig, ax = plt.subplots()
    result = plot.draw_robot_points(ax, [[0,0,1],[2,0,1],[0,0,-1]],[0,1,2],calibration())
    assert result["all_sample_counts"] == {"predicted free":1,"predicted occupied":1,"unknown / ambiguous":1}
    assert result["in_image_display_samples"] == 1
    assert "x-ray" in result["meaning"]
    plt.close(fig)


def test_changed_plot_input_rejected(tmp_path):
    path = tmp_path/"saved.json"; path.write_text("original")
    expected = plot.sha(path); path.write_text("different")
    with pytest.raises(ValueError, match="SHA mismatch"):
        plot.checked(path, expected, {})


def test_case_selection_is_paired_and_prefers_first_oracle_valid_case():
    records=[]
    for case in ("b", "a"):
        for method in plot.METHODS:
            for seed in (29,17):
                accepted=method=="oracle" and case=="b"
                records.append({"snapshot":plot.SNAPSHOTS[0],"case_id":case,"seed":seed,"method":method,
                                "eef_path_world_m":[[0,0,1],[.1,0,1]] if accepted else [],
                                "truth":{"valid_at_all_samples":True} if accepted else None})
    selected=plot.choose_records(records,plot.SNAPSHOTS[0])
    assert all(r["case_id"]=="b" and r["seed"]==17 for r in selected.values())
    assert not plot.returned(selected["astra"])
    assert plot.returned(selected["oracle"])


def test_summary_counts_all_queries_and_only_validated_oracle_paths():
    records=[]
    for snapshot in plot.SNAPSHOTS:
        for method in plot.METHODS:
            for case in ("valid","colliding"):
                accepted=method!="astra"
                records.append({"snapshot":snapshot,"method":method,"case_id":case,"seed":17,
                                "eef_path_world_m":[[0,0,1],[.1,0,1]] if accepted else [],
                                "truth":{"valid_at_all_samples":case=="valid"} if accepted else None,"planning_seconds":.5})
    rows=plot.summary_counts(records)
    astra=rows[0];depthpro=rows[1]
    assert astra["queries"]==2 and astra["refused"]==2
    assert astra["refused_with_oracle_valid_path"]==1
    assert depthpro["returned_truth_collision_free"]==1
    assert depthpro["returned_truth_colliding"]==1
