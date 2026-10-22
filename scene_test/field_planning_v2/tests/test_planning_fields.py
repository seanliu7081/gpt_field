import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning_fields import (AstraEnvironmentField, DepthProVisibilityField, _Adapter,
    environment_csg, load_module, depth_triangle_validity, projective_depth_samples)

PROJECT = Path(__file__).resolve().parents[3]


def primitive(name, op, size, center=(0, 0, 0)):
    return {'name': name, 'op': op, 'center_m': list(center), 'rotation_xyzw': [0, 0, 0, 1],
            'size_m': list(size), 'left': None, 'right': None}


def compound(name, op, a, b):
    return {'name': name, 'op': op, 'center_m': None, 'rotation_xyzw': None,
            'size_m': None, 'left': a, 'right': b}


def original_spec():
    return {'schema_version': 1, 'units': 'm', 'coordinate_frame': 'camera_cv_xright_ydown_zforward',
            'estimated_focal_px': 500., 'uncertainty_notes': 'numeric test only', 'root': 4,
            'nodes': [primitive('environment_outer', 'box', [1., 1., 1.]),
                      primitive('environment_cavity', 'box', [.4, .4, .4]),
                      compound('environment_shell', 'difference', 0, 1),
                      primitive('robot_test', 'sphere', [.7, 0, 0]),
                      compound('scene', 'union', 2, 3)]}


def core():
    return load_module(PROJECT / 'scene_test/astra_scene/runs/fields_001/source/field_csg.py', '_v2_test_core')


def test_environment_removal_preserves_overlapping_environment_and_cavity():
    original = original_spec()
    before = copy.deepcopy(original)
    env, log = environment_csg(original)
    assert original == before
    assert [r['name'] for r in log['removed_components']] == ['robot_test']
    assert env['nodes'] == before['nodes'][:3]
    values = core().evaluate_field(env, [[0, 0, 0], [.6, 0, 0], [.8, 0, 0], [1.2, 0, 0]])
    assert values[0] > 0  # original difference cavity survives
    assert values[1] < 0  # environment overlapping deleted robot is retained
    assert values[2] < 0 and values[3] > 0


def test_robot_named_child_does_not_delete_nonrobot_compound():
    original = original_spec()
    original['nodes'][1]['name'] = 'robot_named_but_cavity_operand'
    env, log = environment_csg(original)
    assert [r['name'] for r in log['removed_components']] == ['robot_test']
    assert env['nodes'][1]['name'] == 'robot_named_but_cavity_operand'
    assert env['nodes'][2]['op'] == 'difference'


def test_empty_environment_after_semantic_selection_is_refused():
    original = original_spec()
    original['nodes'][2]['name'] = 'robot_shell'
    with pytest.raises(ValueError, match='no non-robot'):
        environment_csg(original)


def test_astra_analytic_query_retains_original_domain_and_has_no_self_prior(tmp_path):
    adapter = AstraEnvironmentField.__new__(AstraEnvironmentField)
    _Adapter.__init__(adapter, 'drawer003_closed_step60', project_root=PROJECT)
    adapter.environment, adapter.selection = environment_csg(original_spec())
    adapter.core = core()
    adapter.lower, adapter.upper = np.array([-1.5]*3), np.array([1.5]*3)
    result = adapter.query([[0, 0, 0], [.6, 0, 0], [1.6, 0, 0], [np.nan, 0, 0]])
    assert result.state.tolist() == [1, -1, 0, 0]
    assert not result.known_initial_self.any()
    assert result.source_code.tolist() == [1, 1, 0, 0]
    assert result.conservative_clearance_m[0] > .39
    assert adapter.last_diagnostics['outside_domain_or_nonfinite'] == 2


class FixedSelfGeometry:
    def __init__(self, front=.5, exit=.7):
        self.front, self.exit = front, exit
        self.ray_directions = []

    def contains(self, points):
        return (np.abs(points[:, 0]) <= .1) & (np.abs(points[:, 1]) <= .1) & (points[:, 2] >= .5) & (points[:, 2] <= .7)

    def ray_intervals(self, directions):
        self.ray_directions.append(directions.copy())
        return {'front_t': np.full(len(directions), self.front),
                'exit_t': np.full(len(directions), self.exit), 'geom_id': np.zeros(len(directions), int)}


def depth_adapter(depth=1., front=.5):
    adapter = DepthProVisibilityField.__new__(DepthProVisibilityField)
    _Adapter.__init__(adapter, 'drawer003_closed_step60', project_root=PROJECT)
    adapter.depth = np.full((5, 5), depth)
    adapter.K_pred = np.array([[4., 0., 2.], [0., 4., 2.], [0., 0., 1.]])
    adapter.triangle_valid = depth_triangle_validity(adapter.depth)
    adapter.self_association_image = np.zeros((5, 5), bool)
    adapter.self_geometry = FixedSelfGeometry(front=front)
    adapter.settings = {'override_conflict_reference': 'raw visibility state'}
    return adapter


def test_depthpro_free_prefix_and_unknown_behind_robot():
    adapter = depth_adapter()
    result = adapter.query([[0, 0, .3], [0, 0, .48], [0, 0, .49], [0, 0, .9]])
    assert result.state.tolist() == [1, 0, 0, 0]
    assert result.source_code.tolist() == [2, 0, 0, 0]
    assert adapter.last_diagnostics['ray_inferred_free'] == 1
    assert adapter.last_diagnostics['unknown_robot_occluded'] == 3
    assert not result.conservative_clearance_m.any()  # residual is never ESDF


def test_depthpro_initial_material_prior_does_not_expand_with_robot_or_unknown():
    adapter = depth_adapter()
    result = adapter.query([[0, 0, .5], [0, 0, .7], [0, 0, .700001], [.100001, 0, .6], [9, 0, .6]])
    assert result.known_initial_self.tolist() == [True, True, False, False, False]
    assert result.state.tolist() == [1, 1, 0, 0, 0]
    assert result.source_code.tolist() == [3, 3, 0, 0, 0]
    assert adapter.last_diagnostics['self_prior_free'] == 2


@pytest.mark.parametrize('depth,raw_state,expected_key', [
    (.6, -1, 'initial_self_prior_raw_occupied_conflicts'),
    (.4, 0, 'initial_self_prior_overrides_raw_unknown'),
    (1., 1, 'initial_self_prior_already_raw_free'),
])
def test_initial_self_override_records_conflict_origin(depth, raw_state, expected_key):
    adapter = depth_adapter(depth)
    result = adapter.query([[0, 0, .6]])
    assert result.state.tolist() == [1] and result.raw_predicted_state.tolist() == [raw_state]
    assert adapter.last_diagnostics[expected_key] == 1
    assert adapter.last_diagnostics['self_prior_free'] == 1
    assert adapter.last_diagnostics['ray_inferred_free'] == 0


def test_depthpro_nonself_foreground_surface_remains_occupied():
    adapter = depth_adapter(.3)
    result = adapter.query([[0, 0, .3]])
    assert result.state.tolist() == [-1]
    assert result.source_code.tolist() == [2]


def test_self_associated_surface_becomes_unknown_not_free():
    adapter = depth_adapter(.3, front=np.inf)
    adapter.self_association_image[2, 2] = True
    result = adapter.query([[0, 0, .3], [0, 0, .4], [0, 0, .2]])
    assert result.state.tolist() == [0, 0, 1]
    assert result.self_image_associated.tolist() == [True, True, True]
    assert adapter.last_diagnostics['unknown_self_associated_surface_band'] == 1
    assert adapter.last_diagnostics['unknown_behind_predicted_surface'] == 1


def test_unknown_novel_points_with_invalid_depth_or_out_of_view_never_free():
    adapter = depth_adapter()
    adapter.depth[:] = np.nan
    adapter.triangle_valid = depth_triangle_validity(adapter.depth)
    result = adapter.query([[0, 0, .3], [9, 0, .3], [np.nan, 0, 0], [0, 0, -.3]])
    assert result.state.tolist() == [0, 0, 0, 0]
    assert adapter.last_diagnostics['unknown_out_of_view_or_invalid_depth'] == 4


def test_physical_robot_ray_uses_query_direction_not_rig_pixel_direction():
    adapter = depth_adapter(front=np.inf)
    points = np.array([[.05, .02, .3]])
    adapter.query(points)
    assert np.allclose(adapter.self_geometry.ray_directions[0], points/points[:, 2:3])


def test_projective_interpolation_matches_original_saved_field_math():
    legacy = load_module(PROJECT/'scene_test/rgb_scene/field_rgb.py', '_v2_test_original_projective')
    y, x = np.indices((6, 7))
    depth = 1/(1+.01*x+.02*y)
    K = np.array([[5., 0., 3.], [0., 5., 2.5], [0., 0., 1.]])
    points = np.random.default_rng(8).uniform([-.3, -.3, .6], [.3, .3, 1.4], (500, 3))
    original, old_valid = legacy.sample_projective_depth(depth, K, points)
    actual, valid, _ = projective_depth_samples(depth, K, points, depth_triangle_validity(depth))
    assert np.array_equal(valid, old_valid)
    assert np.allclose(actual, original, equal_nan=True, atol=1e-14, rtol=0)
