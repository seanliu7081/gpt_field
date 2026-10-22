"""Analytic, scale and recorded regression checks for continuous mesh distance."""
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "_deps"))
from geometry import make_scenes
from mesh_queries import MeshDistance, exhaustive_mesh_distance, point_triangle_squared_distances
from sdf_grid import SDFGrid


@pytest.mark.parametrize("scale", [1e-9, 1e-6, 1e-3, 1., 1e3, 1e6])
def test_triangle_face_edge_vertex_at_all_scales(scale):
    tri = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])*scale
    p = np.array([[.2, .3, .7], [.8, .8, 0], [-.3, -.4, 0], [.2, 0, -.3], [.2, .3, 0]])*scale
    expected = np.array([.7, .6/np.sqrt(2), .5, .3, 0])*scale
    d = np.sqrt(point_triangle_squared_distances(np.repeat(tri[None], len(p), axis=0), p))
    np.testing.assert_allclose(d, expected, atol=2e-15*scale, rtol=2e-14)


def test_degenerate_and_nearly_degenerate_faces():
    tri = np.array([[[0., 0, 0], [1, 0, 0], [2, 0, 0]],
                    [[1., 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[0., 0, 0], [1, 0, 0], [.5, 1e-16, 0]]])
    p = np.array([[.5, 1, 0], [2, 4, 5], [.5, .2, 0]])
    np.testing.assert_allclose(np.sqrt(point_triangle_squared_distances(tri, p)), [1, 3, .2], atol=1e-14)


def test_mesh_candidates_match_exhaustive_and_analytic_box():
    box = next(s for s in make_scenes() if s.name == "rotated_box_seed37")
    rng = np.random.default_rng(420)
    points = rng.uniform(-.08, .08, (500, 3))
    query = MeshDistance(box.mesh())
    np.testing.assert_allclose(query(points), np.abs(box.sdf(points)), atol=1e-14)
    np.testing.assert_allclose(query(points[:20]), exhaustive_mesh_distance(query.triangles, points[:20]), atol=1e-14)
    assert query.validate(points)["passed"]


def test_failed_fine_box_regression():
    scene = next(s for s in make_scenes() if s.name == "rotated_box_seed23")
    mesh = SDFGrid.build(scene, .001).surface()
    # Preserve the actual failing run's sample coordinates as a compact fixture.
    points = np.array([
        [0.027710199959402097, 0.03974134097401347, 0.026439632590665847],
        [-0.008447885089841838, 0.01869534135039841, -0.028027012359707983],
        [0.015067673426955069, 0.038757118570286714, -0.02079597245899624],
        [-0.0358045250796612, 0.01963727600738533, 0.0038576380116922375],
        [-0.038636074553428545, -0.014497285993341807, 0.003891224580322072],
        [-0.009549907618463936, -0.03558493480370132, 0.010208116657275317],
        [-0.04017895188102201, -0.010733266736425003, 0.0050522792920182046],
        [0.010474907711704224, -0.028058371283825826, -0.012859051182163149],
        [0.0191990501315897, 0.03897008440873646, 0.014450650086601518],
        [0.019737568779066145, -0.0038703053525989675, 0.020903902850487682],
        [0.0447826243548227, -0.0004202720170495074, 0.018452228080438296],
        [0.027853581204345102, 0.04466324904485891, 0.0037585382209692677],
    ])
    query = MeshDistance(mesh)
    report = query.validate(points)
    assert report["independent_projector_max_difference_m"] < 2e-12
    assert report["broad_phase_max_difference_m"] < 2e-12
    assert np.max(query(points)) < .001
    assert query(points)[7] < 1e-7


def test_empty_and_invalid_queries():
    query = MeshDistance(trimesh.creation.box())
    assert query(np.empty((0, 3))).shape == (0,)
    with pytest.raises(ValueError):
        query([[np.nan, 0, 0]])
    with pytest.raises(ValueError):
        point_triangle_squared_distances(np.zeros((1, 3, 3)), [[np.inf, 0, 0]])
