"""Geometry checks independent of any model response or simulator reference."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "field_csg.py"
module_spec = importlib.util.spec_from_file_location("astra_field_csg", path)
field = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(field)


def primitive(op, size, center=(0, 0, 2), q=(0, 0, 0, 1)):
    return dict(name=op, op=op, center_m=list(center), rotation_xyzw=list(q),
                size_m=list(size), left=None, right=None)


def operation(op, left, right):
    return dict(name=op, op=op, center_m=None, rotation_xyzw=None,
                size_m=None, left=left, right=right)


def specification(nodes):
    return dict(schema_version=1, units="m", coordinate_frame=field.FRAME,
                estimated_focal_px=600., nodes=nodes, root=len(nodes)-1,
                uncertainty_notes="Synthetic test only.")


def hits(spec, origins, directions):
    return field.analytic_first_hits(spec, np.asarray(origins, float), np.asarray(directions, float))


def test_primitive_signed_distances():
    sphere = specification([primitive("sphere", [1, 0, 0])])
    np.testing.assert_allclose(field.evaluate_field(sphere, [[0, 0, 2], [0, 0, 3], [0, 0, 4]]), [-1, 0, 1])
    box = specification([primitive("box", [1, 2, 3], center=(0, 0, 0))])
    np.testing.assert_allclose(field.evaluate_field(box, [[0, 0, 0], [2, 3, 0], [1, 0, 0]]), [-1, np.sqrt(2), 0])
    cylinder = specification([primitive("cylinder", [1, 2, 0], center=(0, 0, 0))])
    np.testing.assert_allclose(field.evaluate_field(cylinder, [[0, 0, 0], [2, 0, 3], [1, 0, 1]]), [-1, np.sqrt(2), 0])


def test_rotated_box_and_cylinder_distances_bounds():
    q = [0, np.sin(np.pi/4), 0, np.cos(np.pi/4)]
    box = specification([primitive("box", [.1, .2, .4], center=(0, 0, 0), q=q)])
    lower, upper = field.field_bounds(box)
    np.testing.assert_allclose(lower, [-.4, -.2, -.1], atol=1e-15)
    np.testing.assert_allclose(upper, [.4, .2, .1], atol=1e-15)
    np.testing.assert_allclose(field.evaluate_field(box, [[.3, 0, 0], [0, 0, .3]]), [-.1, .2], atol=1e-15)
    cylinder = specification([primitive("cylinder", [.1, .4, 0], center=(0, 0, 0), q=q)])
    np.testing.assert_allclose(field.field_bounds(cylinder)[1], [.4, .1, .1], atol=1e-15)
    np.testing.assert_allclose(field.evaluate_field(cylinder, [[.3, 0, 0], [0, 0, .3]]), [-.1, .2], atol=1e-15)


def test_quaternion_normalization_is_logged_and_repeatable():
    raw = specification([primitive("sphere", [1, 0, 0], q=(0, 0, 0, 1.0005))])
    canonical = field.validate_field(raw)
    assert raw["nodes"][0]["rotation_xyzw"][3] == 1.0005
    assert canonical["nodes"][0]["rotation_xyzw"] == [0., 0., 0., 1.]
    assert canonical["_validation"]["normalizations"] == [{"node": 0, "original_norm": 1.0005, "action": "normalize_xyzw"}]
    assert field.validate_field(canonical) == canonical
    raw["nodes"][0]["rotation_xyzw"][3] = 1.01
    with pytest.raises(ValueError, match="quaternion"):
        field.validate_field(raw)


@pytest.mark.parametrize("change", [
    lambda s: s.update(estimated_focal_px=float("nan")),
    lambda s: s.update(units="cm"),
    lambda s: s.update(unexpected="execute_me"),
    lambda s: s["nodes"][0].update(size_m=[1, 1, 0]),
    lambda s: s["nodes"][0].update(center_m=[0, 0, 11]),
    lambda s: s["nodes"][0].update(left=0),
    lambda s: s.update(root=True),
])
def test_validator_rejects_invalid_model_values(change):
    raw = specification([primitive("sphere", [1, 0, 0])])
    change(raw)
    with pytest.raises(ValueError):
        field.validate_field(raw)


def test_graph_rejects_future_refs_and_unreachable_nodes():
    raw = specification([primitive("sphere", [1, 0, 0]), primitive("sphere", [.5, 0, 0])])
    with pytest.raises(ValueError, match="reachable"):
        field.validate_field(raw)
    raw["nodes"].append(operation("difference", 0, 2)); raw["root"] = 2
    with pytest.raises(ValueError, match="prior"):
        field.validate_field(raw)


def test_hollow_bowl_has_void_wall_and_bottom():
    # Outer lower hemisphere, minus a smaller sphere: explicit hollow bowl.
    nodes = [primitive("sphere", [1, 0, 0], center=(0, 0, 0)),
             primitive("box", [2, 2, 1], center=(0, 0, -.9)),
             operation("intersection", 0, 1),
             primitive("sphere", [.8, 0, 0], center=(0, 0, 0)),
             operation("difference", 2, 3)]
    spec = specification(nodes)
    values = field.evaluate_field(spec, [[0, 0, -.5], [0, 0, -.9], [0, 0, -1.1], [0, 0, .5]])
    assert np.all(values[[0, 2, 3]] > 0)
    assert values[1] < 0
    np.testing.assert_allclose(hits(spec, [[0, 0, 2]], [[0, 0, -1]]), [2.8], atol=1e-14)


def test_sphere_rays_outside_inside_miss_tangent_and_direction_scale():
    spec = specification([primitive("sphere", [1, 0, 0])])
    origins = [[0, 0, 0], [0, 0, 2], [2, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]]
    rays = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 2], [0, 0, 1]]
    np.testing.assert_allclose(hits(spec, origins, rays), [1, 1, np.nan, 2, .5, 2], atol=1e-14, equal_nan=True)


def test_box_parallel_tangent_rotated_inside_rays():
    q = [0, 0, np.sin(np.pi/8), np.cos(np.pi/8)]
    box = specification([primitive("box", [1, 1, 1], q=q)])
    expected = [1, 1, np.nan]
    np.testing.assert_allclose(hits(box, [[0, 0, 0], [0, 0, 2], [2, 0, 0]], [[0, 0, 1]]*3), expected, atol=1e-14, equal_nan=True)
    unrotated = specification([primitive("box", [1, 1, 1])])
    np.testing.assert_allclose(hits(unrotated, [[1, 0, 0]], [[0, 0, 1]]), [1])


def test_cylinder_caps_sides_tangent_and_miss():
    spec = specification([primitive("cylinder", [1, .5, 0])])
    origins = [[0, 0, 0], [2, 0, 2], [1, 0, 0], [1, -2, 2], [2, 0, 0], [0, 0, 2]]
    rays = [[0, 0, 1], [-1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    np.testing.assert_allclose(hits(spec, origins, rays), [1.5, 1, 1.5, 2, np.nan, .5], atol=1e-14, equal_nan=True)


def test_boolean_interval_combinations_and_nested_cavity():
    a = primitive("sphere", [1, 0, 0], center=(0, 0, 2))
    b = primitive("sphere", [1, 0, 0], center=(0, 0, 3))
    origin, ray = [[0, 0, 0]], [[0, 0, 1]]
    for op, expected in [("union", 1), ("intersection", 2), ("difference", 1)]:
        spec = specification([a, b, operation(op, 0, 1)])
        np.testing.assert_allclose(hits(spec, origin, ray), [expected])
    shell = specification([a, primitive("sphere", [.7, 0, 0], center=(0, 0, 2)), operation("difference", 0, 1)])
    np.testing.assert_allclose(hits(shell, [[0, 0, 2]], ray), [.7])
    np.testing.assert_allclose(hits(shell, [[0, 0, 1.1]], ray), [.2], atol=1e-14)


def test_union_overlap_removes_internal_boundaries():
    spec = specification([primitive("sphere", [1, 0, 0], center=(0, 0, 1)),
                          primitive("sphere", [1, 0, 0], center=(0, 0, 2)), operation("union", 0, 1)])
    np.testing.assert_allclose(hits(spec, [[0, 0, .5]], [[0, 0, 1]]), [2.5])


def test_zero_set_tangent_inside_boolean_is_retained():
    # A ray tangent to a spherical cavity touches the difference field's zero set.
    spec = specification([primitive("box", [2, 2, 2], center=(0, 0, 0)),
                          primitive("sphere", [1, 0, 0], center=(0, 0, 0)), operation("difference", 0, 1)])
    np.testing.assert_allclose(hits(spec, [[1, 0, -1.5]], [[0, 0, 1]]), [1.5])


def test_analytic_hits_are_field_zeros_and_first_bracket_crossings():
    q = [np.sin(.31), 0, 0, np.cos(.31)]
    spec = specification([primitive("box", [.8, .6, .5], q=q),
                          primitive("cylinder", [.3, 2, 0]), operation("difference", 0, 1)])
    rng = np.random.default_rng(419)
    origins = np.zeros((4000, 3)); rays = np.column_stack((rng.uniform(-.5, .5, (4000, 2)), np.ones(4000)))
    distances = hits(spec, origins, rays)
    valid = np.isfinite(distances)
    assert valid.sum() > 100
    points = rays[valid]*distances[valid, None]
    np.testing.assert_allclose(field.evaluate_field(spec, points), 0, atol=2e-14)
    assert np.all(field.evaluate_field(spec, points-1e-7*rays[valid]) >= -1e-13)
    assert np.all(field.evaluate_field(spec, points+1e-7*rays[valid]) <= 1e-13)


def test_empty_and_invalid_queries():
    spec = specification([primitive("sphere", [1, 0, 0])])
    assert field.evaluate_field(spec, np.empty((0, 3))).shape == (0,)
    assert hits(spec, np.empty((0, 3)), np.empty((0, 3))).shape == (0,)
    with pytest.raises(ValueError, match="nonzero"):
        hits(spec, [[0, 0, 0]], [[0, 0, 0]])
    with pytest.raises(ValueError, match="Nx3"):
        field.evaluate_field(spec, [[0, 0]])


def test_transport_schema_accepts_example_and_rejects_extra_keys():
    jsonschema = pytest.importorskip("jsonschema")
    spec = specification([primitive("sphere", [1, 0, 0])])
    jsonschema.validate(spec, field.field_schema())
    invalid = copy.deepcopy(spec); invalid["executable_python"] = "pass"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, field.field_schema())
