"""Camera/depth conventions and evaluator mask invariants."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'prepare_reference.py'
spec = importlib.util.spec_from_file_location('rgb_prepare_reference', SOURCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def test_depth_conversion_is_axis_z_and_inverts_perspective_projection():
    near, far = .01, 100.
    expected = np.array([near, .13, 1., 2.75, far])
    buffer = (1. - near / expected) / (1. - near / far)
    actual = reference.depth_buffer_to_z(buffer, near, far)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)
    with pytest.raises(ValueError):
        reference.depth_buffer_to_z([1.001], near, far)
    with pytest.raises(ValueError):
        reference.depth_buffer_to_z([np.nan], near, far)
    with pytest.raises(ValueError):
        reference.depth_buffer_to_z([.3], 1., .5)


def test_exact_pixel_centres_and_upright_archived_mirror_are_consistent():
    calibration = reference.camera_calibration(8, 6, 90., [1., 2., 3.], np.eye(3))
    k = np.asarray(calibration['K'])
    np.testing.assert_allclose(k, [[3., 0., 3.5], [0., 3., 2.5], [0., 0., 1.]], atol=1e-14)
    # OpenGL +X/+Y/-Z becomes top-down CV +X/-Y/+Z, a proper rotation.
    transform = np.asarray(calibration['T_world_from_camera_cv'])
    np.testing.assert_array_equal(transform[:3, :3], np.diag([1., -1., -1.]))
    assert np.linalg.det(transform[:3, :3]) == 1
    points = np.array([[.2, -.6, 2.], [-.5, .1, 1.]])
    uv_upright = points @ k.T
    uv_upright = uv_upright[:, :2] / uv_upright[:, 2:3]
    uv_archived = points @ np.asarray(calibration['K_archived_mirrored']).T
    uv_archived = uv_archived[:, :2] / uv_archived[:, 2:3]
    np.testing.assert_allclose(uv_archived[:, 0], 7 - uv_upright[:, 0])
    np.testing.assert_allclose(uv_archived[:, 1], uv_upright[:, 1])
    np.testing.assert_allclose(transform @ np.asarray(calibration['T_camera_cv_from_world']), np.eye(4))


def test_plane_backprojection_has_constant_optical_z_but_variable_range():
    k = reference.camera_calibration(9, 9, 90., [0, 0, 0], np.eye(3))['K']
    points = reference.backproject(np.full((9, 9), 2.), k)
    np.testing.assert_array_equal(points[..., 2], np.full((9, 9), 2.))
    np.testing.assert_allclose(points[4, 4], [0., 0., 2.])
    assert np.linalg.norm(points[0, 0]) > 3.
    assert points[0, 0, 0] < 0 and points[0, 0, 1] < 0
    assert points[-1, -1, 0] > 0 and points[-1, -1, 1] > 0
    with pytest.raises(ValueError):
        reference.backproject(np.zeros((2, 2)), k)
    with pytest.raises(ValueError):
        reference.camera_calibration(4, 4, 90, [0, 0, 0], np.diag([-1., 1., 1.]))


def test_masks_handle_invalid_geom_and_exclude_occluding_robot():
    records = [
        {'object': 'akita_black_bowl_1', 'role': 'scene'},
        {'object': 'wooden_cabinet_1', 'role': 'scene'},
        {'object': 'robot0', 'role': 'robot'},
        {'object': 'main_table', 'role': 'scene'},
    ]
    # Type5 is deliberately not a geometry; negative ID is background.
    seg = np.array([[[5, 0], [5, -1], [5, 1]], [[6, 0], [5, 2], [5, 3]]])
    masks, objects = reference.masks_from_segmentation(seg, records, 5)
    np.testing.assert_array_equal(masks['nonrobot_visible'], [[1, 0, 1], [0, 0, 1]])
    np.testing.assert_array_equal(masks['target_bowl_visible'], [[1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(masks['target_cabinet_visible'], [[0, 0, 1], [0, 0, 0]])
    assert masks['geom_id'][1, 0] == -1
    assert masks['object_id'][0, 1] == -1
    assert objects[masks['object_id'][1, 2]] == 'main_table'
    # A scene with no cabinet must never map background to a target.
    absent, _ = reference.masks_from_segmentation(np.array([[[5, -1]]]), records[:1], 5)
    assert not absent['target_cabinet_visible'].any()


def test_visual_pixel_rays_match_analytic_plane_and_box_without_graphics():
    from types import SimpleNamespace
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="3 3 .01" pos="0 0 -2" group="1"/>
      <geom type="box" size=".2 .2 .2" pos="0 0 -1" group="1"/>
      <geom type="box" size=".1 .1 .1" pos="0 0 -.5" group="0"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data),
        _render_context_offscreen=SimpleNamespace(vopt=SimpleNamespace(geomgroup=np.array([0, 1, 0, 0, 0, 0], np.uint8))))
    cal = reference.camera_calibration(41, 41, 90., [0, 0, 0], np.eye(3))
    cal.update(near_clip_m=.01, far_clip_m=100.)
    depth, gid, valid = reference.pixel_center_ray_depth(sim, cal)
    assert valid.all()
    np.testing.assert_allclose(depth[gid == 0], 2., atol=5e-15, rtol=0)
    np.testing.assert_allclose(depth[gid == 1], .8, atol=5e-15, rtol=0)
    assert depth[20, 20] == pytest.approx(.8, abs=1e-15)
    assert gid[20, 20] == 1  # hidden group-0 collision shape must be excluded
    assert gid[0, 0] == 0
    check = reference.validate_depth_rays(sim, depth, gid, cal, sample_count=256)
    assert check['passed']
    assert check['max_depth_error_m'] < 1e-14


def test_pixel_centres_match_independent_non_msaa_opengl_tilted_face():
    """Independent rasterizer checks centre convention, unlike ray-vs-ray tests."""
    import os
    import json
    import subprocess
    import sys
    code = r'''
import numpy as np,mujoco,json
xml=''' + "'''" + '''<mujoco><statistic extent="1"/><visual><quality offsamples="0"/><map znear=".2" zfar="10"/></visual><worldbody><camera name="test" pos="0 0 0" fovy="60"/><geom type="box" size="2 2 .01" pos="0 0 -2" euler="15 10 0"/></worldbody></mujoco>''' + "'''" + r'''
m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,80,100)
try:
 r.enable_depth_rendering();r.update_scene(d,camera='test');depth=r.render()
 v,u=np.indices(depth.shape);f=40./np.tan(np.deg2rad(60)/2)
 n=d.geom_xmat[0].reshape(3,3)[:,2];p=d.geom_xpos[0]+.01*n
 errors=[]
 for offset in [0.,.5]:
  rays=np.stack([(u-49.5-offset)/f,-(v-39.5-offset)/f,-np.ones_like(u)],axis=-1)
  exact=(n@p)/np.einsum('hwi,i->hw',rays,n)
  # Predeclared 8px frame excludes finite-face silhouette/background pixels.
  errors.append(np.abs(depth-exact)[8:-8,8:-8])
 print(json.dumps({'pixel_count':int(errors[0].size),'max_error_m':float(errors[0].max()),
  'wrong_half_pixel_median_error_m':float(np.median(errors[1])),'offsamples':int(m.vis.quality.offsamples)}))
finally:r.close()
'''
    environment = {**os.environ, 'MUJOCO_GL': 'egl', 'MUJOCO_EGL_DEVICE_ID': '2'}
    completed = subprocess.run([sys.executable, '-B', '-c', code], env=environment,
                               capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result['pixel_count'] == 5376
    assert result['offsamples'] == 0
    assert result['max_error_m'] < 1e-4  # same 0.1mm gate as reference validation
    assert result['wrong_half_pixel_median_error_m'] > 1e-3
    print(result)
