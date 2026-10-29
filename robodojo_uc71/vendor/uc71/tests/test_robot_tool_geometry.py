"""Independent known-robot geometry/observation-frame regressions; no simulator."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from gpt_field.robot_tool_geometry import (
    FINGER_COLLISION_VERTICES_M,
    HAND_COLLISION_VERTICES_M,
    NUMERICAL_PADDING_M,
    ROBOT_ASSET_PROVENANCE,
    geometry_contract,
    world_aabbs,
)

ROBOT_PACKAGE = Path('/venv/fm_dno/lib/python3.10/site-packages/robosuite')


def state(*, q=(.04, -.04), rot=(0., 0., 0.), xyz=(0., 0., 0.)):
    return np.array([*xyz, *rot, *q])


def parts(value, **kwargs):
    return {p['name']: p for p in world_aabbs(value, **kwargs)}


def contains(part, points):
    points = np.asarray(points)
    return np.all(points >= np.asarray(part['min_xyz'])-1e-13) and np.all(
        points <= np.asarray(part['max_xyz'])+1e-13)


def quaternion_matrix(quat_wxyz):
    w, x, y, z = np.asarray(quat_wxyz) / np.linalg.norm(quat_wxyz)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def binary_stl_vertices(path):
    raw = path.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    assert len(raw) == 84+50*count
    return np.unique(np.array([struct.unpack_from('<9f', raw, 84+50*i+12)
                               for i in range(count)]).reshape(-1, 3), axis=0)


def xml_reference_vertices(robot_state):
    """Independently compose actual robot XML hierarchy, including joint axes.

    This parser only reads the gripper's fixed robot assets. It has no task,
    environment, simulation, camera, object-state or controller imports.
    """
    gripper_dir = ROBOT_PACKAGE/'models/assets/grippers'
    tree = ET.parse(gripper_dir/'panda_gripper.xml').getroot()
    meshes = {item.attrib['name']: binary_stl_vertices(gripper_dir/item.attrib['file'])
              for item in tree.findall('./asset/mesh') if item.attrib['name'] in ('hand', 'finger')}
    vec = robot_state[3:6]
    angle = np.linalg.norm(vec)
    axis = vec/angle if angle else np.array([1., 0., 0.])
    hand_rotation = quaternion_matrix(np.r_[math.cos(angle/2), axis*math.sin(angle/2)])
    # Observation position is the grip_site rather than this body origin.
    hand_origin = robot_state[:3] - hand_rotation @ [0., 0., .097]
    found = {}
    def numeric(element, key, default):
        return np.fromstring(element.attrib.get(key, default), sep=' ')
    def descend(body, parent_position, parent_rotation):
        position = parent_position + parent_rotation @ numeric(body, 'pos', '0 0 0')
        rotation = parent_rotation @ quaternion_matrix(numeric(body, 'quat', '1 0 0 0'))
        for joint in body.findall('./joint'):
            assert joint.attrib['type'] == 'slide'
            index = {'finger_joint1': 6, 'finger_joint2': 7}[joint.attrib['name']]
            position += rotation @ (numeric(joint, 'axis', '0 0 1')*robot_state[index])
        for geom in body.findall('./geom'):
            name = geom.attrib.get('name', '')
            if not name.endswith('_collision'):
                continue
            geom_rotation = rotation @ quaternion_matrix(numeric(geom, 'quat', '1 0 0 0'))
            geom_position = position + rotation @ numeric(geom, 'pos', '0 0 0')
            if geom.attrib['type'] == 'mesh':
                vertices = meshes[geom.attrib['mesh']]
            else:
                assert geom.attrib['type'] == 'box'
                vertices = np.array(list(itertools.product((-1.,1.), repeat=3))) * numeric(geom, 'size', '')
            found[name] = vertices @ geom_rotation.T + geom_position
        for child in body.findall('./body'):
            descend(child, position, rotation)
    descend(tree.find('./worldbody/body'), hand_origin, hand_rotation)
    return found


@pytest.mark.parametrize('rot', [(0.,0.,0.), (0.,0.,math.pi/2), (math.pi,0.,0.),
                                (1.41,-.37,2.62), (1e-13,2e-13,-1e-13)])
@pytest.mark.parametrize('q', [(0.,0.), (.04,-.04), (.007,-.032), (.0401,-.0402)])
def test_mesh_and_pad_bounds_match_independent_robot_xml_transforms(rot, q):
    if not (ROBOT_PACKAGE/'models/assets/grippers/panda_gripper.xml').exists():
        pytest.skip('Fixed robot assets absent; literal known-point regressions still run')
    value = state(q=q, rot=rot, xyz=(.12,-.26,.97))
    actual, expected = parts(value), xml_reference_vertices(value)
    assert set(actual) == set(expected) and len(actual) == 5
    for name, points in expected.items():
        assert contains(actual[name], points)
        assert actual[name]['min_xyz'] == pytest.approx(points.min(axis=0)-NUMERICAL_PADDING_M, abs=2e-15)
        assert actual[name]['max_xyz'] == pytest.approx(points.max(axis=0)+NUMERICAL_PADDING_M, abs=2e-15)


def test_embedded_vertices_and_provenance_are_exact_fixed_robot_assets():
    if not ROBOT_PACKAGE.exists():
        pytest.skip('Fixed robot installation not present')
    for item in ROBOT_ASSET_PROVENANCE:
        raw = (ROBOT_PACKAGE/item['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        assert len(raw) == item['size_bytes']
    mesh = ROBOT_PACKAGE/'models/assets/grippers/meshes/panda_gripper'
    assert np.array_equal(HAND_COLLISION_VERTICES_M, binary_stl_vertices(mesh/'hand.stl'))
    assert np.array_equal(FINGER_COLLISION_VERTICES_M, binary_stl_vertices(mesh/'finger.stl'))
    assert len(HAND_COLLISION_VERTICES_M) == 102 and len(FINGER_COLLISION_VERTICES_M) == 18


def test_known_zero_orientation_open_tool_bounds_use_grip_site_origin_not_hand_origin():
    value = parts(state())
    assert value['hand_collision']['min_xyz'] == pytest.approx(
        [-.031635917723178864,-.10398972034454346,-.12292480182647705], abs=2e-12)
    assert value['hand_collision']['max_xyz'] == pytest.approx(
        [.03161578252911568,.10042587667703629,-.03103784930706024], abs=2e-12)
    assert value['finger1_collision']['max_xyz'] == pytest.approx(
        [.010494818910956383,.06640338242053986,.009249034011363983], abs=2e-12)
    assert value['finger2_collision']['min_xyz'] == pytest.approx(
        [-.010494818910956383,-.06640338242053986,-.044468303755647504], abs=2e-12)
    assert value['finger1_pad_collision']['min_xyz'] == pytest.approx([-.008,.0395,-.0116], abs=2e-12)
    assert value['finger2_pad_collision']['max_xyz'] == pytest.approx([.008,-.0395,.0044], abs=2e-12)
    assert value['hand_collision']['max_radius_m'] == pytest.approx(.1514046524714982)
    assert value['finger1_collision']['max_radius_m'] == pytest.approx(.08057374915949184)


def test_known_90_degree_rotation_and_translation_do_not_use_grip_site_yaw():
    value = parts(state(rot=(0.,0.,math.pi/2), xyz=(1.,2.,3.)))
    pad = value['finger1_pad_collision']
    # x=-y_local, y=x_local. An extra gripper-root yaw would put this elsewhere.
    assert pad['min_xyz'] == pytest.approx([.9525,1.992,2.9884], abs=2e-12)
    assert pad['max_xyz'] == pytest.approx([.9605,2.008,3.0044], abs=2e-12)
    assert pad['offset_min_xyz'] == pytest.approx([-.0475,-.008,-.0116], abs=2e-12)


def test_union_keeps_open_grasp_gap_and_demonstrates_55mm_cube_is_not_tool_envelope():
    value = parts(state())
    assert not any(contains(part, [[0.,0.,0.]]) for part in value.values())
    lower = np.min([part['min_xyz'] for part in value.values()], axis=0)
    upper = np.max([part['max_xyz'] for part in value.values()], axis=0)
    assert np.all(lower < 0.) and np.all(upper > 0.)  # One merged box fills the grasp gap.
    # This real open-finger mesh vertex is beyond the old radius AND its axis cube.
    finger = np.array(FINGER_COLLISION_VERTICES_M)
    actual_vertex = finger[np.argmax(finger[:,1])] + [0.,.04,-.0446]
    assert abs(actual_vertex[1]) > .055
    assert np.linalg.norm(actual_vertex) > .055
    assert contains(value['finger1_collision'], [actual_vertex])


def test_independent_finger_motion_changes_only_its_parts_and_never_clips_measurements():
    base = parts(state(q=(.0,.0)))
    moved = parts(state(q=(.012,-.041)))
    assert moved['hand_collision'] == base['hand_collision']
    for name, shift in [('finger1_collision',.012), ('finger1_pad_collision',.012),
                        ('finger2_collision',-.041), ('finger2_pad_collision',-.041)]:
        assert np.array(moved[name]['min_xyz'])-base[name]['min_xyz'] == pytest.approx([0.,shift,0.])
        assert np.array(moved[name]['max_xyz'])-base[name]['max_xyz'] == pytest.approx([0.,shift,0.])
    swept = parts(state(q=(.041,-.042)), full_finger_range=True)
    assert swept['finger1_collision']['finger_range_m'] == [0.,.041]
    assert swept['finger2_collision']['finger_range_m'] == [-.042,0.]
    assert swept['finger2_collision']['min_xyz'][1] < -.068


def test_full_finger_sweep_contains_every_intermediate_pose_without_filling_whole_tool_box():
    nominal = state(rot=(.3,-.6,2.8), xyz=(.31,.24,.91))
    swept = parts(nominal, full_finger_range=True)
    for left in np.linspace(0.,.04,7):
        for right in np.linspace(-.04,0.,7):
            sample = nominal.copy(); sample[6:] = [left,right]
            for name, item in parts(sample).items():
                assert contains(swept[name], [item['min_xyz'],item['max_xyz']])
    assert len(swept) == 5


def test_orientation_margin_contains_rotated_vertices_for_all_sampled_directions():
    nominal = state(rot=(math.pi,0.,0.), xyz=(-.2,.17,.95))
    angle = .04
    bounds = parts(nominal, full_finger_range=True, orientation_margin_rad=angle, padding_m=.001)
    rng = np.random.default_rng(71)
    base_rotation = np.diag([1.,-1.,-1.])
    for _ in range(60):
        axis = rng.normal(size=3);axis /= np.linalg.norm(axis)
        delta = rng.uniform(0.,angle)
        rotation = quaternion_matrix(np.r_[math.cos(delta/2),axis*math.sin(delta/2)]) @ base_rotation
        q = rng.uniform([0.,-.04],[.04,0.])
        local = {
            'hand_collision': np.asarray(HAND_COLLISION_VERTICES_M)-[0.,0.,.097],
            'finger1_collision': np.asarray(FINGER_COLLISION_VERTICES_M)+[0.,q[0],-.0446],
            'finger2_collision': np.asarray(FINGER_COLLISION_VERTICES_M)*[-1.,-1.,1.]+[0.,q[1],-.0446],
        }
        corners = np.array(list(itertools.product((-1.,1.),repeat=3)))*[.008,.004,.008]
        local['finger1_pad_collision'] = corners+[0.,q[0]+.0035,-.0036]
        local['finger2_pad_collision'] = corners+[0.,q[1]-.0035,-.0036]
        for name, vertices in local.items():
            assert contains(bounds[name],vertices@rotation.T+nominal[:3])
    for item in bounds.values():
        assert item['orientation_margin_m'] == pytest.approx(2*item['max_radius_m']*math.sin(angle/2))
        assert item['padding_m'] == .001


def test_positive_margins_only_expand_each_part_and_offsets_remain_world_axes():
    value = state(rot=(.1,.5,-2.7),xyz=(.4,-.2,.9))
    raw, expanded = parts(value), parts(value,orientation_margin_rad=.04,padding_m=.002)
    for name,item in raw.items():
        enlarged = expanded[name]
        added = enlarged['orientation_margin_m']+.002
        assert np.asarray(item['min_xyz'])-enlarged['min_xyz'] == pytest.approx([added]*3)
        assert np.asarray(enlarged['max_xyz'])-item['max_xyz'] == pytest.approx([added]*3)
        assert np.asarray(enlarged['min_xyz'])-value[:3] == pytest.approx(enlarged['offset_min_xyz'])
        assert np.asarray(enlarged['max_xyz'])-value[:3] == pytest.approx(enlarged['offset_max_xyz'])


def test_world_frame_asymmetric_obstacle_inflation_detects_finger_not_eef_center():
    finger = parts(state())['finger1_collision']
    obstacle_low = np.array([-.001,.065,.005]);obstacle_high = np.array([.001,.067,.007])
    assert obstacle_low[1]-.055 > 0.  # Old uniform cube would admit EEF origin.
    forbidden_eef_low = obstacle_low-finger['offset_max_xyz']
    forbidden_eef_high = obstacle_high-finger['offset_min_xyz']
    assert np.all(forbidden_eef_low <= 0.) and np.all(forbidden_eef_high >= 0.)


@pytest.mark.parametrize('bad', [[0.]*7, [[0.]*8], [False]+[0.]*7, ['0']+[0.]*7,
                               [complex(1,0)]+[0.]*7, [float('nan')]+[0.]*7,
                               [0.]*7+[float('inf')], None])
def test_invalid_robot_state_does_not_produce_geometry(bad):
    with pytest.raises(ValueError):
        world_aabbs(bad)


@pytest.mark.parametrize('kwargs', [dict(orientation_margin_rad=-.001),dict(orientation_margin_rad=math.pi+.001),
                                   dict(orientation_margin_rad=float('nan')),dict(padding_m=-.001),
                                   dict(padding_m=float('inf')),dict(padding_m=True),
                                   dict(padding_m='0.1'),dict(full_finger_range=1)])
def test_invalid_margin_contract_cannot_shrink_or_silently_change_envelope(kwargs):
    with pytest.raises(ValueError):
        world_aabbs(state(),**kwargs)


def test_outputs_are_json_serializable_independent_and_explicitly_exclude_payload_environment_arm():
    original = state();saved=original.copy()
    one = world_aabbs(original);one[0]['min_xyz'][0]=100
    two = world_aabbs(original)
    assert two[0]['min_xyz'][0]<0. and np.array_equal(original,saved)
    one_contract = geometry_contract();one_contract['assets'][0]['sha256']='changed'
    contract = geometry_contract()
    assert len(contract['assets'][0]['sha256'])==64
    assert not contract['payload_included'] and not contract['arm_links_included']
    assert not contract['certifies_environment_clearance'] and contract['environment_inputs']==[]
    json.dumps({'contract':contract,'parts':two},allow_nan=False)
