"""CPU-only, operator-only projected finger sweeps; no contact/clearance claims."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[2]
sys.path.insert(0, str(BASE / 'scripts'))
from analyze_joint_probe import URDF, pose_matrix


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def bind(p, expected=None):
    p = Path(p).resolve(strict=True)
    digest = sha(p)
    if expected is not None and digest != expected:
        raise ValueError('Input changed: ' + str(p))
    return {'path': str(p), 'sha256': digest}


comparison_path = OUT.parent / 'controller_clearance_feasibility_004/grid.json'
inputs = [bind(comparison_path, '26f732a8ae6b90e7bb230c14417f46db8195401254bd2d34f20b761a974c4b98')]
comparison = read(comparison_path)
controller = OUT.parent / 'native_controller_characterization_008/worker/controller'
inputs.append(bind(OUT.parent / 'controller_clearance_projection_001/analysis.py', 'a9cfdad2830907e9f5c6880a8d55006111568bcf86b2b94b28f7ff4099fb8ae4'))
SPECS = {'A': (.050, .030, .025, .10), 'B': (.045, .030, .025, .10)}
FIELDS = ('lift_m', 'retreat_m', 'outward_m', 'outward_yaw_rad')
selected = {}
for label, spec in SPECS.items():
    matched = [row for row in comparison['results'] if tuple(row[key] for key in FIELDS) == spec]
    assert len(matched) == 2 and {row['arm'] for row in matched} == {'left', 'right'}
    assert all(row['failure'] is None for row in matched)
    selected[label] = {row['arm']: row for row in matched}

request_path = controller / 'clearance_right/request.json'
request = read(request_path)
inputs.append(bind(request_path))
capture = request['capture']
assert request['stage_index'] == 34 and capture['physics_unchanged_during_capture'] is True
for key in ('camera_inventory', 'robot_measurement'):
    row = capture[key]
    inputs.append(bind(controller / row['path'], row['sha256']))
head_row = next(r for r in capture['images'] if r['camera'] == 'cam_head')
image_path = controller / head_row['path']
inputs.append(bind(image_path, head_row['sha256']))
camera_inventory = read(controller / capture['camera_inventory']['path'])
measurement = read(controller / capture['robot_measurement']['path'])
assert measurement['native_physics_time_s'] == camera_inventory['native_physics_time_s'] == capture['native_physics_time_s']
camera = camera_inventory['cameras']['head']
k = np.asarray(camera['intrinsics'])
optical_from_world = np.asarray(camera['optical_from_world'])
assert np.allclose(optical_from_world @ np.asarray(camera['world_from_optical']), np.eye(4), atol=1e-6)
rgb = np.asarray(Image.open(image_path).convert('RGB'))
assert rgb.shape == (480, 640, 3)
inputs.append(bind(OUT.parent / 'controller_clearance_projection_002/analysis.py', '1db7c0fd52bb02963545a69fbdda642ddb68acf0de38e12ffa8b6c0f16c47cdc'))
left_request_path = controller / 'clearance_left/request.json'
inputs.append(bind(left_request_path)); left_capture = json.loads(left_request_path.read_text())['capture']
assert left_capture['label'] == 'stage_027' and left_capture['physics_unchanged_during_capture'] is True
for name in ('camera_inventory','robot_measurement'):
    row=left_capture[name];inputs.append(bind(controller/row['path'],row['sha256']))
left_measurement=json.loads((controller/left_capture['robot_measurement']['path']).read_text())
left_camera=json.loads((controller/left_capture['camera_inventory']['path']).read_text())['cameras']['head']
left_image=next(row for row in left_capture['images'] if row['camera']=='cam_head')
inputs.append(bind(controller/left_image['path'],left_image['sha256']))
images_by_arm={'right':rgb,'left':np.asarray(Image.open(controller/left_image['path']).convert('RGB'))}
cameras_by_arm={'right':camera,'left':left_camera}

candidate_path = OUT.parent / 'tool_geometry_001/candidate.json'
candidate = read(candidate_path)
inputs.append(bind(candidate_path))
sources = {Path(r['path']).name: Path(r['path']) for r in candidate['source_files']}
inputs.extend(bind(r['path'], r['sha256']) for r in candidate['source_files'])
inputs.append(bind(BASE / 'scripts/analyze_joint_probe.py'))
urdf = URDF(sources['X5A.urdf'])
meshes = {link: trimesh.load_mesh(sources[link + '.STL'], process=False) for link in ('link7', 'link8')}
tool = np.asarray(candidate['link6_from_tool'])
for link, mesh in meshes.items():
    pad = candidate['pad_geometry'][link]
    assert pad['usd_stl_vertices_match'] is True
    actual = np.unique(mesh.triangles[pad['triangle_indices']].reshape(-1, 3), axis=0)
    assert np.allclose(actual, pad['pad_vertices_in_finger_link_m'], rtol=0, atol=1e-10)


def project(points):
    points = np.asarray(points)
    optical = points @ optical_from_world[:3, :3].T + optical_from_world[:3, 3]
    assert np.all(np.isfinite(optical)) and np.all(optical[:, 2] > 0)
    h = optical @ k.T
    return h[:, :2] / h[:, 2, None]


def transform(points, frame):
    return np.asarray(points) @ frame[:3, :3].T + frame[:3, 3]


SCALE = 3


def raster_sweep(open_triangles_uv, closed_triangles_uv):
    """Union projected triangle translation prisms, rasterized at3x RGB size.

    Positive camera depth makes the perspective image of each convex translation
    prism the convex hull of its six projected endpoint vertices. This is a
    visual-mesh footprint, not USD collision geometry or an occlusion/depth test.
    """
    mask = np.zeros((480 * SCALE, 640 * SCALE), np.uint8)
    for a, b in zip(open_triangles_uv, closed_triangles_uv):
        hull = cv2.convexHull(np.asarray(np.r_[a, b], dtype=np.float32))
        cv2.fillConvexPoly(mask, np.round(hull * SCALE).astype(np.int32), 255)
    return mask


robots = {r['arm_name'].removesuffix('_arm'): r for r in measurement['raw_robot_inventory']['robots']}
cases, rows, checks = {}, {}, []
robots['left']=next(row for row in left_measurement['raw_robot_inventory']['robots'] if row['arm_name']=='left_arm')
for arm, robot in robots.items():
    k=np.asarray(cameras_by_arm[arm]['intrinsics']);optical_from_world=np.asarray(cameras_by_arm[arm]['optical_from_world'])
    names = robot['joint_names']
    measured_q = dict(zip(names, robot['joint_positions']['values'][0]))
    poses = np.asarray(robot['body_link_pose_w']['values'])[0]
    root = pose_matrix(poses[robot['body_names'].index('base_link')])
    measured_links, _ = urdf.forward(measured_q, root)
    for link in ('link6', 'link7', 'link8'):
        actual = pose_matrix(poses[robot['body_names'].index(link)])
        error = float(np.max(np.abs(actual - measured_links[link])))
        assert error < 1e-5, (arm, link, error)
        checks.append({'arm': arm, 'link': link, 'measured_fk_matrix_max_abs_error': error})
    cases[arm], rows[arm] = {}, {}
    for distance in ('0', 'A', 'B'):
        q = dict(measured_q)
        if distance != '0':
            endpoint = selected[distance][arm]['candidate_endpoint']
            assert np.allclose(root, endpoint['world_from_base_link'], rtol=0, atol=1e-6)
            q = dict(zip(endpoint['all_joint_names'], endpoint['full_q_rad']))
            # Use actual stage034 opening, not the endpoint's earlier finger sample.
            for name in ('joint7', 'joint8'):
                q[name] = measured_q[name]
        frames = []
        samples = []
        for fraction in np.linspace(0, 1, 41):
            sample_q = dict(q)
            for name in ('joint7', 'joint8'):
                sample_q[name] = measured_q[name] * (1 - float(fraction))
                assert 0 <= sample_q[name] <= .044001
            links, _ = urdf.forward(sample_q, root)
            frames.append(links)
            samples.append({'closing_fraction': float(fraction),
                            'finger_joint_positions_m': {n: sample_q[n] for n in ('joint7', 'joint8')}})
        entry = {'full': np.zeros((480*SCALE, 640*SCALE), np.uint8),
                 'pad': np.zeros((480*SCALE, 640*SCALE), np.uint8), 'paths': {}}
        for link, mesh in meshes.items():
            uv_open = project(transform(mesh.triangles.reshape(-1, 3), frames[0][link])).reshape(-1, 3, 2)
            uv_closed = project(transform(mesh.triangles.reshape(-1, 3), frames[-1][link])).reshape(-1, 3, 2)
            entry['full'] |= raster_sweep(uv_open, uv_closed)
            indices = candidate['pad_geometry'][link]['triangle_indices']
            entry['pad'] |= raster_sweep(uv_open[indices], uv_closed[indices])
            center = np.asarray(candidate['pad_geometry'][link]['centroid_in_finger_link_m'])
            path = np.asarray([project(transform([center], frame[link]))[0] for frame in frames])
            entry['paths'][link] = path
            for sample, frame in zip(samples, frames):
                sample.setdefault('pad_centroid_world_m', {})[link] = transform([center], frame[link])[0].tolist()
        point = (frames[0]['link6'] @ tool)[:3, 3]
        if distance != '0':
            error = float(np.linalg.norm(point - endpoint['target_tool_pose_wxyz'][:3]))
            assert error < 1e-6
        entry['tool_uv'] = project([point])[0]
        cases[arm][distance] = entry
        rows[arm][distance] = {'arm_joint_positions_rad': {n: q[n] for n in names[:6]},
                              'world_from_base_link': root.tolist(),
                              'tool_world_m': point.tolist(), 'tool_uv_pixels': entry['tool_uv'].tolist(),
                              'finger_samples': samples}


def contours(ax, mask, color, linewidth):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cs:
        points = contour[:, 0] / SCALE
        ax.plot(*np.r_[points, points[:1]].T, color=color, lw=linewidth)


def draw(ax, arm, distance, crop=None):
    ax.imshow(images_by_arm[arm], interpolation='nearest')
    entry = cases[arm][distance]
    for key, color, alpha in (('full', '#ffb020', .19), ('pad', '#00d8ff', .5)):
        mask = entry[key]
        rgba = np.zeros((*mask.shape, 4))
        rgba[:, :, :3] = matplotlib.colors.to_rgb(color)
        rgba[:, :, 3] = mask / 255 * alpha
        ax.imshow(rgba, extent=(-.5, 639.5, 479.5, -.5), interpolation='nearest')
        contours(ax, mask, color, 1.4 if key == 'full' else 1.8)
    for path in entry['paths'].values():
        ax.plot(*path.T, color='#0078ff', lw=1.1, ls='--')
        ax.scatter(*path[0], marker='o', s=22, c='#0078ff', edgecolors='white', linewidths=.6)
        ax.scatter(*path[-1], marker='x', s=30, c='#ff40dd', linewidths=1.7)
        ax.annotate('', xy=path[-1], xytext=path[25], arrowprops={'arrowstyle':'->','color':'#0078ff','lw':1.4})
    ax.scatter(*entry['tool_uv'], marker='+', s=60, color='white', linewidths=1.4)
    if crop:
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[3], crop[2])
    ax.set_xlabel('Head image u (pixels)'); ax.set_ylabel('Head image v (pixels)')
    title = 'Recorded008 pose\nhypothetical closing' if distance == '0' else (
        f'{distance}: lift {SPECS[distance][0]*1000:g} / retreat {SPECS[distance][1]*1000:g} mm\n'
        f'outward {SPECS[distance][2]*1000:g} mm / yaw {SPECS[distance][3]:.2f} rad')
    ax.set_title(title, fontsize=11)


legend = [Patch(facecolor='#ffb020', alpha=.5, label='Full finger visual-mesh projected sweep'),
          Patch(facecolor='#00d8ff', alpha=.6, label='Inner pad-face projected sweep'),
          Line2D([], [], color='#0078ff', marker='o', linestyle='--', label='Measured open → physical zero'),
          Line2D([], [], color='#ff40dd', marker='x', linestyle='', label='Pad centroid at q7=q8=0')]
artifacts = []
for arm, crop in (('right', (450, 640, 220, 355)), ('left', (0, 190, 220, 355))):
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.1))
    for ax, distance in zip(axes, ('0', 'A', 'B')):
        draw(ax, arm, distance, crop)
    fig.suptitle(f'{arm.upper()} ARM | source-bound projection on unchanged008 stage{27 if arm == "left" else 34:03d} RGB', fontsize=15)
    fig.legend(handles=legend, loc='lower center', ncol=2, bbox_to_anchor=(.5, .04), fontsize=10)
    fig.text(.5, .012, 'Predictions only. No depth/occlusion or contact test; visual meshes are not collision envelopes. Fresh clearance images remain required.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .16, 1, .93))
    for suffix in ('png', 'pdf'):
        path = OUT / f'{arm}_sweep_comparison.{suffix}'
        fig.savefig(path, dpi=180); artifacts.append(bind(path))
    plt.close(fig)
    print(str(OUT / f'{arm}_sweep_comparison.png'), flush=True)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5))
axes[0].imshow(rgb); axes[0].set_title('Unchanged recorded head RGB, stage034'); axes[0].axis('off')
for ax, distance in zip(axes[1:], ('A', 'B')):
    draw(ax, 'right', distance)
    ax.set_xlim(-.5,639.5);ax.set_ylim(479.5,-.5)
fig.suptitle('Right-arm predictions on the recorded scene — visual review aid only', fontsize=14)
fig.text(.5,.02,'Orange: full finger visual-mesh sweep. Cyan: pad-face sweep. No object geometry or collision/depth inference.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.05,1,.93))
for suffix in ('png','pdf'):
    path=OUT/f'scene_context.{suffix}';fig.savefig(path,dpi=180);artifacts.append(bind(path))
plt.close(fig)

record = {'schema':'robodojo-offline-finger-sweep-projection-1',
          'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'candidate_specifications':{label:dict(zip(FIELDS,spec)) for label,spec in SPECS.items()},
          'baseline_stage_indices':{'left':27,'right':34},
          'cameras_by_arm':cameras_by_arm,
          'hypothetical_closing_only':True,
          'scope':'operator_only_predicted_robot_geometry_on_saved_rgb',
          'source_inputs':inputs, 'analysis_program':bind(__file__), 'artifacts':artifacts,
          'method':'URDF FK at source-bound candidate arm joints;41 finger samples from stage034 measured joint7/8 opening to0m. Whole-finger visual triangles translated between endpoints; union of projected triangle-prism convex hulls rasterized at3x image resolution. Pad-face sweep uses only the two bound distal triangles per finger.',
          'units':'joint1..6 radians; joint7..8 metres despite upstream full_q_rad field name',
          'camera_intrinsics':k.tolist(), 'optical_from_world':optical_from_world.tolist(),
          'native_close_target_control_fact_m':-.01,
          'negative_physical_finger_positions_projected':False,
          'checks':checks, 'cases':rows,
          'limitations':['Visual-mesh footprints are not USD physics collision envelopes.',
                         'Projection ignores scene depth and occlusion; cannot establish contact, separation, or clearance.',
                         'No source object poses, object geometry, contacts, rewards, or model inference were used.',
                         'Only fingers are projected; link6, robot arms, camera housings and dynamic tracking uncertainty are not swept.',
                         'Original recorded robot image remains visible beneath predicted ghost geometry.',
                         'Left uses its actual008 stage027 clearance pose; right uses its actual008 stage034 clearance pose.',
                         'These are hypothetical endpoint closing sweeps, not the swept arm path to the endpoint.',
                         'Fresh future RGB and separate operator review remain mandatory.'],
          'input_image_unchanged':sha(image_path)==head_row['sha256'],
          'clearance_approved':False,'diagnostic_launch_authorized':False,'full_test_authorized':False,
          'calibration_status':'unchanged_candidate','model_input_eligible':False,
          'simulator_starts':0,'model_calls':0,'native_actions':0,
          'packages':{'numpy':np.__version__,'trimesh':trimesh.__version__,'matplotlib':matplotlib.__version__,'opencv':cv2.__version__}}
(OUT/'projection.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
assert all(sha(row['path']) == row['sha256'] for row in inputs)
print(str(OUT/'projection.json'), flush=True)
