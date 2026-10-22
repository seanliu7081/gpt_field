"""Evaluator-only visual depth for three frozen LIBERO RGB images.

Inference receives only the deterministic upright RGB copy. Calibration, depth,
segmentation, frozen simulator and object identities are evaluator-only files.
Existing project/capture/trial files are read-only. Every output directory must
be new. A child process per snapshot isolates archived Python runtimes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np
from PIL import Image

DEFAULT_CAPTURES = Path('/workspace/gpt_field/scene_test/libero_scene/captures_001')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def depth_buffer_to_z(buffer, near, far):
    """Invert standard OpenGL depth to positive camera-axis distance (metres)."""
    values = np.asarray(buffer, dtype=np.float64)
    if not (0 < near < far) or not np.isfinite([near, far]).all():
        raise ValueError('depth clipping planes must satisfy 0 < near < far')
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('OpenGL depth buffer must be finite and in [0,1]')
    return near / (1 - values * (1 - near / far))


def camera_calibration(width, height, fovy_deg, camera_position, camera_rotation):
    """Calibrate top-down upright pixels with integer-valued pixel centres.

    OpenGL window coordinates locate pixel centres at (col+.5,row+.5).
    Hence a symmetric pinhole principal point is ((W-1)/2,(H-1)/2) when
    array indices themselves denote pixel centres. MuJoCo looks along -Z.
    """
    if min(width, height) < 1 or not 0 < fovy_deg < 180:
        raise ValueError('invalid image size or field of view')
    rotation = np.asarray(camera_rotation, float).reshape(3, 3)
    position = np.asarray(camera_position, float).reshape(3)
    if not np.isfinite(position).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-10, rtol=0):
        raise ValueError('camera pose must be a proper rigid transform')
    f = .5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
    k = np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rotation @ np.diag([1., -1., -1.])
    transform[:3, 3] = position
    mirror = np.array([[-1., 0., width - 1.], [0., 1., 0.], [0., 0., 1.]])
    return {'width': int(width), 'height': int(height), 'camera_name': 'agentview',
        'fovy_deg': float(fovy_deg), 'K': k.tolist(),
        'K_archived_mirrored': (mirror @ k).tolist(),
        'T_world_from_camera_cv': transform.tolist(),
        'T_camera_cv_from_world': np.linalg.inv(transform).tolist(),
        'camera_axes': 'x right, y down, z forward; right-handed',
        'pixel_coordinates': 'integer (u=column,v=row) locate pixel centres',
        'upright_from_archived': 'horizontal flip (array[:, ::-1])',
        'upright_from_raw_opengl': 'vertical flip (array[::-1])',
        'depth_convention': 'positive optical-axis z metres, not Euclidean ray range',
        'access': 'evaluation_only; primary reconstruction must estimate its own focal from RGB'}


def backproject(depth_z_m, k):
    depth = np.asarray(depth_z_m, dtype=np.float64)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError('backprojection requires positive finite HxW axis depth')
    k = np.asarray(k, float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or abs(np.linalg.det(k)) < 1e-15:
        raise ValueError('invalid camera intrinsics')
    v, u = np.indices(depth.shape)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(k).T
    return rays * (depth / rays[..., 2])[..., None]


def geometry_labels(worker):
    """Label all visual AND collision geometries by task/robot body ancestry."""
    model = worker.env.sim.model
    core = worker.env.env
    roots = {}
    for name, obj in {**core.objects_dict, **core.fixtures_dict}.items():
        roots[int(model.body_name2id(obj.root_body))] = (name, 'scene')
    for index, robot in enumerate(core.robots):
        roots[int(model.body_name2id(robot.robot_model.root_body))] = (f'robot{index}', 'robot')
        if isinstance(robot.gripper, dict):
            raise ValueError('unexpected multi-arm gripper configuration')
        roots[int(model.body_name2id(robot.gripper.root_body))] = (f'robot{index}_gripper', 'robot')
        # Mount models may not lie within the robot root tree.
        mount = getattr(robot.robot_model, 'mount', None)
        if mount is not None:
            roots[int(model.body_name2id(mount.root_body))] = (f'robot{index}_mount', 'robot')
    records = []
    for gid in range(model.ngeom):
        body_id = int(model.geom_bodyid[gid])
        body = model.body_id2name(body_id) or f'body_{body_id}'
        name = model.geom_id2name(gid) or f'geom_{gid}'
        current, visited = body_id, set()
        owner, role = body, 'scene'
        while current not in visited:
            visited.add(current)
            if current in roots:
                owner, role = roots[current]
                break
            if current == 0:
                break
            current = int(model.body_parentid[current])
        if name.startswith('table_') or body == 'table':
            owner = 'main_table'
        # Explicit fallback for fixed Panda base/stand body names outside root.
        if body.startswith(('robot0_', 'gripper0_', 'mount0_')):
            role = 'robot'
        records.append({'geom_id': gid, 'name': name, 'body': body, 'object': owner,
                        'role': role, 'group': int(model.geom_group[gid]),
                        'rgba': np.asarray(model.geom_rgba[gid], float).tolist(),
                        'geom_type': int(model.geom_type[gid]),
                        'collision_enabled': bool(model.geom_contype[gid] or model.geom_conaffinity[gid])})
    return records


def masks_from_segmentation(segmentation, records, geom_type_id):
    seg = np.asarray(segmentation)
    if seg.ndim != 3 or seg.shape[2] != 2:
        raise ValueError('segmentation must be HxWx2 objtype/objid')
    valid = (seg[..., 0] == geom_type_id) & (seg[..., 1] >= 0) & (seg[..., 1] < len(records))
    geom_id = np.where(valid, seg[..., 1], -1).astype(np.int32)
    object_names = sorted({r['object'] for r in records})
    index = {name: i for i, name in enumerate(object_names)}
    object_lookup = np.array([index[r['object']] for r in records], np.int32)
    nonrobot_lookup = np.array([r['role'] != 'robot' for r in records], bool)
    safe_id = np.maximum(geom_id, 0)
    object_id = np.where(valid, object_lookup[safe_id], -1).astype(np.int32)
    masks = {'geom_id': geom_id, 'object_id': object_id,
        'visible_geometry': valid,
        'nonrobot_visible': valid & nonrobot_lookup[safe_id],
        'target_bowl_visible': valid & (object_id == index.get('akita_black_bowl_1', -2)),
        'target_cabinet_visible': valid & (object_id == index.get('wooden_cabinet_1', -2))}
    return masks, object_names


def pixel_center_ray_depth(sim, calibration):
    """First-hit visual-geometry ray depth at exact pixel centres (no MSAA).

    MuJoCo's default multisample depth resolve need not represent a ray through
    the pixel centre. The image remains the exact original antialiased RGB;
    this separate evaluator reference samples the same geometry at integer
    image centres, using exactly the active rendering geometry groups.
    """
    import mujoco
    model, data = sim.model._model, sim.data._data
    height, width = calibration['height'], calibration['width']
    v, u = np.indices((height, width))
    rays_cv = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3) @ np.linalg.inv(np.asarray(calibration['K'])).T
    norm = np.linalg.norm(rays_cv, axis=1)
    transform = np.asarray(calibration['T_world_from_camera_cv'])
    rays_world = np.ascontiguousarray((rays_cv / norm[:, None]) @ transform[:3, :3].T)
    position = np.ascontiguousarray(transform[:3, 3])
    groups = np.asarray(sim._render_context_offscreen.vopt.geomgroup, np.uint8).copy()
    geom_ids = np.full(len(rays_world), -1, np.int32)
    ranges = np.full(len(rays_world), -1., np.float64)
    # mj_multiRay is deliberately avoided: in the frozen bowl snapshot its
    # broad phase misses robot visual geom 71 for 12/512 checked rays while
    # scalar mj_ray agrees with the raster geom IDs for all 512.
    hit = np.array([-1], np.int32)
    for index, ray in enumerate(rays_world):
        ranges[index] = mujoco.mj_ray(model, data, position, ray, groups, 1, -1, hit)
        geom_ids[index] = hit[0]
    z = (ranges / norm).reshape(height, width)
    valid = (geom_ids.reshape(height, width) >= 0) & (z >= calibration['near_clip_m']) & (z < calibration['far_clip_m'])
    # Use far depth for no-hit pixels; validity is separately and explicitly saved.
    z = np.where(valid, z, calibration['far_clip_m'])
    ids = np.where(valid, geom_ids.reshape(height, width), -1).astype(np.int32)
    return z, ids, valid


def validate_depth_rays(sim, depth, geom_id, calibration, sample_count=512):
    """Independent MuJoCo visual-mesh ray intersections verify K/pose/z-depth.

    Skip silhouette/ID boundaries for raster-versus-analytic comparisons. Ray
    coordinates use exact pixel centres, independently of the depth conversion.
    Error threshold 0.1mm accommodates float32 OpenGL depth quantization at this
    scale, while a half-pixel shift produces millimetre-scale slope errors.
    """
    import mujoco
    model, data = sim.model._model, sim.data._data
    gid = np.asarray(geom_id)
    stable = gid >= 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        stable &= np.roll(gid, (dy, dx), (0, 1)) == gid
    stable[[0, -1], :] = False
    stable[:, [0, -1]] = False
    candidates = np.argwhere(stable)
    rng = np.random.default_rng(44019)
    indices = rng.choice(len(candidates), min(sample_count, len(candidates)), replace=False)
    k = np.asarray(calibration['K'])
    transform = np.asarray(calibration['T_world_from_camera_cv'])
    position = np.ascontiguousarray(transform[:3, 3])
    groups = np.asarray(sim._render_context_offscreen.vopt.geomgroup, np.uint8).copy()
    errors, shifts, mismatch = [], [], []
    for row, col in candidates[indices]:
        ray_cv = np.linalg.solve(k, [col, row, 1.])
        ray_world = np.ascontiguousarray(transform[:3, :3] @ ray_cv)
        # mj_ray parameter is along arbitrary direction, so directly equals Z.
        hit = np.array([-1], np.int32)
        z = mujoco.mj_ray(model, data, position, ray_world, groups, 1, -1, hit)
        if z < 0 or hit[0] != gid[row, col]:
            mismatch.append({'v': int(row), 'u': int(col), 'render_geom': int(gid[row, col]), 'ray_geom': int(hit[0]), 'ray_z_m': float(z)})
            continue
        errors.append(abs(float(depth[row, col]) - z))
        shifted = np.linalg.solve(k, [col - .5, row - .5, 1.])
        hit_shift = np.array([-1], np.int32)
        shifted_z = mujoco.mj_ray(model, data, position, np.ascontiguousarray(transform[:3, :3] @ shifted), groups, 1, -1, hit_shift)
        if shifted_z >= 0 and hit_shift[0] == hit[0]:
            shifts.append(abs(float(depth[row, col]) - shifted_z))
    report = {'method': 'MuJoCo single mj_ray against same visual geom groups; independently projected pixel-centre rays',
        'attempted': len(indices), 'matched_geometry_count': len(errors),
        'mismatches': mismatch, 'max_depth_error_m': max(errors, default=None),
        'median_depth_error_m': float(np.median(errors)) if errors else None,
        'incorrect_half_pixel_shift_median_error_m': float(np.median(shifts)) if shifts else None,
        'tolerance_m': 1e-4, 'passed': bool(len(errors) == len(indices) and len(errors) >= 100 and max(errors) <= 1e-4)}
    return report


def render_snapshot(captures, snapshot, out):
    out.mkdir(parents=True, exist_ok=False)
    evaluator = out / 'evaluator'
    evaluator.mkdir()
    trial = Path(snapshot['source']['trial'])
    configuration = json.loads((trial / 'manifest.json').read_text())['configuration']
    for key, value in configuration['render_environment'].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    runtime = evaluator / 'runtime' / 'gpt_field'
    runtime.mkdir(parents=True)
    sources = [trial / 'manifest.json', trial / 'simulator.json', captures / snapshot['state']['file'],
               captures / snapshot['reference_rgb']['agentview']['file'], captures / 'snapshots.json',
               captures / 'manifest.json', Path(__file__).resolve()]
    sources.extend(sorted((trial / 'frozen_source').glob('*.py')))
    sources.append(trial / 'trace.jsonl')
    before = {str(p): sha256(p) for p in sources}
    state_path = captures / snapshot['state']['file']
    rgb_path = captures / snapshot['reference_rgb']['agentview']['file']
    if sha256(state_path) != snapshot['state']['sha256'] or sha256(rgb_path) != snapshot['reference_rgb']['agentview']['sha256']:
        raise ValueError('frozen capture state/RGB hash mismatch')
    for source in sorted((trial / 'frozen_source').glob('*.py')):
        shutil.copy2(source, runtime / source.name)
    sys.path.insert(0, str(runtime.parent))
    from gpt_field.libero import LiberoConfig, _LiberoWorker
    import mujoco
    import robosuite.macros as macros
    if macros.IMAGE_CONVENTION != 'opengl':
        raise ValueError('frozen image conversion requires raw OpenGL image convention')
    worker = None
    result = {'name': snapshot['name'], 'status': 'running', 'model_calls': 0, 'new_policy_actions': 0, 'archived_raw_actions_replayed': 0,
              'setup_settling_actions': configuration['simulator']['settle_steps'],
              'reference_geometry': 'rendered visual geometries, not collision-box union',
              'source_hashes_before': before}
    write_json(evaluator / 'report.json', result)
    try:
        worker = _LiberoWorker(LiberoConfig(**configuration['simulator']))
        original = json.loads((trial / 'simulator.json').read_text())['metadata']
        for key in ('packages', 'controller_sources', 'libero_source', 'controller'):
            if worker.metadata[key] != original[key]:
                raise ValueError(f'frozen simulator implementation mismatch: {key}')
        # Observable sampling occurs before the control endpoint. Restoring the
        # endpoint qpos alone does NOT restore the cached observation image.
        # Intercept original render calls while replaying only archived actions.
        # The exact original RGB return value remains unchanged; no dynamics or
        # controller method is patched or replaced.
        from robosuite.utils.binding_utils import MjSim
        expected = np.asarray(Image.open(rgb_path).convert('RGB'))
        size = configuration['simulator']['render_size']
        if size != configuration['simulator']['image_size']:
            raise ValueError('exact-reference preparation requires no image resize/padding')
        original_render = MjSim.render
        captured = {}
        capture_count = 0

        def render_with_reference(sim, *args, **kwargs):
            nonlocal capture_count
            if kwargs.get('camera_name') != 'agentview' or kwargs.get('segmentation', False) or kwargs.get('depth', False):
                return original_render(sim, *args, **kwargs)
            rgb, buffer = original_render(sim, *args, **{**kwargs, 'depth': True})
            if np.array_equal(np.asarray(rgb)[::-1, ::-1], expected):
                m, d = sim.model._model, sim.data._data
                cid = sim.model.camera_name2id('agentview')
                cal = camera_calibration(size, size, float(m.cam_fovy[cid]), d.cam_xpos[cid], d.cam_xmat[cid])
                near, far = float(m.vis.map.znear * m.stat.extent), float(m.vis.map.zfar * m.stat.extent)
                cal.update(near_clip_m=near, far_clip_m=far,
                           raw_opengl_depth_conversion='z=near/(1-buffer*(1-near/far))')
                raster_z = depth_buffer_to_z(np.asarray(buffer)[::-1], near, far)
                seg = original_render(sim, width=size, height=size, camera_name='agentview', segmentation=True)
                geom_id = np.where(np.asarray(seg)[::-1, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM), np.asarray(seg)[::-1, :, 1], -1)
                raster_check = validate_depth_rays(sim, raster_z, geom_id, cal)
                z, ray_geom_id, ray_valid = pixel_center_ray_depth(sim, cal)
                ray_check = validate_depth_rays(sim, z, ray_geom_id, cal)
                cal['reference_sampling'] = 'first visual-geometry intersection at exact pixel centre; MSAA-independent scalar mj_ray'
                ray_seg = np.stack([np.where(ray_valid, int(mujoco.mjtObj.mjOBJ_GEOM), -1), ray_geom_id], axis=-1)
                captured.update(rgb=np.asarray(rgb).copy(), depth_buffer=np.asarray(buffer).copy(),
                    segmentation=ray_seg[::-1].copy(), raster_segmentation=np.asarray(seg).copy(),
                    calibration=cal, depth=z, raster_depth=raster_z, ray_validation=ray_check,
                    raster_ray_validation=raster_check, raster_offsamples=int(m.vis.quality.offsamples),
                    sim_time=float(d.time), qpos=d.qpos.copy(), qvel=d.qvel.copy(),
                    geom_xpos=d.geom_xpos.copy(), geom_xmat=d.geom_xmat.copy(),
                    cam_xpos=d.cam_xpos.copy(), cam_xmat=d.cam_xmat.copy())
                capture_count += 1
            return rgb

        MjSim.render = render_with_reference
        try:
            response = worker.reset(configuration['task_id'], configuration['episode_index'])
            target_steps = int(snapshot['state']['pre_action_step'])
            pending = None
            replayed = 0
            if target_steps:
                with (trial / 'trace.jsonl').open() as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row.get('event') == 'reset':
                            if response['observation'] != row['observation']:
                                raise ValueError('reset observation does not match original trace')
                        elif row.get('event') == 'action':
                            if row['step'] != replayed or pending is not None:
                                raise ValueError('archived action order mismatch')
                            if response['observation']['observation/state'] != row['state']:
                                raise ValueError(f'pre-action state mismatch at step {replayed}')
                            response = worker.step(row['action'])
                            pending = row['step']
                        elif row.get('event') == 'executed':
                            if pending != row['step'] or response['observation'] != row['observation']:
                                raise ValueError(f'executed observation mismatch at step {replayed}')
                            replayed += 1
                            pending = None
                            if replayed % 100 == 0:
                                print(f'REPLAY {snapshot["name"]}: {replayed}/{target_steps} observations exact', flush=True)
                            if replayed == target_steps:
                                break
                if replayed != target_steps:
                    raise ValueError('trace ended before requested image')
            result['archived_raw_actions_replayed'] = replayed
            result['replay_observations_exact'] = True
        finally:
            MjSim.render = original_render
        if not captured:
            raise ValueError('no original camera sample matched archived target RGB')
        model, data = worker.env.sim.model._model, worker.env.sim.data._data
        with np.load(state_path, allow_pickle=False) as state:
            position_error = float(np.max(np.abs(data.geom_xpos - state['geom_xpos'])))
            rotation_error = float(np.max(np.abs(data.geom_xmat - state['geom_xmat'])))
            result['state_match'] = {'max_geom_position_error_m': position_error,
                'max_geom_rotation_element_error': rotation_error,
                'qpos_exact': bool(np.array_equal(data.qpos, state['qpos'])),
                'qvel_exact': bool(np.array_equal(data.qvel, state['qvel'])),
                'note': 'endpoint matches prior capture; reference depth uses earlier exact cached image instant',
                'camera_sample_time_s': captured['sim_time'], 'endpoint_time_s': float(data.time)}
        if position_error > 2e-12 or rotation_error > 2e-12 or not result['state_match']['qpos_exact'] or not result['state_match']['qvel_exact']:
            raise ValueError('replayed endpoint does not match archived state')
        rgb_raw, depth_raw, seg_raw = captured['rgb'], captured['depth_buffer'], captured['segmentation']
        calibration, depth, checks = captured['calibration'], captured['depth'], captured['ray_validation']
        actual = rgb_raw[::-1, ::-1]
        result['rgb_match'] = {'pixels_exact': bool(np.array_equal(actual, expected)),
            'differing_pixels': int(np.any(actual != expected, axis=-1).sum()),
            'max_channel_error': int(np.abs(actual.astype(np.int16) - expected.astype(np.int16)).max()),
            'archived_sha256': sha256(rgb_path), 'matching_camera_sample_count': capture_count}
        Image.fromarray(np.ascontiguousarray(actual)).save(evaluator / 'rerendered_archived.png')
        if not result['rgb_match']['pixels_exact']:
            raise ValueError('captured RGB no longer matches the original target')
        np.savez_compressed(evaluator / 'camera_sample_state.npz', **{key: captured[key] for key in
            ('qpos', 'qvel', 'geom_xpos', 'geom_xmat', 'cam_xpos', 'cam_xmat', 'sim_time')})
        upright = np.ascontiguousarray(expected[:, ::-1])
        input_path = out / 'agentview_upright.png'
        Image.fromarray(upright).save(input_path)
        shutil.copy2(rgb_path, evaluator / 'archived_agentview.png')
        records = geometry_labels(worker)
        masks, objects = masks_from_segmentation(np.asarray(seg_raw)[::-1], records, int(mujoco.mjtObj.mjOBJ_GEOM))
        valid = masks['visible_geometry'] & np.isfinite(depth) & (depth > 0)
        masks['valid_depth'] = valid
        for key in ('nonrobot_visible', 'target_bowl_visible', 'target_cabinet_visible'):
            masks[key] &= valid
        if not masks['target_bowl_visible'].any() or ('drawer' in snapshot['name'] and not masks['target_cabinet_visible'].any()):
            raise ValueError('required target has no visible segmentation pixels')
        write_json(evaluator / 'ray_validation.json', checks)
        result['ray_validation'] = checks
        result['raster_ray_validation_diagnostic'] = captured['raster_ray_validation']
        result['raster_offsamples'] = captured['raster_offsamples']
        result['reference_depth_method'] = 'scalar mj_ray first visible visual-geom hit at exact pixel centre; not resolved MSAA raster depth'
        if not checks['passed']:
            raise ValueError('independent camera/depth ray validation failed')
        reference_path = evaluator / 'reference.npz'
        np.savez_compressed(reference_path, depth_z_m=depth.astype(np.float32),
            raster_depth_z_m=captured['raster_depth'].astype(np.float32),
            raster_segmentation=captured['raster_segmentation'][::-1],
            depth_buffer=np.asarray(depth_raw)[::-1], **masks)
        calibration_path = evaluator / 'calibration.json'
        write_json(calibration_path, calibration)
        write_json(evaluator / 'geometry_labels.json', {'object_id_to_name': objects, 'geometries': records,
            'note': 'labels identify first visible visual geom; occluded surfaces are not ground-truth completed'})
        write_json(evaluator / 'simulator_metadata.json', worker.metadata)
        (evaluator / 'render_model.xml').write_text(worker.env.sim.model.get_xml())
        result.update(status='complete', input_rgb={'file': str(input_path.relative_to(out)),
            'sha256': sha256(input_path), 'shape': list(upright.shape), 'orientation': 'upright_cv',
            'transformation': 'horizontal flip of exact archived agentview PNG; no resizing'},
            calibration={'file': str(calibration_path.relative_to(out)), 'sha256': sha256(calibration_path), 'access': 'evaluation_only'},
            evaluator_reference={'file': str(reference_path.relative_to(out)), 'sha256': sha256(reference_path),
                'geometry': 'rendered_visual_geometries', 'orientation': 'upright_cv', 'access': 'evaluation_only',
                'depth_method': result['reference_depth_method']},
            object_id_to_name=objects, mask_pixel_counts={k: int(v.sum()) for k, v in masks.items() if v.dtype == bool},
            render_geom_groups=np.asarray(worker.env.sim._render_context_offscreen.vopt.geomgroup).tolist(),
            bounds_note='depth is first visible surface only; no hidden scene completion is asserted',
            source_hashes_after={str(p): sha256(p) for p in sources})
        result['sources_unchanged'] = result['source_hashes_before'] == result['source_hashes_after']
        if not result['sources_unchanged']:
            raise ValueError('source changed during reference preparation')
        copied = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'report.json'}
        result['artifact_sha256'] = copied
        write_json(evaluator / 'report.json', result)
        return result
    except Exception:
        result.update(status='failed', error=traceback.format_exc())
        write_json(evaluator / 'report.json', result)
        raise
    finally:
        if worker is not None:
            worker.close()


def run(captures, output):
    captures, output = captures.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'capture_directory': str(captures), 'capture_snapshots_sha256': sha256(captures / 'snapshots.json'),
        'script_sha256': sha256(__file__), 'primary_input': 'single upright RGB only; no simulator depth/masks/calibration to inference',
        'reference': 'first-visible visual surfaces; separate from prior collision geometry SDF',
        'snapshots': []}
    start = time.monotonic()
    shutil.copy2(__file__, output / 'prepare_reference_used.py')
    write_json(output / 'manifest.json', manifest)
    try:
        source = json.loads((captures / 'snapshots.json').read_text())
        capture_manifest = json.loads((captures / 'manifest.json').read_text())
        if capture_manifest['status'] != 'complete' or sha256(captures / 'snapshots.json') != capture_manifest['snapshots_sha256']:
            raise ValueError('source capture integrity/status failed')
        for snapshot in source['snapshots']:
            name = snapshot['name']
            child = output / 'snapshots' / name
            child.parent.mkdir(exist_ok=True)
            with (output / f'{name}.log').open('w') as log:
                process = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--captures', str(captures),
                    '--output', str(child), '--snapshot', name], stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f'{name} failed; see {name}.log and snapshot evaluator/report.json')
            report = json.loads((child / 'evaluator' / 'report.json').read_text())
            if report['status'] != 'complete' or not report['sources_unchanged']:
                raise ValueError('child did not pass reference gates')
            item = {key: report[key] for key in ('name', 'status', 'input_rgb', 'calibration', 'evaluator_reference',
                'rgb_match', 'state_match', 'mask_pixel_counts', 'sources_unchanged')}
            for key in ('input_rgb', 'calibration', 'evaluator_reference'):
                item[key]['file'] = str((child / item[key]['file']).relative_to(output))
            item['report'] = str((child / 'evaluator' / 'report.json').relative_to(output))
            manifest['snapshots'].append(item)
            write_json(output / 'manifest.json', manifest)
            print(f'REFERENCE {name} exact RGB; rays={report["ray_validation"]["matched_geometry_count"]} max_error_m={report["ray_validation"]["max_depth_error_m"]:.3g}', flush=True)
        manifest.update(status='complete', elapsed_s=time.monotonic() - start,
            script_unchanged=sha256(__file__) == manifest['script_sha256'],
            capture_unchanged=sha256(captures / 'snapshots.json') == manifest['capture_snapshots_sha256'])
        manifest['artifacts_unchanged'] = {}
        for item in manifest['snapshots']:
            folder = output / 'snapshots' / item['name']
            report = json.loads((output / item['report']).read_text())
            manifest['artifacts_unchanged'][item['name']] = all(sha256(folder / p) == digest for p, digest in report['artifact_sha256'].items())
        if not manifest['script_unchanged'] or not manifest['capture_unchanged'] or not all(manifest['artifacts_unchanged'].values()):
            raise ValueError('final reference integrity check failed')
        write_json(output / 'rgb_inputs.json', {'schema_version': 1, 'inputs': [
            {'name': item['name'], 'rgb': item['input_rgb']['file'], 'sha256': item['input_rgb']['sha256']}
            for item in manifest['snapshots']]})
        write_json(output / 'manifest.json', manifest)
        return manifest
    except Exception:
        manifest.update(status='failed', error=traceback.format_exc(), elapsed_s=time.monotonic() - start)
        write_json(output / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.snapshot:
        snapshots = json.loads((args.captures / 'snapshots.json').read_text())['snapshots']
        selected = [s for s in snapshots if s['name'] == args.snapshot]
        if len(selected) != 1:
            raise ValueError('snapshot name must be unique')
        render_snapshot(args.captures.resolve(), selected[0], args.output.resolve())
    else:
        run(args.captures, args.output)


if __name__ == '__main__':
    main()
