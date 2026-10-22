"""Explicit v2 environment maps built from frozen predictions and robot knowledge.

Astra: original CSG with robot-named top-union components removed, queried
analytically inside its original saved-grid domain. All other components remain.
DepthPro: original predicted depth/intrinsics, visibility-prefix free space,
self-image hypotheses from calibrated robot projection, and a separately marked
initial robot material prior. Unknown space is never globally made free.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

SNAPSHOTS = ('drawer003_closed_step60', 'drawer003_open_step552')
SIGN_EPS_M = 1e-9
DEPTH_MARGIN_M = .020
DEPTH_JUMP_ABS_M = .050
DEPTH_JUMP_REL = .050


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class QueryResult:
    state: np.ndarray
    support: np.ndarray
    raw_phi_m: np.ndarray
    conservative_clearance_m: np.ndarray
    phi_lower_m: np.ndarray
    phi_upper_m: np.ndarray
    known_initial_self: np.ndarray
    raw_predicted_state: np.ndarray
    self_image_associated: np.ndarray
    source_code: np.ndarray  # 0 unknown,1 Astra model,2 DepthPro prediction,3 SELF prior

    @classmethod
    def empty(cls, n):
        return cls(np.zeros(n, np.int8), np.zeros(n, bool), np.full(n, np.nan),
                   np.zeros(n), np.full(n, np.nan), np.full(n, np.nan),
                   np.zeros(n, bool), np.zeros(n, np.int8), np.zeros(n, bool), np.zeros(n, np.int8))

    def summary(self):
        own = self.known_initial_self
        return {'points': len(self.state), 'free': int(np.sum(self.state == 1)),
                'occupied': int(np.sum(self.state == -1)), 'unknown': int(np.sum(self.state == 0)),
                'supported': int(self.support.sum()),
                'supported_ambiguous': int(np.sum(self.support & (self.state == 0))),
                'known_initial_self_points': int(own.sum()),
                'novel_region_points': int((~own).sum()),
                'initial_self_overrides_raw_unknown': int(np.sum(own & (self.raw_predicted_state == 0))),
                'initial_self_overrides_raw_occupied_band': int(np.sum(own & (self.raw_predicted_state == -1))),
                'initial_self_already_raw_free': int(np.sum(own & (self.raw_predicted_state == 1))),
                'self_image_associated_points': int(self.self_image_associated.sum())}


def _points(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('expected camera-CV points of shape [N,3]')
    return points


def flatten_union_components(spec, root=None):
    root = spec['root'] if root is None else root
    node = spec['nodes'][root]
    if node['op'] == 'union':
        return flatten_union_components(spec, node['left']) + flatten_union_components(spec, node['right'])
    return [root]


def environment_csg(original):
    """Remove only robot_* top-level union operands; keep all other expressions."""
    components = flatten_union_components(original)
    removed = [i for i in components if original['nodes'][i]['name'].startswith('robot_')]
    retained = [i for i in components if i not in removed]
    if not retained:
        raise ValueError('no non-robot components remain; refusing an empty environment map')
    reachable = set()
    def visit(index):
        if index in reachable:
            return
        reachable.add(index)
        node = original['nodes'][index]
        if node['left'] is not None:
            visit(node['left']); visit(node['right'])
    for index in retained:
        visit(index)
    order = sorted(reachable)
    mapping = {old: new for new, old in enumerate(order)}
    result = {k: copy.deepcopy(v) for k, v in original.items() if k not in ('nodes', 'root', '_validation')}
    result['nodes'] = []
    for index in order:
        node = copy.deepcopy(original['nodes'][index])
        if node['left'] is not None:
            node['left'], node['right'] = mapping[node['left']], mapping[node['right']]
        result['nodes'].append(node)
    root = mapping[retained[0]]
    for index in retained[1:]:
        new = {'name': f'v2_environment_union_{len(result["nodes"])}', 'op': 'union',
               'center_m': None, 'rotation_xyzw': None, 'size_m': None,
               'left': root, 'right': mapping[index]}
        result['nodes'].append(new)
        root = len(result['nodes']) - 1
    result['root'] = root
    details = {'selection_rule': 'original top-union component name starts with robot_',
               'removed_components': [{'original_node_index': i, 'name': original['nodes'][i]['name']} for i in removed],
               'retained_components': [{'original_node_index': i, 'name': original['nodes'][i]['name']} for i in retained],
               'retained_nodes': [{'original_node_index': i, 'new_node_index': mapping[i]} for i in order],
               'all_nonrobot_components_preserved': True,
               'semantic_labels_verified_against_ground_truth': False}
    return result, details


def depth_triangle_validity(depth):
    d = np.asarray(depth, float)
    if d.ndim != 2 or min(d.shape) < 2:
        raise ValueError('depth must be a 2D image')
    a, b, c, e = d[:-1, :-1], d[:-1, 1:], d[1:, 1:], d[1:, :-1]
    triangles = np.stack((np.stack((a, b, c), axis=-1), np.stack((a, c, e), axis=-1)), axis=-2)
    minimum, maximum = triangles.min(axis=-1), triangles.max(axis=-1)
    return np.all(np.isfinite(triangles) & (triangles > 0), axis=-1) & (maximum - minimum <= DEPTH_JUMP_ABS_M + DEPTH_JUMP_REL * minimum)


def projective_depth_samples(depth, K, points, triangle_valid):
    """Same perspective-correct visible-triangle interpolation as original field."""
    p = _points(points)
    z = p[:, 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        u = K[0, 0] * p[:, 0] / z + K[0, 2]
        v = K[1, 1] * p[:, 1] / z + K[1, 2]
    h, w = depth.shape
    valid = np.isfinite(p).all(axis=1) & (z > 0) & np.isfinite(u) & np.isfinite(v)
    valid &= (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    uu, vv = np.where(valid, u, 0.), np.where(valid, v, 0.)
    ix, iy = np.minimum(np.floor(uu).astype(int), w - 2), np.minimum(np.floor(vv).astype(int), h - 2)
    du, dv = uu - ix, vv - iy
    lower = du >= dv
    ti = np.where(lower, 0, 1)
    valid &= triangle_valid[iy, ix, ti]
    d00, d10, d11, d01 = depth[iy, ix], depth[iy, ix + 1], depth[iy + 1, ix + 1], depth[iy + 1, ix]
    with np.errstate(invalid='ignore', divide='ignore'):
        inv = np.where(lower, (1-du)/d00+(du-dv)/d10+dv/d11,
                       (1-dv)/d00+du/d11+(dv-du)/d01)
        values = 1. / inv
    valid &= np.isfinite(values) & (values > 0)
    return np.where(valid, values, np.nan), valid, (ix, iy, ti)


class _Adapter:
    def __init__(self, snapshot, project_root=None, output=None):
        if snapshot not in SNAPSHOTS:
            raise ValueError('only drawer003 snapshots are allowed')
        self.snapshot = snapshot
        self.project = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[2]
        self.output = Path(output).resolve() if output is not None else None
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
        self.consumed = {}
        self.artifacts = {}
        self.query_calls = 0
        self.query_points = 0
        self.last_diagnostics = {}

    def checked(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError('input hash mismatch: ' + str(path))
        if str(path) in self.consumed and self.consumed[str(path)] != actual:
            raise ValueError('input changed: ' + str(path))
        self.consumed[str(path)] = actual
        return path

    def write_json(self, name, value):
        if self.output is None:
            return
        path = self.output / name
        with path.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        self.artifacts[str(path)] = sha256(path)

    def provenance(self, verify=False):
        unchanged = {p: sha256(p) == h for p, h in self.consumed.items()} if verify else None
        if unchanged is not None and not all(unchanged.values()):
            raise ValueError('planning-map input changed')
        return {'kind': self.kind, 'snapshot': self.snapshot, 'query_calls': self.query_calls,
                'query_points': self.query_points, 'consumed_sha256': dict(self.consumed),
                'inputs_unchanged': unchanged, 'derived_artifact_sha256': dict(self.artifacts),
                'source_codes': {'unknown': 0, 'astra_environment_model': 1, 'depthpro_prediction': 2, 'initial_robot_material_prior': 3},
                'ground_truth_environment_inputs': [], 'settings': self.settings,
                'changes_from_previous_test': self.changes}


class AstraEnvironmentField(_Adapter):
    kind = 'astra'

    def __init__(self, snapshot, project_root=None, output=None):
        super().__init__(snapshot, project_root, output)
        directory = self.project / 'scene_test/astra_scene/runs/fields_001'
        manifest = json.loads(self.checked(directory / 'manifest.json').read_text())
        if manifest.get('status') != 'complete':
            raise ValueError('requires completed original Astra run')
        rows = [r for r in manifest['snapshots'] if r['name'] == snapshot]
        configs = [r for r in manifest['configurations'] if r['name'] == snapshot + '_h2mm']
        if len(rows) != 1 or len(configs) != 1:
            raise ValueError('expected unique original snapshot and 2 mm domain')
        record = rows[0]['field']
        self.original = json.loads(self.checked(directory / record['file'], record['sha256']).read_text())
        source = next(r for r in manifest['sources'] if r['path'].endswith('/field_csg.py'))
        core_path = self.checked(directory / source['copy'], source['copy_sha256'])
        self.core = load_module(core_path, '_v2_original_csg_' + snapshot)
        env, self.selection = environment_csg(self.original)
        self.environment = self.core.validate_field(env)
        allocation = configs[0]['allocation']
        self.lower = np.asarray(allocation['grid_lower_m'], float)
        self.upper = np.asarray(allocation['grid_upper_m'], float)
        self.settings = {'representation': 'analytic original CSG environment components',
                         'domain_lower_m': self.lower.tolist(), 'domain_upper_m': self.upper.tolist(),
                         'sign_epsilon_m': SIGN_EPS_M, 'initial_robot_material_prior': False,
                         'clearance': '1-Lipschitz implicit zero-set and domain-boundary lower bound, not physical guarantee'}
        self.changes = ['Remove only model-named robot_* top-union components.',
                        'Use exact original analytic environment field instead of sampled 2 mm interpolation.',
                        'Keep original saved-grid query bounds; outside remains unknown.']
        self.write_json(snapshot + '_environment_csg.json', self.environment)
        self.write_json(snapshot + '_component_selection.json', self.selection)

    def query(self, points_camera):
        points = _points(points_camera)
        self.query_calls += 1; self.query_points += len(points)
        r = QueryResult.empty(len(points))
        valid = np.isfinite(points).all(axis=1) & np.all((points >= self.lower) & (points <= self.upper), axis=1)
        ids = np.flatnonzero(valid)
        if not len(ids):
            self.last_diagnostics = {'model_free': 0, 'ray_inferred_free': 0, 'self_prior_free': 0, 'outside_domain_or_nonfinite': len(points), 'near_zero_unknown': 0}
            return r
        phi = self.core.evaluate_field(self.environment, points[ids])
        r.raw_phi_m[ids] = phi
        r.phi_lower_m[ids], r.phi_upper_m[ids] = phi-SIGN_EPS_M, phi+SIGN_EPS_M
        r.support[ids] = True
        r.source_code[ids] = 1
        free, occupied = phi > SIGN_EPS_M, phi < -SIGN_EPS_M
        r.state[ids[free]], r.state[ids[occupied]] = 1, -1
        margin = np.minimum(points[ids]-self.lower, self.upper-points[ids]).min(axis=1)
        r.conservative_clearance_m[ids[free]] = np.maximum(0., np.minimum(phi[free]-SIGN_EPS_M, margin[free]))
        r.raw_predicted_state[:] = r.state
        self.last_diagnostics = {'model_free': int((r.state == 1).sum()), 'ray_inferred_free': 0, 'self_prior_free': 0,
                                 'model_occupied': int((r.state == -1).sum()), 'outside_domain_or_nonfinite': int((~valid).sum()),
                                 'near_zero_unknown': int(np.sum(valid & (r.state == 0))), 'removed_robot_named_components': len(self.selection['removed_components'])}
        return r

    def provenance(self, verify=False):
        return {**super().provenance(verify), 'component_selection': self.selection}


class DepthProVisibilityField(_Adapter):
    kind = 'depthpro'

    def __init__(self, snapshot, robot, project_root=None, output=None, self_geometry=None):
        super().__init__(snapshot, project_root, output)
        directory = self.project / 'scene_test/rgb_scene/predictions_001'
        manifest = json.loads(self.checked(directory / 'manifest.json').read_text())
        if manifest.get('status') != 'complete':
            raise ValueError('requires completed original DepthPro predictions')
        rows = [r for r in manifest['predictions'] if r['name'] == snapshot]
        if len(rows) != 1:
            raise ValueError('expected unique original DepthPro snapshot')
        record = rows[0]
        with np.load(self.checked(directory / record['file'], record['sha256']), allow_pickle=False) as archive:
            self.depth = archive['depth_m'].astype(float)
            self.K_pred = archive['K_pred'].astype(float)
        if not np.array_equal(self.K_pred, np.asarray(record['K_pred'], float)):
            raise ValueError('prediction intrinsics differ from their manifest')
        self.triangle_valid = depth_triangle_validity(self.depth)
        calibration_path = Path(__file__).resolve().parent / 'preparation_001' / snapshot / 'rig_calibration.json'
        calibration = json.loads(self.checked(calibration_path).read_text())
        self.K_rig = np.asarray(calibration['K'], float)
        if (calibration['height'], calibration['width']) != self.depth.shape:
            raise ValueError('known rig calibration shape differs from input RGB')
        if not np.allclose(np.asarray(calibration['T_camera_cv_from_world']), robot.camera_from_world, atol=1e-12, rtol=0.):
            raise ValueError('known rig extrinsics differ from robot package')
        if self_geometry is None:
            from self_geometry import InitialRobotSelfGeometry
            self_geometry = InitialRobotSelfGeometry(robot, rig_calibration_path=calibration_path)
        self.self_geometry = self_geometry
        rig_image = self.self_geometry.ray_image(self.K_rig, self.depth.shape)
        self.rig_front = np.asarray(rig_image['front_depth_m'], float)
        self.rig_exit = np.asarray(rig_image['exit_depth_m'], float)
        self.self_association_image = np.isfinite(self.rig_front) & np.isfinite(self.rig_exit)
        self.self_association_image &= (self.depth >= self.rig_front-DEPTH_MARGIN_M) & (self.depth <= self.rig_exit+DEPTH_MARGIN_M)
        self.settings = {
            'representation': 'original projective predicted depth plus explicit visibility and initial SELF prior',
            'predicted_geometry_K': self.K_pred.tolist(), 'self_image_rig_K': self.K_rig.tolist(),
            'depth_margin_m': DEPTH_MARGIN_M, 'occupied_band_half_width_m': DEPTH_MARGIN_M,
            'depth_triangle_jump_abs_m': DEPTH_JUMP_ABS_M, 'depth_triangle_jump_rel': DEPTH_JUMP_REL,
            'calibration_role': 'known rig K is used only for pixel self association; predicted geometry retains original K_pred',
            'occlusion_rule': 'query physical ray p/z uses exact initial robot geometry; never substitute a true-K ray at the same pixel',
            'known_initial_material_assumption': 'initial native robot material is environment-free by explicit noninterpenetration/calibration prior',
            'self_prior_geometry_dilation_m': 0.,
            'clearance': 'zero conservative Euclidean bound; projective residual is not ESDF',
            'override_conflict_reference': 'raw unmasked DepthPro ray state before SELF prior and robot-occlusion masking, not old sparse TSDF query',
        }
        self.changes = ['Extend free evidence from near-surface saved blocks to visible ray prefixes of the original prediction.',
                        'Associate possible robot pixels using separately supplied known-rig calibration and exact initial native geometry.',
                        'Keep self-associated surface band and occluded/out-of-view novel space unknown.',
                        'Mark exact initial robot material as SELF prior, with separate conflict and override counters.']
        if self.output is not None:
            path = self.output / (snapshot + '_self_projection.npz')
            with path.open('xb') as stream:
                np.savez_compressed(stream, K_rig=self.K_rig, K_pred=self.K_pred,
                                    known_robot_front_m=self.rig_front, known_robot_exit_m=self.rig_exit,
                                    self_association_image=self.self_association_image)
            self.artifacts[str(path)] = sha256(path)
        self.write_json(snapshot + '_visibility_settings.json', self.settings)
        # Register helper-consumed known robot assets/calibration for runner audits.
        helper_provenance = self.self_geometry.provenance(verify=True)
        self.consumed.update(helper_provenance.get('consumed_sha256', {}))

    def query(self, points_camera):
        points = _points(points_camera)
        self.query_calls += 1; self.query_points += len(points)
        r = QueryResult.empty(len(points))
        finite = np.isfinite(points).all(axis=1)
        own = np.zeros(len(points), bool)
        own[finite] = self.self_geometry.contains(points[finite])
        d, valid, (ix, iy, ti) = projective_depth_samples(self.depth, self.K_pred, points, self.triangle_valid)
        residual = d-points[:, 2]
        r.raw_phi_m[valid] = residual[valid]
        r.phi_lower_m[valid], r.phi_upper_m[valid] = residual[valid], residual[valid]
        raw_free = valid & (residual > DEPTH_MARGIN_M+SIGN_EPS_M)
        raw_band = valid & (np.abs(residual) <= DEPTH_MARGIN_M)
        r.raw_predicted_state[raw_free], r.raw_predicted_state[raw_band] = 1, -1
        # Any self-associated source triangle vertex makes its surface evidence
        # ambiguous. This only removes obstacle evidence to UNKNOWN, never FREE.
        mask = self.self_association_image
        associated = np.where(ti == 0, mask[iy, ix] | mask[iy, ix+1] | mask[iy+1, ix+1],
                              mask[iy, ix] | mask[iy+1, ix+1] | mask[iy+1, ix]) & valid
        r.self_image_associated[:] = associated
        novel = valid & ~own
        ids = np.flatnonzero(novel)
        occluded_by_robot = np.zeros(len(points), bool)
        if len(ids):
            directions = points[ids] / points[ids, 2:3]
            hits = self.self_geometry.ray_intervals(directions)
            front = np.asarray(hits['front_t'], float)
            effective_surface = np.minimum(d[ids], front)
            free = points[ids, 2] < effective_surface-DEPTH_MARGIN_M-SIGN_EPS_M
            before_robot = points[ids, 2] < front-DEPTH_MARGIN_M
            occluded_by_robot[ids] = ~before_robot
            occupied = raw_band[ids] & ~associated[ids] & before_robot
            r.state[ids[free]], r.state[ids[occupied]] = 1, -1
            # Support denotes usable visible prediction, not physically verified free space.
            r.support[ids[free | occupied]] = True
            r.source_code[ids[free | occupied]] = 2
        # Explicit prior is never inferred from camera unknowns or a clearance shell.
        r.known_initial_self[:] = own
        r.state[own], r.support[own], r.source_code[own] = 1, True, 3
        unknown = r.state == 0
        unknown_no_depth = unknown & ~valid
        unknown_robot = unknown & valid & occluded_by_robot
        unknown_self_band = unknown & valid & ~occluded_by_robot & associated & raw_band
        unknown_behind_depth = unknown & valid & ~occluded_by_robot & ~unknown_self_band & (residual < -DEPTH_MARGIN_M)
        unknown_other = unknown & ~unknown_no_depth & ~unknown_robot & ~unknown_self_band & ~unknown_behind_depth
        self.last_diagnostics = {
            'model_free': 0, 'ray_inferred_free': int(np.sum((r.state == 1) & (r.source_code == 2))),
            'self_prior_free': int(own.sum()), 'predicted_occupied': int((r.state == -1).sum()),
            'initial_self_prior_raw_occupied_conflicts': int(np.sum(own & (r.raw_predicted_state == -1))),
            'initial_self_prior_overrides_raw_unknown': int(np.sum(own & (r.raw_predicted_state == 0))),
            'initial_self_prior_already_raw_free': int(np.sum(own & (r.raw_predicted_state == 1))),
            'unknown_out_of_view_or_invalid_depth': int(unknown_no_depth.sum()),
            'unknown_robot_occluded': int(unknown_robot.sum()),
            'unknown_self_associated_surface_band': int(unknown_self_band.sum()),
            'unknown_behind_predicted_surface': int(unknown_behind_depth.sum()),
            'unknown_boundary_or_other': int(unknown_other.sum()),
            'self_image_associated_query_points': int(associated.sum()),
            'initial_self_prior_is_assumption_not_camera_observation': True,
            'raw_conflict_definition': self.settings['override_conflict_reference'],
        }
        return r

    def provenance(self, verify=False):
        return {**super().provenance(verify), 'known_robot_geometry': self.self_geometry.provenance(verify=verify),
                'self_associated_image_pixels': int(self.self_association_image.sum()),
                'self_mask_is_verified_segmentation': False}


def load_planning_field(kind, snapshot, robot=None, project_root=None, output=None):
    if kind == 'astra':
        return AstraEnvironmentField(snapshot, project_root, output)
    if kind == 'depthpro':
        if robot is None:
            raise ValueError('DepthPro planning map requires the known initial native robot')
        return DepthProVisibilityField(snapshot, robot, project_root, output)
    raise ValueError('kind must be astra or depthpro')
