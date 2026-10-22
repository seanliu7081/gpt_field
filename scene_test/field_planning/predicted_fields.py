"""Read-only, saved-field collision queries for the two drawer003 RGB snapshots.

No model, geometry renderer, CSG evaluator, reference depth, or GT geometry is
used here. Positive means predicted free, not physically verified free space.
Unknown and numerical sign ambiguity always stay unknown. The adapters do not
remove the robot depicted in either source image.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

SNAPSHOTS = ('drawer003_closed_step60', 'drawer003_open_step552')
SIGN_EPS_M = 1e-9
CORNERS = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)])


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass
class QueryResult:
    state: np.ndarray  # -1 occupied, 0 unknown/ambiguous, +1 free
    support: np.ndarray  # a saved observation/certificate supports this point
    raw_phi_m: np.ndarray  # interpolated samples; NaN for certificate-only points
    conservative_clearance_m: np.ndarray  # free-space lower bound, else zero
    phi_lower_m: np.ndarray
    phi_upper_m: np.ndarray

    @classmethod
    def empty(cls, n):
        return cls(np.zeros(n, np.int8), np.zeros(n, bool), np.full(n, np.nan),
                   np.zeros(n), np.full(n, np.nan), np.full(n, np.nan))

    def summary(self):
        return {'points': len(self.state), 'free': int(np.sum(self.state == 1)),
                'occupied': int(np.sum(self.state == -1)), 'unknown': int(np.sum(self.state == 0)),
                'supported': int(self.support.sum()),
                'supported_ambiguous': int(np.sum(self.support & (self.state == 0)))}


def _points(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('points must have shape [N,3] in camera-CV metres')
    return p


def _cell_samples(values, lower, spacing, points):
    """Return 8 corners, local fraction, cell index and distance to cell faces."""
    ijk = (points - lower) / spacing
    base = np.minimum(np.floor(ijk).astype(np.int64), np.array(values.shape) - 2)
    base = np.maximum(base, 0)
    fraction = np.clip(ijk - base, 0., 1.)
    indices = base[:, None, :] + CORNERS[None, :, :]
    corners = values[indices[..., 0], indices[..., 1], indices[..., 2]].astype(float)
    weights = np.prod(np.where(CORNERS[None, :, :], fraction[:, None, :], 1. - fraction[:, None, :]), axis=2)
    interpolated = np.sum(corners * weights, axis=1)
    distance_to_face = spacing * np.minimum(fraction, 1. - fraction).min(axis=1)
    return corners, fraction, base, interpolated, distance_to_face, indices


class _SavedAdapter:
    def __init__(self, directory, snapshot, spacing_m=.002, cache_bytes=128 * 1024**2):
        if snapshot not in SNAPSHOTS:
            raise ValueError('only the two drawer003 snapshots are permitted')
        if not np.isclose(spacing_m, .002, rtol=0., atol=1e-12):
            raise ValueError('this experiment uses the frozen 2 mm fields only')
        self.directory = Path(directory).resolve()
        self.snapshot = snapshot
        self.spacing_m = float(spacing_m)
        self.cache_bytes = int(cache_bytes)
        if self.cache_bytes <= 0:
            raise ValueError('cache_bytes must be positive')
        self._consumed = {}
        self._cache = OrderedDict()
        self._cached_bytes = 0
        self._queries = 0
        self._query_points = 0

    def _checked(self, file, expected=None):
        path = (self.directory / file).resolve()
        if self.directory not in path.parents:
            raise ValueError('artifact path escapes its saved-run directory')
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f'saved artifact hash mismatch: {path}')
        if str(path) in self._consumed and self._consumed[str(path)] != digest:
            raise ValueError(f'saved artifact changed during use: {path}')
        self._consumed[str(path)] = digest
        return path

    def _json(self, file, expected=None):
        return json.loads(self._checked(file, expected).read_text())

    def _load_block(self, record):
        key = record['file']
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key][0]
        path = self._checked(key, record['sha256'])
        with np.load(path, allow_pickle=False) as archive:
            block = {k: archive[k] for k in archive.files}
        values = block['values']
        if values.ndim != 3 or min(values.shape) < 2 or not np.all(np.isfinite(values)):
            raise ValueError('invalid saved scalar block')
        if not np.isclose(float(block['spacing_m']), self.spacing_m, rtol=0., atol=1e-12):
            raise ValueError('saved block spacing mismatch')
        nbytes = sum(a.nbytes for a in block.values())
        while self._cache and self._cached_bytes + nbytes > self.cache_bytes:
            _, (_, removed) = self._cache.popitem(last=False)
            self._cached_bytes -= removed
        # At most one block can exceed the cap; it is needed for this query.
        self._cache[key] = (block, nbytes)
        self._cached_bytes += nbytes
        return block

    def _start_query(self, points):
        p = _points(points)
        self._queries += 1
        self._query_points += len(p)
        return p, QueryResult.empty(len(p))

    def provenance(self, verify=False):
        unchanged = None
        if verify:
            unchanged = {path: sha256(path) == digest for path, digest in self._consumed.items()}
            if not all(unchanged.values()):
                raise ValueError('one or more consumed saved-field inputs changed')
        return {'kind': self.kind, 'snapshot': self.snapshot, 'spacing_m': self.spacing_m,
                'coordinate_frame': 'camera_cv_xright_ydown_zforward_metres',
                'state_codes': {'occupied': -1, 'unknown_or_ambiguous': 0, 'free': 1},
                'sign_epsilon_m': SIGN_EPS_M, 'query_calls': self._queries,
                'query_points': self._query_points, 'consumed_sha256': dict(self._consumed),
                'inputs_unchanged': unchanged, 'cache_limit_bytes': self.cache_bytes,
                'cache_current_bytes': self._cached_bytes,
                'robot_image_geometry_removed': False, 'ground_truth_inputs': [],
                'limitations': self.limitations}


class AstraSavedField(_SavedAdapter):
    kind = 'astra'
    limitations = [
        'Predicted CSG geometry, including hidden completion, is not observed truth.',
        'Only saved scalar samples and persisted Lipschitz block certificates are queried.',
        'Near-zero values without a certified sign are unknown; outside the saved grid is unknown.',
        'Free clearance is a lower bound for the model implicit zero set and grid-domain boundary.',
        'raw_phi_m is NaN in skipped blocks: the certificate is not a sample at the query point.',
    ]

    def __init__(self, directory, snapshot, spacing_m=.002, cache_bytes=128 * 1024**2):
        super().__init__(directory, snapshot, spacing_m, cache_bytes)
        manifest = self._json('manifest.json')
        if manifest.get('status') != 'complete':
            raise ValueError('Astra field run is not complete')
        name = snapshot + '_h2mm'
        matches = [c for c in manifest['configurations'] if c['name'] == name]
        if len(matches) != 1 or matches[0].get('status') != 'ok':
            raise ValueError('missing unique completed Astra configuration')
        self.config = matches[0]
        allocation = self.config['allocation']
        self.lower_m = np.asarray(allocation['grid_lower_m'], float)
        self.upper_m = np.asarray(allocation['grid_upper_m'], float)
        self.block_cells = int(allocation['block_cells'])
        self.records = {}
        for record in self.config['fields']:
            stem = Path(record['file']).stem
            key = tuple(map(int, stem.removeprefix('block_').split('_')))
            if key in self.records:
                raise ValueError('duplicate Astra block')
            self.records[key] = record
        path = self._checked('fields/' + name + '/' + allocation['certificate_file'], allocation['certificate_sha256'])
        with np.load(path, allow_pickle=False) as archive:
            self.certificate = {k: archive[k] for k in archive.files}
        c = self.certificate
        if not np.allclose(c['lower_m'], self.lower_m, atol=1e-12, rtol=0) or not np.isclose(float(c['spacing_m']), self.spacing_m, atol=1e-12, rtol=0):
            raise ValueError('certificate grid mismatch')
        self.total_cells = c['total_cells'].astype(int)
        self.counts = (self.total_cells + self.block_cells - 1) // self.block_cells
        expected_ids = np.stack(np.meshgrid(*(np.arange(x) for x in self.counts), indexing='ij'), axis=-1).reshape(-1, 3)
        if not np.array_equal(expected_ids, c['block_ids']):
            raise ValueError('certificate block ordering/coverage mismatch')
        expected_keys = {tuple(x) for x in c['block_ids'][c['retained']]}
        if expected_keys != set(self.records):
            raise ValueError('saved blocks do not match retained certificates')
        self.centers = self.lower_m + (c['start_cells'] + c['cell_shapes'] / 2.) * self.spacing_m
        if np.any(np.abs(c['center_values_m'][~c['retained']]) <= c['half_diagonal_m'][~c['retained']] + c['certificate_tolerance_m'][~c['retained']]):
            raise ValueError('invalid skipped-block sign certificate')

    def query(self, points_camera):
        p, result = self._start_query(points_camera)
        inside = np.all(np.isfinite(p) & (p >= self.lower_m) & (p <= self.upper_m), axis=1)
        ids = np.flatnonzero(inside)
        if not len(ids):
            return result
        cell = np.minimum(np.floor((p[ids] - self.lower_m) / self.spacing_m).astype(int), self.total_cells - 1)
        block_id = cell // self.block_cells
        linear = np.ravel_multi_index(block_id.T, self.counts)
        c = self.certificate
        for number in np.unique(linear):
            selection = ids[linear == number]
            q = p[selection]
            domain_margin = np.minimum(q - self.lower_m, self.upper_m - q).min(axis=1)
            if not c['retained'][number]:
                value = c['center_values_m'][number]
                radius = np.linalg.norm(q - self.centers[number], axis=1)
                tolerance = c['certificate_tolerance_m'][number]
                lo, hi = value - radius - tolerance, value + radius + tolerance
            else:
                record = self.records[tuple(c['block_ids'][number])]
                block = self._load_block(record)
                corners, fraction, _, phi, _, _ = _cell_samples(block['values'], block['lower_m'], self.spacing_m, q)
                distances = self.spacing_m * np.linalg.norm(fraction[:, None, :] - CORNERS[None, :, :], axis=2)
                tolerance = float(block['vertex_quantization_max_m']) + SIGN_EPS_M
                lo = np.max(corners - distances - tolerance, axis=1)
                hi = np.min(corners + distances + tolerance, axis=1)
                result.raw_phi_m[selection] = phi
            result.support[selection] = True
            result.phi_lower_m[selection], result.phi_upper_m[selection] = lo, hi
            free, occupied = lo > SIGN_EPS_M, hi < -SIGN_EPS_M
            result.state[selection[free]] = 1
            result.state[selection[occupied]] = -1
            result.conservative_clearance_m[selection[free]] = np.maximum(0., np.minimum(lo[free] - SIGN_EPS_M, domain_margin[free]))
        return result


class DepthProSavedField(_SavedAdapter):
    kind = 'depthpro'
    limitations = [
        'Single-view predicted support is not physically verified free space.',
        'Truncated camera-Z residuals are not Euclidean distances.',
        'A point needs its owner block, all eight observed corners and cell_valid; absent blocks remain unknown.',
        'Behind-surface regions beyond saved support and outside the image stay unknown.',
        'Clearance bounds only the saved trilinear field inside this supported cell and owner block.',
        'Clearance can be zero on cell boundaries even when the predicted sign is free.',
    ]

    def __init__(self, directory, snapshot, spacing_m=.002, cache_bytes=128 * 1024**2):
        super().__init__(directory, snapshot, spacing_m, cache_bytes)
        manifest = self._json('manifest.json')
        if manifest.get('status') != 'complete':
            raise ValueError('DepthPro field run is not complete')
        metrics = self._json('metrics.json', manifest['metrics_sha256'])
        matches = [c for c in metrics['configurations'] if c['name'] == snapshot + '_h2mm']
        if len(matches) != 1 or matches[0].get('status') != 'ok':
            raise ValueError('missing unique completed DepthPro configuration')
        self.config = matches[0]
        self.records = {}
        self.metadata = {}
        for record in self.config['fields']:
            path = self._checked(record['file'], record['sha256'])
            with np.load(path, allow_pickle=False) as archive:
                m = json.loads(str(archive['metadata_json']))
            if m.get('representation') != 'masked_camera_z_projective_tsdf':
                raise ValueError('unexpected DepthPro field representation')
            slab_size = m['core_z_m'][1] - m['core_z_m'][0]
            key = (*m['tile_yx'], int(np.rint(m['core_z_m'][0] / slab_size)))
            if key in self.records:
                raise ValueError('duplicate DepthPro owner block')
            self.records[key], self.metadata[key] = record, m
        if not self.records:
            raise ValueError('no saved DepthPro blocks')
        first = next(iter(self.metadata.values()))
        self.K = np.asarray(first['K_pred'], float)
        self.image_shape = tuple(first['image_shape'])
        self.slab_m = first['core_z_m'][1] - first['core_z_m'][0]
        self.u_edges = np.unique([x for m in self.metadata.values() for x in m['core_uv'][:2]])
        self.v_edges = np.unique([x for m in self.metadata.values() for x in m['core_uv'][2:]])
        for m in self.metadata.values():
            if not np.array_equal(m['K_pred'], self.K) or tuple(m['image_shape']) != self.image_shape or not np.isclose(m['core_z_m'][1] - m['core_z_m'][0], self.slab_m):
                raise ValueError('DepthPro blocks do not share one calibration and slab convention')
        if len(self.u_edges) != 5 or len(self.v_edges) != 5:
            raise ValueError('expected frozen four-by-four pixel partition')

    def query(self, points_camera):
        p, result = self._start_query(points_camera)
        z = p[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = self.K[0, 0] * p[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * p[:, 1] / z + self.K[1, 2]
        valid = np.all(np.isfinite(p), axis=1) & (z > 0) & np.isfinite(u) & np.isfinite(v)
        valid &= (u >= self.u_edges[0]) & (u <= self.u_edges[-1]) & (v >= self.v_edges[0]) & (v <= self.v_edges[-1])
        ids = np.flatnonzero(valid)
        if not len(ids):
            return result
        tx = np.minimum(np.searchsorted(self.u_edges, u[ids], side='right') - 1, 3)
        ty = np.minimum(np.searchsorted(self.v_edges, v[ids], side='right') - 1, 3)
        slab = np.floor(z[ids] / self.slab_m + 1e-11).astype(int)
        owner = np.stack((ty, tx, slab), axis=1)
        keys, inverse = np.unique(owner, axis=0, return_inverse=True)
        for group, key_array in enumerate(keys):
            key = tuple(key_array)
            if key not in self.records:
                continue
            selected = ids[inverse == group]
            block = self._load_block(self.records[key])
            lower = block['lower_m']
            upper = lower + self.spacing_m * (np.array(block['values'].shape) - 1)
            q = p[selected]
            in_grid = np.all((q >= lower) & (q <= upper), axis=1)
            selected, q = selected[in_grid], q[in_grid]
            if not len(selected):
                continue
            corners, fraction, base, phi, cell_margin, indices = _cell_samples(block['values'], lower, self.spacing_m, q)
            observed = block['observed'][indices[..., 0], indices[..., 1], indices[..., 2]].all(axis=1)
            cell_valid = block['cell_valid'][tuple(base.T)]
            planes = np.asarray(self.metadata[key]['clip_planes'], float)
            plane_margin = ((q @ planes[:, :3].T + planes[:, 3]) / np.linalg.norm(planes[:, :3], axis=1)).min(axis=1)
            supported = observed & cell_valid & (plane_margin >= -1e-12)
            result.raw_phi_m[selected[supported]] = phi[supported]
            result.support[selected[supported]] = True
            tolerance = SIGN_EPS_M + self.metadata[key]['vertex_quantization_max_m']
            lo, hi = phi - tolerance, phi + tolerance
            result.phi_lower_m[selected[supported]] = lo[supported]
            result.phi_upper_m[selected[supported]] = hi[supported]
            free, occupied = supported & (lo > SIGN_EPS_M), supported & (hi < -SIGN_EPS_M)
            result.state[selected[free]] = 1
            result.state[selected[occupied]] = -1
            # Max derivative in each axis bounds the gradient of the stored
            # trilinear polynomial throughout this one supported voxel cell.
            c = corners.reshape(-1, 2, 2, 2)
            grad_bound = np.stack([np.max(np.abs(np.diff(c, axis=a)), axis=(1, 2, 3)) / self.spacing_m for a in (1, 2, 3)], axis=1)
            lipschitz = np.linalg.norm(grad_bound, axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                zero_margin = np.where(lipschitz > 0., np.maximum(0., lo - SIGN_EPS_M) / lipschitz, np.inf)
            margin = np.minimum(np.minimum(cell_margin, np.maximum(0., plane_margin)), zero_margin)
            result.conservative_clearance_m[selected[free]] = margin[free]
        return result


def load_predicted_field(kind, snapshot, project_root=None, spacing_m=.002, cache_bytes=128 * 1024**2):
    project = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    if kind == 'astra':
        return AstraSavedField(project / 'scene_test/astra_scene/runs/fields_001', snapshot, spacing_m, cache_bytes)
    if kind == 'depthpro':
        return DepthProSavedField(project / 'scene_test/rgb_scene/runs/fields_full_001', snapshot, spacing_m, cache_bytes)
    raise ValueError('kind must be astra or depthpro')
