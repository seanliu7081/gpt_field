"""Single-view projective TSDF, with explicit unknown space and saved-only decode.

Coordinates are camera CV (x right, y down, z forward), metres. Values are
camera-Z residuals, NOT Euclidean signed distances or certified free space.
Only predicted depth and predicted intrinsics enter field construction.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import trimesh
from skimage.measure import marching_cubes

TAU_M = .020
JUMP_ABS_M = .050
JUMP_REL = .050
TILES = 4
SLAB_M = .064
GRID_ALIGNMENT_M = .004


def validate_depth(depth, K):
    depth, K = np.asarray(depth, float), np.asarray(K, float)
    if depth.ndim != 2 or min(depth.shape) < 2:
        raise ValueError('depth must be HxW with each dimension >=2')
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError('finite K must be3x3')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[[0, 1], [1, 0]], 0) or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('K must be a positive-focal zero-skew pinhole calibration')
    return depth, K


def backproject(depth, K):
    depth, K = validate_depth(depth, K)
    v, u = np.indices(depth.shape)
    return np.stack(((u-K[0, 2])*depth/K[0, 0], (v-K[1, 2])*depth/K[1, 1], depth), axis=-1)


def depth_triangles(depth):
    """Two image triangles/cell; depth-jump gate frozen independent of results."""
    depth = np.asarray(depth, float)
    h, w = depth.shape
    ids = np.arange(h*w).reshape(h, w)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]
    faces = np.stack((np.stack((a, b, c), -1), np.stack((a, c, d), -1)), -2)
    values = depth.ravel()[faces]
    valid = np.all(np.isfinite(values) & (values > 0), axis=-1)
    minimum, maximum = np.min(values, -1), np.max(values, -1)
    valid &= maximum-minimum <= JUMP_ABS_M + JUMP_REL*minimum
    return faces, valid


def visible_mesh(depth, K, pixel_mask=None):
    """Open visible surface; never bridge invalid/depth-discontinuous triangles.

    Optional masks are evaluator-only: all three pixel vertices must match.
    """
    depth, K = validate_depth(depth, K)
    faces, valid = depth_triangles(depth)
    if pixel_mask is not None:
        mask = np.asarray(pixel_mask, bool)
        if mask.shape != depth.shape:
            raise ValueError('pixel mask shape mismatch')
        valid &= np.all(mask.ravel()[faces], axis=-1)
    vertices = backproject(depth, K).reshape(-1, 3)
    # Unused invalid vertices cannot be serialized as NaN in geometry artifacts.
    vertices[~np.all(np.isfinite(vertices), axis=1)] = 0
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces[valid], process=False)
    mesh.remove_unreferenced_vertices()
    return mesh, {'pixels': int(depth.size), 'finite_positive_pixels': int(np.sum(np.isfinite(depth) & (depth > 0))),
                  'candidate_triangles': int(valid.size), 'retained_triangles': int(valid.sum()),
                  'rejected_triangle_fraction': float(1-valid.mean()), 'surface_area_m2': float(mesh.area)}


def sample_projective_depth(depth, K, points, triangle_valid=None):
    """Perspective-correct depth of exactly the visible_mesh triangles.

    Inverse depth is affine in image barycentric coordinates. Consequently the
    continuous zero set matches the mesh, avoiding a bilinear-depth confound.
    """
    depth, K = validate_depth(depth, K)
    points = np.asarray(points, float)
    z = points[:, 2]
    with np.errstate(invalid='ignore', divide='ignore'):
        u = K[0, 0]*points[:, 0]/z + K[0, 2]
        v = K[1, 1]*points[:, 1]/z + K[1, 2]
    h, w = depth.shape
    valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 0) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)
    safe_u, safe_v = np.where(valid, u, 0), np.where(valid, v, 0)
    ix, iy = np.minimum(np.floor(safe_u).astype(int), w-2), np.minimum(np.floor(safe_v).astype(int), h-2)
    du, dv = safe_u-ix, safe_v-iy
    lower = du >= dv
    triangle = np.where(lower, 0, 1)
    if triangle_valid is None:
        _, triangle_valid = depth_triangles(depth)
    valid &= triangle_valid[iy, ix, triangle]
    d00, d10, d11, d01 = depth[iy, ix], depth[iy, ix+1], depth[iy+1, ix+1], depth[iy+1, ix]
    with np.errstate(invalid='ignore', divide='ignore'):
        inv = np.where(lower, (1-du)/d00+(du-dv)/d10+dv/d11,
                       (1-dv)/d00+du/d11+(dv-du)/d01)
        result = 1/inv
    valid &= np.isfinite(result) & (result > 0)
    return np.where(valid, result, np.nan), valid


def frustum_planes(core, K):
    """Halfspace planes n dot p + offset >=0; decoder owns upper coplanars."""
    u0, u1, v0, v1 = core
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([[fx, 0, cx-u0, 0], [-fx, 0, u1-cx, 0],
                     [0, fy, cy-v0, 0], [0, -fy, v1-cy, 0]], float)


def clip_mesh_planes(mesh, planes, exclude_upper_z_coplanar=False):
    """Clip existing faces without manufacturing caps at image/slab boundaries."""
    triangles = np.asarray(mesh.triangles)
    if not len(triangles):
        return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), int), process=False)
    distances = np.einsum('tvc,pc->tvp', triangles, planes[:, :3]) + planes[:, 3]
    whole = np.all(distances >= -1e-12, axis=(1, 2))
    possible = np.all(np.max(distances, axis=1) >= -1e-12, axis=1)
    pieces = [triangles[whole]]
    for triangle in triangles[possible & ~whole]:
        polygon = triangle
        for plane in planes:
            if not len(polygon):
                break
            output = []
            for a, b in zip(np.roll(polygon, 1, axis=0), polygon):
                da, db = np.dot(a, plane[:3])+plane[3], np.dot(b, plane[:3])+plane[3]
                ina, inb = da >= 0, db >= 0
                if ina != inb:
                    output.append(a+(da/(da-db))*(b-a))
                if inb:
                    output.append(b)
            polygon = np.asarray(output).reshape(-1, 3)
        if len(polygon) >= 3:
            pieces.append(np.array([[polygon[0], polygon[i], polygon[i+1]] for i in range(1, len(polygon)-1)]))
    triangles = np.concatenate(pieces)
    area2 = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    triangles = triangles[area2 > 1e-18]
    if exclude_upper_z_coplanar:
        for plane in planes[1::2]:
            # Upper u, upper v and upper z each belong to the adjacent block.
            distance = (np.einsum('tvc,c->tv', triangles, plane[:3])+plane[3])/np.linalg.norm(plane[:3])
            triangles = triangles[~np.all(np.abs(distance) <= 1e-10, axis=1)]
    return trimesh.Trimesh(vertices=triangles.reshape(-1, 3), faces=np.arange(len(triangles)*3).reshape(-1, 3), process=False)


def plan_blocks(depth, K, tiles=TILES, slab_m=SLAB_M):
    """Prediction-only bounds; camera-Z slabs avoid dense empty depth spans.

    Core image boundaries lie on pixel centres, partitioning the triangulated
    image domain [0,W-1]x[0,H-1]. A one-pixel halo catches crossing triangles.
    """
    depth, K = validate_depth(depth, K)
    h, w = depth.shape
    vertices = backproject(depth, K).reshape(-1, 3)
    faces, valid = depth_triangles(depth)
    us, vs = np.linspace(0, w-1, tiles+1), np.linspace(0, h-1, tiles+1)
    blocks = []
    for ty in range(tiles):
        for tx in range(tiles):
            core = [us[tx], us[tx+1], vs[ty], vs[ty+1]]
            x0, x1 = max(0, int(np.floor(core[0]))-1), min(w-1, int(np.ceil(core[1]))+1)
            y0, y1 = max(0, int(np.floor(core[2]))-1), min(h-1, int(np.ceil(core[3]))+1)
            selected = faces[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
            if not len(selected):
                continue
            tri = vertices[selected]
            zmin, zmax = tri[:, :, 2].min(1), tri[:, :, 2].max(1)
            first, last = int(np.floor(zmin.min()/slab_m)), int(np.floor(zmax.max()/slab_m))
            for slab in range(first, last+1):
                lowz, highz = slab*slab_m, (slab+1)*slab_m
                take = (zmax >= lowz) & (zmin <= highz)
                if not np.any(take):
                    continue
                selected_points = tri[take].reshape(-1, 3)
                lower = np.floor((selected_points.min(0)-TAU_M)/GRID_ALIGNMENT_M)*GRID_ALIGNMENT_M
                upper = np.ceil((selected_points.max(0)+TAU_M)/GRID_ALIGNMENT_M)*GRID_ALIGNMENT_M
                lower[2] = max(GRID_ALIGNMENT_M, lowz-GRID_ALIGNMENT_M)
                upper[2] = highz+GRID_ALIGNMENT_M
                if upper[2] <= lower[2]:
                    continue
                planes = np.vstack((frustum_planes(core, K), [0, 0, 1, -lowz], [0, 0, -1, highz]))
                blocks.append({'id': f'tile{ty:02d}_{tx:02d}_slab{slab:04d}', 'tile_yx': [ty, tx],
                               'core_uv': core, 'core_z_m': [lowz, highz], 'lower_m': lower.tolist(),
                               'upper_m': upper.tolist(), 'clip_planes': planes.tolist()})
    return blocks


def block_shape(block, spacing):
    intervals = (np.asarray(block['upper_m'])-block['lower_m'])/spacing
    if not np.allclose(intervals, np.rint(intervals), rtol=0, atol=1e-7):
        raise ValueError('bounds must be integral grid intervals')
    return tuple((np.rint(intervals).astype(int)+1).tolist())


@dataclass
class ProjectiveField:
    values: np.ndarray
    observed: np.ndarray
    cell_valid: np.ndarray
    lower: np.ndarray
    spacing: float
    metadata: dict

    @classmethod
    def build(cls, depth, K, block, spacing, max_vertices=50_000_000, chunk_size=250_000):
        depth, K = validate_depth(depth, K)
        shape = block_shape(block, spacing)
        count = int(np.prod(shape))
        if count > max_vertices:
            raise MemoryError(f'block {block["id"]}: {count} vertices exceed declared {max_vertices}')
        lower = np.asarray(block['lower_m'], float)
        values = np.empty(count, np.float32)
        observed = np.empty(count, bool)
        source_z = np.empty(count, np.float32)
        _, triangle_valid = depth_triangles(depth)
        quantization = 0.
        for start in range(0, count, chunk_size):
            indices = np.array(np.unravel_index(np.arange(start, min(start+chunk_size, count)), shape)).T
            points = lower+spacing*indices
            d, valid = sample_projective_depth(depth, K, points, triangle_valid)
            residual = d-points[:, 2]
            valid &= residual >= -TAU_M
            phi = np.where(valid, np.clip(residual, -TAU_M, TAU_M), TAU_M)
            values[start:start+len(points)] = phi
            observed[start:start+len(points)] = valid
            source_z[start:start+len(points)] = np.where(valid, d, 0.)
            quantization = max(quantization, float(np.max(np.abs(values[start:start+len(points)]-phi))))
        values, observed, source_z = values.reshape(shape), observed.reshape(shape), source_z.reshape(shape)
        cell_shape = tuple(s-1 for s in shape)
        cell_valid = np.ones(cell_shape, bool)
        dmin = np.full(cell_shape, np.inf, np.float32)
        dmax = np.full(cell_shape, -np.inf, np.float32)
        for x in (0, 1):
            for y in (0, 1):
                for z in (0, 1):
                    sl = (slice(x, shape[0]-1+x), slice(y, shape[1]-1+y), slice(z, shape[2]-1+z))
                    cell_valid &= observed[sl]
                    np.minimum(dmin, source_z[sl], out=dmin)
                    np.maximum(dmax, source_z[sl], out=dmax)
        cell_valid &= dmax-dmin <= JUMP_ABS_M+JUMP_REL*dmin
        # Corner tests alone can jump over an unsupported image pixel when one
        # voxel projects to several pixels. Require the whole projected box's
        # image-cell rectangle to contain valid triangles (conservative erosion).
        invalid = ~np.all(triangle_valid, axis=2)
        integral = np.pad(invalid.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        flat_valid = cell_valid.reshape(-1)
        for start in range(0, len(flat_valid), chunk_size):
            ids = np.flatnonzero(flat_valid[start:start+chunk_size])+start
            if not len(ids):
                continue
            ijk = np.array(np.unravel_index(ids, cell_shape)).T
            lo, hi = lower+spacing*ijk, lower+spacing*(ijk+1)
            us = np.stack((lo[:, 0]/lo[:, 2], lo[:, 0]/hi[:, 2], hi[:, 0]/lo[:, 2], hi[:, 0]/hi[:, 2]), 1)*K[0, 0]+K[0, 2]
            vs = np.stack((lo[:, 1]/lo[:, 2], lo[:, 1]/hi[:, 2], hi[:, 1]/lo[:, 2], hi[:, 1]/hi[:, 2]), 1)*K[1, 1]+K[1, 2]
            x0 = np.clip(np.floor(us.min(1)).astype(int), 0, depth.shape[1]-2)
            x1 = np.clip(np.floor(us.max(1)).astype(int)+1, 1, depth.shape[1]-1)
            y0 = np.clip(np.floor(vs.min(1)).astype(int), 0, depth.shape[0]-2)
            y1 = np.clip(np.floor(vs.max(1)).astype(int)+1, 1, depth.shape[0]-1)
            unsupported = integral[y1, x1]-integral[y0, x1]-integral[y1, x0]+integral[y0, x0]
            flat_valid[ids] &= unsupported == 0
        metadata = {**block, 'schema_version': 1, 'representation': 'masked_camera_z_projective_tsdf',
                    'coordinate_frame': 'predicted_camera_cv_m', 'tau_m': TAU_M, 'K_pred': K.tolist(),
                    'image_shape': list(depth.shape), 'jump_abs_m': JUMP_ABS_M, 'jump_rel': JUMP_REL,
                    'unknown_semantics': 'unobserved or behind predicted surface band; not free',
                    'observed_semantics': 'single-prediction support, not physically verified free space',
                    'depth_interpolation': 'perspective_correct_inverse_depth_on_pixel_triangles',
                    'cell_support': 'all8_observed_plus_depth_jump_and_projected_rectangle_all_valid_triangles',
                    'block_boundary_ownership': 'lower_uvz_included_upper_uvz_coplanar_excluded_with_0.1nm_plane_tolerance',
                    'vertex_quantization_max_m': quantization,
                    'observed_vertices': int(observed.sum()), 'valid_cells': int(cell_valid.sum())}
        return cls(values, observed, cell_valid, lower, float(spacing), metadata)

    def save(self, path):
        path = Path(path)
        with path.open('xb') as stream:
            np.savez_compressed(stream, values=self.values, observed=self.observed,
                                cell_valid=self.cell_valid, lower_m=self.lower, spacing_m=self.spacing,
                                metadata_json=np.array(json.dumps(self.metadata, allow_nan=False)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            result = cls(data['values'].copy(), data['observed'].copy(), data['cell_valid'].copy(),
                         data['lower_m'].copy(), float(data['spacing_m']), json.loads(str(data['metadata_json'])))
        if result.values.ndim != 3 or result.observed.shape != result.values.shape or result.cell_valid.shape != tuple(s-1 for s in result.values.shape):
            raise ValueError('field array shape mismatch')
        if not np.all(np.isfinite(result.values)) or result.spacing <= 0:
            raise ValueError('invalid saved field')
        return result

    def decode(self):
        """Uses ONLY serialized field state; no RGB/depth/source surface reads."""
        if not self.cell_valid.any() or self.values.min() >= 0 or self.values.max() <= 0:
            return trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3), int), process=False)
        # Artificial unknown values may cause candidate faces; cell validity
        # below excludes every face without a fully observed generating cell.
        verts, faces, _, _ = marching_cubes(self.values, 0., gradient_direction='descent', allow_degenerate=False)
        triangle = verts[faces].astype(float)
        centroid = triangle.mean(1)
        base = np.floor(centroid).astype(int)
        accepted = np.zeros(len(faces), bool)
        shape = np.array(self.cell_valid.shape)
        # MC triangles on a grid plane have two possible generating cells.
        # Test adjacent candidates and require containment, not just centroid.
        epsilon = 2e-4  # in voxel-index units; MC vertex arithmetic is float32.
        for dx in (0, -1):
            for dy in (0, -1):
                for dz in (0, -1):
                    index = base+np.array([dx, dy, dz])
                    in_grid = np.all((index >= 0) & (index < shape), axis=1)
                    inside = np.all((triangle >= index[:, None, :]-epsilon) & (triangle <= index[:, None, :]+1+epsilon), axis=(1, 2))
                    candidates = np.flatnonzero(in_grid & inside & ~accepted)
                    if len(candidates):
                        accepted[candidates] = self.cell_valid[tuple(index[candidates].T)]
        mesh = trimesh.Trimesh(vertices=self.lower+self.spacing*verts.astype(float), faces=faces[accepted], process=False)
        return clip_mesh_planes(mesh, np.asarray(self.metadata['clip_planes'], float), exclude_upper_z_coplanar=True)
