"""Robot-only convex collision envelopes, evaluated without a simulator.

Collision meshes are read from the same static Panda asset as the kinematics.
Their convex hulls conservatively enclose the mesh; they are not scene geometry.
Scene geometry remains predicted and incomplete, so this is not a safety proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..robodojo_direct_sim import _transform, rotation_matrix


def transform_points(pose, points):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def _vertices(path):
    data = path.read_bytes()
    if len(data) >= 84 and len(data) == 84 + 50 * struct.unpack_from('<I', data, 80)[0]:
        rows = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), offset=84)
        points = rows['v'].reshape(-1, 3).astype(float)
    elif path.suffix.lower() == '.obj':
        points = np.array([[float(v) for v in line.split()[1:4]] for line in data.decode().splitlines() if line.startswith('v ')])
    else:
        points = np.array([[float(v) for v in line.split()[1:4]] for line in data.decode().splitlines() if line.strip().startswith('vertex ')])
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
        raise ValueError(f'invalid robot mesh: {path}')
    return np.unique(points, axis=0), hashlib.sha256(data).hexdigest()


@dataclass
class CollisionShape:
    name: str
    group: int
    vertices: np.ndarray
    equations: np.ndarray
    pose: np.ndarray
    kind: str = 'arm'

    @property
    def center(self):
        return transform_points(self.pose, self.vertices.mean(axis=0))

    @property
    def radius(self):
        return float(np.linalg.norm(self.vertices - self.vertices.mean(axis=0), axis=1).max())

    def contains(self, points, margin=0.):
        local = (np.asarray(points) - self.pose[:3, 3]) @ self.pose[:3, :3]
        return np.all(local @ self.equations[:, :3].T + self.equations[:, 3] <= margin, axis=1)

    def ray_entry_distances(self, origin, points, margin=0.):
        """First entry into the convex envelope along calibrated camera rays.

        Used only to classify current-pose robot surface predictions. A point
        behind the first modeled robot surface is not automatically self-masked.
        """
        points = np.asarray(points, float)
        origin = np.asarray(origin, float)
        rays = points-origin
        lengths = np.linalg.norm(rays, axis=1)
        rays = rays / np.maximum(lengths[:, None], 1e-12)
        local_origin = (origin-self.pose[:3, 3]) @ self.pose[:3, :3]
        local_rays = rays @ self.pose[:3, :3]
        offset = local_origin @ self.equations[:, :3].T + self.equations[:, 3]-margin
        denominator = local_rays @ self.equations[:, :3].T
        parallel = np.abs(denominator) < 1e-12
        invalid = np.any(parallel & (offset[None, :] > 0), axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            fractions = -offset[None, :] / denominator
        lower = np.max(np.where(denominator < -1e-12, fractions, -np.inf), axis=1)
        upper = np.min(np.where(denominator > 1e-12, fractions, np.inf), axis=1)
        return np.where(~invalid & (upper >= np.maximum(lower, 0.)), np.maximum(lower, 0.), np.inf)

    def surface_points(self):
        # Convex hull vertices and facet centers are finite coverage probes.
        # The result is deliberately not labelled a continuous-volume proof.
        return transform_points(self.pose, self.vertices)

    def support(self, direction, margin=0.):
        local_direction = self.pose[:3, :3].T @ direction
        p = transform_points(self.pose, self.vertices[np.argmax(self.vertices @ local_direction)])
        return p + margin * direction / max(float(np.linalg.norm(direction)), 1e-12)


def _simplex(simplex):
    """GJK simplex update; newest vertex is last."""
    a = simplex[-1]; ao = -a
    if len(simplex) == 2:
        b = simplex[-2]; ab = b-a
        if ab @ ao > 0:
            direction = np.cross(np.cross(ab, ao), ab)
            return False, [b, a], direction
        return False, [a], ao
    b, c = simplex[-2], simplex[-3]
    ab, ac = b-a, c-a
    abc = np.cross(ab, ac)
    if len(simplex) == 3:
        if np.cross(abc, ac) @ ao > 0:
            if ac @ ao > 0:
                return False, [c, a], np.cross(np.cross(ac, ao), ac)
            return _simplex([b, a])
        if np.cross(ab, abc) @ ao > 0:
            return _simplex([b, a])
        if abc @ ao > 0:
            return False, [c, b, a], abc
        return False, [b, c, a], -abc
    d = simplex[-4]
    for p, q, opposite in ((b, c, d), (c, d, b), (d, b, c)):
        normal = np.cross(p-a, q-a)
        if normal @ (opposite-a) > 0:
            normal = -normal
            p, q = q, p
        if normal @ ao > 0:
            return _simplex([q, p, a])
    return True, simplex, np.zeros(3)


def intersect(first, second, margin=0.):
    if np.linalg.norm(first.center-second.center) > first.radius+second.radius+2*margin:
        return False
    direction = second.center-first.center
    if np.linalg.norm(direction) < 1e-10:
        direction = np.array([1., 0., 0.])
    def support(d):
        return first.support(d, margin)-second.support(-d, margin)
    simplex = [support(direction)]
    direction = -simplex[0]
    for _ in range(40):
        if np.linalg.norm(direction) < 1e-10:
            return True
        p = support(direction)
        if p @ direction < -1e-10:
            return False
        simplex.append(p)
        hit, simplex, direction = _simplex(simplex)
        if hit:
            return True
    # Numerical nonconvergence is conservatively treated as potential overlap.
    return True


class RobotCollisionModel:
    def __init__(self, kinematics, *, include_visual=False):
        """Load collision envelopes, or combined body envelopes for self masks.

        ``include_visual=True`` is a separate current-pose image projection
        model. Its convex body silhouettes also enclose visual-only meshes; they
        must not replace the ordinary collision model used for planning.
        """
        self.kinematics = kinematics
        self.include_visual = bool(include_visual)
        self.robot_path = Path(kinematics.metadata['robot_xml'])
        self.gripper_path = Path(kinematics.metadata['gripper_xml'])
        self.mesh_hashes = {}
        self._mesh_cache = {}
        self.skipped_visual_geometries = []
        self.geometry_counts = {'collision_geometries': 0, 'visual_geometries': 0,
                                'merged_body_envelopes': 0}
        self.robot = self._load(self.robot_path, 'arm')
        self.gripper = self._load(self.gripper_path, 'tool')
        self.metadata = {'representation': ('static_robot_visual_silhouettes_with_support_plane_depth_enclosures'
                                            if self.include_visual else 'static_robot_collision_mesh_convex_hulls'),
                         'robot_xml': str(self.robot_path), 'gripper_xml': str(self.gripper_path),
                         'mesh_sha256': dict(self.mesh_hashes), 'scene_truth_used': False,
                         'include_visual': self.include_visual,
                         'intended_use': 'current_robot_image_self_mask' if self.include_visual else 'collision_checks',
                         'geometry_counts': dict(self.geometry_counts),
                         'skipped_visual_geometries': list(self.skipped_visual_geometries),
                         'visual_silhouette_is_convex_approximation': self.include_visual,
                         'self_mask_depth_plane_limit': 98 if self.include_visual else None,
                         'continuous_collision_certified': False}

    def _load(self, path, category):
        root = ET.parse(path).getroot()
        compiler = root.find('compiler')
        meshdir = path.parent / (compiler.get('meshdir', '') if compiler is not None else '')
        assets = {m.get('name'): m for m in root.findall('./asset/mesh')}
        def visit(body, inherited_group=0):
            joint = body.find('joint')
            group = inherited_group
            if joint is not None and joint.get('name', '').startswith('joint'):
                group = int(joint.get('name')[5:])
            shapes, body_vertices, body_geometry_names = [], [], []
            for index, geom in enumerate(body.findall('geom')):
                visual_only = geom.get('contype', '1') == '0' and geom.get('conaffinity', '1') == '0'
                if visual_only and not self.include_visual:
                    continue
                name = geom.get('name', f'{body.get("name")}_{index}')
                kind = geom.get('type', 'sphere')
                if kind == 'mesh':
                    asset = assets[geom.get('mesh')]
                    mesh_path = (meshdir / asset.get('file')).resolve()
                    if mesh_path not in self._mesh_cache:
                        self._mesh_cache[mesh_path] = _vertices(mesh_path)
                    vertices, digest = self._mesh_cache[mesh_path]
                    self.mesh_hashes[str(mesh_path)] = digest
                    vertices = vertices * np.asarray(asset.get('scale', '1 1 1').split(), float)
                elif kind == 'box':
                    size = np.asarray(geom.get('size').split(), float)
                    vertices = np.array([[x, y, z] for x in (-size[0], size[0]) for y in (-size[1], size[1]) for z in (-size[2], size[2])])
                else:
                    if visual_only:
                        self.skipped_visual_geometries.append({
                            'xml': str(path), 'body': body.get('name'), 'name': name,
                            'type': kind, 'reason': 'unsupported_visual_decoration'})
                        continue
                    raise ValueError(f'unsupported static robot collision geom {name}: {kind}')
                self.geometry_counts['visual_geometries' if visual_only else 'collision_geometries'] += 1
                if self.include_visual:
                    # Combine visual fragments in body coordinates *after* each
                    # geom's own placement. One envelope per body avoids casting
                    # camera rays against hundreds of little visual mesh hulls.
                    body_vertices.append(transform_points(_transform(geom), vertices))
                    body_geometry_names.append(name)
                    continue
                hull = ConvexHull(vertices)
                vertices = vertices[hull.vertices]
                # Recompute equations after reducing to hull vertices.
                equations = ConvexHull(vertices).equations
                shapes.append((name, vertices, equations, _transform(geom),
                               'pad' if '_pad_' in name else 'finger' if 'finger' in name else category))
            if body_vertices:
                vertices = np.unique(np.concatenate(body_vertices), axis=0)
                try:
                    hull = ConvexHull(vertices)
                    vertices = vertices[hull.vertices]
                    # Keep the mesh's projected convex silhouette, but bound its
                    # depth with a fixed angular sampling of supporting planes.
                    # Thousands of nearly coplanar visual mesh facets add no useful
                    # mask precision and otherwise dominate every scene refresh.
                    directions = np.array([[x, y, z] for x in range(-2, 3)
                                           for y in range(-2, 3) for z in range(-2, 3)
                                           if x or y or z], float)
                    directions /= np.linalg.norm(directions, axis=1)[:, None]
                    directions = np.unique(np.round(directions, 12), axis=0)
                    equations = np.column_stack((directions, -np.max(vertices @ directions.T, axis=0)))
                except QhullError:
                    self.skipped_visual_geometries.append({
                        'xml': str(path), 'body': body.get('name'), 'names': body_geometry_names,
                        'reason': 'degenerate_combined_visual_envelope'})
                else:
                    name = f'{body.get("name")}_visual_envelope'
                    shapes.append((name, vertices, equations, np.eye(4),
                                   'finger' if 'finger' in str(body.get('name')) else category))
                    self.geometry_counts['merged_body_envelopes'] += 1
            return {'transform': _transform(body), 'joint': None if joint is None else dict(joint.attrib),
                    'group': group, 'shapes': shapes, 'name': body.get('name'),
                    'children': [visit(child, group) for child in body.findall('body')]}
        return [visit(body) for body in root.findall('./worldbody/body')]

    def shapes(self, bound, joints, finger_positions=(.04, -.04)):
        output = []; hand_pose = None
        def visit(node, parent, tool=False):
            nonlocal hand_pose
            pose = parent @ node['transform']
            joint = node['joint']
            if joint is not None:
                name = joint.get('name', '')
                axis = np.asarray(joint.get('axis', '0 0 1').split(), float)
                anchor = np.asarray(joint.get('pos', '0 0 0').split(), float)
                if name.startswith('finger_joint'):
                    value = finger_positions[int(name[-1])-1]
                    pose[:3, 3] += pose[:3, :3] @ (axis*value)
                elif name.startswith('joint'):
                    value = joints[int(name[5:])-1]
                    R = rotation_matrix(axis*value)
                    move = np.eye(4); move[:3, :3] = R; move[:3, 3] = anchor-R@anchor
                    pose = pose @ move
                else:
                    raise ValueError(f'unsupported robot joint {name}')
            if node['name'] == 'right_hand':
                hand_pose = pose.copy()
            for name, vertices, equations, offset, kind in node['shapes']:
                output.append(CollisionShape(name, 8 if tool else node['group'], vertices, equations, pose@offset, kind))
            for child in node['children']:
                visit(child, pose, tool)
        for node in self.robot:
            visit(node, bound.world_from_base)
        if hand_pose is None:
            raise ValueError('right_hand not found in static robot model')
        for node in self.gripper:
            visit(node, hand_pose, True)
        return output

    @staticmethod
    def self_collision(shapes):
        for i, first in enumerate(shapes):
            for second in shapes[i+1:]:
                # Same rigid body and direct neighboring arm bodies are expected
                # to overlap at their joints. Tools are rigidly adjacent to link7.
                if abs(first.group-second.group) <= 1:
                    continue
                if intersect(first, second):
                    return {'reason': 'robot_self_collision', 'parts': [first.name, second.name]}
        return None
