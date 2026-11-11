"""Connect Astra key poses one stage at a time using current robot/scene evidence.

The model owns target poses and task semantics. The planner checks only the next
segment, never gates approach on hypothetical later grasp or placement geometry.
Unknown space and weak surface hypotheses are reported and permitted under
measured small-step execution. Cross-view supported or coherent identified
obstacles, joint limits and robot self-collisions still reject motion.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math

import numpy as np
from scipy.spatial import cKDTree

from ..robodojo_direct_sim import quaternion_matrix, rotation_matrix
from .robot_geometry import RobotCollisionModel, CollisionShape, transform_points, intersect
from .types import KeyPose, OperationPlan


@dataclass(frozen=True)
class MotionConfig:
    collision_margin_m: float = .006
    max_joint_step_rad: float = .055
    max_planning_s: float = 12.
    rrt_iterations: int = 100
    rrt_step_rad: float = .18
    max_geometry_age_s: float = 5.
    allow_unknown_space: bool = True
    free_probe_limit: int = 160
    upright_tolerance_rad: float = math.radians(5.)


class PlanningFailure(RuntimeError):
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}


def _unit(value):
    vector = np.asarray(value, float)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm < 1e-8:
        raise PlanningFailure('invalid_direction')
    return vector/norm


def _pose(state8):
    pose = np.eye(4); pose[:3, 3] = np.asarray(state8[:3]); pose[:3, :3] = rotation_matrix(state8[3:6])
    return pose


def _payload_shape(points):
    if len(points) < 4:
        return None
    from scipy.spatial import ConvexHull, QhullError
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    vertices = points[hull.vertices]
    return CollisionShape('held_visible_surface_envelope', 9, vertices, hull.equations, np.eye(4), 'payload')


class CandidatePlanner:
    def __init__(self, kinematics, config=None):
        self.kinematics = kinematics
        self.config = config or MotionConfig()
        self.robot = RobotCollisionModel(kinematics)
        self.visual_robot = RobotCollisionModel(kinematics, include_visual=True)
        self.bound = kinematics
        self._scene_key = None
        self._cloud = np.empty((0, 3)); self._owners = np.empty(0, object)
        self._strong = np.empty(0, bool)
        self._tree = None; self._initial_shapes = []
        self._deadline = float('inf')
        self._payload_cache = {}
        self.last_candidates = []
        self.last_rejections = []
        self._contact_parts = ("finger", "pad", "payload")
        self._unknown_sample_visits = 0
        self._connection_diagnostics = {}
        self._uncertain_obstacle_samples = 0
        self._uncertain_obstacle_examples = []

    def _budget(self):
        if perf_counter() > self._deadline:
            raise PlanningFailure('planning_compute_budget', {
                'rejections': self.last_rejections[-12:], **self._connection_diagnostics,
                'uncertain_obstacle_samples': self._uncertain_obstacle_samples,
                'uncertain_obstacle_examples': list(self._uncertain_obstacle_examples)})

    def _prepare(self, scene, state8, joints):
        self.bound = self.kinematics.bind(state8, joints)
        key = (id(scene), scene.frame_id)
        if key == self._scene_key:
            return
        self._scene_key = key
        points = np.asarray(scene.scene_points, float).reshape(-1, 3)
        owners = getattr(scene, 'scene_point_object_ids', None)
        owners = np.full(len(points), '', object) if owners is None else np.asarray(owners, object)
        if owners.shape != (len(points),):
            raise PlanningFailure('scene_point_ownership_shape')
        # GeometryEvidence and scene_points share the source-view ordering.
        # Unassigned single-view/contradictory hypotheses remain in the cloud,
        # but are uncertain obstacles rather than equivalent to supported ones.
        strong = np.zeros(len(points), bool)
        from .surface import planning_surface
        for object_id, obj in scene.objects.items():
            ids = np.flatnonzero((owners == object_id) & np.isfinite(points).all(axis=1))
            if not len(ids):
                continue
            surface, _ = planning_surface(obj)
            if len(surface):
                distances, _ = cKDTree(surface).query(points[ids])
                # This is source-point correspondence, not a depth-error bound.
                strong[ids[distances <= 1e-6]] = True
        source_offset = 0
        strength_alignment = True
        for view in getattr(scene.geometry, '_views', {}).values():
            count = len(view.points_world)
            if source_offset+count > len(points) or len(view.support) != count or len(view.contradictions) != count:
                strength_alignment = False
                break
            strong[source_offset:source_offset+count] |= (np.asarray(view.support) >= 2) & (np.asarray(view.contradictions) == 0)
            source_offset += count
        strength_alignment &= source_offset == len(points)
        valid = np.isfinite(points).all(axis=1)
        source_indices = np.flatnonzero(valid)
        points, owners, strong = points[valid], owners[valid], strong[valid]
        self._initial_shapes = self.robot.shapes(self.bound, joints, state8[6:8])
        self_shapes = self.visual_robot.shapes(self.bound, joints, state8[6:8])
        # Project the CURRENT robot into each source view. Depth predictions on
        # or behind that silhouette cannot establish an external visible surface.
        # Masking only a 3-D shell is insufficient: biased robot depths can lie
        # outside that shell and later block the moving arm. Preserve predicted
        # foreground and explicitly assigned material; never use future poses.
        margin = self.config.collision_margin_m
        keep = np.ones(len(points), bool)
        view_slices = scene.evidence.get('scene_point_view_slices', {})
        cameras = {camera.name: camera for camera in getattr(scene.frame, 'cameras', ())}
        ray_candidates = 0
        camera_envelopes_skipped = []
        for name, interval in view_slices.items():
            if name not in cameras or len(interval) != 2:
                continue
            ids = np.flatnonzero((owners == '') & (source_indices >= interval[0]) & (source_indices < interval[1]))
            if not len(ids):
                continue
            camera = cameras[name]
            origin = -camera.world_to_camera[:3, :3].T @ camera.world_to_camera[:3, 3]
            camera_points = points[ids] @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
            projected = camera_points @ camera.intrinsics.T
            uv = projected[:, :2]/projected[:, 2:3]
            height, width = camera.rgb.shape[:2]
            raster_valid = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height-1)
            pixels = np.rint(np.where(raster_valid[:, None], uv, 0)).astype(int)
            rays = points[ids]-origin
            measured_distances = np.linalg.norm(rays, axis=1)
            rays /= np.maximum(measured_distances[:, None], 1e-12)
            for shape in self_shapes:
                # A convex envelope can contain the wrist camera even though
                # the actual mesh has an aperture. It must not mask the full view.
                if shape.contains(origin[None], 1e-7)[0]:
                    camera_envelopes_skipped.append([name, shape.name])
                    continue
                vertices = shape.surface_points() @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
                if np.all(vertices[:, 2] <= 1e-7):
                    continue
                silhouette = raster_valid
                if np.all(vertices[:, 2] > 1e-7):
                    import cv2
                    projected_vertices = vertices @ camera.intrinsics.T
                    polygon = projected_vertices[:, :2]/projected_vertices[:, 2:3]
                    polygon = cv2.convexHull(np.float32(polygon))
                    mask = np.zeros((height, width), np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(np.rint(np.clip(polygon, -1_000_000, 1_000_000))), 1)
                    silhouette = raster_valid & mask[pixels[:, 1], pixels[:, 0]].astype(bool)
                center = shape.center-origin
                along = np.einsum('ij,j->i', rays, center)
                radius = shape.radius
                broad = (keep[ids] & silhouette & (along+radius >= 0)
                         & (float(center@center)-along*along <= radius*radius)
                         & (measured_distances >= np.maximum(along-radius, 0.)-margin))
                local_ids = np.flatnonzero(broad)
                ray_candidates += len(local_ids)
                for offset in range(0, len(local_ids), 1024):
                    chunk = local_ids[offset:offset+1024]
                    entry = shape.ray_entry_distances(origin, points[ids[chunk]])
                    occluded = np.isfinite(entry) & (measured_distances[chunk] >= entry-margin)
                    keep[ids[chunk[occluded]]] = False
        self._cloud, self._owners, self._strong = points[keep], owners[keep], strong[keep]
        self._tree = cKDTree(self._cloud) if len(self._cloud) else None
        self._self_removed = int((~keep).sum())
        self._self_filter_evidence = {
            'method': 'current_robot_silhouette_ray_occlusion', 'tolerance_m': margin,
            'removed_points': self._self_removed, 'ray_candidate_visits': ray_candidates,
            'camera_containing_envelopes_skipped': camera_envelopes_skipped,
            'assigned_object_points_removed': 0, 'future_robot_poses_remove_points': False,
            'fixed_base_environment_overlap': 'existing_occupied_region_not_new_motion',
            'strong_obstacle_points': int(self._strong.sum()),
            'uncertain_obstacle_points_retained': int((~self._strong).sum()),
            'strength_source_alignment': bool(strength_alignment),
            'strong_definition': 'owned_coherent_surface_or_crossview_support_at_least_2_without_contradictions',
            'limitation': 'unassigned foreground objects with depth incorrectly predicted behind the robot may be confused with self/occlusion'}

    def _known_start_volume(self, points):
        known = np.zeros(len(points), bool)
        for shape in self._initial_shapes:
            ids = np.flatnonzero(~known)
            if len(ids):
                known[ids] |= shape.contains(points[ids], 1e-5)
        return known

    def _point_owners(self, points, distance=.012):
        if self._tree is None or not len(points):
            return np.full(len(points), '', object)
        distances, indices = self._tree.query(points)
        return np.where(distances <= distance, self._owners[indices], '')

    def _configuration(self, q, scene, fingers, held, allowed_contacts=(), *, free_check=True):
        self._budget()
        if not np.isfinite(q).all() or np.any(q < self.bound.low) or np.any(q > self.bound.high):
            return {'safe': False, 'reason': 'joint_limit'}
        shapes = self.robot.shapes(self.bound, q, fingers)
        self_hit = self.robot.self_collision(shapes)
        if self_hit:
            return {'safe': False, **self_hit}
        pose, _ = self.bound.forward(q)
        if held is not None and held.get('upright_in_gripper') is not None:
            reference_up = _unit(held['upright_in_gripper'])
            world_up = pose[:3, :3] @ reference_up
            tilt = float(np.arccos(np.clip(world_up @ np.array([0., 0., 1.]), -1., 1.)))
            if tilt > self.config.upright_tolerance_rad:
                return {'safe': False, 'reason': 'upright_constraint_violation',
                        'tilt_rad': tilt, 'allowed_tilt_rad': self.config.upright_tolerance_rad}
        payload = None
        payload_points = np.empty((0, 3))
        if held is not None:
            cache_key = id(held)
            cached = self._payload_cache.get(cache_key)
            if cached is None or cached[0] is not held:
                local_shape = _payload_shape(np.asarray(held.get('points_in_gripper', []), float).reshape(-1, 3))
                self._payload_cache[cache_key] = (held, local_shape)
                if len(self._payload_cache) > 32:
                    self._payload_cache.pop(next(iter(self._payload_cache)))
            else:
                local_shape = cached[1]
            if local_shape is not None:
                payload = CollisionShape(local_shape.name, local_shape.group, local_shape.vertices,
                                         local_shape.equations, pose, 'payload')
                payload_points = payload.surface_points()
        payload_geometry_missing = held is not None and payload is None
        object_id = None if held is None else held.get('object_id')
        owner_ids = [] if held is None else held.get('owner_ids', [object_id])
        if payload is not None:
            for shape in shapes:
                if shape.kind in ('finger', 'pad'):
                    continue  # gripped material intersects the finger envelope by design
                if intersect(payload, shape, self.config.collision_margin_m):
                    return {'safe': False, 'reason': 'payload_robot_collision', 'part': shape.name}
        uncertain_count = 0
        uncertain_examples = []
        for shape in shapes + ([payload] if payload is not None else []):
            # Fixed base volume already exists at the measured start. Its static
            # mount/table overlap is not a newly introduced collision. Keep base
            # geometry in self-collision and payload-vs-robot checks above.
            if shape.kind == 'arm' and shape.group == 0:
                continue
            if self._tree is None:
                continue
            ids = np.asarray(self._tree.query_ball_point(shape.center, shape.radius+self.config.collision_margin_m), int)
            if not len(ids):
                continue
            inside = shape.contains(self._cloud[ids], self.config.collision_margin_m)
            ids = ids[inside]
            if object_id is not None and payload is not None:
                # The held object's observed points move with its payload model.
                ids = ids[~np.isin(self._owners[ids], owner_ids)]
            if len(ids) and shape.kind in self._contact_parts:
                ids = ids[~np.isin(self._owners[ids], list(allowed_contacts))]
            if len(ids):
                strong_ids = ids[self._strong[ids]]
                if len(strong_ids):
                    index = int(strong_ids[0])
                    return {'safe': False, 'reason': 'observed_obstacle_intersection', 'part': shape.name,
                            'point_world_m': self._cloud[index].tolist(),
                            'point_object_id': str(self._owners[index]), 'obstacle_evidence': 'strong',
                            'uncertain_obstacle_samples': uncertain_count,
                            'uncertain_obstacle_examples': uncertain_examples}
                uncertain_count += len(ids)
                self._uncertain_obstacle_samples += len(ids)
                example = {'part': shape.name, 'point_world_m': self._cloud[int(ids[0])].tolist(),
                           'point_object_id': str(self._owners[int(ids[0])]),
                           'obstacle_evidence': 'uncertain_retained_surface'}
                if len(uncertain_examples) < 2:
                    uncertain_examples.append(example)
                if len(self._uncertain_obstacle_examples) < 2 and example not in self._uncertain_obstacle_examples:
                    self._uncertain_obstacle_examples.append(example)
                if not self.config.allow_unknown_space:
                    return {'safe': False, 'reason': 'uncertain_obstacle_intersection', **example,
                            'uncertain_obstacle_samples': uncertain_count,
                            'uncertain_obstacle_examples': uncertain_examples}
        warnings = {'uncertain_obstacle_samples': uncertain_count,
                    'uncertain_obstacle_examples': uncertain_examples,
                    'uncertain_obstacles_allowed': self.config.allow_unknown_space}
        if not free_check:
            return {'safe': True, 'reason': 'uncertain_obstacle_evidence' if uncertain_count else 'sampled_geometry_clear',
                    'unknown_samples': 0, 'payload_geometry_unknown': payload_geometry_missing, **warnings}
        probes = np.concatenate([s.surface_points() for s in shapes if not (s.kind == 'arm' and s.group == 0)]
                                + ([payload_points] if payload is not None else []))
        if len(probes) > self.config.free_probe_limit:
            probes = probes[np.linspace(0, len(probes)-1, self.config.free_probe_limit, dtype=int)]
        known = self._known_start_volume(probes)
        # Expected contact samples are exempt only at pads / payload; all arm
        # contact is still checked above. Unknown clearance elsewhere stays unknown.
        remaining = probes[~known]
        if not len(remaining):
            return {'safe': True, 'reason': 'uncertain_obstacle_evidence' if uncertain_count else 'within_measured_robot_volume',
                    'unknown_samples': 0, 'payload_geometry_unknown': payload_geometry_missing, **warnings}
        now = max(scene.generated_at_s, scene.captured_at_s)
        rows = scene.geometry.query(remaining, now_s=now, max_age_s=self.config.max_geometry_age_s)
        owners = self._point_owners(remaining)
        expected = np.isin(owners, list(allowed_contacts) + owner_ids)
        unknown = sum(row['state'] != 'estimated_free' and not permitted for row, permitted in zip(rows, expected))
        self._unknown_sample_visits += int(unknown)
        return {'safe': not unknown or self.config.allow_unknown_space,
                'reason': 'unknown_swept_space' if unknown else 'uncertain_obstacle_evidence' if uncertain_count else 'sampled_geometry_clear',
                'unknown_samples': int(unknown), 'unknown_space_allowed': self.config.allow_unknown_space,
                'payload_geometry_unknown': payload_geometry_missing, **warnings}

    def _edge(self, a, b, scene, fingers, held, contacts=(), *, free_check=True):
        count = max(1, int(np.ceil(np.max(np.abs(np.asarray(b)-a)) / self.config.max_joint_step_rad)))
        worst = 0; payload_unknown = False; uncertain_count = 0; uncertain_examples = []
        for q in np.linspace(a, b, count+1):
            result = self._configuration(q, scene, fingers, held, contacts, free_check=free_check)
            worst = max(worst, result.get('unknown_samples', 0))
            payload_unknown |= result.get('payload_geometry_unknown', False)
            uncertain_count += result.get('uncertain_obstacle_samples', 0)
            for example in result.get('uncertain_obstacle_examples', []):
                if len(uncertain_examples) < 2 and example not in uncertain_examples:
                    uncertain_examples.append(example)
            if not result['safe']:
                return {**result, 'uncertain_obstacle_samples': uncertain_count,
                        'uncertain_obstacle_examples': uncertain_examples}
        return {'safe': True, 'reason': 'sampled_joint_edge_allowed_with_uncertain_obstacles' if uncertain_count else
                'sampled_joint_edge_allowed_with_unknown_space' if worst else 'sampled_joint_edge_clear', 'samples': count+1,
                'unknown_samples': worst, 'payload_geometry_unknown': payload_unknown,
                'uncertain_obstacle_samples': uncertain_count, 'uncertain_obstacle_examples': uncertain_examples,
                'unknown_space_allowed': self.config.allow_unknown_space,
                'continuous_collision_certified': False}

    def check_motion(self, startq, endq, scene, state8, held=None, allowed_contacts=(), fingers=None, contact_parts=None):
        old_deadline = self._deadline
        self._deadline = perf_counter()+self.config.max_planning_s
        self._connection_diagnostics = {}
        self._uncertain_obstacle_samples = 0
        self._uncertain_obstacle_examples = []
        self._contact_parts = tuple(contact_parts or ())
        try:
            self._prepare(scene, state8, startq)
            return self._edge(np.asarray(startq, float), np.asarray(endq, float), scene,
                              np.asarray(state8[6:8] if fingers is None else fingers, float), held, allowed_contacts)
        except PlanningFailure as exc:
            return {'safe': False, 'reason': exc.reason, 'evidence': exc.evidence}
        finally:
            self._deadline = old_deadline

    def check_gripper(self, joints, scene, state8, target_fingers=None, held=None, allowed_contacts=(),
                      incremental_closure=False, contact_parts=None):
        old_deadline = self._deadline
        self._deadline = perf_counter()+self.config.max_planning_s
        self._connection_diagnostics = {}
        self._uncertain_obstacle_samples = 0
        self._uncertain_obstacle_examples = []
        self._contact_parts = tuple(contact_parts or ())
        try:
            self._prepare(scene, state8, joints)
            start = np.asarray(state8[6:8], float)
            target = np.asarray([0., 0.] if target_fingers is None else target_fingers, float)
            if incremental_closure:
                target = start+np.clip(target-start, -.003, .003)
            count = max(2, int(np.ceil(np.max(np.abs(target-start))/.003))+1)
            worst_unknown = 0
            payload_unknown = False
            for fingers in np.linspace(start, target, count):
                result = self._configuration(np.asarray(joints), scene, fingers, held, allowed_contacts)
                worst_unknown = max(worst_unknown, result.get('unknown_samples', 0))
                payload_unknown |= result.get('payload_geometry_unknown', False)
                if not result['safe']:
                    return {**result, 'uncertain_obstacle_samples': self._uncertain_obstacle_samples,
                            'uncertain_obstacle_examples': list(self._uncertain_obstacle_examples)}
            return {'safe': True, 'reason': 'sampled_gripper_sweep_allowed_with_uncertain_obstacles'
                    if self._uncertain_obstacle_samples else 'sampled_gripper_sweep_clear', 'samples': count,
                    'unknown_samples': worst_unknown, 'payload_geometry_unknown': payload_unknown,
                    'uncertain_obstacle_samples': self._uncertain_obstacle_samples,
                    'uncertain_obstacle_examples': list(self._uncertain_obstacle_examples),
                    'unknown_space_allowed': self.config.allow_unknown_space}
        except PlanningFailure as exc:
            return {'safe': False, 'reason': exc.reason, 'evidence': exc.evidence}
        finally:
            self._deadline = old_deadline

    def _densify(self, path):
        result = [np.asarray(path[0]).tolist()]
        for first, second in zip(path[:-1], path[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(np.asarray(second)-first))/.025)))
            result.extend(np.linspace(first, second, count+1)[1:].tolist())
        return result

    def _connect(self, start, goal, scene, fingers, held, contacts, contact_motion=False):
        self._connection_diagnostics = {'direct_check': None, 'last_blocked_edge': None}
        # A detour cannot repair a colliding endpoint. Report the original
        # body/point/constraint evidence immediately instead of exhausting RRT.
        for name, joints in (('start', start), ('goal', goal)):
            endpoint = self._configuration(np.asarray(joints), scene, fingers, held, contacts)
            self._connection_diagnostics[name + '_check'] = dict(endpoint)
            if not endpoint['safe']:
                failed = {**endpoint, 'segment_location': name}
                self._connection_diagnostics['direct_check'] = failed
                self._connection_diagnostics['last_blocked_edge'] = failed
                raise PlanningFailure(endpoint['reason'], {
                    **endpoint, 'endpoint': name, **self._connection_diagnostics})
        check = self._edge(start, goal, scene, fingers, held, contacts)
        self._connection_diagnostics['direct_check'] = dict(check)
        if check['safe']:
            return self._densify([np.asarray(start).tolist(), np.asarray(goal).tolist()])
        self._connection_diagnostics['last_blocked_edge'] = {**check, 'segment_location': 'direct'}
        if contact_motion or check['reason'] == 'unknown_swept_space':
            raise PlanningFailure(check['reason'], {**check, **self._connection_diagnostics})
        # A bounded joint-space RRT connects stages. All edges use the same robot,
        # payload and observed-space checks. No cached demonstration path exists.
        rng = np.random.default_rng(0)
        nodes = [np.asarray(start, float)]; parents = [-1]
        for iteration in range(self.config.rrt_iterations):
            self._budget()
            target = np.asarray(goal) if iteration % 4 == 0 else rng.uniform(self.bound.low, self.bound.high)
            nearest = int(np.argmin([np.linalg.norm(node-target) for node in nodes]))
            delta = target-nodes[nearest]
            delta *= min(1., self.config.rrt_step_rad/max(float(np.linalg.norm(delta)), 1e-12))
            trial = nodes[nearest]+delta
            extension = self._edge(nodes[nearest], trial, scene, fingers, held, contacts)
            if not extension['safe']:
                self._connection_diagnostics['last_blocked_edge'] = {
                    **extension, 'segment_location': 'rrt_extension', 'iteration': iteration}
                continue
            nodes.append(trial); parents.append(nearest)
            connection = self._edge(trial, goal, scene, fingers, held, contacts)
            if connection['safe']:
                path = [np.asarray(goal).tolist(), trial.tolist()]; index = parents[-1]
                while index >= 0:
                    path.append(nodes[index].tolist()); index = parents[index]
                return self._densify(path[::-1])
            self._connection_diagnostics['last_blocked_edge'] = {
                **connection, 'segment_location': 'rrt_goal_connection', 'iteration': iteration}
        raise PlanningFailure('no_connected_joint_path_found', {
            **check, **self._connection_diagnostics})

    def _sequence(self, operation, scene, state8, joints, keyposes, held):
        paths = []; current = np.asarray(joints, float); payload = held
        fingers = np.asarray(state8[6:8], float); cost = 0.
        for keypose in keyposes:
            self._budget()
            self._contact_parts = tuple(keypose.metadata.get('contact_parts', ()))
            result = self.bound.solve(keypose.pose[:3, 3], keypose.pose[:3, :3], current, max_iterations=80)
            if not result.ok:
                raise PlanningFailure('ik_no_solution_found', {'keypose': keypose.name,
                                      'position_error_m': result.position_error_m,
                                      'orientation_error_deg': result.orientation_error_deg})
            goal = np.asarray(result.joints)
            path = self._connect(current, goal, scene, fingers, payload, keypose.allowed_contacts,
                                 contact_motion=keypose.profile == 'contact')
            paths.append(path)
            cost += sum(float(np.linalg.norm(np.asarray(b)-a)) for a, b in zip(path[:-1], path[1:]))
            # A close command is force-limited physical interaction, not an empty
            # gripper trajectory through the object to q=0. Runtime checks the
            # next measured 3 mm finger increment while waiting for settling.
            if keypose.gripper == 'open':
                new_fingers = np.array([.04, -.04])
                for intermediate in np.linspace(fingers, new_fingers, 6):
                    check = self._configuration(goal, scene, intermediate, payload, keypose.allowed_contacts)
                    if not check['safe']:
                        raise PlanningFailure(check['reason'], {'keypose': keypose.name, **check})
                fingers = new_fingers
            current = goal
        return OperationPlan(operation, keyposes, paths, cost, {
            'planner': 'astra_keypose_next_segment_joint_connector', 'scene_frame_id': scene.frame_id,
            'robot_geometry': self.robot.metadata, 'self_filter': self._self_filter_evidence,
            'self_surface_points_removed': self._self_removed,
            'continuous_collision_certified': False, 'complete_object_shape': False,
            'unknown_space_allowed': self.config.allow_unknown_space,
            'unknown_sample_visits': self._unknown_sample_visits,
            'uncertain_obstacle_samples': self._uncertain_obstacle_samples,
            'uncertain_obstacle_examples': list(self._uncertain_obstacle_examples),
            'uncertain_count_semantics': 'surface_sample_intersections_including_repeated_planning_checks',
            'future_keyposes_checked': False,
            'payload_geometry_unknown': held is not None and (held.get('geometry_unknown', False)
                or len(held.get('points_in_gripper', [])) < 4)})

    def _astra_keypose(self, operation, scene, state8, held=None):
        entries = operation.get('keyposes')
        index = operation.get('_keypose_index', 0)
        if type(index) is not int or index < 0 or index >= len(entries):
            raise PlanningFailure('keypose_index_out_of_range', {'index': index, 'count': len(entries)})
        entry = entries[index]
        pose_data = entry['pose']
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(pose_data['position'], float)
        pose[:3, :3] = quaternion_matrix(pose_data['quaternion_wxyz'])
        if not np.isfinite(pose).all():
            raise PlanningFailure('invalid_astra_keypose', {'index': index})
        name = entry['name']
        object_id = operation.get('object')
        source_ids = tuple(dict.fromkeys(value for value in (object_id, operation.get('grasp_region')) if value))
        contacts = source_ids if name in ('grasp', 'close', 'lift') else (
            (operation['target'],) if name in ('place', 'release') and operation.get('target') else ())
        parts = ['finger', 'pad'] if name in ('grasp', 'close', 'lift') else (
            ['payload'] if name in ('place', 'release') else [])
        metadata = {'astra_keypose_index': index, 'observe_after': bool(entry.get('observe_after', False)),
                    'pose_source': 'astra', 'contact_parts': parts}
        gripper = entry.get('gripper', 'keep')
        if gripper == 'open':
            metadata['fingers'] = [.04, -.04]
        if name == 'release' or (gripper == 'open' and held is not None):
            metadata['release'] = True
        if gripper == 'close':
            metadata['incremental_closure'] = True
            metadata['fingers'] = [0., 0.]
        if gripper == 'close' and operation.get('contact_mode') != 'articulated':
            payload = {'object_id': object_id, 'owner_ids': list(source_ids),
                       'points_in_gripper': [], 'confirmed': False,
                       'source': 'astra_close_hypothesis', 'geometry_unknown': True}
            obj = scene.objects.get(object_id)
            if obj is not None:
                from .surface import planning_surface
                points, evidence = planning_surface(obj)
                metadata['payload_surface_evidence'] = evidence
                if len(points) >= 4:
                    local = (np.asarray(points)-pose[:3, 3]) @ pose[:3, :3]
                    payload.update(points_in_gripper=local.tolist(),
                                   source='astra_close_visible_surface_hypothesis', geometry_unknown=False)
                else:
                    metadata['payload_geometry'] = 'unknown'
            else:
                metadata['payload_geometry'] = 'unknown'
            if operation.get('keep_upright'):
                payload['upright_in_gripper'] = (pose[:3, :3].T @ np.array([0., 0., 1.])).tolist()
            metadata['payload_after'] = payload
        if operation.get('contact_mode') == 'articulated' and name == 'close':
            metadata['articulated_engagement'] = True
        return KeyPose(name, pose, gripper, entry.get('profile', 'transit'), contacts, metadata)

    def plan(self, operation, scene, state8, joints, held=None):
        started = perf_counter(); self._deadline = started+self.config.max_planning_s
        self.last_candidates = []; self.last_rejections = []; self._unknown_sample_visits = 0
        self._connection_diagnostics = {}
        self._uncertain_obstacle_samples = 0
        self._uncertain_obstacle_examples = []
        try:
            self._prepare(scene, state8, joints)
            if operation.get('keyposes'):
                keypose = self._astra_keypose(operation, scene, state8, held)
            else:
                # Compatibility for standalone directed movement/observation.
                # Manipulation now requires model-specified key poses.
                kind = operation['kind']; goal = _pose(state8)
                metadata = {'pose_source': 'astra', 'contact_parts': [], 'observe_after': False}
                if kind in ('move', 'contact_move'):
                    value = operation.get('goal_pose')
                    if value is not None:
                        goal[:3, 3] = value['position']
                        goal[:3, :3] = quaternion_matrix(value['quaternion_wxyz'])
                    elif operation.get('displacement') is not None:
                        goal[:3, 3] += operation['displacement']
                    else:
                        raise PlanningFailure('motion_goal_required')
                    contacts = tuple(value for value in (operation.get('object'), operation.get('grasp_region')) if value) if kind == 'contact_move' else ()
                    if kind == 'contact_move':
                        metadata['contact_parts'] = ['finger', 'pad']
                    keypose = KeyPose(kind, goal, profile='contact' if kind == 'contact_move' else 'transit',
                                      allowed_contacts=contacts, metadata=metadata)
                elif kind == 'release':
                    metadata.update(release=True, fingers=[.04, -.04], contact_parts=['payload'])
                    keypose = KeyPose('release', goal, 'open', 'contact', tuple(filter(None, [operation.get('target')])), metadata)
                elif kind == 'observe':
                    metadata.update(request_observation=True, observe_after=True)
                    keypose = KeyPose('observe', goal, metadata=metadata)
                else:
                    raise PlanningFailure('astra_keyposes_required', {'kind': kind})
            self.last_candidates = [[keypose.to_dict()]]
            result = self._sequence(operation, scene, state8, joints, [keypose], held)
            result.evidence.update(planning_s=perf_counter()-started, considered=1, candidate_count=1,
                                   astra_keypose_index=keypose.metadata.get('astra_keypose_index'))
            return result
        except PlanningFailure as exc:
            # Keep the original geometric cause; callers need part/point/IK
            # details rather than a generic no_complete_operation_path label.
            detail = {'reason': exc.reason, 'evidence': exc.evidence,
                      'astra_keypose_index': operation.get('_keypose_index', 0),
                      'planning_s': perf_counter()-started}
            self.last_rejections.append(detail)
            raise PlanningFailure(exc.reason, detail) from exc
        finally:
            self._deadline = float('inf')
