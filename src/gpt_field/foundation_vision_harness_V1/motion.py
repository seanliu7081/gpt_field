"""Continuous Cartesian tracking with measured-joint, short-horizon checks.

The far route uses nominal kinematics for candidate screening only. Each command
is checked again from measured q/qd using an OSC response approximation and an
explicit joint prediction-error envelope. This is sampled predicted-geometry
avoidance, not a certified dynamic/continuous collision guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree, ConvexHull

from ..foundation_harness.robot_geometry import RobotCollisionModel, CollisionShape, intersect
from ..foundation_harness.types import KeyPose, jsonable
from ..robodojo_direct_sim import PandaKinematics, rotation_matrix, rotation_vector, matrix_quaternion
from .geometry import pose_from_state


class PlanningError(RuntimeError):
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}


@dataclass(frozen=True)
class MotionConfig:
    control_period_s: float = .05
    # Generic execution settings, not benchmark-tuned geometry.
    transit_command_m: float = .06
    contact_command_m: float = .004
    rotation_command_rad: float = .08
    lookahead_m: float = .035
    contact_lookahead_m: float = .008
    position_tolerance_m: float = .003
    rotation_tolerance_rad: float = .05
    collision_margin_m: float = .003
    voxel_m: float = .006
    maximum_scene_points: int = 100000
    planning_configuration_budget: int = 40
    prediction_steps: int = 3
    joint_error_floor_rad: float = .003
    joint_prediction_limit_rad: float = .10
    # Fit from public robot response logs, inherited from uc7.2 EXC-0 plan.
    translation_a: float = .711
    translation_b: float = .0732
    rotation_a: float = .90
    rotation_b: float = .027
    jacobian_damping: float = .008
    stale_slow_s: float = 5.
    unavailable_scene_s: float = 20.


@dataclass
class RollingPlan:
    keypose: KeyPose
    waypoints: list[np.ndarray]
    scene_revision: int
    index: int = 1
    evidence: dict = field(default_factory=dict)
    last_command: dict | None = None
    reference_up: np.ndarray | None = None

    def to_dict(self):
        return jsonable({"keypose": self.keypose.to_dict(), "waypoints": self.waypoints,
            "scene_revision": self.scene_revision, "index": self.index, "evidence": self.evidence})


def _interpolate(a, b, fraction):
    pose = np.eye(4)
    pose[:3, 3] = a[:3, 3] + fraction*(b[:3, 3]-a[:3, 3])
    pose[:3, :3] = rotation_matrix(rotation_vector(b[:3, :3] @ a[:3, :3].T)*fraction) @ a[:3, :3]
    return pose


def _payload_shape(held, pose):
    if held is None:
        return None
    points = np.asarray(held.get("points_in_gripper", []), float).reshape(-1, 3)
    if held.get("extent_unknown") or len(points) < 4:
        raise PlanningError("payload_extent_unknown", {"object_id": held.get("object_id")})
    points = points[np.isfinite(points).all(1)]
    if len(points) < 4:
        raise PlanningError("payload_extent_unknown")
    # A visible extent box fills internal holes conservatively; it does NOT
    # assign all environment points inside that box to the carried object.
    sigma = np.asarray(held.get("sigma_m", 0.), float)
    lo, hi = points.min(0)-sigma, points.max(0)+sigma
    hi = np.maximum(hi, lo+.001)
    vertices = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    hull = ConvexHull(vertices)
    return CollisionShape("held_observed_extent", 9, vertices, hull.equations, pose, "payload")


class RollingPlanner:
    def __init__(self, kinematics=None, config=None, *, robot_model=None):
        self.kinematics = kinematics or PandaKinematics.from_xml()
        self.config = config or MotionConfig()
        self.robot = robot_model or RobotCollisionModel(self.kinematics)
        self._scene_key = None
        self._scene_ref = None
        self._tree = None
        self._points = np.empty((0,3))
        self._owners = np.empty(0, object)
        self._sigma = np.empty(0)
        self._maximum_sigma = 0.
        self._bound = self.kinematics
        self._predicted_next_q = None
        self._last_q = None
        self._error = np.full(7, self.config.joint_error_floor_rad)
        self._last_velocity = np.zeros(6)
        self._evaluations = 0
        self.stats = {"scene_preparations": 0, "plan_calls": 0, "command_calls": 0,
                      "planning_s": 0., "command_s": 0.}

    def _prepare(self, scene, state8, q7):
        self._bound = self.kinematics.bind(state8, q7)
        if scene.evidence.get("perception_status") in {"unavailable", "error"}:
            raise PlanningError("scene_unavailable")
        key = (id(scene), scene.evidence.get("revision", scene.frame_id))
        if key == self._scene_key:
            return
        points = np.asarray(scene.scene_points, float).reshape(-1,3)
        owners = getattr(scene, "scene_point_object_ids", None)
        owners = np.full(len(points), "", object) if owners is None else np.asarray(owners, object)
        if owners.shape != (len(points),):
            raise PlanningError("scene_ownership_misaligned")
        sigma = np.asarray(scene.evidence.get("point_sigma_m", .005), float)
        if sigma.ndim == 0:
            sigma = np.full(len(points), float(sigma))
        elif sigma.shape == (len(points), 3):
            sigma = np.linalg.norm(sigma, axis=1)
        if sigma.shape != (len(points),) or not np.isfinite(sigma).all() or np.any(sigma < 0):
            raise PlanningError("scene_uncertainty_misaligned")
        valid = np.isfinite(points).all(1)
        points, owners, sigma = points[valid], owners[valid], sigma[valid]
        # Deterministic surface voxelisation retains every occupied cell. Keep
        # owner identities distinct in shared cells; a target may not erase an
        # obstacle merely because their predicted surfaces overlap.
        if len(points):
            voxel = np.floor(points/self.config.voxel_m).astype(np.int64)
            labels, owner_index = np.unique(owners.astype(str), return_inverse=True)
            keys = np.column_stack((voxel, owner_index))
            _, indices, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            conservative_sigma = np.zeros(len(indices))
            regroup_distance=np.linalg.norm(points-points[indices][inverse],axis=1)
            np.maximum.at(conservative_sigma, inverse, sigma+regroup_distance)
            points, owners = points[indices], owners[indices]
            sigma = conservative_sigma
        # Do not drop occupied cells to meet an arbitrary point-count target.
        # Source worker can adapt resolution; this limit diagnoses runaway data.
        if len(points) > self.config.maximum_scene_points:
            raise PlanningError("scene_resolution_budget", {"occupied_cells": len(points)})
        self._points, self._owners, self._sigma = points, owners, sigma
        self._maximum_sigma = float(sigma.max(initial=0.))
        self._tree = cKDTree(points) if len(points) else None
        self._scene_key = key
        self._scene_ref = scene
        self.stats["scene_preparations"] += 1

    def _joint_origins(self, q):
        bound = self._bound
        if not all(hasattr(bound, name) for name in ("fixed", "axes", "anchors", "world_from_base")):
            return None
        pose = bound.world_from_base.copy()
        origins, axes = [], []
        for index in range(7):
            pose = pose @ bound.fixed[index]
            origins.append(pose[:3, 3]+pose[:3, :3]@bound.anchors[index])
            axes.append(pose[:3, :3]@bound.axes[index])
            R = rotation_matrix(bound.axes[index]*q[index])
            transform = np.eye(4)
            transform[:3,:3] = R
            transform[:3,3] = bound.anchors[index]-R@bound.anchors[index]
            pose = pose@transform
        return np.asarray(origins), np.asarray(axes)

    def _part_band(self, shape, joint_band, joint_geometry):
        if joint_band is None:
            return 0.
        if joint_geometry is None:
            return float(np.sum(joint_band))*.5
        origins, axes = joint_geometry
        count = min(7, max(0, shape.group))
        if count == 0:
            return 0.
        lever = np.linalg.norm(np.cross(axes[:count], shape.center-origins[:count]), axis=1)+shape.radius
        # First-order joint-to-body uncertainty plus a second-order allowance.
        angles = np.asarray(joint_band)[:count]
        return float(lever @ (angles+.5*angles**2))

    def _check(self, q, fingers, keypose, held, *, joint_band=None, enforce_budget=False):
        if enforce_budget:
            self._evaluations += 1
            if self._evaluations > self.config.planning_configuration_budget:
                raise PlanningError("planning_evaluation_budget", {"evaluations": self._evaluations-1})
        q = np.asarray(q, float)
        if np.any(q <= self.kinematics.low) or np.any(q >= self.kinematics.high):
            violation=np.maximum(self.kinematics.low-q,0.)+np.maximum(q-self.kinematics.high,0.)
            return {"safe": False, "reason": "joint_limit", "conflicts": {"joint_limit":float(violation.max())+1e-9}, "examples": []}
        pose, _ = self._bound.forward(q)
        shapes = self.robot.shapes(self._bound, q, fingers)
        conflicts, examples = {}, []
        self_hit = self.robot.self_collision(shapes)
        if self_hit:
            conflicts["self:"+"/".join(self_hit.get("parts", []))] = 1.
            examples.append(self_hit)
        payload = _payload_shape(held, pose)
        held_ids = set() if held is None else set(held.get("owner_ids", [held.get("object_id")]))
        permitted_ids = set(keypose.allowed_contacts)
        permitted_parts = set(keypose.metadata.get("contact_parts", []))
        joint_geometry = self._joint_origins(q) if joint_band is not None else None
        if payload is not None:
            for shape in shapes:
                if shape.kind in {"finger", "pad"}:
                    continue
                if intersect(payload, shape, self.config.collision_margin_m):
                    conflicts["payload_robot:"+shape.name] = 1.
                    examples.append({"part": shape.name, "reason": "payload_robot_collision"})
        for shape in shapes+([payload] if payload is not None else []):
            # Static base mounting contact cannot be changed by an arm command.
            if shape.kind == "arm" and shape.group == 0:
                continue
            if self._tree is None:
                continue
            motion_band = self._part_band(shape, joint_band, joint_geometry)
            margin = self.config.collision_margin_m+motion_band
            indices = np.asarray(self._tree.query_ball_point(shape.center,
                shape.radius+margin+self._maximum_sigma), int)
            if not len(indices):
                continue
            own = np.isin(self._owners[indices], list(held_ids)) if held_ids else np.zeros(len(indices), bool)
            if shape.kind in permitted_parts:
                permissions = keypose.metadata.get("contact_permissions")
                part_ids = permitted_ids if permissions is None else {
                    owner for owner, parts in permissions.items() if shape.kind in parts}
                permitted = np.isin(self._owners[indices], list(part_ids))
                plane = keypose.metadata.get("payload_contact_plane")
                if shape.kind == "payload" and plane:
                    n = np.asarray(plane["normal"],float)
                    origin = np.asarray(plane["point"],float)
                    permitted &= np.abs((self._points[indices]-origin)@n)<=float(plane["thickness_m"])
                own |= permitted
            indices = indices[~own]
            if not len(indices):
                continue
            local = (self._points[indices]-shape.pose[:3,3])@shape.pose[:3,:3]
            phi = np.max(local@shape.equations[:,:3].T+shape.equations[:,3], axis=1)
            depths = margin+self._sigma[indices]-phi
            bad = np.flatnonzero(depths >= 0.)
            if not len(bad):
                continue
            # Group by physical owner; unlabelled samples retain spatial cells.
            for local_index in bad:
                index = int(indices[local_index]); owner = str(self._owners[index])
                token = owner or "cell:"+",".join(map(str, np.floor(self._points[index]/.02).astype(int)))
                key = shape.name+"|"+token
                conflicts[key] = max(conflicts.get(key, 0.), float(depths[local_index]))
            deepest = int(bad[np.argmax(depths[bad])]); index = int(indices[deepest])
            examples.append({"part": shape.name, "kind": shape.kind,
                "point_object_id": str(self._owners[index]), "point_world_m": self._points[index].tolist(),
                "overlap_m": float(depths[deepest]), "motion_band_m": motion_band,
                "sigma_m": float(self._sigma[index]), "reason": "observed_surface_intersection"})
        return {"safe": not conflicts, "reason": "sampled_clear" if not conflicts else "observed_collision_risk",
                "conflicts": conflicts, "examples": examples[:8], "continuous_collision_certified": False}

    @staticmethod
    def _nonworsening(start, result):
        before, after = start["conflicts"], result["conflicts"]
        # Boolean convex overlap has no measured penetration depth. It cannot
        # justify a claim that persisting self/payload collision is improving.
        if any(key.startswith(("self:","payload_robot:")) for key in after):
            return False
        return (not (set(after)-set(before)) and all(value <= before[key]+1e-6 for key,value in after.items()))

    def _coarse_route(self, route, q0, state8, keypose, held):
        q = np.asarray(q0, float)
        first = self._check(q, state8[6:8], keypose, held, enforce_budget=True)
        previous = first
        last = first
        for a, b in zip(route[:-1], route[1:]):
            distance = np.linalg.norm(b[:3,3]-a[:3,3])
            angle = np.linalg.norm(rotation_vector(b[:3,:3]@a[:3,:3].T))
            samples = max(1, min(5, int(np.ceil(max(distance/.05, angle/.4)))))
            for fraction in np.linspace(0., 1., samples+1)[1:]:
                sample = _interpolate(a,b,fraction)
                solved = self._bound.solve(sample[:3,3], sample[:3,:3], q, max_iterations=60)
                if not solved.ok:
                    return False, {"reason": "nominal_reachability_unresolved", "position_error_m": solved.position_error_m}
                q = np.asarray(solved.joints)
                last = self._check(q, state8[6:8], keypose, held, enforce_budget=True)
                if not last["safe"]:
                    if first["safe"] or not self._nonworsening(previous,last):
                        return False, last
                previous = last
        if not last["safe"]:
            return False, {**last, "reason": "escape_did_not_clear_existing_conflicts"}
        return True, {"reason": "nominal_cartesian_route_clear", "start_conflicts": len(first["conflicts"]),
                      "future_arm_path_is_nominal_only": True}

    def _routes(self, current, goal, held):
        # Only real bends/rotation sites are waypoints; no stop-and-go dense list.
        rotate_here = current.copy(); rotate_here[:3,:3] = goal[:3,:3]
        translate_first = goal.copy(); translate_first[:3,:3] = current[:3,:3]
        yield [current, rotate_here, goal], "rotate_then_translate"
        yield [current, translate_first, goal], "translate_then_rotate"
        displacement = goal[:3,3]-current[:3,3]
        lateral = np.cross(displacement, [0.,0.,1.])
        if np.linalg.norm(lateral) < 1e-8:
            lateral = np.array([1.,0.,0.])
        lateral /= np.linalg.norm(lateral)
        payload_radius = 0.
        if held is not None:
            points = np.asarray(held.get("points_in_gripper", []), float).reshape(-1,3)
            if len(points):
                payload_radius = float(np.max(np.linalg.norm(points, axis=1)))
        offset = max(.08, payload_radius)+2*self._maximum_sigma
        for direction in (np.array([0.,0.,1.]), lateral, -lateral):
            a, b = current.copy(), goal.copy()
            a[:3,3] += direction*offset
            b[:3,3] += direction*offset
            a[:3,:3] = current[:3,:3]
            turn = a.copy(); turn[:3,:3] = goal[:3,:3]
            yield [current,a,turn,b,goal], "clearance_detour"

    def plan(self, keypose, scene, state8, q7, *, held=None):
        started = perf_counter(); self.stats["plan_calls"] += 1
        try:
            self._prepare(scene,state8,q7)
            _payload_shape(held,pose_from_state(state8))
            current = pose_from_state(state8)
            candidates = keypose.metadata.get("candidate_poses") or [keypose.pose]
            failures = []
            self._evaluations = 0
            # Candidate permutations are local measured surface alternatives.
            # Expensive broad searches are deliberately outside the control loop.
            for candidate in candidates:
                goal = np.asarray(candidate,float)
                trial = KeyPose(keypose.name,goal,keypose.gripper,keypose.profile,keypose.allowed_contacts,dict(keypose.metadata))
                for route,kind in self._routes(current,goal,held):
                    if trial.profile == "contact" and kind == "clearance_detour":
                        continue
                    try:
                        valid, detail = self._coarse_route(route,q7,np.asarray(state8),trial,held)
                    except PlanningError as exc:
                        raise PlanningError(exc.reason,{**exc.evidence,"attempts":failures[-6:]}) from exc
                    if valid:
                        # Remove exactly redundant points, preserving rotations.
                        compact = [route[0]]
                        compact.extend(p for p in route[1:] if not np.allclose(p,compact[-1],atol=1e-10))
                        if len(compact)==1:
                            compact.append(goal)
                        return RollingPlan(trial,compact,int(scene.evidence.get("revision",scene.frame_id)),
                            evidence={"route_kind":kind,"evaluations":self._evaluations,
                                "nominal_check":detail,"actual_control":"measured_joint_short_horizon_OSC",
                                "continuous_collision_certified":False,"planning_s":perf_counter()-started},
                            reference_up=current[:3,:3].T@np.array([0.,0.,1.]) if trial.metadata.get("keep_upright") else None)
                    failures.append({"route":kind,**detail})
            raise PlanningError("no_local_route",{"attempts":failures[-6:],"evaluations":self._evaluations})
        finally:
            self.stats["planning_s"] += perf_counter()-started

    def _target(self, plan, current):
        # Advance through a transit bend when its incoming segment has been
        # passed; close/final endpoints still use measured arrival tolerances.
        while plan.index < len(plan.waypoints)-1:
            a,b = plan.waypoints[plan.index-1],plan.waypoints[plan.index]
            distance = np.linalg.norm(b[:3,3]-current[:3,3])
            angle = np.linalg.norm(rotation_vector(b[:3,:3]@current[:3,:3].T))
            direction = b[:3,3]-a[:3,3]
            passed = np.linalg.norm(direction)>1e-8 and (current[:3,3]-b[:3,3])@direction>=0
            if (distance <= self.config.position_tolerance_m or passed) and angle <= self.config.rotation_tolerance_rad:
                plan.index += 1
            else:
                break
        waypoint = plan.waypoints[plan.index]
        return waypoint

    def _prediction(self, state8, q7, qd7, delta, rotation_delta):
        q = np.asarray(q7,float).copy()
        qd = np.zeros(7) if qd7 is None else np.asarray(qd7,float)
        if qd.shape!=(7,) or not np.isfinite(qd).all():
            qd=np.zeros(7)
        _, J = self._bound.forward(q)
        velocity = J@qd
        velocity = velocity if qd7 is not None else self._last_velocity.copy()
        output=[]; first_velocity=None
        for step in range(1,self.config.prediction_steps+1):
            velocity[:3] = self.config.translation_a*velocity[:3]+self.config.translation_b*delta/self.config.control_period_s
            velocity[3:] = self.config.rotation_a*velocity[3:]+self.config.rotation_b*rotation_delta/self.config.control_period_s
            if first_velocity is None:
                first_velocity=velocity.copy()
            _, J = self._bound.forward(q)
            pinv = J.T@np.linalg.solve(J@J.T+self.config.jacobian_damping**2*np.eye(6),np.eye(6))
            # Preserve observed redundant motion instead of assuming zero
            # nullspace. Unmodelled posture response enters per-joint bands.
            null_velocity = (np.eye(7)-pinv@J)@qd
            next_velocity = pinv@velocity+null_velocity
            q += next_velocity*self.config.control_period_s
            band = self._error*np.sqrt(step)+np.abs(null_velocity)*self.config.control_period_s*.25*step
            output.append((q.copy(),band))
            qd=next_velocity
        return output,first_velocity

    def command(self, plan, scene, state8, q7, qd7=None, *, held=None):
        started=perf_counter(); self.stats["command_calls"]+=1
        state=np.asarray(state8,float); q=np.asarray(q7,float)
        current=pose_from_state(state)
        def stopped(reason,evidence=None):
            return {"delta_xyz":np.zeros(3),"quaternion_xyzw":matrix_quaternion(current[:3,:3]),
                "gripper":"keep","arrived":False,"blocked":reason,
                "evidence":{"reason":reason,**(evidence or {}),"continuous_collision_certified":False}}
        try:
            self._prepare(scene,state,q)
            _payload_shape(held,current)
            age=max(0.,perf_counter()-scene.captured_at_s)
            if age>self.config.unavailable_scene_s:
                return stopped("scene_expired",{"age_s":age})
            if self._predicted_next_q is not None and self._last_q is not None and not np.array_equal(q,self._last_q):
                residual=np.abs(q-self._predicted_next_q)
                if np.max(residual)>self.config.joint_prediction_limit_rad:
                    self._predicted_next_q=None
                    return stopped("OSC_prediction_outside_validity",{"joint_residual_rad":residual.tolist()})
                self._error=np.maximum(self.config.joint_error_floor_rad,np.maximum(.95*self._error,residual))
            target=self._target(plan,current)
            goal=plan.keypose.pose
            distance=float(np.linalg.norm(goal[:3,3]-current[:3,3]))
            angle=float(np.linalg.norm(rotation_vector(goal[:3,:3]@current[:3,:3].T)))
            arrived=(plan.index==len(plan.waypoints)-1 and distance<=self.config.position_tolerance_m
                     and angle<=self.config.rotation_tolerance_rad)
            gripper=plan.keypose.gripper if arrived else "keep"
            delta=target[:3,3]-current[:3,3]
            remaining=float(np.linalg.norm(delta))
            cap=self.config.contact_command_m if plan.keypose.profile=="contact" or age>self.config.stale_slow_s else self.config.transit_command_m
            # A nonzero command remains available inside intermediate arrival
            # bands; there is no deadband that can strand a moving controller.
            speed=0. if qd7 is None else float(np.linalg.norm(self._bound.forward(q)[1][:3]@np.asarray(qd7)))
            coast=speed*self.config.control_period_s*self.config.translation_a/(1-self.config.translation_a)
            braking=max(self.config.position_tolerance_m*.25,remaining-coast)
            lookahead=self.config.contact_lookahead_m if plan.keypose.profile=="contact" else self.config.lookahead_m
            magnitude=min(cap,lookahead,braking,remaining)
            delta=delta*magnitude/remaining if remaining>1e-10 else np.zeros(3)
            dr=rotation_vector(target[:3,:3]@current[:3,:3].T)
            dr*=min(1.,self.config.rotation_command_rad/max(float(np.linalg.norm(dr)),1e-12))
            if arrived:
                delta=np.zeros(3); dr=np.zeros(3)
            start=self._check(q,state[6:8],plan.keypose,held)
            best=None; failures=[]
            for scale in (1.,.5,.25):
                prediction,velocity=self._prediction(state,q,qd7,delta*scale,dr*scale)
                previous=start; valid=True
                reports=[]
                for predicted,band in prediction:
                    checked=self._check(predicted,state[6:8],plan.keypose,held,joint_band=band)
                    reports.append(checked)
                    # To compare an initially overlapping uncertainty envelope,
                    # compare it at the same prediction band, not against a
                    # zero-band start (which would spuriously reject every exit).
                    if not checked["safe"]:
                        baseline=self._check(q,state[6:8],plan.keypose,held,joint_band=band)
                        if baseline["safe"] or not self._nonworsening(baseline,checked):
                            valid=False; failures.append(checked); break
                    previous=checked
                if valid:
                    best=(scale,prediction,velocity,reports);break
            if best is None:
                return stopped("rolling_sweep_blocked",{"attempts":failures[-3:]})
            scale,prediction,velocity,reports=best
            if gripper in {"open","close"}:
                destination=np.array([.04,-.04]) if gripper=="open" else np.zeros(2)
                # Check the next physical jaw increment, not an instantaneous
                # fully closed hand that would lie inside the intended object.
                next_fingers=state[6:8]+np.clip(destination-state[6:8],-.003,.003)
                for fraction in (.5,1.):
                    report=self._check(q,state[6:8]+fraction*(next_fingers-state[6:8]),plan.keypose,held)
                    if not report["safe"] and not self._nonworsening(start,report):
                        return stopped("jaw_sweep_blocked",report)
            commanded_R=rotation_matrix(dr*scale)@current[:3,:3]
            if plan.reference_up is not None:
                up=commanded_R@plan.reference_up
                if np.arccos(np.clip(up@[0.,0.,1.],-1.,1.))>.10:
                    return stopped("upright_constraint_violation")
            self._predicted_next_q=prediction[0][0]
            self._last_q=q.copy(); self._last_velocity=velocity
            evidence={"position_error_m":distance,"orientation_error_rad":angle,"waypoint_index":plan.index,
                "scene_revision":scene.evidence.get("revision",scene.frame_id),"scene_age_s":age,
                "prediction_samples":len(prediction),"joint_error_band_rad":self._error.tolist(),
                "velocity_source":"measured_qd" if qd7 is not None else "previous_OSC_prediction",
                "redundant_motion":"measured_nullspace_velocity_with_error_band",
                "scale":scale,"existing_conflicts":len(start["conflicts"]),
                "unknown_space_is_not_certified_free":True,"continuous_collision_certified":False}
            plan.last_command=evidence
            return {"delta_xyz":delta*scale,"quaternion_xyzw":matrix_quaternion(commanded_R),
                    "gripper":gripper,"arrived":arrived,"blocked":None,"evidence":evidence}
        except PlanningError as exc:
            return stopped(exc.reason,exc.evidence)
        finally:
            self.stats["command_s"]+=perf_counter()-started
