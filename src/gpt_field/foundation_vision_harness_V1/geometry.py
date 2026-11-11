"""Ground image-selected material and support regions into local tool poses.

No task registry, scene truth, object class dimensions, or learned task policy is
used here. A region denotes observed material, not the centre of its parent.
Every call solves ONE execution phase; missing future placement geometry cannot
prevent a useful approach. Numbers describe the Panda tool or generic numerical
resolution, and are not fitted to a benchmark layout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from ..foundation_harness.types import KeyPose
from ..robodojo_direct_sim import rotation_matrix, rotation_vector


class GroundingError(RuntimeError):
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence or {}


@dataclass(frozen=True)
class GroundingConfig:
    # Static Panda tool dimensions, in the public right_hand/grip_site frame.
    max_gap_m: float = .08
    pad_depth_m: float = .016
    fingertip_forward_m: float = .00925
    palm_back_m: float = .03104
    numerical_margin_m: float = .003
    # Approach/lift margins scale with the tool and measured uncertainty.
    approach_tool_lengths: float = 2.
    lift_tool_lengths: float = 2.
    maximum_surface_points: int = 4096


def pose_from_state(state8):
    state = np.asarray(state8, float)
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix(state[3:6])
    pose[:3, 3] = state[:3]
    return pose


def unit(vector, fallback=None):
    value = np.asarray(vector, float) if vector is not None else np.asarray(fallback, float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-9:
        raise GroundingError("direction_unavailable")
    return value / np.linalg.norm(value)


def observed_points(obj, limit=4096):
    """Retain all supported components; never use the old all-or-none veto."""
    if obj is None:
        return np.empty((0, 3))
    points = np.asarray(getattr(obj, "points", []), float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    if len(points) > limit:
        points = points[np.linspace(0, len(points)-1, limit, dtype=int)]
    return points


def object_sigma(obj):
    evidence = getattr(obj, "evidence", {}) or {}
    raw = evidence.get("sigma_m", evidence.get("position_sigma_m", .01))
    sigma = np.asarray(raw, float)
    if sigma.ndim == 0:
        sigma = np.repeat(sigma, 3)
    if sigma.shape != (3,) or not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise GroundingError("invalid_geometry_uncertainty")
    return sigma


def _basis(approach, jaw_axis):
    """Panda jaw motion is right_hand local x; local z points to the tips."""
    z = unit(approach)
    x = np.asarray(jaw_axis, float) - z * float(np.asarray(jaw_axis) @ z)
    if np.linalg.norm(x) < 1e-7:
        reference = np.eye(3)[np.argmin(np.abs(z))]
        x = np.cross(reference, z)
    x = unit(x)
    return np.column_stack((x, unit(np.cross(z, x)), z))


def _support_frame(points, sigma, evidence):
    supplied = evidence.get("support_plane") or {}
    if supplied:
        normal = unit(supplied.get("normal"), [0., 0., 1.])
        center = np.asarray(supplied.get("point", np.median(points, axis=0)), float)
        if "offset" in supplied and "point" not in supplied:
            center -= normal * (float(center @ normal) + float(supplied["offset"]))
    else:
        # Find a locally horizontal observed patch. Whole-cloud PCA on a rim or
        # wall can otherwise invent a tilted support through empty space.
        low, high = np.quantile(points[:, 2], [.1, .9])
        bandwidth = max(float(sigma[2])*2, .003)
        levels = np.linspace(low, high, min(16, max(2, len(points)//8)))
        groups = [points[np.abs(points[:, 2]-z) <= bandwidth] for z in levels]
        patch = max(groups, key=len)
        if len(patch) < 6:
            raise GroundingError("support_surface_unavailable")
        center = np.median(patch, axis=0)
        _, _, axes = np.linalg.svd(patch-center, full_matrices=False)
        normal = unit(axes[-1])
        if normal[2] < 0:
            normal *= -1
        if normal[2] < .7:
            raise GroundingError("support_not_observed", {"normal": normal.tolist()})
        residual = np.abs((patch-center) @ normal)
        if np.quantile(residual, .9) > max(.003, 2*float(np.linalg.norm(sigma))):
            raise GroundingError("support_surface_inconsistent")
        points = patch
    x = unit(np.cross([0., 1., 0.], normal), [1., 0., 0.])
    y = unit(np.cross(normal, x))
    axes = np.column_stack((x, y))
    xy = (points-center) @ axes
    try:
        hull = ConvexHull(xy)
    except QhullError as exc:
        raise GroundingError("support_footprint_unavailable") from exc
    return center, normal, axes, hull


class Grounder:
    """A small candidate generator, grounded entirely in the current scene."""

    def __init__(self, kinematics=None, config=None):
        self.kinematics = kinematics
        self.config = config or GroundingConfig()
        self._contact_cache = {}

    def _region(self, operation, scene, *, required=True):
        object_id = operation.get("grasp_region") or operation.get("object")
        obj = scene.objects.get(object_id)
        points = observed_points(obj, self.config.maximum_surface_points)
        unavailable = obj is None or getattr(obj, "status", "") in {"missing", "ambiguous", "conflict", "stale", "unobserved"}
        if unavailable or len(points) < 6:
            if required:
                raise GroundingError("contact_region_unresolved", {"region": object_id,
                    "parent": operation.get("object"), "whole_object_fallback": False})
            return object_id, obj, np.empty((0, 3))
        return object_id, obj, points

    def _contacts(self, operation, scene, state8):
        region_id, obj, points = self._region(operation, scene)
        sigma = object_sigma(obj)
        center = np.median(points, axis=0)
        # Keep the target on actual selected material, not in the hole of a loop.
        _, index = cKDTree(points).query(center)
        material = points[int(index)]
        radii = np.linalg.norm(points-material, axis=1)
        local = points[radii <= max(self.config.pad_depth_m, float(np.quantile(radii, .35)))]
        if len(local) < 6:
            local = points
        _, _, axes = np.linalg.svd(local-np.mean(local, axis=0), full_matrices=False)
        approach = unit(operation.get("approach_direction"), [0., 0., -1.])
        jaw = operation.get("jaw_axis")
        if jaw is None:
            # Pinch across the selected material's dominant tangent rather than
            # along it. Candidate rotations cover degenerate/symmetric patches.
            jaw = np.cross(axes[0], approach)
            if np.linalg.norm(jaw) < 1e-7:
                jaw = axes[1]
        R = _basis(approach, jaw)
        candidates = []
        contact_depths = (self.config.fingertip_forward_m-self.config.numerical_margin_m,
                          self.config.fingertip_forward_m-.5*self.config.pad_depth_m)
        offsets = [np.zeros(3)]
        # Material samples are local alternatives; no class-specific rim/handle
        # dimensions or hidden target centers are used.
        for sign in (-1., 1.):
            target = material + sign * axes[0] * self.config.pad_depth_m * .25
            _, i = cKDTree(points).query(target)
            offsets.append(points[int(i)]-material)
        for angle in (0., np.pi/2, -np.pi/2):
            candidate_R = rotation_matrix(approach*angle) @ R
            width = float(np.ptp(local @ candidate_R[:, 0]))
            if width > self.config.max_gap_m + 2*float(np.linalg.norm(sigma)):
                continue
            for contact_depth in contact_depths:
                for offset in offsets:
                    pose = np.eye(4)
                    pose[:3, :3] = candidate_R
                    pose[:3, 3] = material + offset - approach*contact_depth
                    candidates.append(pose)
        if not candidates:
            raise GroundingError("contact_patch_exceeds_jaw_span", {"region": region_id})
        current_R = pose_from_state(state8)[:3, :3]
        candidates.sort(key=lambda p: np.linalg.norm(rotation_vector(p[:3, :3] @ current_R.T)))
        return candidates, {"region": region_id, "material_point_m": material.tolist(),
            "sigma_m": sigma.tolist(), "region_points": len(points),
            "contact_frame_source": "observed_local_surface_and_tool_geometry",
            "candidate_count": len(candidates), "complete_object_shape": False}

    def _beside_destination(self, operation, scene, relative_payload):
        """Ground a side preference using a distinct observed support surface."""
        neighbour=scene.objects.get(operation.get("target"))
        neighbour_points=observed_points(neighbour)
        offset=np.asarray(operation.get("offset_m",[0.,0.,0.]),float)
        lateral=offset.copy();lateral[2]=0.
        if len(neighbour_points)<6 or getattr(neighbour,"status","") in {"unobserved","stale","missing","ambiguous","conflict"}:
            raise GroundingError("beside_neighbour_unresolved")
        if np.linalg.norm(lateral)<1e-6:
            raise GroundingError("beside_direction_required")
        direction=lateral/np.linalg.norm(lateral)
        neighbour_center=np.median(neighbour_points,axis=0)
        payload_center=.5*(relative_payload.min(0)+relative_payload.max(0))
        forward=float(np.max((neighbour_points-neighbour_center)@direction))
        backward=float(-np.min((relative_payload-payload_center)@direction))
        options=[]
        for support_id,support in scene.objects.items():
            if support_id in {operation.get("object"),operation.get("target")} or support.evidence.get("role")!="support":
                continue
            points=observed_points(support)
            if len(points)<6 or support.status in {"unobserved","stale","missing","ambiguous","conflict"}:
                continue
            sigma=object_sigma(support)
            try:
                center,normal,axes,hull=_support_frame(points,sigma,support.evidence)
            except GroundingError:
                continue
            # Supports above the neighbouring object's visible bottom do not
            # establish a surface beside that object.
            plane_z=float(center[2])
            if plane_z>float(neighbour_points[:,2].min())+2*float(sigma[2]):
                continue
            margin=self.config.numerical_margin_m+2*float(np.max(sigma+object_sigma(neighbour)))
            distance=max(float(np.linalg.norm(lateral)),forward+backward+margin)
            desired=neighbour_center+direction*distance
            desired-=normal*float((desired-center)@normal)
            support_xy=(desired-center)@axes
            if np.max(support_xy@hull.equations[:,:2].T+hull.equations[:,2])>0:
                continue
            options.append((abs(float(neighbour_points[:,2].min())-plane_z),support_id,support,
                            desired,direction,neighbour_center))
        if not options:
            raise GroundingError("beside_support_unresolved",{"required":"observed support-role surface"})
        _,support_id,support,desired,direction,neighbour_center=min(options,key=lambda item:item[0])
        return support_id,support,desired,direction,neighbour_center

    def _placement(self, operation, scene, state8, held, phase):
        target_id = operation.get("target")
        obj = scene.objects.get(target_id)
        points = observed_points(obj, self.config.maximum_surface_points)
        if obj is None or len(points) < 6 or getattr(obj,"status","") in {"missing","stale","unobserved","ambiguous","conflict"}:
            raise GroundingError("placement_region_unresolved", {"target": target_id})
        if held is None or held.get("extent_unknown"):
            raise GroundingError("payload_extent_unknown")
        local_payload = np.asarray(held.get("points_in_gripper", []), float).reshape(-1, 3)
        if len(local_payload) < 4:
            raise GroundingError("payload_extent_unknown")
        relation = operation.get("relation", operation.get("placement_relation"))
        pose = pose_from_state(state8)
        relative = local_payload @ pose[:3, :3].T
        support_id=target_id
        desired_center=None
        if relation=="beside":
            support_id,obj,desired_center,_,_=self._beside_destination(operation,scene,relative)
            points=observed_points(obj,self.config.maximum_surface_points)
        evidence = getattr(obj, "evidence", {}) or {}
        sigma = object_sigma(obj)
        center, normal, axes, hull = _support_frame(points, sigma, evidence)
        opening = evidence.get("opening")
        if relation == "inside":
            if not opening:
                # A top opening can be inferred only when observed raised wall
                # material surrounds the selected support patch. Front openings
                # require explicit observed opening geometry from perception.
                parent = scene.objects.get(evidence.get("parent_id")) or obj
                wall_points = observed_points(parent)
                raised = wall_points[(wall_points-center)@normal > max(self.config.pad_depth_m,2*float(sigma@np.abs(normal)))]
                if len(raised)>=12:
                    wall_xy=(raised-center)@axes
                    angles=np.sort(np.arctan2(wall_xy[:,1],wall_xy[:,0]))
                    angular_gap=float(np.max(np.diff(np.r_[angles,angles[0]+2*np.pi])))
                    if angular_gap<np.pi:
                        opening={"observed_walls":True,"polygon_world":raised.tolist(),
                                 "normal":normal.tolist(),"source":"observed_surrounding_wall_patches"}
            if not opening or not opening.get("observed_walls"):
                raise GroundingError("inside_geometry_unavailable", {"target": target_id,
                    "required": "observed_support_and_opening_with_walls", "whole_box_contact_exemption": False})
            polygon = np.asarray(opening.get("polygon_world", []), float).reshape(-1, 3)
            if len(polygon) < 3:
                raise GroundingError("inside_opening_unresolved")
            try:
                hull = ConvexHull((polygon-center) @ axes)
            except QhullError as exc:
                raise GroundingError("inside_opening_unresolved") from exc
        payload_sigma = np.asarray(held.get("sigma_m", 0.), float)
        margin = self.config.numerical_margin_m + float(np.max(sigma)) + float(np.max(payload_sigma))
        offset = np.asarray(operation.get("offset_m", [0., 0., 0.]), float)
        # Centre the complete observed footprint, including off-axis payloads
        # (a handle pinch must not centre the grip site on the plate).
        footprint_center = .5*(relative.min(0)+relative.max(0))
        position = (center - footprint_center + offset) if desired_center is None else (desired_center-footprint_center)
        bottom = float(np.min(relative @ normal))
        position += normal*(float(center @ normal)-float(position @ normal)-bottom)
        footprint = (relative+position-center) @ axes
        distances = footprint @ hull.equations[:, :2].T + hull.equations[:, 2]
        if np.any(distances > -margin):
            raise GroundingError("payload_does_not_fit_observed_support", {"target": target_id,
                "margin_m": margin, "max_boundary_residual_m": float(distances.max()),
                "region_only": True})
        if phase in {"transport", "preplace"}:
            approach = unit(operation.get("approach_direction"), -normal)
            position -= approach*(self.config.palm_back_m*self.config.approach_tool_lengths + 2*margin)
        else:
            # Visible-surface contact with the permitted support is intentional.
            position += normal*self.config.numerical_margin_m
        pose[:3, 3] = position
        return pose, {"target": target_id, "relation": relation, "support_normal": normal.tolist(),
            "support_point": center.tolist(), "sigma_m": sigma.tolist(), "footprint_checked": True,
            "contact_parts": ["payload"], "payload_support_ids": [support_id],
            "payload_contact_plane": {"normal":normal.tolist(),"point":center.tolist(),
                "thickness_m":max(self.config.numerical_margin_m,float(sigma@np.abs(normal)))}}

    def check_release_relation(self, operation, scene, object_points):
        """Check observed placement geometry, never physical contact truth.

        ``supported`` means the visible geometry supports the requested relation;
        release and temporal stability must be established separately by the
        executor. Missing/occluded support remains ``unknown``.
        """
        target_id=operation.get("target")
        target=scene.objects.get(target_id)
        points=observed_points(target)
        payload=np.asarray(object_points,float).reshape(-1,3)
        payload=payload[np.isfinite(payload).all(1)]
        result={"status":"unknown","target":target_id,"relation":operation.get("relation"),
                "source":"observed_predicted_geometry","physical_contact_proven":False}
        if target is None or len(points)<6 or len(payload)<4 or getattr(target,"status","") in {
                "missing","stale","unobserved","ambiguous","conflict"}:
            return {**result,"reason":"current_release_geometry_unavailable"}
        relation=operation.get("relation",operation.get("placement_relation"))
        side_details=None
        if relation=="beside":
            try:
                support_id,target,_,direction,neighbour_center=self._beside_destination(
                    operation,scene,payload-np.median(payload,axis=0))
                points=observed_points(target)
                neighbour_points=observed_points(scene.objects[operation["target"]])
                separation=float(np.min(payload@direction)-np.max(neighbour_points@direction))
                side_details={"support_id":support_id,"side_clearance_m":separation}
            except GroundingError as exc:
                return {**result,"reason":exc.reason}
        sigma=object_sigma(target)
        source=scene.objects.get(operation.get("object"))
        source_sigma=object_sigma(source) if source is not None else sigma
        try:
            center,normal,axes,hull=_support_frame(points,sigma,getattr(target,"evidence",{}))
        except GroundingError as exc:
            return {**result,"reason":exc.reason}
        if relation=="inside":
            opening=target.evidence.get("opening")
            if not opening or not opening.get("observed_walls"):
                parent=scene.objects.get(target.evidence.get("parent_id")) or target
                walls=observed_points(parent)
                raised=walls[(walls-center)@normal>max(self.config.pad_depth_m,2*float(sigma@np.abs(normal)))]
                if len(raised)<12:
                    return {**result,"reason":"inside_geometry_unavailable"}
                xy=(raised-center)@axes
                angles=np.sort(np.arctan2(xy[:,1],xy[:,0]))
                if np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))>=np.pi:
                    return {**result,"reason":"inside_geometry_unavailable"}
                polygon=raised
            else:
                polygon=np.asarray(opening.get("polygon_world",[]),float).reshape(-1,3)
            if len(polygon)<3:
                return {**result,"reason":"inside_opening_unresolved"}
            try:
                hull=ConvexHull((polygon-center)@axes)
            except QhullError:
                return {**result,"reason":"inside_opening_unresolved"}
        elif relation not in {"on", "beside"}:
            return {**result,"reason":"placement_relation_unavailable"}
        height=(payload-center)@normal
        uncertainty=self.config.numerical_margin_m+2*float((sigma+source_sigma)@np.abs(normal))
        xy=(payload-center)@axes
        boundary=xy@hull.equations[:,:2].T+hull.equations[:,2]
        horizontal_uncertainty=2*float(np.linalg.norm(sigma[:2]+source_sigma[:2]))+self.config.numerical_margin_m
        details={"lowest_visible_height_m":float(height.min()),
                 "max_footprint_residual_m":float(boundary.max()),
                 "height_uncertainty_m":uncertainty,"footprint_uncertainty_m":horizontal_uncertainty}
        if side_details is not None:
            details.update(side_details)
            if side_details["side_clearance_m"] < -horizontal_uncertainty:
                return {**result,**details,"status":"contradicted","reason":"beside_direction_or_separation_contradicted"}
        if height.min() < -uncertainty or boundary.max()>horizontal_uncertainty:
            return {**result,**details,"status":"contradicted","reason":"observed_geometry_outside_support"}
        if height.min()>uncertainty:
            return {**result,**details,"reason":"support_gap_or_unseen_bottom"}
        return {**result,**details,"status":"supported","reason":"released_geometry_matches_support_region"}

    def ground(self, operation, scene, state8, q7, *, phase, held=None):
        state = np.asarray(state8, float)
        pose = pose_from_state(state)
        meta = {"phase": phase, "pose_source": "measured_region_geometry",
                "scene_revision": scene.evidence.get("revision", scene.frame_id),
                "contact_parts": [], "stop_on": operation.get("stop_on", "goal")}
        contacts = tuple(dict.fromkeys(x for x in (operation.get("object"), operation.get("grasp_region")) if x))
        profile, gripper, allowed = "transit", "keep", ()
        if phase in {"approach", "align", "close"}:
            try:
                candidates, details = self._contacts(operation, scene, state)
            except GroundingError:
                if phase != "approach":
                    raise
                parent = scene.objects.get(operation.get("object"))
                points = observed_points(parent)
                if len(points) < 6:
                    raise
                approach = unit(operation.get("approach_direction"), [0., 0., -1.])
                pose[:3, 3] = np.median(points, axis=0)-approach*(self.config.palm_back_m*3 + 2*np.max(object_sigma(parent)))
                meta.update(viewpoint_only=True, unresolved_region=operation.get("grasp_region"),
                            automatic_close_allowed=False)
                return KeyPose(phase, pose, "open", "transit", (), meta)
            pose = candidates[0].copy()
            meta.update(details)
            if phase == "approach":
                standoff = self.config.palm_back_m*self.config.approach_tool_lengths + 2*max(details["sigma_m"])
                for candidate in candidates:
                    candidate[:3, 3] -= candidate[:3, 2]*standoff
                pose = candidates[0].copy()
                gripper = "open"
            else:
                profile = "contact"
                allowed = contacts
                meta["contact_parts"] = ["finger", "pad"]
                gripper = "close" if phase == "close" else "open"
            meta["candidate_poses"] = [candidate.tolist() for candidate in candidates]
        elif phase in {"transport", "preplace", "place"}:
            pose, details = self._placement(operation, scene, state, held, phase)
            meta.update(details)
            if phase == "place":
                allowed = tuple(details["payload_support_ids"])
                profile = "contact"
        elif phase == "lift":
            if held is None or held.get("extent_unknown"):
                raise GroundingError("payload_extent_unknown")
            clearance = self.config.palm_back_m*self.config.lift_tool_lengths + 2*float(np.max(held.get("sigma_m", 0.)))
            pose[:3, 3] += np.array([0., 0., clearance])
            meta.update(contact_parts=["finger", "pad"], payload_lift=True,
                        pick_support_ids=list(held.get("support_ids", [])), keep_upright=operation.get("keep_upright", False))
            allowed = contacts
        elif phase == "release":
            gripper, profile = "open", "contact"
            allowed = contacts
            meta.update(contact_parts=["finger", "pad"], release=True)
            meta["contact_permissions"] = {owner:["finger", "pad"] for owner in contacts}
            # Release keeps only the observed support contact permission. A
            # container identity never exempts its walls from payload checks.
            support_id = operation.get("target")
            support = scene.objects.get(support_id)
            if held is not None:
                try:
                    if operation.get("relation") == "beside":
                        relative = np.asarray(held.get("points_in_gripper", []), float).reshape(-1, 3) @ pose[:3, :3].T
                        support_id, support, _, _, _ = self._beside_destination(operation, scene, relative)
                    if support is not None and support.status not in {"unobserved", "stale", "missing", "ambiguous", "conflict"}:
                        points, sigma = observed_points(support), object_sigma(support)
                        center, normal, _, _ = _support_frame(points, sigma, support.evidence)
                        allowed = tuple(dict.fromkeys(contacts+(support_id,)))
                        meta["contact_parts"].append("payload")
                        meta["contact_permissions"].setdefault(support_id, []).append("payload")
                        meta["payload_contact_plane"] = {"normal":normal.tolist(), "point":center.tolist(),
                            "thickness_m":max(self.config.numerical_margin_m, float(sigma@np.abs(normal)))}
                except (GroundingError, ValueError):
                    meta["support_contact_permission"] = "unresolved"
        elif phase == "retreat":
            approach = unit(operation.get("approach_direction"), [0., 0., -1.])
            pose[:3, 3] -= approach*self.config.palm_back_m*self.config.approach_tool_lengths
            # Leaving a just released object permits fingertip separation only;
            # other contacts continue to be checked and cannot worsen.
            allowed, meta["contact_parts"] = contacts, ["finger", "pad"]
        elif phase in {"move", "contact_move"}:
            displacement = operation.get("displacement")
            target = operation.get("target")
            if displacement is not None:
                pose[:3, 3] += np.asarray(displacement, float)
                meta["relative_origin_m"] = state[:3].tolist()
            elif target:
                obj = scene.objects.get(target)
                points = observed_points(obj)
                if len(points) < 6:
                    raise GroundingError("motion_anchor_unresolved", {"target": target})
                pose[:3, 3] = np.median(points, axis=0)+np.asarray(operation.get("offset_m", [0., 0., 0.]))
            elif phase == "move":
                raise GroundingError("motion_anchor_required")
            axis, angle = operation.get("rotation_axis"), operation.get("rotation_angle_rad")
            if axis is not None and angle is not None:
                pose[:3, :3] = rotation_matrix(unit(axis)*float(angle)) @ pose[:3, :3]
            if phase == "contact_move":
                self._region(operation, scene)  # preserve explicit material identity
                allowed, profile = contacts, "contact"
                raw = operation.get("contact_parts", ["fingers", "pads"])
                meta["contact_parts"] = [{"fingers": "finger", "pads": "pad", "palm": "tool"}.get(x, x) for x in raw]
                meta["contact_region"] = operation.get("grasp_region")
        else:
            raise GroundingError("unsupported_phase", {"phase": phase})
        meta["keep_upright"] = bool(operation.get("keep_upright") and held is not None)
        return KeyPose(phase, pose, gripper, profile, allowed, meta)
