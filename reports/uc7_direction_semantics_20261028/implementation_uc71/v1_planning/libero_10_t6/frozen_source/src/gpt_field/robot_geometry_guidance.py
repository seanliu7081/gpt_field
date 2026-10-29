"""Optional prompt-only description of the installed Panda's robot geometry.

No scene, camera, simulator, observation or network access occurs here. The
definitions are shared text, so a future matched Direct arm can use exactly the
same information. They do not alter a controller or supply object geometry.
"""
from __future__ import annotations

import hashlib

from .robot_tool_geometry import GEOMETRY_VERSION, ROBOT_ASSET_PROVENANCE


ROBOT_GEOMETRY_GUIDANCE_VERSION = "panda-grasp-frame-v1"


def robot_geometry_instructions() -> str:
    return (
        f"Robot geometry clarification ({ROBOT_GEOMETRY_GUIDANCE_VERSION}; known Panda robot assets only). "
        "Let p be the measured state[0:3] in world meters. It is already the grip_site position "
        "near the finger gap, NOT the wrist/hand-body origin. Let R be the world rotation of the "
        "right_hand body, obtained from the measured state[3:6] rotation vector (axis-angle, "
        "not three Euler angles). When Field is used, orientation_target_xyzw is a unit "
        "quaternion [x,y,z,w] for this SAME right_hand orientation; goal_offset_m specifies "
        "a displacement of grip_site from p. The simulator's grip_site orientation differs "
        "from right_hand by a fixed local -90-degree Z rotation. Do not apply that rotation "
        "to the reported state or the requested Field quaternion; the existing world-delta "
        "controller already preserves the stated convention. "
        "The hand-body origin is p - R*[0,0,0.097] meters. Do not add another 0.097 m "
        "wrist-to-tip offset to p or to a grip_site waypoint. In right_hand coordinates, "
        "+Z extends from the hand toward the fingertips, and the fingers open/close along "
        "the Y axis. Their world directions are R*[0,0,1] and R*[0,1,0]; they are not fixed "
        "world down or image horizontal. Changing orientation rotates the finger approach "
        "direction and the opening axis as well as any fixed offset. "
        "Let q1=state[6] and q2=state[7] be the measured finger slide positions in meters. "
        "Their nominal fully open values are +0.04 and -0.04; closing moves both toward zero. "
        "q1-q2 is a joint-separation proxy, not an object size or a measured contact gap. "
        "The two nominal finger-pad box centers are p + R*[0,q1+0.0035,-0.0036] and "
        "p + R*[0,q2-0.0035,-0.0036]. Their midpoint is p + R*[0,(q1+q2)/2,-0.0036]. "
        "These are robot geometry references, not observed object contact points. Align the "
        "intended interaction region with the space between the pads and the appropriate "
        "jaw orientation; reaching a grip_site waypoint alone does not establish engagement. "
        "Direct translation and rotation increments remain in WORLD axes, with the existing "
        "0.05 m and 0.5 rad per unit scaling; do not rotate them as if they were tool-frame "
        "increments. Use the measured robot feedback to assess actual response. This geometry "
        "provides no camera calibration, depth, object pose, obstacle clearance or grasp proof."
    )


def robot_geometry_metadata() -> dict:
    return {
        "version": ROBOT_GEOMETRY_GUIDANCE_VERSION,
        "text_sha256": hashlib.sha256(robot_geometry_instructions().encode()).hexdigest(),
        "source": "known_robot_assets_only",
        "robot_geometry_version": GEOMETRY_VERSION,
        "robot_asset_provenance": [dict(item) for item in ROBOT_ASSET_PROVENANCE],
        "scope": "static_prompt_only",
        "extra_observation_fields": [],
        "controller_changed": False,
    }
