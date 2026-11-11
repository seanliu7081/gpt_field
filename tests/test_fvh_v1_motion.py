"""Focused CPU regressions for the new geometry/control contract.

Fixtures are synthetic software scenes, not demos or benchmark-policy inputs.
No network, model weights, physics simulator or rollout is used.
"""
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from gpt_field.foundation_harness.types import ObjectGeometry, SceneSnapshot, KeyPose
from gpt_field.foundation_harness.robot_geometry import CollisionShape
from gpt_field.foundation_vision_harness_V1.geometry import Grounder, GroundingError
from gpt_field.foundation_vision_harness_V1.motion import RollingPlanner, MotionConfig, PlanningError
from gpt_field.robodojo_direct_sim import rotation_matrix, rotation_vector


def cube(center, half):
    c=np.asarray(center); h=np.broadcast_to(half,(3,))
    return np.array([c+[x,y,z] for x in (-h[0],h[0]) for y in (-h[1],h[1]) for z in (-h[2],h[2])])


def obj(name, points, **evidence):
    points=np.asarray(points,float)
    return ObjectGeometry(name,name,points,np.zeros_like(points),"observed",evidence={"sigma_m":[.001]*3,**evidence})


def scene(objects=None, points=(), owners=None, sigma=0.):
    points=np.asarray(points,float).reshape(-1,3)
    return SceneSnapshot(1,perf_counter(),perf_counter(),objects or {},points,None,None,
        evidence={"revision":1,"point_sigma_m":np.full(len(points),sigma)},
        scene_point_object_ids=np.full(len(points),"",object) if owners is None else np.asarray(owners,object))


def operation(**updates):
    result={"kind":"pick_place","object":"body","target":"support","grasp_region":"region",
        "relation":"on","keep_upright":True,"approach_direction":[0.,0.,-1.],"jaw_axis":[1.,0.,0.],
        "offset_m":[0.,0.,0.],"displacement":None,"rotation_axis":None,"rotation_angle_rad":None,
        "contact_parts":["fingers","pads"],"stop_on":"goal"}
    result.update(updates); return result


def state(position=(0.,0.,.4)):
    return np.r_[position,[0.,0.,0.,.04,-.04]]


class CartesianChain:
    """Analytic test actuator; first 3 joints translate, next 3 rotate."""
    low=np.full(7,-5.); high=np.full(7,5.)
    def bind(self,state,q): return self
    def forward(self,q):
        pose=np.eye(4); pose[:3,3]=q[:3];pose[:3,:3]=rotation_matrix(q[3:6])
        return pose,np.column_stack((np.eye(6),np.zeros(6)))
    def solve(self,position,orientation,seed,max_iterations=60):
        return SimpleNamespace(ok=True,joints=np.r_[position,rotation_vector(orientation),0.],position_error_m=0.)


class ToolOnlyRobot:
    def __init__(self,half=.01,kind="tool"):
        self.vertices=cube([0,0,0],half);self.equations=ConvexHull(self.vertices).equations;self.kind=kind
    def shapes(self,bound,q,fingers):
        pose,_=bound.forward(q)
        return [CollisionShape("test_tool",8,self.vertices,self.equations,pose,self.kind)]
    @staticmethod
    def self_collision(shapes): return None


def planner(kind="tool",**config):
    return RollingPlanner(CartesianChain(),MotionConfig(voxel_m=.001,collision_margin_m=.001,
        joint_error_floor_rad=0.,**config),robot_model=ToolOnlyRobot(kind=kind))


def test_missing_part_can_approach_but_never_closes_at_parent_center():
    grounder=Grounder()
    s=scene({"body":obj("body",cube([.2,0,.1],.02))})
    approach=grounder.ground(operation(),s,state(),np.zeros(7),phase="approach")
    assert approach.metadata["viewpoint_only"] and approach.gripper=="open"
    with pytest.raises(GroundingError,match="contact_region_unresolved"):
        grounder.ground(operation(),s,state(),np.zeros(7),phase="close")
    s.objects["region"]=obj("region",cube([.2,0,.1],.005));s.objects["region"].status="unobserved"
    with pytest.raises(GroundingError,match="contact_region_unresolved"):
        grounder.ground(operation(),s,state(),np.zeros(7),phase="align")


def test_contact_frame_keeps_selected_material_and_explicit_permissions():
    points=cube([.3,0,.15],[.006,.01,.003])
    s=scene({"body":obj("body",cube([0,0,.1],.05)),"region":obj("region",points,parent_id="body")})
    pose=Grounder().ground(operation(),s,state(),np.zeros(7),phase="align")
    assert pose.pose[0,3]>.28
    assert pose.allowed_contacts==("body","region")
    assert pose.metadata["contact_parts"]==["finger","pad"]
    assert pose.metadata["candidate_count"]>1


def test_placement_centres_off_axis_payload_and_rejects_unknown_extent():
    xy=np.array([[x,y,.1]for x in np.linspace(-.15,.15,9)for y in np.linspace(-.15,.15,9)])
    s=scene({"support":obj("support",xy,role="support")})
    payload={"object_id":"body","points_in_gripper":cube([.07,0,-.05],[.02,.02,.03]),"sigma_m":.001}
    result=Grounder().ground(operation(),s,state(),np.zeros(7),phase="place",held=payload)
    assert result.pose[0,3]==pytest.approx(-.07)
    assert result.pose[2,3]==pytest.approx(.183)
    assert result.metadata["footprint_checked"]
    payload["extent_unknown"]=True
    with pytest.raises(GroundingError,match="payload_extent_unknown"):
        Grounder().ground(operation(),s,state(),np.zeros(7),phase="transport",held=payload)


def test_inside_cannot_be_granted_by_a_filled_box_permission():
    xy=np.array([[x,y,.1]for x in np.linspace(-.15,.15,9)for y in np.linspace(-.15,.15,9)])
    s=scene({"support":obj("support",xy)})
    payload={"object_id":"body","points_in_gripper":cube([0,0,-.05],.02)}
    with pytest.raises(GroundingError,match="inside_geometry_unavailable"):
        Grounder().ground(operation(relation="inside"),s,state(),np.zeros(7),phase="place",held=payload)


def test_single_view_surface_is_not_silently_traversable():
    p=planner();x=state([0,0,0]);q=np.zeros(7)
    s=scene(points=[[.1,0,0]],owners=["obstacle"])
    p._prepare(s,x,q)
    goal=np.eye(4);goal[0,3]=.1
    checked=p._check(np.array([.1,0,0,0,0,0,0]),x[6:8],KeyPose("move",goal),None)
    assert not checked["safe"]
    assert checked["examples"][0]["point_object_id"]=="obstacle"


def test_held_membership_never_swallows_a_neighbour_inside_its_envelope():
    p=planner();x=state([0,0,0]);q=np.zeros(7)
    points=[[.1,0,0],[.105,0,0]]
    s=scene(points=points,owners=["body","neighbour"])
    p._prepare(s,x,q)
    held={"object_id":"body","points_in_gripper":cube([.1,0,0],.02)}
    checked=p._check(q,x[6:8],KeyPose("transport",np.eye(4)),held)
    assert not checked["safe"]
    assert any(e.get("point_object_id")=="neighbour"for e in checked["examples"])
    s=scene(points=points[:1],owners=["body"])
    p._prepare(s,x,q)
    assert p._check(q,x[6:8],KeyPose("transport",np.eye(4)),held)["safe"]


def test_finger_contact_permission_does_not_exempt_the_palm():
    p=planner();x=state([0,0,0]);q=np.zeros(7)
    s=scene(points=[[0,0,0]],owners=["body"]);p._prepare(s,x,q)
    key=KeyPose("align",np.eye(4),allowed_contacts=("body",),metadata={"contact_parts":["finger","pad"]})
    assert not p._check(q,x[6:8],key,None)["safe"]
    fingers=planner(kind="finger");fingers._prepare(s,x,q)
    assert fingers._check(q,x[6:8],key,None)["safe"]


def test_continuous_command_and_scene_cache_do_not_wait_for_dense_waypoints():
    p=planner();x=state([0,0,0]);q=np.zeros(7);s=scene()
    goal=np.eye(4);goal[0,3]=.3
    path=p.plan(KeyPose("move",goal),s,x,q)
    result=p.command(path,s,x,q,np.zeros(7))
    assert result["blocked"] is None
    assert result["delta_xyz"][0]>.02
    assert len(path.waypoints)==2
    p.command(path,s,x,q,np.zeros(7))
    assert p.stats["scene_preparations"]==1


def test_escape_predicate_requires_no_new_or_deeper_conflicts():
    start={"conflicts":{"tool|a":.01}}
    assert RollingPlanner._nonworsening(start,{"conflicts":{"tool|a":.005}})
    assert not RollingPlanner._nonworsening(start,{"conflicts":{"tool|a":.02}})
    assert not RollingPlanner._nonworsening(start,{"conflicts":{"tool|b":.001}})


def test_release_relation_uses_current_support_and_reports_geometry_only():
    xy=np.array([[x,y,.1]for x in np.linspace(-.15,.15,9)for y in np.linspace(-.15,.15,9)])
    s=scene({"support":obj("support",xy),"body":obj("body",cube([0,0,.13],.03))})
    grounder=Grounder();op=operation()
    supported=grounder.check_release_relation(op,s,s.objects["body"].points)
    assert supported["status"]=="supported" and not supported["physical_contact_proven"]
    contradicted=grounder.check_release_relation(op,s,cube([.5,0,.13],.03))
    assert contradicted["status"]=="contradicted"
    s.objects["support"].status="unobserved"
    assert grounder.check_release_relation(op,s,s.objects["body"].points)["status"]=="unknown"


def test_beside_uses_a_distinct_support_and_measured_object_extents():
    floor=np.array([[x,y,.1]for x in np.linspace(-.4,.4,12)for y in np.linspace(-.4,.4,12)])
    s=scene({"table":obj("table",floor,role="support"),
             "neighbour":obj("neighbour",cube([0,0,.13],.03))})
    payload={"object_id":"body","points_in_gripper":cube([0,0,-.05],.02)}
    op=operation(target="neighbour",relation="beside",offset_m=[.01,0,0])
    g=Grounder();p=g.ground(op,s,state(),np.zeros(7),phase="place",held=payload)
    assert p.pose[0,3]>.05 and p.allowed_contacts==("table",)
    world=payload["points_in_gripper"]@p.pose[:3,:3].T+p.pose[:3,3]
    assert g.check_release_relation(op,s,world)["status"]=="supported"
    with pytest.raises(GroundingError,match="beside_direction_required"):
        g.ground({**op,"offset_m":[0,0,0]},s,state(),np.zeros(7),phase="place",held=payload)


def test_release_support_permission_keeps_container_walls_and_fingers_checked():
    floor=np.array([[x,y,0.]for x in np.linspace(.05,.15,6)for y in np.linspace(-.05,.05,6)])
    payload={"object_id":"body","points_in_gripper":cube([.1,0,.025],[.015,.015,.025])}
    x=state([0,0,0]);q=np.zeros(7)
    s=scene({"support":obj("support",floor)},points=[[.1,0,.025]],owners=["support"])
    key=Grounder().ground(operation(relation="inside"),s,x,q,phase="release",held=payload)
    assert key.metadata["contact_permissions"]["support"]==["payload"]
    p=planner();p._prepare(s,x,q)
    assert not p._check(q,x[6:8],key,payload)["safe"]  # wall above allowed plane
    s=scene(s.objects,points=[[.1,0,0]],owners=["support"]);p._prepare(s,x,q)
    assert p._check(q,x[6:8],key,payload)["safe"]  # observed support plane
    fingers=planner(kind="finger");s=scene(s.objects,points=[[0,0,0]],owners=["support"])
    fingers._prepare(s,x,q)
    assert not fingers._check(q,x[6:8],key,None)["safe"]
